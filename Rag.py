from fastapi import FastAPI
from fastapi import BackgroundTasks
from pydantic import BaseModel
import os
from fastapi.middleware.cors import CORSMiddleware
import boto3
from datetime import datetime

from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA
import openai
from pymongo import MongoClient



mongo_uri = os.getenv("MONGODB_ATLAS_URI")
mongo_db = "qna_logs"
mongo_collection = "qa_archive"

mongo_client = MongoClient(mongo_uri)
db = mongo_client[mongo_db]
collection = db[mongo_collection]


openai.api_key = os.getenv("OPENAI_API_KEY")
s3_bucket = os.getenv("S3_BUCKET_NAME")

s3_client = boto3.client("s3")


loader = PyPDFLoader("resume.pdf")
documents = loader.load()
for doc in documents:
    doc.metadata["source"] = "resume"
    
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
docs = text_splitter.split_documents(documents)


embeddings = OpenAIEmbeddings()
vectorstore = FAISS.from_documents(docs, embeddings)
llm = ChatOpenAI(temperature=0)
qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=None)


app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  
    allow_credentials=False,
    allow_methods=["*"],  
    allow_headers=["*"], 
)

class QueryRequest(BaseModel):
    query: str

def log_to_s3_and_update_embeddings(question: str, answer: str):
    filename = "q&a.txt"
    local_path = f"/tmp/{filename}"

    try:
        s3_client.download_file(s3_bucket, filename, local_path)
    except s3_client.exceptions.NoSuchKey:
        open(local_path, "w").close()

    with open(local_path, "r") as f:
        content = f.read().strip()
        qna_pairs = content.split("\n\n") if content else []

    if len(qna_pairs) >= 10:
        to_offload = qna_pairs[:8]
        remaining = qna_pairs[8:]

        for entry in to_offload:
            lines = entry.split("\n")
            q, a = lines[0].replace("Question: ", ""), lines[1].replace("Answer: ", "")
            collection.insert_one({
                "question": q,
                "answer": a,
                "archived_at": datetime.utcnow()
            })

        with open(local_path, "w") as f:
            f.write("\n\n".join(remaining))

    with open(local_path, "a") as f:
        f.write(f"Question: {question}\n")
        f.write(f"Answer: {answer}\n\n")  
    with open(local_path, "rb") as f:
        s3_client.upload_fileobj(f, s3_bucket, filename)

    loader = TextLoader(local_path)
    new_docs = loader.load()
    new_splits = text_splitter.split_documents(new_docs)

    for doc in new_splits:
        doc.metadata["source"] = "log"
        
    vectorstore.add_documents(new_splits)


@app.post("/ask")
def ask_question(request: QueryRequest, background_tasks: BackgroundTasks):

    resume_docs = vectorstore.similarity_search(request.query, k=4, filter={"source": "resume"})
    log_docs = vectorstore.similarity_search(request.query, k=1, filter={"source": "log"})

    combined_docs = resume_docs + log_docs  
    response = qa_chain.combine_documents_chain.run({
        "input_documents": combined_docs,
        "question": request.query
    })
    
    background_tasks.add_task(log_to_s3_and_update_embeddings, request.query, response)

    return {"answer": response}
