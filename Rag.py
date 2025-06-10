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



openai.api_key = os.getenv("OPENAI_API_KEY")
s3_bucket = os.getenv("S3_BUCKET_NAME")

s3_client = boto3.client("s3")


loader = PyPDFLoader("resume.pdf")
documents = loader.load()
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
docs = text_splitter.split_documents(documents)


embeddings = OpenAIEmbeddings()
vectorstore = FAISS.from_documents(docs, embeddings)
llm = ChatOpenAI(temperature=0)
qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=vectorstore.as_retriever())


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

    with open(local_path, "a") as f:
        f.write(f"Question: {question}\n")
        f.write(f"Answer: {answer}\n\n")  
    with open(local_path, "rb") as f:
        s3_client.upload_fileobj(f, s3_bucket, filename)

    loader = TextLoader(local_path)
    new_docs = loader.load()
    new_splits = text_splitter.split_documents(new_docs)
    vectorstore.add_documents(new_splits)


@app.post("/ask")
def ask_question(request: QueryRequest, background_tasks: BackgroundTasks):
    response = qa_chain.run(request.query)
    
    background_tasks.add_task(log_to_s3_and_update_embeddings, request.query, response)

    return {"answer": response}
