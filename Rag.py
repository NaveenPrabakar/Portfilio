from fastapi import FastAPI, BackgroundTasks
from pydantic import BaseModel
import os
from fastapi.middleware.cors import CORSMiddleware
import boto3
from botocore.exceptions import ClientError
from datetime import datetime

from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.chat_models import ChatOpenAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
import openai

from pymongo import MongoClient


# ========== SYSTEM PROMPT SETUP ==========

system_prompt = (
    "You are a career assistant who gives helpful, concise answers about Naveen Prabakar "
    "based on his resume and prior questions. Any irrelevant question asked should be dismissed professionally."
)

system_msg = SystemMessagePromptTemplate.from_template(system_prompt)

# 🔧 Inject both question and context into the prompt
human_msg = HumanMessagePromptTemplate.from_template(
    "Given the following context:\n{context}\n\nAnswer the question:\n{question}"
)
chat_prompt = ChatPromptTemplate.from_messages([system_msg, human_msg])


# ========== ENVIRONMENT SETUP ==========

mongo_uri = os.getenv("MONGODB_ATLAS_URI")
mongo_db = "qna_logs"
mongo_collection = "qa_archive"

mongo_client = MongoClient(mongo_uri)
db = mongo_client[mongo_db]
collection = db[mongo_collection]

openai.api_key = os.getenv("OPENAI_API_KEY")
s3_bucket = os.getenv("S3_BUCKET_NAME")
s3_client = boto3.client("s3")


# ========== LOAD RESUME & BUILD VECTORSTORE ==========

loader = PyPDFLoader("cv.pdf")
documents = loader.load()
for doc in documents:
    doc.metadata["source"] = "resume"

text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
docs = text_splitter.split_documents(documents)

embeddings = OpenAIEmbeddings()
vectorstore = FAISS.from_documents(docs, embeddings)

# Create Chat LLM with system prompt
llm = ChatOpenAI(temperature=0)
qa_chain = load_qa_chain(llm=llm, chain_type="stuff", prompt=chat_prompt)


# ========== FASTAPI SETUP ==========

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


# ========== BACKGROUND TASK TO LOG TO S3 AND RE-EMBED ==========

def log_to_s3_and_update_embeddings(question: str, answer: str):
    filename = "q&a.txt"
    local_path = f"/tmp/{filename}"

    try:
        s3_client.download_file(s3_bucket, filename, local_path)
    except ClientError as e:
        if e.response['Error']['Code'] == '404':
            open(local_path, "w").close()  # Create empty file if not exists
        else:
            raise

    # Read and parse existing Q&A log
    with open(local_path, "r") as f:
        content = f.read().strip()
        qna_pairs = content.split("\n\n") if content else []

    # Offload old entries to MongoDB if too many
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

    # Append current question and answer
    with open(local_path, "a") as f:
        f.write(f"Question: {question}\n")
        f.write(f"Answer: {answer}\n\n")

    # Upload updated file
    with open(local_path, "rb") as f:
        s3_client.upload_fileobj(f, s3_bucket, filename)

    # Re-embed updated log content
    loader = TextLoader(local_path)
    new_docs = loader.load()
    new_splits = text_splitter.split_documents(new_docs)

    for doc in new_splits:
        doc.metadata["source"] = "log"

    vectorstore.add_documents(new_splits)


# ========== MAIN Q&A ENDPOINT ==========

@app.post("/ask")
def ask_question(request: QueryRequest, background_tasks: BackgroundTasks):
    # Search relevant documents
    resume_docs = vectorstore.similarity_search(request.query, k=4, filter={"source": "resume"})
    log_docs = vectorstore.similarity_search(request.query, k=1, filter={"source": "log"})

    combined_docs = resume_docs + log_docs

    # Run chain with injected context + question
    response = qa_chain.run({
        "input_documents": combined_docs,
        "question": request.query
    })

    # Background log task
    background_tasks.add_task(log_to_s3_and_update_embeddings, request.query, response)

    return {"answer": response}
