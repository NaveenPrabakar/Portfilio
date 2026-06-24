from fastapi import FastAPI, BackgroundTasks, Request
from pydantic import BaseModel
import os
from fastapi.middleware.cors import CORSMiddleware
import boto3
from botocore.exceptions import ClientError
from datetime import datetime
# import redis  
import json
from uuid import uuid4
import openai
from pymongo import MongoClient

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.chat_models import ChatOpenAI
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain.prompts import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)

# ====== SYSTEM PROMPT ======
system_prompt = (
    "You are a career assistant who gives helpful, concise answers about Naveen Prabakar "
    "based on his resume and prior questions. Any irrelevant question should be dismissed professionally."
)

system_msg = SystemMessagePromptTemplate.from_template(system_prompt)

human_msg = HumanMessagePromptTemplate.from_template(
    "Here is the chat history so far:\n{context}\n\n"
    "Now answer the user's latest question:\n{question}"
)

chat_prompt = ChatPromptTemplate.from_messages([system_msg, human_msg])

# ====== ENVIRONMENT ======
openai.api_key = os.getenv("OPENAI_API_KEY")

mongo_client = MongoClient(os.getenv("MONGODB_ATLAS_URI"))
collection = mongo_client["qna_logs"]["qa_archive"]

s3_client = boto3.client("s3")
s3_bucket = os.getenv("S3_BUCKET_NAME")

# DISABLED REDIS
# redis_client = redis.Redis.from_url(
#     os.getenv("REDIS_URL"),
#     decode_responses=True
# )

SESSION_TTL_SECONDS = 600

# ====== LOAD RESUME ======
loader = PyPDFLoader("cv.pdf")
documents = loader.load()

for doc in documents:
    doc.metadata["source"] = "resume"

splitter = RecursiveCharacterTextSplitter(
    chunk_size=800,
    chunk_overlap=100,
)

docs = splitter.split_documents(documents)

embeddings = OpenAIEmbeddings()
vectorstore = FAISS.from_documents(docs, embeddings)

# ====== BUILD CHAINS ======
llm = ChatOpenAI(model_name="gpt-4o", temperature=0)

document_chain = create_stuff_documents_chain(llm, chat_prompt)

retriever = vectorstore.as_retriever(search_kwargs={"k": 8})

qa_chain = create_retrieval_chain(retriever, document_chain)

# ====== FASTAPI ======
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.middleware("http")
async def attach_session_id(request: Request, call_next):
    session_id = request.cookies.get("session_id") or str(uuid4())
    response = await call_next(request)
    response.set_cookie("session_id", session_id)
    return response

class QueryRequest(BaseModel):
    query: str

# ====== BACKGROUND TASK ======
def log_to_s3_and_update_embeddings(question: str, answer: str):
    filename = "q&a.txt"
    local_path = f"/tmp/{filename}"

    try:
        s3_client.download_file(s3_bucket, filename, local_path)
    except ClientError:
        open(local_path, "w").close()

    with open(local_path, "a") as f:
        f.write(f"Question: {question}\nAnswer: {answer}\n\n")

    with open(local_path, "rb") as f:
        s3_client.upload_fileobj(f, s3_bucket, filename)

    loader = TextLoader(local_path)
    new_docs = splitter.split_documents(loader.load())

    for doc in new_docs:
        doc.metadata["source"] = "log"

    vectorstore.add_documents(new_docs)

# ====== MAIN ENDPOINT ======
@app.post("/ask")
def ask_question(
    request: QueryRequest,
    background_tasks: BackgroundTasks,
    http_request: Request,
):
    session_id = http_request.cookies.get("session_id")

    # redis_key = f"session:{session_id}"
    # history = [
    #     json.loads(x)
    #     for x in redis_client.lrange(redis_key, 0, -1)
    # ]

    # fallback: no persistent history
    history = []

    history.append({"role": "user", "content": request.query})
    history = history[-5:]

    context = ""
    for msg in history:
        role = "User" if msg["role"] == "user" else "Assistant"
        context += f"{role}: {msg['content']}\n"

    response = qa_chain.invoke({
        "input": request.query,
        "question": request.query,
        "context": context
    })

    answer = response["answer"]

    # redis_client.rpush(
    #     redis_key,
    #     json.dumps({"role": "assistant", "content": answer})
    # )
    # redis_client.expire(redis_key, SESSION_TTL_SECONDS)

    background_tasks.add_task(
        log_to_s3_and_update_embeddings,
        request.query,
        answer
    )

    return {"answer": answer}

# ====== CLEAR SESSION ======
@app.post("/clear_session")
def clear_session(request: Request):
    # redis_client.delete(f"session:{request.cookies.get('session_id')}")
    return {"message": "Session cleared (no-op, Redis disabled)"}
