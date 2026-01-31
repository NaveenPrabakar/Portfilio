from fastapi import FastAPI, BackgroundTasks, Request
from pydantic import BaseModel
import os
from fastapi.middleware.cors import CORSMiddleware
import boto3
from botocore.exceptions import ClientError
from datetime import datetime
import redis
import json
from uuid import uuid4
import openai
from pymongo import MongoClient

# ====== LangChain & LLM Imports ======
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

# ====== SYSTEM PROMPT SETUP ======
system_prompt = (
    "You are a career assistant who gives helpful, concise answers about Naveen Prabakar "
    "based on his resume and prior questions. Any irrelevant question asked should be dismissed professionally."
)

system_msg = SystemMessagePromptTemplate.from_template(system_prompt)

human_msg = HumanMessagePromptTemplate.from_template(
    "Here is the chat history so far:\n{context}\n\n"
    "Now answer the user's latest question:\n{question}"
)

chat_prompt = ChatPromptTemplate.from_messages([system_msg, human_msg])

# ====== ENVIRONMENT SETUP ======
mongo_uri = os.getenv("MONGODB_ATLAS_URI")
mongo_client = MongoClient(mongo_uri)
db = mongo_client["qna_logs"]
collection = db["qa_archive"]

openai.api_key = os.getenv("OPENAI_API_KEY")

s3_bucket = os.getenv("S3_BUCKET_NAME")
s3_client = boto3.client("s3")

redis_url = os.getenv("REDIS_URL")
if not redis_url:
    raise ValueError("REDIS_URL not set")
redis_client = redis.Redis.from_url(redis_url, decode_responses=True)

SESSION_TTL_SECONDS = 600  # 10 minutes

# ====== LOAD RESUME & BUILD VECTORSTORE ======
loader = PyPDFLoader("cv.pdf")
documents = loader.load()

for doc in documents:
    doc.metadata["source"] = "resume"

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=800,
    chunk_overlap=100,
)

docs = text_splitter.split_documents(documents)

embeddings = OpenAIEmbeddings()
vectorstore = FAISS.from_documents(docs, embeddings)

# ====== BUILD LLM + CHAINS ======
llm = ChatOpenAI(
    temperature=0,
    model_name="gpt-4o",
)

document_chain = create_stuff_documents_chain(llm, chat_prompt)

retriever = vectorstore.as_retriever(search_kwargs={"k": 8})
qa_chain = create_retrieval_chain(retriever, document_chain)

# ====== FASTAPI SETUP ======
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
    session_id = request.cookies.get("session_id")
    if not session_id:
        session_id = str(uuid4())
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
    except ClientError as e:
        if e.response["Error"]["Code"] == "404":
            open(local_path, "w").close()
        else:
            raise

    with open(local_path, "r") as f:
        content = f.read().strip()
        qna_pairs = content.split("\n\n") if content else []

    if len(qna_pairs) >= 10:
        to_offload = qna_pairs[:8]
        remaining = qna_pairs[8:]

        for entry in to_offload:
            lines = entry.split("\n")
            q = lines[0].replace("Question: ", "")
            a = lines[1].replace("Answer: ", "")
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

# ====== MAIN ENDPOINT ======
@app.post("/ask")
def ask_question(
    request: QueryRequest,
    background_tasks: BackgroundTasks,
    http_request: Request,
):
    session_id = http_request.cookies.get("session_id")
    redis_key = f"session:{session_id}"

    history_json = redis_client.lrange(redis_key, 0, -1)
    history = [json.loads(msg) for msg in history_json]

    history.append({"role": "user", "content": request.query})
    trimmed_history = history[-5:]

    redis_client.delete(redis_key)
    for msg in trimmed_history:
        redis_client.rpush(redis_key, json.dumps(msg))
    redis_client.expire(redis_key, SESSION_TTL_SECONDS)

    conversational_context = ""
    for msg in trimmed_history:
        role = "User" if msg["role"] == "user" else "Assistant"
        conversational_context += f"{role}: {msg['content']}\n"

    # ✅ FIXED: pass expected variables
    response = qa_chain.invoke({
        "context": conversational_context,
        "question": request.query,
    })

    answer = response.get("answer", str(response))

    redis_client.rpush(
        redis_key,
        json.dumps({"role": "assistant", "content": answer}),
    )
    redis_client.expire(redis_key, SESSION_TTL_SECONDS)

    background_tasks.add_task(
        log_to_s3_and_update_embeddings,
        request.query,
        answer,
    )

    return {"answer": answer}

# ====== CLEAR SESSION ======
@app.post("/clear_session")
def clear_session(http_request: Request):
    session_id = http_request.cookies.get("session_id")
    redis_client.delete(f"session:{session_id}")
    return {"message": "Session memory cleared."}
