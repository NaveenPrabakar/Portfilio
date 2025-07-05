# 🧠 AI Resume Assistant API

A FastAPI-based backend powered by OpenAI and LangChain that answers questions about **Naveen Prabakar's** resume. It reads a PDF resume, stores previous questions, embeds Q&A logs, and returns smart, context-aware answers with a professional tone.

![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)
![FastAPI](https://img.shields.io/badge/FastAPI-async-green.svg)
![OpenAI](https://img.shields.io/badge/OpenAI-GPT4-ff69b4.svg)
![MongoDB](https://img.shields.io/badge/MongoDB-Atlas-47A248.svg)
![LangChain](https://img.shields.io/badge/LangChain-FAISS-yellow.svg)
![AWS](https://img.shields.io/badge/AWS-S3-orange.svg)

---

## 🚀 Features

- 🔍 **Semantic Resume Search**: Ask questions about the resume and get intelligent answers.
- 🧠 **OpenAI GPT-4**: Powered by a system-prompted assistant for concise, professional responses.
- 📄 **PDF Resume Loader**: Parses resume with `PyPDFLoader`.
- 🗂️ **Log Memory Embedding**: Stores and re-embeds past Q&A from S3 and MongoDB.
- ⏳ **Background Archival**: Automatically archives older logs to Mongo and updates embeddings.
- 🌐 **FastAPI**: Built as a clean RESTful API with CORS enabled for frontend access.

## 📊 How It Works

1. Parses `resume.pdf` and creates semantic vector chunks.
2. Embeds Q&A logs from `q&a.txt` in S3 and resumes.
3. Uses LangChain’s `ChatPromptTemplate` to inject a system prompt and custom context.
4. Answers questions using GPT-4 and returns JSON responses.
5. Logs new Q&A to S3 and MongoDB, re-embeds content in the background.

---

## 📦 Tech Stack

| Component      | Tech                                       |
|----------------|--------------------------------------------|
| Backend        | FastAPI                                    |
| Embeddings     | OpenAI Embeddings (`text-embedding-ada-002`) |
| LLM            | OpenAI GPT-4 via LangChain                 |
| Vector Store   | FAISS (in-memory)                          |
| Storage        | AWS S3 (`q&a.txt` log)                     |
| Database       | MongoDB Atlas                              |

---

## 📁 Project Structure

```
.
├── main.py              # FastAPI app with Q&A endpoint
├── resume.pdf           # Resume used as the primary source
├── requirements.txt     # Python dependencies
```

---

## 🛠️ Setup Instructions

### 1. Clone the repo

```bash
git clone https://github.com/NaveenPrabakar/Portfilio.git
cd Portfilio
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Set environment variables

Create a `.env` file or export manually:

```bash
export OPENAI_API_KEY=your-openai-key
export MONGODB_ATLAS_URI=your-mongo-uri
export S3_BUCKET_NAME=your-s3-bucket
```

### 4. Run the API

```bash
uvicorn main:app --reload
```

---

## 🔍 Example API Call

### Endpoint: `POST /ask`

```json
{
  "query": "What are Naveen Prabakar’s technical skills?"
}
```

### Example Response:

```json
{
  "answer": "Naveen Prabakar is skilled in Python, Java, Docker, SQL, AWS, among others."
}
```


