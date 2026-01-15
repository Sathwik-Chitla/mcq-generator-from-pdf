# 📘 PDF to MCQ Generator using RAG

A **Retrieval Augmented Generation (RAG)**–based web application that generates **topic-specific, difficulty-aware multiple-choice questions (MCQs)** from user-uploaded PDF documents.  
Built with **Streamlit**, **LangChain**, **Groq LLaMA-3.1**, and **Hugging Face sentence embeddings**, the system ensures questions are grounded strictly in the source document, minimizing hallucinations.

🔗 **Live Demo:**  
👉 https://mcq-generator-from-pdf-2.streamlit.app/

---

## 🚀 Features

- 📄 Upload any PDF and generate MCQs directly from its content  
- 🎯 Topic-conditioned question generation  
- 🧠 Difficulty levels: **Easy / Medium / Hard** (with real cognitive differences)  
- 🔍 Retrieval Augmented Generation (RAG) to ground questions in source text  
- 🧪 Semantic similarity search using Hugging Face embeddings  
- ✅ Automated scoring with detailed explanations  
- 🔁 Session-safe handling to prevent question repetition across documents  
- ⚡ Low-latency, real-time interaction  
- ☁️ Fully deployed on Streamlit Community Cloud  

---

## 🏗️ System Architecture

PDF Upload

↓

PyMuPDF (Text Extraction)

↓

Chunking & Embedding (Hugging Face)

↓

Similarity Search (Cosine Similarity)

↓

Relevant Context

↓

Groq LLaMA-3.1

↓

MCQs + Explanations



---

## 🧠 Tech Stack

- **Frontend:** Streamlit  
- **LLM:** Groq `llama-3.1-8b-instant`  
- **Embeddings:** Hugging Face Sentence Transformers  
- **RAG Framework:** LangChain  
- **PDF Parsing:** PyMuPDF  
- **Similarity Search:** Cosine Similarity (scikit-learn)  
- **Deployment:** Streamlit Community Cloud  

---

## ⚙️ Setup & Installation (Local)

### 1️⃣ Clone the repository
```bash
git clone https://github.com/your-username/mcq-generator-from-pdf.git
cd mcq-generator-from-pdf
```
### 2️⃣ Install dependencies
```bash
pip install -r requirements.txt
```
### 3️⃣ Set environment variables
```bash
export HUGGINGFACE_API_TOKEN=hf_xxxxxxxxxxxxxxxxx
export GROQ_API_KEY=gsk_xxxxxxxxxxxxxxxxx
```

### 4️⃣ Run the app
```bash
streamlit run app.py
```
### 📌 Difficulty Design

Difficulty	                                                
- Easy:                                                 Definition-based, directly stated facts
- Medium:                                          Conceptual understanding and comparisons
- Hard:                                               Application-based questions combining multiple facts

### 🧪 Example Workflow

- Upload lecture notes or textbook PDF

- Enter a topic (e.g., Bayes Theorem)

- Select difficulty and number of questions

- Attempt the quiz

- Review score and explanations

### 🎯 Why RAG?

- Prevents hallucinations
 
- Ensures questions are derived only from the document

- Improves trustworthiness and assessment quality

### 📎 Live Demo
```bash
https://mcq-generator-from-pdf-2.streamlit.app/
```

📈 Future Improvements

- Per-question source citations

- Question diversity scoring

- Support for figures

- Export quiz as PDF / JSON

- User authentication and quiz history



