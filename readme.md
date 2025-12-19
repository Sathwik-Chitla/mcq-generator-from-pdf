📘 PDF → RAG-Based MCQ Generator

A Streamlit web app that generates difficulty-aware Multiple Choice Questions (MCQs) strictly from an uploaded PDF using Retrieval-Augmented Generation (RAG) powered by LangChain + Pinecone + Ollama (LLaMA 3.1).

🚀 Features

Upload any PDF and extract text automatically

Generate difficulty-aware MCQs (Easy / Medium / Hard)

Generate a customizable number of MCQs (1–10)

Interactive quiz interface with score calculation

Detailed explanations for every question

Questions generated only from PDF content (RAG)

Powered by Ollama (LLaMA-3.1) + Pinecone + LangChain

⚙️ Installation

Clone this repository

git clone https://github.com/your-username/rag-mcq-quiz-generator.git
cd rag-mcq-quiz-generator


Create a virtual environment (optional but recommended)

python -m venv .venv
.venv\Scripts\activate   # On Windows
source .venv/bin/activate  # On Mac/Linux


Install dependencies

pip install -r requirements.txt


Install & run Ollama

Download Ollama: https://ollama.ai

Pull required models:

ollama pull llama3.1
ollama pull nomic-embed-text


Start the Ollama server:

ollama serve


Configure Pinecone

Set your Pinecone API key

Windows (PowerShell)

setx PINECONE_API_KEY "your_pinecone_api_key"


Mac / Linux

export PINECONE_API_KEY=your_pinecone_api_key


Ensure a Pinecone index exists with the name:

rag-pdf-index

▶️ Usage

Run the Streamlit app:

streamlit run app.py


Then open http://localhost:8501
 in your browser.

📂 Project Structure
rag-mcq-quiz-generator/
│── app.py                 # Streamlit frontend
│── rag_backend.py         # RAG pipeline + Pinecone + LLM logic
│── requirements.txt       # Dependencies
│── README.md              # Project description + usage

📸 Screenshots
<img width="1903" height="809" alt="image" src="https://github.com/user-attachments/assets/ba20a2ec-3773-4ea4-a9f7-c3636ec7ec72" /> <img width="1827" height="895" alt="image" src="https://github.com/user-attachments/assets/9830d110-8a7f-4046-8178-6fa3488b6c4e" /> <img width="1688" height="944" alt="image" src="https://github.com/user-attachments/assets/cbc57777-af96-4e12-af42-37326b1a2313" /> <img width="1348" height="529" alt="image" src="https://github.com/user-attachments/assets/3c687a9d-68e2-4cd3-a3d0-1d6a17884a58" />
🎯 Demo

Example MCQs generated from an uploaded ML PDF:

Q1. What does Maximum Likelihood Estimation (MLE) aim to maximize?

Prior probability of parameters

Likelihood of observed data

Posterior distribution

Regularization term
✅ Answer: Likelihood of observed data

Q2. How does MAP differ from MLE?

MAP ignores prior information

MAP incorporates prior probability

MAP minimizes likelihood

MAP is unsupervised
✅ Answer: MAP incorporates prior probability