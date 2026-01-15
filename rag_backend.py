import os
import re
import json
import fitz
import streamlit as st

from huggingface_hub import InferenceClient
from sklearn.metrics.pairwise import cosine_similarity

from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser


# =========================================================
# HF Embeddings Client
# =========================================================
hf_client = InferenceClient(
    model="sentence-transformers/all-MiniLM-L6-v2",
    token=os.getenv("HUGGINGFACE_API_TOKEN")
)


# =========================================================
# Difficulty Control (REAL DIFFERENCES)
# =========================================================
DIFFICULTY_CONFIG = {
    "Easy": {
        "retrieval_k": 2,
        "instruction": """
Ask factual, definition-based questions.
Answers must be explicitly stated.
Avoid comparisons or reasoning.
"""
    },
    "Medium": {
        "retrieval_k": 4,
        "instruction": """
Ask conceptual or comparison-based questions.
Test understanding, not recall.
No multi-step reasoning.
"""
    },
    "Hard": {
        "retrieval_k": 6,
        "instruction": """
Ask application or implication-based questions.
Require combining multiple facts.
Avoid surface-level recall.
"""
    }
}


# =========================================================
# Prompt
# =========================================================
PROMPT = PromptTemplate(
    input_variables=["context", "num_q", "difficulty_instruction"],
    template="""
You are an expert instructor creating assessment-quality MCQs.

DIFFICULTY RULES:
{difficulty_instruction}

STRICT RULES:
- Use ONLY the study material
- Do NOT repeat questions
- Do NOT ask trivial questions
- Generate fewer questions if needed

OUTPUT ONLY JSON:
[
  {{
    "question": "...",
    "options": ["A) ...", "B) ...", "C) ...", "D) ..."],
    "answer": "A",
    "explanation": "Explain using evidence from the material"
  }}
]

STUDY MATERIAL:
{context}
"""
)


# =========================================================
# PDF Loader
# =========================================================
def load_pdf(file):
    doc = fitz.open(stream=file.read(), filetype="pdf")
    chunks = []

    for page in doc:
        blocks = page.get_text("blocks")
        for b in blocks:
            txt = b[4].strip()
            if txt:
                chunks.append(re.sub(r"\s+", " ", txt))

    return chunks


# =========================================================
# Embeddings
# =========================================================
def embed_texts(texts):
    return [hf_client.feature_extraction(t) for t in texts]


# =========================================================
# Ingest
# =========================================================
def ingest_pdf(chunks):
    st.session_state.chunks = chunks
    st.session_state.vectors = embed_texts(chunks)


# =========================================================
# Retrieve Context
# =========================================================
def retrieve_context(query, k):
    q_vec = embed_texts([query])[0]
    sims = cosine_similarity([q_vec], st.session_state.vectors)[0]
    top_idx = sims.argsort()[-k:][::-1]
    return "\n\n".join(st.session_state.chunks[i] for i in top_idx)


# =========================================================
# MCQ Generation
# =========================================================
def generate_mcqs(query, difficulty, num_q):
    cfg = DIFFICULTY_CONFIG[difficulty]
    context = retrieve_context(query, cfg["retrieval_k"])

    llm = ChatGroq(
        model="llama-3.1-8b-instant",
        temperature=0
    )

    chain = PROMPT | llm | StrOutputParser()

    raw = chain.invoke({
        "context": context,
        "num_q": num_q,
        "difficulty_instruction": cfg["instruction"]
    })

    objects = re.findall(r"\{[^{}]*\}", raw, re.DOTALL)
    mcqs = []

    for o in objects:
        try:
            mcq = json.loads(o)
            if {"question", "options", "answer", "explanation"}.issubset(mcq):
                mcqs.append(mcq)
        except:
            pass

    return mcqs
