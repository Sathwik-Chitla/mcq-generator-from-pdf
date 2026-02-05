import os
import json
import re
import fitz
import streamlit as st

from huggingface_hub import InferenceClient
from sklearn.metrics.pairwise import cosine_similarity

from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser


hf_client = InferenceClient(
    model="sentence-transformers/all-MiniLM-L6-v2",
    token=os.getenv("HUGGINGFACE_API_TOKEN")
)


def embed_texts(texts):
    return [hf_client.feature_extraction(t) for t in texts]


DIFFICULTY_CONFIG = {
    "Easy": {"retrieval_k": 6},
    "Medium": {"retrieval_k": 10},
    "Hard": {"retrieval_k": 14},
}


PROMPT = PromptTemplate(
    input_variables=["context", "topic", "num_q", "difficulty"],
    template="""
You are an expert instructor.

TASK:
Generate EXACTLY {num_q} UNIQUE MCQs about the topic:
"{topic}"

DIFFICULTY:
{difficulty}

RULES:
- Every question MUST be directly related to the topic
- Use ONLY the provided study material
- Each question must test a DIFFERENT idea
- Do NOT repeat concepts
- Do NOT invent facts

Return VALID JSON only.

FORMAT:
[
  {
    "question": "...",
    "options": ["A) ...", "B) ...", "C) ...", "D) ..."],
    "answer": "A",
    "explanation": "Explain using the material"
  }
]

STUDY MATERIAL:
{context}
"""
)


def load_pdf(file):
    doc = fitz.open(stream=file.read(), filetype="pdf")
    chunks = []

    for page in doc:
        for block in page.get_text("blocks"):
            text = block[4].strip()
            if text:
                chunks.append(re.sub(r"\s+", " ", text))

    return chunks


def ingest_pdf(chunks):
    st.session_state.chunks = chunks
    st.session_state.vectors = embed_texts(chunks)


def retrieve_context(topic, k):
    q_vec = embed_texts([topic])[0]
    sims = cosine_similarity([q_vec], st.session_state.vectors)[0]

    top_idx = sims.argsort()[-k:][::-1]
    return "\n\n".join(st.session_state.chunks[i] for i in top_idx)


def generate_mcqs(query, difficulty, num_q):
    cfg = DIFFICULTY_CONFIG[difficulty]
    context = retrieve_context(query, cfg["retrieval_k"])

    llm = ChatGroq(
        model="llama-3.1-8b-instant",
        temperature=0
    )

    parser = JsonOutputParser()

    chain = PROMPT | llm | parser

    try:
        mcqs = chain.invoke({
            "context": context,
            "topic": query,
            "num_q": num_q,
            "difficulty": difficulty
        })
    except Exception:
        return []

    valid = []
    seen = set()

    for q in mcqs:
        if (
            isinstance(q, dict)
            and {"question", "options", "answer", "explanation"}.issubset(q)
            and q["question"] not in seen
        ):
            seen.add(q["question"])
            valid.append(q)

    return valid[:num_q]
