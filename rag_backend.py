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
# HuggingFace Embedding Client
# =========================================================
hf_client = InferenceClient(
    model="sentence-transformers/all-MiniLM-L6-v2",
    token=os.getenv("HUGGINGFACE_API_TOKEN")
)


# =========================================================
# Difficulty Configuration (REAL EFFECT)
# =========================================================
DIFFICULTY_CONFIG = {
    "Easy": {
        "retrieval_k": 2,
        "instruction": """
Ask definition-based or direct factual questions.
Answers must be explicitly stated in one place.
Avoid reasoning or comparisons.
"""
    },
    "Medium": {
        "retrieval_k": 4,
        "instruction": """
Ask conceptual or comparison-based questions.
Test understanding across nearby statements.
Avoid multi-step reasoning.
"""
    },
    "Hard": {
        "retrieval_k": 6,
        "instruction": """
Ask application or implication-based questions.
Each question must combine at least TWO facts.
Avoid surface-level recall.
"""
    }
}


# =========================================================
# Prompt (ENFORCES QUESTION COUNT)
# =========================================================
PROMPT = PromptTemplate(
    input_variables=["context", "num_q", "difficulty_instruction"],
    template="""
You are an expert instructor creating assessment-quality MCQs.

DIFFICULTY RULES:
{difficulty_instruction}

TASK:
Generate EXACTLY {num_q} UNIQUE MCQs from the study material below.
If fewer than {num_q} are possible, generate as many as you can.

STRICT RULES:
- Use ONLY the study material
- Each question must test a DIFFERENT concept
- Do NOT repeat questions
- Do NOT invent facts

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
            text = b[4].strip()
            if text:
                chunks.append(re.sub(r"\s+", " ", text))

    return chunks


# =========================================================
# Embeddings (HF Inference API)
# =========================================================
def embed_texts(texts):
    vectors = []
    for text in texts:
        vec = hf_client.feature_extraction(text)
        vectors.append(vec)
    return vectors


# =========================================================
# Ingest PDF (RESET-SAFE)
# =========================================================
def ingest_pdf(chunks):
    st.session_state.chunks = chunks
    st.session_state.vectors = embed_texts(chunks)


# =========================================================
# Retrieve Context
# =========================================================
def retrieve_context(query, k):
    q_vec = embed_texts([query])[0]

    sims = cosine_similarity(
        [q_vec],
        st.session_state.vectors
    )[0]

    top_idx = sims.argsort()[-k:][::-1]
    return "\n\n".join(st.session_state.chunks[i] for i in top_idx)


# =========================================================
# MCQ Generation (RESPECTS num_q)
# =========================================================
def generate_mcqs(query, difficulty, num_q):
    cfg = DIFFICULTY_CONFIG[difficulty]

    llm = ChatGroq(
        model="llama-3.1-8b-instant",
        temperature=0
    )

    all_mcqs = []
    attempts = 0
    max_attempts = 3  # prevents infinite loops

    while len(all_mcqs) < num_q and attempts < max_attempts:
        attempts += 1

        context = retrieve_context(query, cfg["retrieval_k"])

        chain = PROMPT | llm | StrOutputParser()

        raw = chain.invoke({
            "context": context,
            "num_q": num_q - len(all_mcqs),
            "difficulty_instruction": cfg["instruction"]
        })

        objects = re.findall(r"\{[^{}]*\}", raw, re.DOTALL)

        for obj in objects:
            try:
                mcq = json.loads(obj)
                if (
                    {"question", "options", "answer", "explanation"}
                    .issubset(mcq)
                    and len(mcq["options"]) == 4
                    and mcq["answer"] in ["A", "B", "C", "D"]
                ):
                    # Deduplicate by question text
                    if mcq["question"] not in {
                        q["question"] for q in all_mcqs
                    }:
                        all_mcqs.append(mcq)
            except json.JSONDecodeError:
                continue

        # If model can't produce new questions, stop early
        if not objects:
            break

    return all_mcqs[:num_q]
