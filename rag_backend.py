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
# HuggingFace Inference Client (FREE embeddings)
# =========================================================
HF_TOKEN = os.getenv("HUGGINGFACE_API_TOKEN")

hf_client = InferenceClient(
    model="sentence-transformers/all-MiniLM-L6-v2",
    token=HF_TOKEN
)


# =========================================================
# DIFFICULTY RULES
# =========================================================
DIFF_RULES = {
    "Easy": "Generate simple definition-based questions.",
    "Medium": "Generate understanding or comparison-based questions.",
    "Hard": """
Generate HARD questions.
- Combine at least TWO facts from the material
- Use comparisons or implications
- No external knowledge
"""
}


# =========================================================
# PROMPT
# =========================================================
PROMPT = PromptTemplate(
    input_variables=["context", "num_q", "difficulty"],
    template="""
You are an expert instructor creating assessment-quality MCQs.

DIFFICULTY:
{difficulty}

Generate up to {num_q} MCQs strictly from the study material.

Rules:
- Correct answer must be supported by the text
- Wrong options must be plausible misconceptions
- If insufficient info exists, generate fewer questions

Output ONLY valid JSON.

Format:
[
  {{
    "question": "...",
    "options": ["A) ...", "B) ...", "C) ...", "D) ..."],
    "answer": "A",
    "explanation": "Explain using the material"
  }}
]

STUDY MATERIAL:
{context}
"""
)


# =========================================================
# PDF LOADING
# =========================================================
def load_pdf(file):
    doc = fitz.open(stream=file.read(), filetype="pdf")
    chunks = []

    for page in doc:
        blocks = page.get_text("blocks")
        for b in blocks:
            content = b[4].strip()
            if not content:
                continue

            # Heuristic table handling
            if content.count("  ") >= 2 or "\t" in content:
                chunks.append("Table Fact: " + re.sub(r"\s+", " ", content))
            else:
                chunks.append(content)

    return chunks


# =========================================================
# EMBEDDINGS (CORRECT HF API USAGE)
# =========================================================
def embed_texts(texts):
    """
    Generate embeddings using HuggingFace Inference API.
    HF does NOT support batch embeddings reliably on free tier,
    so we embed one-by-one for stability.
    """
    vectors = []
    for text in texts:
        vec = hf_client.feature_extraction(text)
        vectors.append(vec)
    return vectors


# =========================================================
# INGEST (IN-MEMORY)
# =========================================================
def ingest_pdf_to_pinecone(chunks):
    vectors = embed_texts(chunks)

    st.session_state.chunks = chunks
    st.session_state.vectors = vectors


# =========================================================
# RETRIEVE CONTEXT
# =========================================================
def retrieve_context(query, k=4):
    query_vec = embed_texts([query])[0]

    sims = cosine_similarity(
        [query_vec],
        st.session_state.vectors
    )[0]

    top_idx = sims.argsort()[-k:][::-1]
    return "\n\n".join(st.session_state.chunks[i] for i in top_idx)


# =========================================================
# MCQ GENERATION
# =========================================================
def generate_mcqs(query, difficulty, num_q):
    context = retrieve_context(query)

    llm = ChatGroq(
    model="llama-3.1-8b-instant",
    temperature=0
)
    chain = PROMPT | llm | StrOutputParser()

    raw = chain.invoke({
        "context": context,
        "num_q": num_q,
        "difficulty": DIFF_RULES[difficulty]
    })

    # Salvage valid JSON objects only
    objects = re.findall(r"\{[^{}]*\}", raw, re.DOTALL)
    mcqs = []

    for obj in objects:
        try:
            mcq = json.loads(obj)
            if (
                {"question", "options", "answer", "explanation"}
                .issubset(mcq)
                and len(mcq["options"]) == 4
                and mcq["answer"] in ["A", "B", "C", "D"]
            ):
                mcqs.append(mcq)
        except json.JSONDecodeError:
            continue

    return mcqs

