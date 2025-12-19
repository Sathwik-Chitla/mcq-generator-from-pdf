import os
import re
import json
import fitz

from pinecone import Pinecone
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser


# =========================================================
# CONFIG
# =========================================================
INDEX_NAME = "rag-pdf-index"
EMBED_MODEL = "nomic-embed-text"
OLLAMA_BASE_URL = "http://127.0.0.1:11434"


# =========================================================
# DIFFICULTY RULES
# =========================================================
DIFF_RULES = {
    "Easy": """
Generate EASY questions.
Focus only on definitions and explicitly stated facts.
Avoid reasoning or application.
""",
    "Medium": """
Generate MEDIUM questions.
Test understanding, intuition, or comparisons.
No multi-step reasoning.
""",
    "Hard": """
Generate HARD questions.
Test application, assumptions, or implications.
Only use what is clearly inferable from the material.
"""
}


# =========================================================
# PROMPT (BALANCED & HALLUCINATION-RESISTANT)
# =========================================================
PROMPT = PromptTemplate(
    input_variables=["context", "num_q", "difficulty"],
    template="""
You are an expert instructor designing assessment-quality MCQs.

DIFFICULTY INSTRUCTIONS:
{difficulty}

TASK:
Generate up to {num_q} MCQs strictly from the study material below.

MANDATORY QUESTION RULES:
1. Each question must test ONE clear concept explicitly present in the material.
2. The correct answer must be directly supported by the material.
3. At least TWO incorrect options must be plausible misconceptions.
4. If the material does not support all questions, generate FEWER valid questions.

MANDATORY EXPLANATION STRUCTURE:
- Sentence 1: Restate the relevant information from the study material.
- Sentence 2: Explain why the correct option follows from this information.
- Sentence 3: Explain why one incorrect option contradicts or misuses the material.

STRICT OUTPUT RULES:
- Output ONLY valid JSON.
- No markdown, no commentary, no extra text.
- Do NOT invent facts.

JSON FORMAT:
[
  {{
    "question": "Question text",
    "options": ["A) ...", "B) ...", "C) ...", "D) ..."],
    "answer": "A",
    "explanation": "Structured explanation as required"
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
    return " ".join(page.get_text() for page in doc)


# =========================================================
# INGEST PDF INTO PINECONE
# =========================================================
def ingest_pdf_to_pinecone(text):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=120
    )
    chunks = splitter.split_text(text)

    pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])
    index = pc.Index(INDEX_NAME)

    embeddings = OllamaEmbeddings(
        model=EMBED_MODEL,
        base_url=OLLAMA_BASE_URL
    )

    PineconeVectorStore.from_texts(
        texts=chunks,
        embedding=embeddings,
        index_name=INDEX_NAME
    )


# =========================================================
# RAG RETRIEVAL (ROBUST & SAFE)
# =========================================================
def retrieve_context(topic, k_retrieve=5, k_use=2):
    retrieval_query = f"""
{topic}
Use exact terminology from the study material.
Focus on definitions, assumptions, and explanations.
"""

    pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])
    index = pc.Index(INDEX_NAME)

    embeddings = OllamaEmbeddings(
        model=EMBED_MODEL,
        base_url=OLLAMA_BASE_URL
    )

    store = PineconeVectorStore(
        index=index,
        embedding=embeddings,
        text_key="text"
    )

    docs = store.similarity_search(retrieval_query, k=k_retrieve)

    # Primary context
    context = "\n\n".join(d.page_content for d in docs[:k_use])

    # Safety net: ensure sufficient context
    if len(context.strip()) < 300:
        context = "\n\n".join(d.page_content for d in docs[:3])

    return context


# =========================================================
# MCQ GENERATION (SAFE + SALVAGE)
# =========================================================
def generate_mcqs(query, difficulty, num_q):
    context = retrieve_context(query)

    llm = ChatOllama(
        model="llama3.1",
        temperature=0,
        base_url=OLLAMA_BASE_URL,
        system="""
You are a strict JSON generator.
Only use information from the provided study material.
"""
    )

    chain = PROMPT | llm | StrOutputParser()

    raw_output = chain.invoke({
        "context": context,
        "num_q": num_q,
        "difficulty": DIFF_RULES[difficulty]
    })

    # -----------------------------------------------------
    # Salvage valid MCQs even if output is partially broken
    # -----------------------------------------------------
    objects = re.findall(r"\{[^{}]*\}", raw_output, re.DOTALL)
    mcqs = []

    for obj in objects:
        try:
            mcq = json.loads(obj)
            if (
                isinstance(mcq, dict)
                and {"question", "options", "answer", "explanation"}.issubset(mcq)
                and isinstance(mcq["options"], list)
                and len(mcq["options"]) == 4
                and mcq["answer"] in ["A", "B", "C", "D"]
            ):
                mcqs.append(mcq)
        except json.JSONDecodeError:
            continue

    return mcqs
