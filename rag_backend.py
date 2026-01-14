import os
import re
import json
import fitz

from pinecone import Pinecone
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser


# =========================================================
# CONFIG
# =========================================================
INDEX_NAME = "rag-pdf-index"
EMBED_DIM = 1536  # OpenAI embeddings


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
# PDF LOADING (CLOUD SAFE)
# =========================================================
def load_pdf(file):
    doc = fitz.open(stream=file.read(), filetype="pdf")
    text = []

    for page in doc:
        blocks = page.get_text("blocks")
        for b in blocks:
            content = b[4].strip()
            if not content:
                continue

            if content.count("  ") >= 2 or "\t" in content:
                text.append("Table Fact: " + re.sub(r"\s+", " ", content))
            else:
                text.append(content)

    return "\n".join(text)


# =========================================================
# INGEST INTO PINECONE
# =========================================================
def ingest_pdf_to_pinecone(text):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=120
    )
    chunks = splitter.split_text(text)

    embeddings = OpenAIEmbeddings()

    pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])

    if INDEX_NAME not in [i["name"] for i in pc.list_indexes()]:
        pc.create_index(
            name=INDEX_NAME,
            dimension=EMBED_DIM,
            metric="cosine"
        )

    PineconeVectorStore.from_texts(
        texts=chunks,
        embedding=embeddings,
        index_name=INDEX_NAME
    )


# =========================================================
# RETRIEVAL
# =========================================================
def retrieve_context(topic, k=6):
    embeddings = OpenAIEmbeddings()
    pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])

    store = PineconeVectorStore(
        index=pc.Index(INDEX_NAME),
        embedding=embeddings,
        text_key="text"
    )

    docs = store.similarity_search(topic, k=k)
    return "\n\n".join(d.page_content for d in docs[:4])


# =========================================================
# MCQ GENERATION
# =========================================================
def generate_mcqs(query, difficulty, num_q):
    context = retrieve_context(query)

    llm = ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0
    )

    chain = PROMPT | llm | StrOutputParser()

    raw = chain.invoke({
        "context": context,
        "num_q": num_q,
        "difficulty": DIFF_RULES[difficulty]
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
