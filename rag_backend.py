import re
import json
import fitz

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.vectorstores import FAISS


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
# INGEST (FAISS)
# =========================================================
def ingest_pdf_to_pinecone(text):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=120
    )
    chunks = splitter.split_text(text)

    embeddings = OpenAIEmbeddings()
    vectorstore = FAISS.from_texts(chunks, embedding=embeddings)

    return vectorstore


# =========================================================
# MCQ GENERATION
# =========================================================
def generate_mcqs(query, difficulty, num_q):
    # vectorstore is rebuilt per upload (fine for demo)
    context_docs = st.session_state.vectorstore.similarity_search(query, k=4)
    context = "\n\n".join(d.page_content for d in context_docs)

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
