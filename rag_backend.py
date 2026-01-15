import streamlit as st
import hashlib
from rag_backend import load_pdf, ingest_pdf, generate_mcqs


# --------------------------------------------------
# Helpers
# --------------------------------------------------
def get_pdf_hash(file):
    file.seek(0)
    data = file.read()
    file.seek(0)
    return hashlib.md5(data).hexdigest()


# --------------------------------------------------
# Page Config
# --------------------------------------------------
st.set_page_config(
    page_title="RAG MCQ Quiz Generator",
    page_icon="📘",
    layout="wide"
)

st.title("📘 RAG-Based MCQ Quiz Generator")
st.caption("Difficulty-aware MCQs using RAG + HuggingFace Embeddings + Groq (Free APIs)")


# --------------------------------------------------
# Sidebar
# --------------------------------------------------
with st.sidebar:
    st.header("⚙️ Quiz Settings")

    pdf = st.file_uploader("Upload PDF", type="pdf")

    topic = st.text_input(
        "Enter topic / concept",
        placeholder="e.g. MLE vs MAP"
    )

    difficulty = st.selectbox(
        "Select difficulty",
        ["Easy", "Medium", "Hard"]
    )

    num_q = st.slider("Number of questions", 1, 10, 5)

    generate_btn = st.button("🚀 Generate Quiz")


# --------------------------------------------------
# Session State
# --------------------------------------------------
if "pdf_hash" not in st.session_state:
    st.session_state.pdf_hash = None

if "vector_ready" not in st.session_state:
    st.session_state.vector_ready = False

if "mcqs" not in st.session_state:
    st.session_state.mcqs = []

if "answers" not in st.session_state:
    st.session_state.answers = {}

if "submitted" not in st.session_state:
    st.session_state.submitted = False


# --------------------------------------------------
# Generate Quiz
# --------------------------------------------------
if generate_btn:
    if not pdf:
        st.warning("Please upload a PDF.")
    elif not topic.strip():
        st.warning("Please enter a topic.")
    else:
        current_hash = get_pdf_hash(pdf)

        # 🔥 NEW PDF → RESET STATE
        if st.session_state.pdf_hash != current_hash:
            st.session_state.pdf_hash = current_hash
            st.session_state.vector_ready = False
            st.session_state.mcqs = []
            st.session_state.answers = {}
            st.session_state.submitted = False

            for k in ["chunks", "vectors"]:
                if k in st.session_state:
                    del st.session_state[k]

        if not st.session_state.vector_ready:
            with st.spinner("Processing PDF and building knowledge base..."):
                chunks = load_pdf(pdf)
                ingest_pdf(chunks)
                st.session_state.vector_ready = True

        with st.spinner("Generating MCQs..."):
            mcqs = generate_mcqs(
                query=topic,
                difficulty=difficulty,
                num_q=num_q
            )

        if mcqs:
            st.session_state.mcqs = mcqs
            st.session_state.answers = {i: None for i in range(len(mcqs))}
            st.session_state.submitted = False
            st.success("Quiz generated successfully!")
        else:
            st.error("No valid questions could be generated from this document.")


# --------------------------------------------------
# Quiz UI
# --------------------------------------------------
if st.session_state.mcqs and not st.session_state.submitted:
    st.header("📝 Quiz")

    for i, q in enumerate(st.session_state.mcqs):
        st.markdown(f"### Q{i+1}. {q['question']}")

        choice = st.radio(
            "Choose an option:",
            q["options"],
            index=None,
            key=f"q_{i}"
        )

        st.session_state.answers[i] = choice
        st.divider()

    if all(v is not None for v in st.session_state.answers.values()):
        st.button("✅ Submit Answers",
                  on_click=lambda: st.session_state.update({"submitted": True}))


# --------------------------------------------------
# Results
# --------------------------------------------------
if st.session_state.submitted:
    st.header("📊 Results")
    score = 0

    for i, q in enumerate(st.session_state.mcqs):
        user_choice = st.session_state.answers[i]
        correct = next(o for o in q["options"] if o.startswith(q["answer"]))

        st.markdown(f"### Q{i+1}. {q['question']}")
        st.write(f"🧑 Your answer: {user_choice}")
        st.write(f"✅ Correct answer: {correct}")

        if user_choice == correct:
            st.success("Correct")
            score += 1
        else:
            st.error("Wrong")

        st.info(f"📘 Explanation: {q['explanation']}")
        st.divider()

    st.subheader(f"🎯 Final Score: {score} / {len(st.session_state.mcqs)}")
