import streamlit as st
from rag_backend import load_pdf, ingest_pdf, generate_mcqs

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

st.info(
    "ℹ️ **Demo Notice:** This application uses **free-tier APIs** for embeddings and LLM inference. "
    "It is intended **for demonstration and learning purposes only**. "
    "Occasional latency, limited throughput, or variability in responses may occur."
)


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

    num_q = st.slider(
        "Number of questions",
        min_value=1,
        max_value=10,
        value=5
    )

    generate_btn = st.button("🚀 Generate Quiz")

# --------------------------------------------------
# Session State Initialization
# --------------------------------------------------
if "mcqs" not in st.session_state:
    st.session_state.mcqs = []

if "answers" not in st.session_state:
    st.session_state.answers = {}

if "submitted" not in st.session_state:
    st.session_state.submitted = False

if "pdf_ingested" not in st.session_state:
    st.session_state.pdf_ingested = False

# --------------------------------------------------
# Generate Quiz
# --------------------------------------------------
if generate_btn:
    if not pdf:
        st.warning("Please upload a PDF.")
    elif not topic.strip():
        st.warning("Please enter a topic.")
    else:
        # Ingest PDF only once per upload
        if not st.session_state.pdf_ingested:
            with st.spinner("Processing PDF and building knowledge base..."):
                chunks = load_pdf(pdf)
                ingest_pdf(chunks)
                st.session_state.pdf_ingested = True

        with st.spinner("Generating MCQs..."):
            mcqs = generate_mcqs(
                query=topic,
                difficulty=difficulty,
                num_q=num_q
            )

        if mcqs:
            st.session_state.mcqs = mcqs
            st.session_state.submitted = False
            st.session_state.answers = {i: None for i in range(len(mcqs))}
            st.success("Quiz generated successfully!")
        else:
            st.error("No MCQs generated. Try a different topic.")

# --------------------------------------------------
# Quiz Display
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
        st.button(
            "✅ Submit Answers",
            on_click=lambda: st.session_state.update({"submitted": True})
        )

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
        st.write(f"🧑 **Your answer:** {user_choice}")
        st.write(f"✅ **Correct answer:** {correct}")

        if user_choice == correct:
            st.success("Correct")
            score += 1
        else:
            st.error("Wrong")

        st.info(f"📘 **Explanation:** {q['explanation']}")
        st.divider()

    st.subheader(f"🎯 Final Score: {score} / {len(st.session_state.mcqs)}")


