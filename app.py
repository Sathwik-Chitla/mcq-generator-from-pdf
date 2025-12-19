import streamlit as st
from rag_backend import load_pdf, ingest_pdf_to_pinecone, generate_mcqs

# --------------------------------------------------
# Page Config
# --------------------------------------------------
st.set_page_config(
    page_title="RAG MCQ Quiz Generator",
    page_icon="📘",
    layout="wide"
)

st.title("📘 RAG-Based MCQ Quiz Generator")
st.caption("Difficulty-aware MCQs with explanations using RAG + Ollama")

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

# --------------------------------------------------
# Placeholder (Before Quiz)
# --------------------------------------------------
if not st.session_state.mcqs:
    st.info("""
👋 **How this quiz works**

1. Upload a PDF  
2. Enter a topic (e.g. *MLE vs MAP*)  
3. Choose difficulty  
4. Generate MCQs using **RAG**  
5. Answer all questions  
6. Submit to see score + explanations  

📌 Questions are generated **only from your PDF**
""")

# --------------------------------------------------
# Generate Quiz
# --------------------------------------------------
if generate_btn:
    if not pdf:
        st.warning("Please upload a PDF.")
    elif not topic.strip():
        st.warning("Please enter a topic.")
    else:
        with st.spinner("Processing PDF and building knowledge base..."):
            text = load_pdf(pdf)
            ingest_pdf_to_pinecone(text)

        with st.spinner("Generating MCQs using RAG..."):
            mcqs = generate_mcqs(
                query=topic,
                difficulty=difficulty,
                num_q=num_q
            )

        if not mcqs:
            st.error("Failed to generate MCQs. Try a different topic.")
        else:
            st.session_state.mcqs = mcqs
            st.session_state.submitted = False

            # Reset answers cleanly
            st.session_state.answers = {}
            for i in range(len(mcqs)):
                st.session_state.answers[i] = None

            st.success("Quiz generated successfully!")

# --------------------------------------------------
# Quiz Display
# --------------------------------------------------
if st.session_state.mcqs and not st.session_state.submitted:
    st.header("📝 Quiz")

    answered_count = sum(
        1 for v in st.session_state.answers.values() if v is not None
    )
    st.caption(f"Answered {answered_count} / {len(st.session_state.mcqs)}")

    for i, q in enumerate(st.session_state.mcqs):
        st.markdown(f"### Q{i+1}. {q['question']}")

        choice = st.radio(
            "Choose an option:",
            q["options"],
            index=None,                     # IMPORTANT: no preselection
            key=f"q_{i}"
        )

        st.session_state.answers[i] = choice
        st.divider()

    all_answered = all(
        st.session_state.answers[i] is not None
        for i in range(len(st.session_state.mcqs))
    )

    st.button(
        "✅ Submit Answers",
        disabled=not all_answered,
        on_click=lambda: st.session_state.update({"submitted": True})
    )

# --------------------------------------------------
# Results View
# --------------------------------------------------
if st.session_state.submitted:
    st.header("📊 Results")

    score = 0
    total = len(st.session_state.mcqs)

    for i, q in enumerate(st.session_state.mcqs):
        user_choice = st.session_state.answers.get(i)
        correct_option = next(
            opt for opt in q["options"]
            if opt.startswith(q["answer"])
        )

        st.markdown(f"### Q{i+1}. {q['question']}")

        st.write(f"🧑 **Your answer:** {user_choice}")
        st.write(f"✅ **Correct answer:** {correct_option}")

        if user_choice == correct_option:
            st.success("✔️ Correct")
            score += 1
        else:
            st.error("❌ Wrong")

        st.info(f"📘 **Explanation:** {q['explanation']}")
        st.divider()

    st.subheader(f"🎯 Final Score: {score} / {total}")

    if score == total:
        st.success("🔥 Excellent! Perfect score.")
    elif score >= total * 0.6:
        st.warning("👍 Good job! Review explanations to improve.")
    else:
        st.error("📘 Needs revision. Go through the explanations carefully.")
