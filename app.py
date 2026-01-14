import streamlit as st
from rag_backend import load_pdf, ingest_pdf_to_pinecone, generate_mcqs

st.set_page_config(
    page_title="RAG MCQ Quiz Generator",
    page_icon="📘",
    layout="wide"
)

st.title("📘 RAG-Based MCQ Quiz Generator")
st.caption("Difficulty-aware MCQs with explanations using RAG + OpenAI")

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
        1, 10, 5
    )

    generate_btn = st.button("🚀 Generate Quiz")

if "mcqs" not in st.session_state:
    st.session_state.mcqs = []

if "answers" not in st.session_state:
    st.session_state.answers = {}

if "submitted" not in st.session_state:
    st.session_state.submitted = False

if generate_btn:
    if not pdf:
        st.warning("Please upload a PDF.")
    elif not topic.strip():
        st.warning("Please enter a topic.")
    else:
        with st.spinner("Processing PDF..."):
            text = load_pdf(pdf)
            ingest_pdf_to_pinecone(text)

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
            st.error("No MCQs generated. Try another topic.")

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

    st.subheader(f"🎯 Score: {score}/{len(st.session_state.mcqs)}")
