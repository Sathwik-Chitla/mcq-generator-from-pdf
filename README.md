# 📘 PDF → MCQ Generator  

A Streamlit web app that generates **Multiple Choice Questions (MCQs)** from any uploaded PDF using **LangChain + Ollama (LLaMA 3.1)**.  

## 🚀 Features  
- Upload any PDF and extract text automatically  
- Generate **customizable number of MCQs** (1–20)  
- Export MCQs to **Word (.docx)** or **PDF**  
- Powered by **Ollama (LLaMA-3.1) + LangChain**  

---

## ⚙️ Installation  

1. **Clone this repository**  
   ```bash
   git clone https://github.com/your-username/mcq-generator.git
   cd mcq-generator
   ```

2. **Create a virtual environment (optional but recommended)**  
   ```bash
   python -m venv .venv
   .venv\Scripts\activate   # On Windows
   source .venv/bin/activate  # On Mac/Linux
   ```

3. **Install dependencies**  
   ```bash
   pip install -r requirements.txt
   ```

4. **Install & run Ollama**  
   - Download Ollama: [https://ollama.ai](https://ollama.ai)  
   - Pull the LLaMA 3.1 model:  
     ```bash
     ollama pull llama3.1
     ```
   - Start the Ollama server:  
     ```bash
     ollama serve
     ```

---

## ▶️ Usage  

Run the Streamlit app:  
```bash
streamlit run mcq_app.py
```

Then open [http://localhost:8501](http://localhost:8501) in your browser.  

---

## 📂 Project Structure  
```
mcq-generator/
│── mcq_app.py              # Main Streamlit app
│── mcq_from_pdf_min.py     # Minimal script version
│── requirements.txt        # Dependencies
│── README.md               # Project description + usage
```

---

## 📸 Screenshots  
*(Add screenshots of your app here, e.g., upload PDF, generated MCQs, download buttons)*  

---

## 🎯 Demo  

Example MCQs generated from a sample internship PDF:  

**Q1. What is the duration of the internship?**  
- 3 Months  
- 6 Months  
- 9 Months  
- 12 Months  
✅ Answer: **6 Months**  

**Q2. What is the stipend for this internship?**  
- ₹10,000/month  
- ₹15,000/month  
- ₹20,000/month  
- ₹25,000/month  
✅ Answer: **₹15,000/month**  

---

## 📜 License  
MIT License – feel free to use and modify.  
