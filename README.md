# Financial Chatbot - Advanced RAG

## 🚀 Setup Instructions

### 1️⃣ Create and Activate Virtual Environment
```sh
python -m venv env_financial_report_Chatbot
env_financial_report_Chatbot\Scripts\activate
```

### 2️⃣ Install Dependencies
```sh
pip install -r requirements.txt
```

### 3️⃣ Run the Application
#### Run the Basic RAG Version:
```sh
streamlit run app.py
```
#### Run the Advanced RAG Version:
```sh
streamlit run C:\Robert\GitWorkspace\CAI_Financial_Chatbot_AdavancedRAG\AdvanceRAG.py
```

---

## 🌍 Project Links
- 📂 **GitHub Repository**: [CAI Financial Chatbot](https://github.com/robertjohnson88/CAI_Financial_Chatbot)
- 💰 **Live RAG Chatbot**: [Basic RAG](https://cai-financial-chatbot.streamlit.app/)
- 📈 **Advanced RAG Chatbot**: [Advanced RAG](https://cai-financial-advancedrag.streamlit.app/)

---

## 📝 Query Examples

### ✅ High-Confidence Query (Relevant Financial Question)
> **Example:** "GOOG company revenue in year 2020"

### ⚠️ Low-Confidence Query (Future Financial Data)
> **Example:** "GOOG company revenue in year 2045"

### ❌ Irrelevant Query (General Knowledge - Not Supported)
> **Example:** "What is the capital of France?"

🔹 _Note: The chatbot does not handle general knowledge queries unrelated to financial data._

