# CAI Assignment 2 - Group 87

## Financial Chatbot

### Team Members:
- **Robert Johnson S** (2023aa05813)
- **Sathish Kumar .S** (2023aa05698)
- **Sathish N M** (2023aa05812)
- **Ritesh Ranjan** (2023aa05888)
- **Vibhav Raman** (2023aa05805)

---

## 📌 Problem Statement
Develop a **Retrieval-Augmented Generation (RAG)** model to answer financial questions based on company financial statements from the last two years.

### Key Requirements:
- **Use only open-source embedding models**
- **Small Open-Source Language Model (SLM)** for response generation (no proprietary APIs)
- **Implement a Guardrail** (Input-side or Output-side)
- **Develop a user-friendly application** (Web-based, CLI, or GUI)
- **Implement an Advanced RAG Technique** based on the group number

### Advanced RAG Techniques Implemented:
- **Hybrid Search** (Sparse + Dense Retrieval)
- **Re-Ranking with Cross-Encoders**
- **Chunk Merging & Adaptive Retrieval**

---

## 🚀 Setup Instructions

### 1️⃣ Create and Activate Virtual Environment
```sh
python -m venv env_financial_report_Chatbot
source env_financial_report_Chatbot/bin/activate  # For Mac/Linux
env_financial_report_Chatbot\Scripts\activate  # For Windows
```

### 2️⃣ Install Dependencies
```sh
pip install -r requirements.txt
```

### 3️⃣ Run the Application
#### Basic RAG Version
```sh
streamlit run app.py
```
#### Advanced RAG Version
```sh
streamlit run AdvanceRAG.py
```

---

## 📊 Data Collection & Preprocessing
- Dataset: **financial_statements.csv** (last two years)
- **Preprocessing Steps:**
  - Handling missing values (interpolation, mean imputation, or deletion)
  - Standardizing text-based financial data
  - **Chunking Strategy:**
    - Column-Level (financial metrics like Net Income, Revenue)
    - Time-Series Chunks (grouping by year & company)
    - **Adaptive Merging** based on similarity scores

---

## 🔍 Basic RAG Implementation
1. Load financial data from CSV
2. Chunk text into smaller sections
3. Generate embeddings and store them in FAISS
4. Retrieve relevant chunks based on user query
5. Generate a response using a **small LLM**
6. Display response with **confidence score**

---

## 📈 Advanced RAG Implementation
- **Step 1:** **Sparse Retrieval** (BM25) for keyword filtering
- **Step 2:** **Dense Retrieval** (FAISS/ChromaDB) for semantic search
- **Step 3:** **Adaptive Chunk Merging** to improve retrieval quality
- **Step 4:** **Re-Ranking** using Cross-Encoders

---

## 🖥 UI Development (Streamlit)
- Accept user queries
- Display answer & confidence score
- Ensure clear formatting & responsiveness

---

## 🛡 Guardrail Implementation
### ✅ **Input Guardrail (Pre-filtering Questions)**
- Regex-based filtering (prevent ambiguous/irrelevant questions)
- Question rewriter (rewrite poorly phrased financial queries)

### ✅ **Output Guardrail (Fact Verification)**
- **Confidence Thresholding** (prevent hallucinated responses)
- **Fact-checking** against retrieved financial data

---

## 🛠 Testing & Validation
Test with **3 types of queries:**
1. **High-confidence question:** "What was Google’s revenue in 2020?"
2. **Low-confidence question:** "What will be Google’s revenue in 2045?"
3. **Irrelevant question:** "What is the capital of France?"

---

## 🔗 Project Links
- 📂 **GitHub Repository:** [CAI Financial Chatbot](#)
- 💰 **Live RAG Chatbot**: [Basic RAG](https://cai-financial-chatbot.streamlit.app/)
- 📈 **Advanced RAG Chatbot**: [Advanced RAG](https://cai-financial-advancedrag.streamlit.app/)

---

## 📝 Query Examples
✅ **High-Confidence Query:** "GOOG company revenue in year 2020"
⚠️ **Low-Confidence Query:** "GOOG company revenue in year 2045"
❌ **Irrelevant Query:** "What is the capital of France?"

**Note:** This chatbot only handles financial queries.

---

