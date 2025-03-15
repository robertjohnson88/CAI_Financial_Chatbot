import pandas as pd
import faiss
import numpy as np
import re
import rank_bm25
from sentence_transformers import SentenceTransformer
import streamlit as st
from collections import defaultdict

# Load dataset
def load_dataset(csv_path):
    df = pd.read_csv(csv_path)
    df.fillna(value='', inplace=True)
    for col in df.select_dtypes(include=['float64']).columns:
        df[col] = df[col].astype(str)
    return df

# Convert data to text format
def prepare_text_data(df):
    text_data = df.apply(lambda row: f"{row['Year']} {row['Company']} {row['Category']} Market Cap: {row['Market Cap(in B USD)']}B Revenue: {row['Revenue']} Net Income: {row['Net Income']} EPS: {row['Earning Per Share']}", axis=1)
    return text_data.tolist()

# Compute BM25 index
def compute_bm25_index(text_data):
    tokenized_corpus = [text.split() for text in text_data]
    bm25 = rank_bm25.BM25Okapi(tokenized_corpus)
    return bm25, tokenized_corpus

# Save embeddings to FAISS
def save_to_faiss(embeddings, df, index_path="financial_index.faiss", metadata_path="metadata.npy"):
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)
    faiss.write_index(index, index_path)
    
    metadata = df.to_dict(orient="records")
    np.save(metadata_path, metadata)
    print("Saved embeddings and metadata.")

# Load FAISS and metadata
def load_faiss(index_path="financial_index.faiss", metadata_path="metadata.npy"):
    index = faiss.read_index(index_path)
    metadata = np.load(metadata_path, allow_pickle=True)
    return index, metadata

# Extract year from query
def extract_year(query):
    match = re.search(r'\b(19\d{2}|20\d{2})\b', query)
    return match.group(0) if match else None

# Extract company from query
def extract_company(query, company_list):
    for company in company_list:
        if company.lower() in query.lower():
            return company
    return None

# Check for irrelevant questions
def is_irrelevant_question(query):
    irrelevant_keywords = ["capital", "president", "weather", "movie", "country"]
    return any(word in query.lower() for word in irrelevant_keywords)

# Retrieve results with hybrid retrieval
def retrieve_info(query, model, index, metadata, bm25, tokenized_corpus, top_k=10):
    if is_irrelevant_question(query):
        return "I'm here to assist with financial data. Please ask relevant financial questions."
    
    year = extract_year(query)
    company_list = list(set([entry['Company'] for entry in metadata]))
    company = extract_company(query, company_list)
    
    query_embedding = model.encode([query])
    distances, indices = index.search(query_embedding, top_k)
    faiss_results = [metadata[i] for i in indices[0] if i < len(metadata)]
    
    bm25_scores = bm25.get_scores(query.split())
    bm25_indices = np.argsort(bm25_scores)[::-1][:top_k]
    bm25_results = [metadata[i] for i in bm25_indices]
    
    combined_results = list({(res['Company'], res['Year']): res for res in faiss_results + bm25_results}.values())
    
    # Apply filtering based on company and year
    if company:
        combined_results = [res for res in combined_results if res["Company"].lower() == company.lower()]
    if year:
        combined_results = [res for res in combined_results if str(res["Year"]) == str(year)]
    
    return sorted(combined_results, key=lambda x: float(x.get('Year', 0)), reverse=True)

# Load model and data
csv_path = "./data/financial_statements.csv"
model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

df = load_dataset(csv_path)
text_data = prepare_text_data(df)
embeddings = model.encode(text_data, convert_to_numpy=True)

save_to_faiss(embeddings, df)
index, metadata = load_faiss()
bm25, tokenized_corpus = compute_bm25_index(text_data)

# Streamlit UI - Improved Chatbox Style
st.markdown("""
    <style>
        .chatbox-container {
            border: 2px solid #007bff;
            padding: 15px;
            border-radius: 10px;
            background-color: #f8f9fa;
            max-width: 700px;
            margin: auto;
        }
        .user-message {
            background-color: #d1e7fd;
            padding: 10px;
            border-radius: 10px;
            margin-bottom: 5px;
        }
        .bot-message {
            background-color: #e2e3e5;
            padding: 10px;
            border-radius: 10px;
            margin-bottom: 5px;
        }
    </style>
""", unsafe_allow_html=True)

st.markdown("<h3>CAI Assignment 2 - Group 87</h3>", unsafe_allow_html=True)
st.markdown("<h3>Adavnced RAG Financial Data Retrieval Chatbot</h3>", unsafe_allow_html=True)

with st.container():
    query = st.text_input("Enter your financial query:")
    if st.button("Search"):
        if "chat_history" not in st.session_state:
            st.session_state.chat_history = []
        if query:
            st.session_state.chat_history.insert(0, ("user", query))
            results = retrieve_info(query, model, index, metadata, bm25, tokenized_corpus)
            
            if isinstance(results, str):
                response = f"<strong>Bot:</strong> {results}"
            elif results:
                response = "<strong>Query Results:</strong><br><table><tr><th>Year</th><th>Company</th><th>Category</th><th>Revenue</th><th>Net Income</th></tr>"
                for res in results:
                    response += f"<tr><td>{res['Year']}</td><td>{res['Company']}</td><td>{res['Category']}</td><td>{res['Revenue']}</td><td>{res['Net Income']}</td></tr>"
                response += "</table>"
            else:
                response = "<strong>No results found.</strong>"
            
            st.session_state.chat_history.insert(1, ("bot", response))
            st.rerun()

# Display chat history
st.markdown('<div class="chatbox-container">', unsafe_allow_html=True)
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

for chat in st.session_state.chat_history:
    role, text = chat
    if role == "user":
        st.markdown(f'<div class="user-message"><strong>User:  </strong> {text}</div>', unsafe_allow_html=True)
    else:
        st.markdown(f'<div class="bot-message"><strong>Bot:  </strong> {text}</div>', unsafe_allow_html=True)
st.markdown('</div>', unsafe_allow_html=True)