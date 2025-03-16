import pandas as pd
import faiss
import numpy as np
import re
import rank_bm25
from sentence_transformers import SentenceTransformer, util
import streamlit as st
from collections import defaultdict

def load_dataset(csv_path):
    df = pd.read_csv(csv_path)
    for col in df.select_dtypes(include=['float64']).columns:
        df[col] = df[col].astype(str)
    df.fillna('', inplace=True)
    return df

def prepare_text_data(df):
    text_data = df.apply(lambda row: f"{row['Year']} {row['Company']} {row['Category']} Market Cap: {row['Market Cap(in B USD)']}B Revenue: {row['Revenue']} Net Income: {row['Net Income']} EPS: {row['Earning Per Share']}", axis=1)
    return text_data.tolist()

def compute_bm25_index(text_data):
    tokenized_corpus = [text.split() for text in text_data]
    bm25 = rank_bm25.BM25Okapi(tokenized_corpus)
    return bm25, tokenized_corpus

def extract_year(query):
    match = re.search(r'\b(19\d{2}|20\d{2})\b', query)
    return match.group(0) if match else None

def extract_company(query, company_list):
    for company in company_list:
        if company.lower() in query.lower():
            return company
    return None

def is_valid_query(query):
    blocked_keywords = ["hack", "exploit", "illegal", "scam", "cheat"]
    return not any(word in query.lower() for word in blocked_keywords)

def is_irrelevant_question(query):
    irrelevant_keywords = ["capital", "president", "weather", "movie", "country"]
    return any(word in query.lower() for word in irrelevant_keywords)

def compute_confidence(faiss_scores, bm25_scores):
    faiss_confidence = (1 - (faiss_scores / np.max(faiss_scores))) if len(faiss_scores) > 0 else np.array([])
    bm25_confidence = bm25_scores / np.max(bm25_scores) if np.max(bm25_scores) > 0 else np.array([])
    combined_confidence = (faiss_confidence + bm25_confidence) / 2
    return combined_confidence

def retrieve_info(query, model, index, metadata, bm25, tokenized_corpus, top_k=5):
    if not is_valid_query(query):
        return "Your query contains restricted terms and cannot be processed."
    if is_irrelevant_question(query):
        return "I'm here to assist with financial data. Please ask relevant financial questions."
    
    year = extract_year(query)
    company_list = list(set([entry['Company'] for entry in metadata]))
    company = extract_company(query, company_list)
    
    query_embedding = model.encode([query])
    distances, indices = index.search(query_embedding, top_k)
    faiss_results = [metadata[i] for i in indices[0] if i < len(metadata)]
    faiss_scores = distances[0]
    
    bm25_scores = bm25.get_scores(query.split())
    bm25_indices = np.argsort(bm25_scores)[::-1][:top_k]
    bm25_results = [metadata[i] for i in bm25_indices]
    
    confidence_scores = compute_confidence(faiss_scores, bm25_scores[bm25_indices])
    combined_results = list({(res['Company'], res['Year']): res for res in faiss_results + bm25_results}.values())
    
    if company:
        combined_results = [res for res in combined_results if res["Company"].lower() == company.lower()]
    if year:
        combined_results = [res for res in combined_results if str(res["Year"]) == str(year)]
    
    for i, res in enumerate(combined_results[:top_k]):
        res['confidence'] = confidence_scores[i] if i < len(confidence_scores) else 0
    
    return sorted(combined_results, key=lambda x: x.get('confidence', 0), reverse=True)

csv_path = "./data/financial_statements.csv"
model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

df = load_dataset(csv_path)
text_data = prepare_text_data(df)
embeddings = model.encode(text_data, convert_to_numpy=True)

index = faiss.IndexFlatL2(embeddings.shape[1])
index.add(embeddings)
metadata = df.to_dict(orient="records")

bm25, tokenized_corpus = compute_bm25_index(text_data)

st.markdown("<h3>Financial Data Retrieval Chatbot</h3>", unsafe_allow_html=True)

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
            response = "<strong>Query Results:</strong><br><table><tr><th>Year</th><th>Company</th><th>Category</th><th>Revenue</th><th>Net Income</th><th>Confidence</th></tr>"
            for res in results:
                response += f"<tr><td>{res['Year']}</td><td>{res['Company']}</td><td>{res['Category']}</td><td>{res['Revenue']}</td><td>{res['Net Income']}</td><td>{res['confidence']:.2f}</td></tr>"
            response += "</table>"
        else:
            response = "<strong>No results found.</strong>"
        
        st.session_state.chat_history.insert(1, ("bot", response))
        st.rerun()
