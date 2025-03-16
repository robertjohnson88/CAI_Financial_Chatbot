import pandas as pd
import faiss
import numpy as np
import re
import rank_bm25
from sentence_transformers import SentenceTransformer, util
import streamlit as st
from collections import defaultdict


# Load dataset from a CSV file
def load_dataset(csv_path):
    """Reads a CSV file and processes financial data."""
    df = pd.read_csv(csv_path)
    for col in df.select_dtypes(include=['float64']).columns:
        df[col] = df[col].astype(str)  # Convert float columns to string to avoid dtype issues
    df.fillna('', inplace=True)  # Replace NaN values with empty strings
    return df

# Convert dataset to textual format for retrieval
def prepare_text_data(df):
    """Formats financial data into a textual representation for retrieval."""
    text_data = df.apply(lambda row: f"{row['Year']} {row['Company']} {row['Category']} Market Cap: {row['Market Cap(in B USD)']}B Revenue: {row['Revenue']} Net Income: {row['Net Income']} EPS: {row['Earning Per Share']}", axis=1)
    return text_data.tolist()

# Compute BM25 index for keyword-based retrieval
def compute_bm25_index(text_data):
    """Creates a BM25 index for ranking documents based on keyword relevance."""
    tokenized_corpus = [text.split() for text in text_data]
    bm25 = rank_bm25.BM25Okapi(tokenized_corpus)
    return bm25, tokenized_corpus

# Save embeddings to FAISS for efficient similarity search
def save_to_faiss(embeddings, df, index_path="financial_index.faiss", metadata_path="metadata.npy"):
    """Save sentence embeddings and metadata to FAISS for fast retrieval."""
    dimension = embeddings.shape[1]  # Get the embedding dimension
    index = faiss.IndexFlatL2(dimension)  # Create a FAISS index for L2 distance search
    index.add(embeddings)  # Add embeddings to the index
    faiss.write_index(index, index_path)  # Save FAISS index to disk
    
    metadata = df.to_dict(orient="records")  # Convert DataFrame to a list of dictionaries
    np.save(metadata_path, metadata)  # Save metadata as a NumPy file
    print("Saved embeddings and metadata.")

# Load FAISS index and metadata
def load_faiss(index_path="financial_index.faiss", metadata_path="metadata.npy"):
    """Loads the FAISS index and metadata from disk."""
    index = faiss.read_index(index_path)  # Load FAISS index
    metadata = np.load(metadata_path, allow_pickle=True)  # Load metadata as a NumPy object
    return index, metadata

# Extract the year from a user query
def extract_year(query):
    """Extracts a four-digit year from the query if present."""
    match = re.search(r'\b(19\d{2}|20\d{2})\b', query)
    return match.group(0) if match else None

# Extract company name from a user query
def extract_company(query, company_list):
    """Identifies a company name in the query from a predefined list."""
    for company in company_list:
        if company.lower() in query.lower():
            return company
    return None

# Validate user query for security and relevance
def is_invalid_query(query):
    """Checks if the query contains restricted or unsafe terms."""
    blocked_keywords = ["hack", "exploit", "illegal", "scam", "cheat"]
    return any(word in query.lower() for word in blocked_keywords)

# Check if the user query is irrelevant
def is_irrelevant_question(query):
    """Detects if the query is unrelated to financial data."""
    irrelevant_keywords = ["capital", "president", "weather", "movie", "country"]
    return any(word in query.lower() for word in irrelevant_keywords)

# Compute confidence scores using a hardcoded value
def compute_confidence(faiss_scores, bm25_scores):
    """Returns a hardcoded confidence score of 0.85 for all results."""
    return np.full_like(faiss_scores, 0.85)  # Fixed confidence score of 0.85

# Function to display results including confidence scores
def display_results(results):
    """Formats and prints the retrieved results along with confidence scores."""
    for res in results:
        print(f"Company: {res['Company']}, Year: {res['Year']}, Revenue: {res['Revenue']}, Confidence: 0.85")

# Retrieve results with hybrid retrieval
def retrieve_info(query, model, index, metadata, bm25, tokenized_corpus, top_k=10):
    if is_irrelevant_question(query):
        return "I'm here to assist with financial data. Please ask relevant financial questions."+"<strong> Confidence: 0.00</strong>"
    
    if is_invalid_query(query):
        return "Restricted or unsafe terms was used. Please ask relevant financial questions."+"<strong> Confidence: 0.00</strong>"
    
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
        .stApp {background-color: #9ca3a9;}
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
st.markdown("<h3>Advanced RAG Financial Data Retrieval Chatbot</h3>", unsafe_allow_html=True)


with st.container():
    query = st.text_input("Enter your financial query:")
    if st.button("Search"):
        if "chat_history" not in st.session_state:
            st.session_state.chat_history = []
        if query:
            st.session_state.chat_history.insert(0, ("user", query))
            results = retrieve_info(query, model, index, metadata, bm25, tokenized_corpus)
            
            if isinstance(results, str):
                response = f"<strong></strong> {results}"
            elif results:
                response = "<strong>Query Results:</strong><br><table><tr><th>Year</th><th>Company</th><th>Category</th><th>Revenue</th><th>Net Income</th><th>Confidence</th></tr>"
                for res in results:
                    response += f"<tr><td>{res['Year']}</td><td>{res['Company']}</td><td>{res['Category']}</td><td>{res['Revenue']}</td><td>{res['Net Income']}</td><td>{1/len(results):.3f}</td></tr>"
                response += "</table>"
            else:
                response = "<strong>No results found.</strong><strong> Confidence: 0.50</strong>"
            
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