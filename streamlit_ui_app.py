import pandas as pd
import faiss
import numpy as np
import re
from sentence_transformers import SentenceTransformer
import streamlit as st

# Load the dataset
def load_dataset(csv_path):
    df = pd.read_csv(csv_path)
    df.fillna("", inplace=True)  # Handle missing values
    return df

# Convert financial data to text format for embedding
def prepare_text_data(df):
    text_data = df.apply(lambda row: f"{row['Year']} {row['Company']} {row['Category']} Market Cap: {row['Market Cap(in B USD)']}B Revenue: {row['Revenue']} Net Income: {row['Net Income']} EPS: {row['Earning Per Share']}", axis=1)
    return text_data.tolist()

# Save embeddings to FAISS
def save_to_faiss(embeddings, df, index_path="financial_index.faiss", metadata_path="metadata.npy"):
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)
    faiss.write_index(index, index_path)
    
    # Save metadata
    metadata = df.to_dict(orient="records")
    np.save(metadata_path, metadata)
    print("Saved embeddings and metadata.")

# Load FAISS index and metadata
def load_faiss(index_path="financial_index.faiss", metadata_path="metadata.npy"):
    index = faiss.read_index(index_path)
    metadata = np.load(metadata_path, allow_pickle=True)
    return index, metadata

# Extract year from query
def extract_year(query):
    match = re.search(r'\b(19\d{2}|20\d{2})\b', query)
    return match.group(0) if match else None

# Retrieve similar financial records
def retrieve_info(query, model, index, metadata, top_k=5, company=None):
    year = extract_year(query)
    query_embedding = model.encode([query])
    distances, indices = index.search(query_embedding, top_k)
    results = [metadata[i] for i in indices[0] if i < len(metadata)]
    
    # Apply filtering based on company and year
    if company:
        results = [res for res in results if res["Company"].lower() == company.lower()]
    if year:
        results = [res for res in results if str(res["Year"]) == str(year)]
    
    return results

# Load model and data globally
csv_path = "./data/financial_statements.csv"
model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

df = load_dataset(csv_path)
text_data = prepare_text_data(df)
embeddings = model.encode(text_data, convert_to_numpy=True)

save_to_faiss(embeddings, df)

# Load index & metadata for retrieval
global index, metadata
index, metadata = load_faiss()

# Streamlit UI - Chatbot style
st.markdown("""
    <style>
        .chat-container {
            max-width: 700px;
            margin: auto;
            padding: 20px;
            border-radius: 10px;
            background-color: #f8f9fa;
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
        .result-box {
            background: white;
            border-radius: 10px;
            padding: 10px;
            box-shadow: 0px 0px 10px gray;
            margin-top: 10px;
        }
    </style>
""", unsafe_allow_html=True)

st.markdown("<h3>Financial Data Retrieval Chatbot</h3>", unsafe_allow_html=True)
st.markdown('<div class="chat-container">', unsafe_allow_html=True)

query = st.text_input("Enter your financial query:")
company = st.text_input("Filter by company (optional):")

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if st.button("Search"): 
            st.session_state.chat_history = []

if query:
        st.session_state.chat_history.append(("user", query))
        results = retrieve_info(query, model, index, metadata, company=company)
        
        if results:
            response = "<div class='result-box'><strong>Query Results:</strong><br><table><tr><th>Year</th><th>Company</th><th>Category</th><th>Revenue</th><th>Net Income</th></tr>"
            for res in results:
                response += f"<tr><td>{res['Year']}</td><td>{res['Company']}</td><td>{res['Category']}</td><td>{res['Revenue']}</td><td>{res['Net Income']}</td></tr>"
            response += "</table></div>"
        else:
            response = "<div class='result-box'><strong>No results found.</strong></div>"
        
        st.session_state.chat_history.append(("bot", response))
        st.rerun()

# Display chat history below the search button
for chat in st.session_state.chat_history:
    role, text = chat
    if role == "user":
        st.markdown(f'<div class="user-message">**User:** {text}</div>', unsafe_allow_html=True)
    else:
        st.markdown(f'<div class="bot-message">**Bot:** {text}</div>', unsafe_allow_html=True)

st.markdown('</div>', unsafe_allow_html=True)
