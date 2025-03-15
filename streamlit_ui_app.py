import pandas as pd
import faiss
import numpy as np
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

# Retrieve similar financial records
def retrieve_info(query, model, index, metadata, top_k=5, company=None, year=None):
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

# Streamlit UI
st.title("Financial Data Retrieval Chatbot")
query = st.text_input("Enter your financial query:")
company = st.text_input("Filter by company (optional):")
year = st.text_input("Filter by year (optional):")

if st.button("Search"):
    results = retrieve_info(query, model, index, metadata, company=company, year=year)
    
    if results:
        for res in results:
            st.write(f"**Company:** {res['Company']}, **Year:** {res['Year']}")
            st.write(f"**Market Cap:** {res['Market Cap(in B USD)']}B, **Revenue:** {res['Revenue']}, **Net Income:** {res['Net Income']}")
            st.write(f"**EPS:** {res['Earning Per Share']}")
            st.write("---")
    else:
        st.write("No results found.")
