import os
import streamlit as st
import chromadb
import ollama
import fitz  # PyMuPDF for PDF text extraction
import pytesseract  # OCR for images
from PIL import Image
from sentence_transformers import SentenceTransformer

# Initialize ChromaDB
chroma_client = chromadb.PersistentClient(path="./chroma_db")
collection = chroma_client.get_or_create_collection(name="laws_db")
forms_collection = chroma_client.get_or_create_collection(name="legal_forms")

# Load embedding model
embedding_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

# Streamlit UI
st.set_page_config(page_title="Legal Chatbot", layout="wide")
st.title("⚖️ AI-Powered Legal Assistant")

# Sidebar Navigation
st.sidebar.header("🔍 Features")
feature = st.sidebar.radio("Choose an option:", ["Legal Chat", "Upload & Identify Form"])

### 💬 LEGAL CHAT FUNCTIONALITY ###
def process_query_with_gemma(query):
    """
    Convert user queries into structured legal search terms.
    """
    prompt = f"""
    You are an AI trained in Indian law. A user has described their situation:
    "{query}"
    Your task:
    - Identify the key legal issue.
    - Convert it into a structured query to search Indian laws (IPC, Motor Vehicles Act, etc.).
    - DO NOT generate explanations or opinions.
    - Only return the refined legal search query.
    """
    response = ollama.chat(model="gemma2:2b", messages=[{"role": "user", "content": prompt}])
    return response['message']['content'].strip()

def generate_final_response(query, law_results):
    """
    Generate a legal response based on retrieved law sections.
    """
    law_text = "\n".join(law_results)
    prompt = f"""
    You are an AI specializing in Indian law. A user has asked:
    "{query}"
    Your task:
    - Provide a structured, user-friendly legal explanation.
    - Use ONLY the given legal sections (IPC, Motor Vehicles Act, Family Law, etc.).
    - DO NOT generate false information.
    - Explain in simple terms.
    Relevant Laws:
    {law_text}
    """
    response = ollama.chat(model="gemma2:2b", messages=[{"role": "user", "content": prompt}])
    return response['message']['content'].strip()

def search_laws(query):
    """
    Retrieve relevant legal sections from ChromaDB.
    """
    refined_query = process_query_with_gemma(query)
    results = collection.query(query_texts=[refined_query], n_results=5)
    
    if results["documents"]:
        metadata_list = results["metadatas"]
        if metadata_list and isinstance(metadata_list[0], list):
            metadata_list = metadata_list[0]

        law_results = [
            f"🔹 **{match.get('law_type', 'Unknown')} {match.get('section', 'Unknown')}**: {match.get('section_title', 'No Title')}\n"
            f"{match.get('chapter_title', 'No Chapter')}\n"
            f"{match.get('description', 'No Description')}"
            for match in metadata_list
        ]
        return generate_final_response(query, law_results)
    
    return "❌ No matching legal section found!"

### 📄 FORM HANDLING FUNCTIONALITY ###
def extract_text_from_pdf(pdf_path):
    """ Extract text from PDF forms. """
    doc = fitz.open(pdf_path)
    text = "".join([page.get_text("text") for page in doc])
    return text.strip()

def extract_text_from_image(image):
    """ Extract text from images using OCR. """
    return pytesseract.image_to_string(image)

def identify_form(text):
    """
    Identify form and retrieve full metadata from ChromaDB.
    """
    query_embedding = embedding_model.encode([text])[0]
    results = forms_collection.query(query_embeddings=[query_embedding], n_results=1)

    if results["documents"]:
        matched_metadata = results["metadatas"][0][0]
        return {
            "form_type": matched_metadata.get("form_type", "Unknown Form"),
            "text": matched_metadata.get("text", "No description available.")
        }
    return None

def generate_form_filling_guidance(form_metadata):
    """
    Provide instructions based on form metadata.
    """
    if not form_metadata:
        return "Could not identify the form. Please try another document."
    
    prompt = f"""
    You are an AI assistant specializing in **Indian legal documentation**. A user has uploaded:
    **📄 Form Name:** {form_metadata['form_type']}
    **📜 Extracted Text:** {form_metadata['text']}
    Your task:
    - Provide relevant legal context and filling instructions based on extracted text.
    - Ensure accuracy by referring strictly to the retrieved form text.
    """
    response = ollama.chat(model="gemma2:2b", messages=[{"role": "user", "content": prompt}])
    return response['message']['content'].strip()

### 🚀 STREAMLIT INTERFACE ###
if feature == "Legal Chat":
    st.subheader("💬 Legal Chat")
    user_query = st.text_area("Ask a legal question:")
    if st.button("Search Law") and user_query:
        response = search_laws(user_query)
        st.markdown(f"**📝 Response:**\n\n{response}")
    elif not user_query:
        st.warning("Please enter a legal question.")

elif feature == "Upload & Identify Form":
    st.subheader("📄 Upload Form for Identification")
    uploaded_file = st.file_uploader("Upload a legal form (PDF or Image)", type=["pdf", "png", "jpg", "jpeg"])
    if uploaded_file:
        extracted_text = extract_text_from_pdf("temp.pdf") if uploaded_file.type == "application/pdf" else extract_text_from_image(Image.open(uploaded_file))
        if extracted_text:
            form_metadata = identify_form(extracted_text)
            if form_metadata:
                st.success(f"✅ Identified Form: **{form_metadata['form_type']}**")
                if st.button("Get Form Filling Guidance"):
                    guidance = generate_form_filling_guidance(form_metadata)
                    st.markdown(f"**📌 How to Fill:**\n\n{guidance}")
            else:
                st.error("Could not identify the form. Try another file.")
        else:
            st.error("Could not extract text. Try another file.")
