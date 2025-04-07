import os
import fitz  # PyMuPDF for PDF text extraction
import chromadb
from sentence_transformers import SentenceTransformer

# Initialize ChromaDB client
chroma_client = chromadb.PersistentClient(path="./chroma_db")
forms_collection = chroma_client.get_or_create_collection(name="legal_forms")

# Load embedding model
embedding_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

def extract_text_from_pdf(pdf_path):
    """
    Extracts meaningful text sections from a PDF form.
    """
    doc = fitz.open(pdf_path)
    sections = []

    for page in doc:
        text = page.get_text("text")
        if text.strip():  # Ensure non-empty sections
            sections.extend(text.split("\n\n"))  # Split based on paragraph spacing

    return [section.strip() for section in sections if section.strip()]

def store_form_in_chroma(pdf_path, form_name):
    """
    Extracts text from a PDF form, converts to embeddings, and stores in ChromaDB.
    """
    print(f"📄 Processing Form: {form_name}...")
    
    form_sections = extract_text_from_pdf(pdf_path)

    # Generate embeddings for sections
    embeddings = embedding_model.encode(form_sections)

    # Store each section in ChromaDB
    for i, section in enumerate(form_sections):
        forms_collection.add(
            ids=[f"{form_name}_section_{i}"],
            embeddings=[embeddings[i].tolist()],
            metadatas=[{
                "form_type": form_name,
                "text": section
            }]
        )

    print(f"✅ {form_name} stored in ChromaDB!")

def process_multiple_forms(folder_path):
    """
    Processes all PDF forms in a given folder and stores them.
    """
    for file in os.listdir(folder_path):
        if file.endswith(".pdf"):
            pdf_path = os.path.join(folder_path, file)
            form_name = os.path.splitext(file)[0]  # Remove .pdf extension
            store_form_in_chroma(pdf_path, form_name)

# Example: Process all PDFs inside "forms" folder
process_multiple_forms("./forms")
