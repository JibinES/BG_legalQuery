import os
import glob
import fitz  # PyMuPDF for PDF extraction
import chromadb
from sentence_transformers import SentenceTransformer
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Initialize ChromaDB client
chroma_client = chromadb.PersistentClient(path="./chroma_db")
collection = chroma_client.get_or_create_collection(name="laws_db")

# Load embedding model
embedding_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

# Function to extract text from PDF
def extract_text_from_pdf(pdf_path):
    """Extracts text from a PDF file."""
    text = ""
    try:
        doc = fitz.open(pdf_path)
        for page in doc:
            text += page.get_text("text") + "\n"
    except Exception as e:
        print(f"❌ Error reading {pdf_path}: {e}")
    return text.strip()

# Function to split text into chunks
def split_text_into_chunks(text, chunk_size=500, overlap=50):
    """Splits text into chunks for embedding."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=overlap,
        separators=["\n", " ", ""]  # Prioritize splitting at newlines
    )
    return splitter.split_text(text)

# Function to add PDF content to ChromaDB
def add_pdf_to_chromadb(pdf_path):
    """Processes a PDF and adds its text chunks to ChromaDB with embeddings."""
    pdf_name = os.path.basename(pdf_path).replace(".pdf", "")
    text = extract_text_from_pdf(pdf_path)
    if not text:
        print(f"⚠️ No text found in {pdf_path}")
        return
    
    chunks = split_text_into_chunks(text)
    for idx, chunk in enumerate(chunks):
        embedding = embedding_model.encode(chunk).tolist()
        
        collection.add(
            ids=[f"{pdf_name}-chunk-{idx}"],
            embeddings=[embedding],
            documents=[chunk],
            metadatas=[{"source": pdf_name, "chunk_index": idx}]
        )
    
    print(f"✅ {pdf_name} loaded into ChromaDB with {len(chunks)} chunks!")

# Process all PDF files dynamically
pdf_files = glob.glob("*.pdf")  # Finds all PDFs in the current directory
for pdf_file in pdf_files:
    add_pdf_to_chromadb(pdf_file)

print("🚀 All PDFs processed and stored in ChromaDB!")