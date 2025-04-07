import json
import chromadb
import glob
from sentence_transformers import SentenceTransformer

# Initialize ChromaDB client (Persistent Storage)
chroma_client = chromadb.PersistentClient(path="./chroma_db")
collection = chroma_client.get_or_create_collection(name="laws_db")

# Load an embedding model (lightweight for legal documents)
embedding_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")  # 384-dim vector

def extract_law_data(law, law_type):
    """Extracts relevant fields from various legal JSON structures."""
    if law_type == "HMA" and isinstance(law, str):
        # Handle HMA.json format (comma-separated string)
        try:
            chapter, section, section_title, section_desc = law.split(",", 3)
            return f"{law_type}-{section.strip()}", section_title.strip(), section_desc.strip(), chapter.strip(), "No Chapter"
        except ValueError:
            return f"{law_type}-Unknown", "No Title", "Malformed Data", "Unknown", "No Chapter"
    
    section_id = f"{law_type}-{law.get('section', law.get('section_number', 'Unknown'))}"
    title = law.get("title") or law.get("section_title") or "No Title"
    description = law.get("description") or law.get("section_desc") or "No description available"
    chapter = law.get("chapter", "Unknown")
    chapter_title = law.get("chapter_title", law.get("Act", "No Chapter"))
    
    return section_id, title, description, chapter, chapter_title

def add_laws_to_chromadb(law_data, law_type):
    """Insert law data into ChromaDB with embeddings."""
    for law in law_data:
        section_id, title, description, chapter, chapter_title = extract_law_data(law, law_type)
        
        # Generate embeddings for the section description
        embedding = embedding_model.encode(description).tolist()
        
        collection.add(
            ids=[section_id],
            embeddings=[embedding],  # Store generated embedding
            documents=[description],
            metadatas=[{
                "law_type": law_type,
                "chapter": chapter,
                "chapter_title": chapter_title,
                "section": law.get("section", law.get("section_number", "Unknown")) if isinstance(law, dict) else section_id.split("-")[1],
                "section_title": title
            }]
        )
    print(f"✅ {law_type} data loaded into ChromaDB with embeddings!")

# Process all JSON files dynamically
json_files = glob.glob("*.json")  # Finds all JSON files in the current directory
for json_file in json_files:
    law_type = json_file.replace(".json", "").upper()  # Extract file name as law type
    with open(json_file, "r", encoding="utf-8") as file:
        try:
            law_data = json.load(file)
            if isinstance(law_data, dict):
                law_data = [law_data]  # Convert single dict to list
            if isinstance(law_data, str):
                law_data = [law_data]  # Ensure HMA data is a list
            add_laws_to_chromadb(law_data, law_type)
        except json.JSONDecodeError:
            print(f"❌ Error reading {json_file}. Ensure it is properly formatted JSON.")
