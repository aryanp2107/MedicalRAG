import os
import json
import pickle
import gzip

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_core.documents import Document

CORPUS_DIR = r"C:\Users\USER\.vscode\Arxelos_Phase03\medical_corpus"
PKL_FILE = "vector_store.pkl.gz"

def print_json_schema(data, depth=0, max_depth=3):
    """Prints the structural keys of the JSON file to see how it's built."""
    if depth > max_depth:
        return
    indent = "  " * depth
    if isinstance(data, dict):
        for key, value in data.items():
            print(f"{indent}🔑 Key: '{key}' ({type(value).__name__})")
            if isinstance(value, (dict, list)):
                print_json_schema(value, depth + 1, max_depth)
    elif isinstance(data, list) and len(data) > 0:
        print(f"{indent}📋 List containing {len(data)} items:")
        print_json_schema(data[0], depth + 1, max_depth)

def extract_text_by_keys(data, target_keys=["text", "content", "body", "abstract", "title"]):
    """
    Structurally extracts text fields by matching known text keys.
    Handles case-insensitivity completely.
    """
    text_content = ""
    
    if isinstance(data, list):
        for item in data:
            text_content += extract_text_by_keys(item, target_keys)
            
    elif isinstance(data, dict):
        for key, value in data.items():
            # If the key matches any of our target structural fields and is a string
            if key.lower() in target_keys and isinstance(value, str):
                if value.strip():
                    text_content += value.strip() + "\n\n"
            # Keep navigating the tree structure
            elif isinstance(value, (dict, list)):
                text_content += extract_text_by_keys(value, target_keys)
                
    return text_content

def load_bioc_jsons(directory):
    print(f"Scanning target directory: {directory}")
    documents = []
    
    if not os.path.exists(directory):
        print(f"❌ Error: Directory '{directory}' NOT FOUND!")
        return documents
        
    all_files = os.listdir(directory)
    json_files = [f for f in all_files if f.lower().endswith(".json")]
    
    if not json_files:
        print("❌ No JSON files found in the directory.")
        return documents

    # --- STRUCTURAL SCHEMA DIAGNOSIS ---
    print("\n--- DIAGNOSING SCHEMA OF THE FIRST FILE ---")
    first_filepath = os.path.join(directory, json_files[0])
    try:
        with open(first_filepath, 'r', encoding='utf-8') as f:
            sample_data = json.load(f)
            print(f"Analyzing keys for {json_files[0]}:")
            print_json_schema(sample_data)
    except Exception as e:
        print(f"Could not read sample file for schema diagnosis: {e}")
    print("-------------------------------------------\n")

    print("Beginning structural parsing...")
    for filename in json_files:
        filepath = os.path.join(directory, filename)
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)
                
            # Extract text matching our structural data keys
            full_text = extract_text_by_keys(data)
            
            if full_text.strip():
                snippet = full_text.strip()[:60].replace('\n', ' ')
                print(f"✅ Parsed {filename} -> \"{snippet}...\"")
                doc = Document(page_content=full_text, metadata={"source": filename})
                documents.append(doc)
            else:
                print(f"⚠️ Warning: Skipped {filename} (Matched keys were empty or missing)")
                
        except Exception as e:
            print(f"Error reading {filename}: {e}")
            
    print(f"\nSuccessfully structurally loaded {len(documents)} papers into memory.")
    return documents

def build_database():
    docs = load_bioc_jsons(CORPUS_DIR)
    if not docs:
        print("No documents found. Structural parsing failed. Exiting.")
        return
    
    print("Splitting text into chunks...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, 
        chunk_overlap=200,
        separators=["\n\n", "\n", ".", " ", ""]
    )
    chunks = text_splitter.split_documents(docs)
    print(f"Created {len(chunks)} text chunks.")
    
    print("Initializing embedding model (Downloading ~90MB if first time)...")
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    
    print("Building In-Memory Vector Store...")
    vector_store = InMemoryVectorStore.from_documents(chunks, embeddings)
    
    print(f"Saving database to {PKL_FILE}...")
    with open(PKL_FILE, "wb") as f:
        pickle.dump(vector_store, f)
    
    print("\nTask 02 Complete! Vector database is saved as a local pickle file.")

if __name__ == "__main__":
    build_database()