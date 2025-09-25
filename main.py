import os
from llama_index.core import (
    VectorStoreIndex,
    SimpleDirectoryReader,
    StorageContext,
    load_index_from_storage,
)
from llama_index.llms.ollama import Ollama

# --- Configuration ---
DOCUMENTS_DIR = "./documents"
STORAGE_DIR = "./storage"
MODEL_NAME = "llama3" # The model you downloaded with Ollama

# --- 1. Set up the LLM ---
llm = Ollama(model=MODEL_NAME, request_timeout=120.0)
print(f"LLM '{MODEL_NAME}' initialized.")

# --- 2. Load or Build the Index ---
if not os.path.exists(STORAGE_DIR):
    print(f"Storage directory not found. Creating a new index from documents in '{DOCUMENTS_DIR}'.")
    documents = SimpleDirectoryReader(DOCUMENTS_DIR).load_data()
    index = VectorStoreIndex.from_documents(documents)
    index.storage_context.persist(persist_dir=STORAGE_DIR)
    print(f"Index created and saved to '{STORAGE_DIR}'.")
else:
    print(f"Loading existing index from '{STORAGE_DIR}'.")
    storage_context = StorageContext.from_defaults(persist_dir=STORAGE_DIR)
    index = load_index_from_storage(storage_context)
    print("Checking for document updates...")
    refreshed_docs = index.refresh_ref_docs()
    if any(refreshed_docs):
        print(f"Index refreshed with new or updated documents.")
        index.storage_context.persist(persist_dir=STORAGE_DIR)
    else:
        print("No new documents found. Index is up to date.")

# --- 3. Create the Query Engine ---
query_engine = index.as_query_engine(llm=llm)
print("\n--- Query Engine is Ready. Ask a question! (Type 'exit' to quit) ---\n")

# --- 4. Start the Chat Loop ---
while True:
    prompt = input("Your Question: ")
    if prompt.lower() == 'exit':
        break
    
    response = query_engine.query(prompt)
    print("\nAI Answer:", response)
    print("-" * 50)
