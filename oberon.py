import os
import sys
import time
import threading
import docx2txt
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader, TextLoader, Docx2txtLoader
from langchain_huggingface import HuggingFaceEmbeddings
from llama_cpp import Llama

# ------------------- SETTINGS -------------------
MODEL_PATH = "models/q4_0-orca-mini-3b.gguf"  # Your local model file
DOCS_DIR = "docs"
INDEX_DIR = "faiss_index"
EMBED_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 150
# -------------------------------------------------

# Utility to chunk text
def chunk_text(text, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    chunks = []
    start = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        chunks.append(text[start:end])
        start += chunk_size - overlap
    return chunks

# Loader for any supported file
def load_and_chunk_single_file(file_path):
    ext = file_path.lower()
    try:
        if ext.endswith(".pdf"):
            loader = PyPDFLoader(file_path)
            pages = loader.load()
            docs = []
            for page in pages:
                docs.extend(chunk_text(page.page_content))
            return [{"page_content": c, "metadata": {"source": file_path}} for c in docs]

        elif ext.endswith(".txt"):
            loader = TextLoader(file_path, encoding="utf-8")
            data = loader.load()
            docs = []
            for d in data:
                docs.extend(chunk_text(d.page_content))
            return [{"page_content": c, "metadata": {"source": file_path}} for c in docs]

        elif ext.endswith(".docx"):
            text = docx2txt.process(file_path)
            chunks = chunk_text(text)
            return [{"page_content": c, "metadata": {"source": file_path}} for c in chunks]

        else:
            return []
    except Exception as e:
        print(f"❌ Error reading {file_path}: {e}")
        return []

# Load all docs on startup
def load_all_docs():
    all_chunks = []
    for file in os.listdir(DOCS_DIR):
        path = os.path.join(DOCS_DIR, file)
        if os.path.isfile(path) and path.lower().endswith((".pdf", ".txt", ".docx")):
            all_chunks.extend(load_and_chunk_single_file(path))
    return all_chunks

# ------------------- Watchdog for live updates -------------------
class DocsFolderHandler(FileSystemEventHandler):
    def __init__(self, app):
        self.app = app

    def on_created(self, event):
        if not event.is_directory and event.src_path.lower().endswith((".pdf", ".txt", ".docx")):
            self.app.add_system_message(f"📄 New file detected: {event.src_path}")
            threading.Thread(target=self.app.add_new_file, args=(event.src_path,)).start()

# ------------------- Main Oberon App -------------------
class OberonApp:
    def __init__(self):
        print("⚡ Loading local LLaMA model...")
        if not os.path.exists(MODEL_PATH):
            print(f"❌ Model not found: {MODEL_PATH}")
            sys.exit(1)
        self.llm = Llama(model_path=MODEL_PATH, n_ctx=2048, n_threads=6, verbose=False)

        print("💾 Loading or creating FAISS index...")
        self.embeddings = HuggingFaceEmbeddings(model_name=EMBED_MODEL_NAME)

        if os.path.exists(INDEX_DIR):
            self.vectorstore = FAISS.load_local(INDEX_DIR, self.embeddings, allow_dangerous_deserialization=True)
        else:
            chunks = load_all_docs()
            if chunks:
                self.vectorstore = FAISS.from_documents(chunks, self.embeddings)
                self.vectorstore.save_local(INDEX_DIR)
            else:
                self.vectorstore = FAISS.from_documents([], self.embeddings)

    def add_system_message(self, text):
        print(text)

    def add_new_file(self, file_path):
        try:
            chunks = load_and_chunk_single_file(file_path)
            if chunks:
                self.vectorstore.add_documents(chunks)
                self.vectorstore.save_local(INDEX_DIR)
                self.add_system_message(f"✅ Added {len(chunks)} chunks from {file_path}")
            else:
                self.add_system_message(f"⚠ No text found in {file_path}")
        except Exception as e:
            self.add_system_message(f"❌ Error processing {file_path}: {e}")

    def start_watcher(self):
        event_handler = DocsFolderHandler(self)
        observer = Observer()
        observer.schedule(event_handler, DOCS_DIR, recursive=False)
        observer.start()
        threading.Thread(target=self._watchdog_thread, args=(observer,), daemon=True).start()

    def _watchdog_thread(self, observer):
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            observer.stop()
        observer.join()

    def chat(self):
        print("\n💬 Oberon Local RAG Chat Ready (type 'quit' to exit)")
        while True:
            query = input("\nYou: ")
            if query.lower() in ["quit", "exit"]:
                break
            docs = self.vectorstore.similarity_search(query, k=3)
            context = "\n".join([d.page_content for d in docs])
            prompt = f"Context:\n{context}\n\nQuestion: {query}\nAnswer:"
            output = self.llm(prompt, max_tokens=512, stop=["\nUser:"])
            print(f"Oberon: {output['choices'][0]['text'].strip()}")

# ------------------- Run -------------------
if __name__ == "__main__":
    if not os.path.exists(DOCS_DIR):
        os.makedirs(DOCS_DIR)

    app = OberonApp()
    app.start_watcher()
    app.chat()
