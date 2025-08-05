import os
import time
import glob
import tkinter as tk
from tkinter import scrolledtext, END
from datetime import datetime
from llama_cpp import Llama
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import pickle

# ====== CONFIG ======
MODEL_PATH = "models/tinyllama-1.1b-1t-openorca.Q4_K_M.gguf"  # Your model
DOCS_DIR = "docs"
INDEX_FILE = "faiss_index"
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 150
EMBED_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

# ====== INIT EMBEDDINGS ======
embeddings = HuggingFaceEmbeddings(model_name=EMBED_MODEL_NAME)

# ====== LOAD LOCAL MODEL ======
def load_llm():
    print("⚡ Loading local LLaMA model...")
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model not found: {MODEL_PATH}")
    return Llama(model_path=MODEL_PATH, n_ctx=2048, n_threads=6, verbose=False)

# ====== LOAD & CHUNK DOCS ======
def load_and_chunk_documents():
    print("📄 Loading and chunking documents...")
    docs = []

    for file in glob.glob(f"{DOCS_DIR}/*"):
        ext = file.lower()
        if ext.endswith(".pdf"):
            loader = PyPDFLoader(file)
        elif ext.endswith(".docx"):
            loader = Docx2txtLoader(file)
        elif ext.endswith(".txt"):
            loader = TextLoader(file, encoding="utf-8")
        else:
            continue
        docs.extend(loader.load())

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    chunks = text_splitter.split_documents(docs)
    print(f"✂ Created {len(chunks)} chunks.")
    return chunks

# ====== BUILD INDEX ======
def build_faiss_index():
    chunks = load_and_chunk_documents()
    vectorstore = FAISS.from_documents(chunks, embeddings)
    vectorstore.save_local(INDEX_FILE)
    with open(INDEX_FILE + "_meta.pkl", "wb") as f:
        pickle.dump({"last_index_time": datetime.now(), "files": get_docs_list()}, f)
    print("💾 FAISS index saved.")
    return vectorstore

# ====== LOAD INDEX ======
def load_faiss_index():
    if not os.path.exists(INDEX_FILE):
        return build_faiss_index()
    return FAISS.load_local(INDEX_FILE, embeddings, allow_dangerous_deserialization=True)

# ====== CHECK FOR NEW FILES ======
def get_docs_list():
    return sorted([os.path.basename(f) for f in glob.glob(f"{DOCS_DIR}/*")])

def check_new_files():
    meta_file = INDEX_FILE + "_meta.pkl"
    if not os.path.exists(meta_file):
        return True
    with open(meta_file, "rb") as f:
        meta = pickle.load(f)
    old_files = meta.get("files", [])
    return old_files != get_docs_list()

# ====== MAIN QUERY FUNCTION ======
def query_oberon(llm, vectorstore, query):
    docs = vectorstore.similarity_search(query, k=3)
    context = "\n\n".join([d.page_content for d in docs])
    prompt = f"Use the following context to answer the question:\n\n{context}\n\nQuestion: {query}\nAnswer:"
    output = llm(prompt, max_tokens=512, stop=["\n", "Question:"])
    return output["choices"][0]["text"].strip()

# ====== TKINTER UI ======
class OberonApp:
    def __init__(self, root, llm, vectorstore):
        self.llm = llm
        self.vectorstore = vectorstore
        root.title("🪐 Oberon Local AI")
        root.geometry("800x500")

        self.chat_display = scrolledtext.ScrolledText(root, wrap=tk.WORD, state="disabled")
        self.chat_display.pack(padx=10, pady=10, fill=tk.BOTH, expand=True)

        self.entry = tk.Entry(root, width=100)
        self.entry.pack(side=tk.LEFT, padx=10, pady=5, fill=tk.X, expand=True)
        self.entry.bind("<Return>", self.send_message)

        send_button = tk.Button(root, text="Send", command=self.send_message)
        send_button.pack(side=tk.RIGHT, padx=10, pady=5)

    def send_message(self, event=None):
        query = self.entry.get().strip()
        if not query:
            return
        self.display_message("You", query)
        self.entry.delete(0, END)

        if query.lower() == "!quit":
            root.quit()
            return
        if query.lower() == "!reindex":
            self.vectorstore = build_faiss_index()
            self.display_message("Oberon", "Index rebuilt successfully.")
            return

        response = query_oberon(self.llm, self.vectorstore, query)
        self.display_message("Oberon", response)

    def display_message(self, sender, message):
        self.chat_display.config(state="normal")
        self.chat_display.insert(END, f"{sender}: {message}\n\n")
        self.chat_display.config(state="disabled")
        self.chat_display.yview(END)

# ====== MAIN ======
if __name__ == "__main__":
    llm = load_llm()
    if check_new_files():
        vectorstore = build_faiss_index()
    else:
        vectorstore = load_faiss_index()

    root = tk.Tk()
    app = OberonApp(root, llm, vectorstore)
    root.mainloop()
