import os
import sys
import time
import glob
import tkinter as tk
from tkinter.scrolledtext import ScrolledText
from datetime import datetime

from llama_cpp import Llama
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from langchain.embeddings.huggingface import HuggingFaceEmbeddings

import docx2txt

# ------------------ CONFIG ------------------
DOCS_PATH = "docs"
INDEX_PATH = "faiss_index"
MODEL_PATH = "models/tinyllama-1.1b-1t-openorca.Q4_K_M.gguf"  # Change if needed
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
# ---------------------------------------------

class OberonApp:
    def __init__(self):
        print("⚡ Loading local LLaMA model...")
        if not os.path.exists(MODEL_PATH):
            print(f"❌ Model not found: {MODEL_PATH}")
            sys.exit(1)

        self.llm = Llama(model_path=MODEL_PATH, n_ctx=2048, n_threads=4, verbose=False)

        print("🔍 Loading embeddings model...")
        self.embeddings = HuggingFaceEmbeddings(model_name=EMBED_MODEL)

        os.makedirs(DOCS_PATH, exist_ok=True)
        os.makedirs(INDEX_PATH, exist_ok=True)

        self.vectorstore = None
        self.last_index_time = None

        self.build_or_load_index(force=False)

    # ------------------ DOCUMENT LOADING ------------------
    def load_and_chunk_docs(self):
        print("📄 Loading and chunking documents...")
        raw_docs = []

        for file in os.listdir(DOCS_PATH):
            full_path = os.path.join(DOCS_PATH, file)

            if file.lower().endswith(".txt"):
                with open(full_path, "r", encoding="utf-8", errors="ignore") as f:
                    text = f.read()
                    raw_docs.append(Document(page_content=text, metadata={"source": file}))

            elif file.lower().endswith(".pdf"):
                loader = PyPDFLoader(full_path)
                pdf_docs = loader.load()
                raw_docs.extend(pdf_docs)

            elif file.lower().endswith(".docx"):
                text = docx2txt.process(full_path)
                raw_docs.append(Document(page_content=text, metadata={"source": file}))

        print(f"✔ Loaded {len(raw_docs)} raw documents")

        splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
        chunks = splitter.split_documents(raw_docs)
        print(f"✔ Created {len(chunks)} chunks")
        return chunks

    # ------------------ INDEX HANDLING ------------------
    def build_or_load_index(self, force=False):
        if force or self.new_files_detected():
            print("💾 Rebuilding FAISS index...")
            chunks = self.load_and_chunk_docs()
            self.vectorstore = FAISS.from_documents(chunks, self.embeddings)
            self.vectorstore.save_local(INDEX_PATH)
            self.last_index_time = datetime.now()
        else:
            print("💾 Loading existing FAISS index...")
            self.vectorstore = FAISS.load_local(INDEX_PATH, self.embeddings, allow_dangerous_deserialization=True)

    def new_files_detected(self):
        """Check if any file in DOCS_PATH is newer than last_index_time."""
        if self.last_index_time is None:
            return True
        for file in os.listdir(DOCS_PATH):
            full_path = os.path.join(DOCS_PATH, file)
            if os.path.getmtime(full_path) > self.last_index_time.timestamp():
                return True
        return False

    # ------------------ QUERY ------------------
    def query(self, question):
        if question.strip().lower() == "force reindex":
            self.build_or_load_index(force=True)
            return "✅ Index rebuilt."

        retriever = self.vectorstore.as_retriever(search_kwargs={"k": 3})
        docs = retriever.get_relevant_documents(question)
        context = "\n\n".join([doc.page_content for doc in docs])

        prompt = f"Answer the question using the context below.\n\nContext:\n{context}\n\nQuestion: {question}\nAnswer:"
        output = self.llm(prompt, max_tokens=512, stop=["\n", "User:"])
        return output["choices"][0]["text"].strip()

# ------------------ GUI ------------------
class OberonGUI:
    def __init__(self, app):
        self.app = app
        self.root = tk.Tk()
        self.root.title("Oberon Local RAG Chat")

        self.chat_area = ScrolledText(self.root, wrap=tk.WORD, height=25, width=100)
        self.chat_area.pack(padx=10, pady=10)
        self.chat_area.insert(tk.END, "💬 Oberon Local RAG Chat Ready (type 'quit' to exit)\n")

        self.entry = tk.Entry(self.root, width=80)
        self.entry.pack(side=tk.LEFT, padx=10, pady=5)
        self.entry.bind("<Return>", self.send_message)

        self.send_button = tk.Button(self.root, text="Send", command=self.send_message)
        self.send_button.pack(side=tk.LEFT)

    def send_message(self, event=None):
        user_input = self.entry.get().strip()
        if not user_input:
            return
        if user_input.lower() == "quit":
            self.root.quit()
            return

        self.chat_area.insert(tk.END, f"\nYou: {user_input}\n")
        self.entry.delete(0, tk.END)

        self.root.update()
        start_time = time.time()
        answer = self.app.query(user_input)
        elapsed = time.time() - start_time

        self.chat_area.insert(tk.END, f"Oberon: {answer}\n⏱ Processing time: {elapsed:.2f} sec\n")
        self.chat_area.see(tk.END)

    def run(self):
        self.root.mainloop()

# ------------------ MAIN ------------------
if __name__ == "__main__":
    app = OberonApp()
    gui = OberonGUI(app)
    gui.run()
