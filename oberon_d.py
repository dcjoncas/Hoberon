import os
import time
import threading
import queue
import tkinter as tk
from tkinter import scrolledtext, filedialog, messagebox
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader, TextLoader, Docx2txtLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from llama_cpp import Llama

# -------------------------
# CONFIG
# -------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DOCS_DIR = os.path.join(BASE_DIR, "docs")
INDEX_FILE = os.path.join(BASE_DIR, "faiss_index")
MODEL_PATH = os.path.join(BASE_DIR, "models", "q4_0-orca-mini-3b.gguf")  # Correct model name
EMBED_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

# -------------------------
# HELPER FUNCTIONS
# -------------------------
def load_and_embed_documents():
    """Load docs, chunk, embed, and save FAISS index."""
    docs = []
    for file in os.listdir(DOCS_DIR):
        path = os.path.join(DOCS_DIR, file)
        if file.lower().endswith(".pdf"):
            loader = PyPDFLoader(path)
        elif file.lower().endswith(".txt"):
            loader = TextLoader(path)
        elif file.lower().endswith(".docx"):
            loader = Docx2txtLoader(path)
        else:
            continue
        docs.extend(loader.load())

    if not docs:
        return None

    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.split_documents(docs)

    print(f"✂ Created {len(chunks)} chunks.")
    embeddings = HuggingFaceEmbeddings(model_name=EMBED_MODEL_NAME)
    vectorstore = FAISS.from_documents(chunks, embeddings)
    vectorstore.save_local(INDEX_FILE)

    return chunks


def load_vectorstore():
    """Load FAISS index if exists."""
    if os.path.exists(INDEX_FILE):
        embeddings = HuggingFaceEmbeddings(model_name=EMBED_MODEL_NAME)
        return FAISS.load_local(INDEX_FILE, embeddings, allow_dangerous_deserialization=True)
    return None


# -------------------------
# LLAMA LOADER
# -------------------------
def load_model():
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model not found at: {MODEL_PATH}")
    return Llama(model_path=MODEL_PATH, n_ctx=2048, n_threads=6, verbose=False)


# -------------------------
# CHAT PROCESSOR
# -------------------------
def answer_question(llm, vs, question, progress_callback=None):
    """Answer question using RAG."""
    if not vs:
        return "No document index found."

    # Retrieve relevant chunks
    retriever = vs.as_retriever(search_kwargs={"k": 3})
    relevant_docs = retriever.get_relevant_documents(question)
    context = "\n".join([doc.page_content for doc in relevant_docs])

    # Estimate processing time based on token size
    est_time = max(2, len(context.split()) // 30)
    if progress_callback:
        for sec in range(est_time):
            progress_callback(sec + 1, est_time)
            time.sleep(1)

    # Ask model
    prompt = f"Use the following context to answer:\n{context}\n\nQuestion: {question}\nAnswer:"
    output = llm(prompt, max_tokens=512, stop=["\n"], echo=False)
    return output["choices"][0]["text"].strip()


# -------------------------
# TKINTER FRONTEND
# -------------------------
class OberonApp:
    def __init__(self, master):
        self.master = master
        master.title("Oberon Local RAG Chat")
        master.geometry("800x600")

        self.chat_area = scrolledtext.ScrolledText(master, wrap=tk.WORD, state="disabled", height=20)
        self.chat_area.pack(padx=10, pady=10, fill=tk.BOTH, expand=True)

        self.input_box = tk.Entry(master)
        self.input_box.pack(padx=10, pady=(0,10), fill=tk.X)
        self.input_box.bind("<Return>", self.send_message)

        self.progress_label = tk.Label(master, text="")
        self.progress_label.pack()

        self.send_button = tk.Button(master, text="Send", command=self.send_message)
        self.send_button.pack()

        self.chunks_button = tk.Button(master, text="View Chunks", command=self.show_chunks)
        self.chunks_button.pack()

        self.message_queue = queue.Queue()

        # Load model + vectorstore
        self.llm = load_model()
        self.vectorstore = load_vectorstore()

        if not self.vectorstore:
            self.add_message("System", "No index found. Building from docs folder...")
            chunks = load_and_embed_documents()
            if chunks:
                self.vectorstore = load_vectorstore()
                self.all_chunks = chunks
                self.add_message("System", f"Indexed {len(chunks)} chunks.")
            else:
                self.add_message("System", "No supported documents found in docs folder.")
                self.all_chunks = []
        else:
            self.all_chunks = []

    def progress_callback(self, current, total):
        self.progress_label.config(text=f"Processing... {current}/{total} sec")
        self.master.update_idletasks()

    def send_message(self, event=None):
        question = self.input_box.get().strip()
        if not question:
            return
        self.add_message("You", question)
        self.input_box.delete(0, tk.END)

        threading.Thread(target=self.process_question, args=(question,)).start()

    def process_question(self, question):
        answer = answer_question(self.llm, self.vectorstore, question, self.progress_callback)
        self.add_message("Oberon", answer)
        self.progress_label.config(text="")

    def add_message(self, sender, message):
        self.chat_area.config(state="normal")
        self.chat_area.insert(tk.END, f"{sender}: {message}\n\n")
        self.chat_area.config(state="disabled")
        self.chat_area.see(tk.END)

    def show_chunks(self):
        if not self.all_chunks:
            messagebox.showinfo("Chunks", "No chunks available. Try rebuilding index.")
            return
        chunk_win = tk.Toplevel(self.master)
        chunk_win.title("Document Chunks")
        chunk_text = scrolledtext.ScrolledText(chunk_win, wrap=tk.WORD)
        chunk_text.pack(fill=tk.BOTH, expand=True)
        for i, chunk in enumerate(self.all_chunks):
            chunk_text.insert(tk.END, f"--- Chunk {i+1} ---\n{chunk.page_content}\n\n")


# -------------------------
# MAIN
# -------------------------
if __name__ == "__main__":
    root = tk.Tk()
    app = OberonApp(root)
    root.mainloop()
