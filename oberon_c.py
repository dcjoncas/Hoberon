import os
import sys
from llama_cpp import Llama
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import PyPDFLoader, TextLoader, Docx2txtLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import tkinter as tk
from tkinter import scrolledtext, messagebox

# -------------------------
# SETTINGS
# -------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "models", "q4_0-orca-mini-3b.gguf")
DOCS_PATH = os.path.join(BASE_DIR, "docs")
INDEX_PATH = os.path.join(BASE_DIR, "faiss_index")

# -------------------------
# MODEL PATH VALIDATION
# -------------------------
if not os.path.isfile(MODEL_PATH):
    message = f"❌ ERROR: Model file not found:\n{MODEL_PATH}\n\n" \
              "Please make sure the file exists and the path is correct."
    print(message)
    sys.exit(1)

# -------------------------
# LOAD LOCAL LLaMA MODEL
# -------------------------
print("⚡ Loading local LLaMA model...")
try:
    llm = Llama(model_path=MODEL_PATH, n_ctx=2048, n_threads=6, verbose=False)
except Exception as e:
    print(f"❌ Failed to load LLaMA model: {e}")
    sys.exit(1)

# -------------------------
# LOAD & EMBED DOCUMENTS
# -------------------------
print("📄 Loading and chunking documents...")

docs = []
if os.path.isdir(DOCS_PATH):
    for filename in os.listdir(DOCS_PATH):
        file_path = os.path.join(DOCS_PATH, filename)
        if filename.lower().endswith(".pdf"):
            loader = PyPDFLoader(file_path)
        elif filename.lower().endswith(".txt"):
            loader = TextLoader(file_path)
        elif filename.lower().endswith(".docx"):
            loader = Docx2txtLoader(file_path)
        else:
            continue
        docs.extend(loader.load())

if not docs:
    print("⚠ No documents found to index. Put your PDFs, TXT, or DOCX in the 'docs' folder.")

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
chunks = text_splitter.split_documents(docs)
print(f"✂ Created {len(chunks)} chunks.")

embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vectorstore = FAISS.from_documents(chunks, embeddings)
vectorstore.save_local(INDEX_PATH)
print("💾 FAISS index saved.")

# -------------------------
# QUERY FUNCTION
# -------------------------
def query_oberon(user_input):
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
    relevant_docs = retriever.get_relevant_documents(user_input)

    context = "\n".join([doc.page_content for doc in relevant_docs])
    prompt = f"Context:\n{context}\n\nQuestion: {user_input}\nAnswer:"

    output = llm(prompt, max_tokens=512, stop=["\n", "User:"], echo=False)
    return output["choices"][0]["text"].strip()

# -------------------------
# TKINTER CHAT UI
# -------------------------
def send_query():
    user_input = entry.get().strip()
    if not user_input:
        return

    chat_window.insert(tk.END, f"You: {user_input}\n", "user")
    entry.delete(0, tk.END)

    try:
        answer = query_oberon(user_input)
    except Exception as e:
        answer = f"⚠ ERROR: {str(e)}"

    chat_window.insert(tk.END, f"Oberon: {answer}\n\n", "ai")
    chat_window.see(tk.END)

# GUI Setup
root = tk.Tk()
root.title("Oberon - Local AI RAG Assistant")
root.geometry("700x500")

chat_window = scrolledtext.ScrolledText(root, wrap=tk.WORD, state=tk.NORMAL)
chat_window.pack(padx=10, pady=10, fill=tk.BOTH, expand=True)
chat_window.tag_config("user", foreground="blue")
chat_window.tag_config("ai", foreground="green")

entry = tk.Entry(root)
entry.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=10, pady=5)

send_button = tk.Button(root, text="Send", command=send_query)
send_button.pack(side=tk.RIGHT, padx=10, pady=5)

root.mainloop()
