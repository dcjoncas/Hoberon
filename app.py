import os
import sys
import requests
from tqdm import tqdm
from datetime import datetime
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from llama_cpp import Llama
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document
from PyPDF2 import PdfReader
import docx2txt

# ---------------- CONFIG ---------------- #
MODEL_REPO = "TheBloke/orca_mini_3B-GGUF"
MODEL_FILE = "q4_0-orca-mini-3b.gguf"
MODEL_PATH = os.path.join("models", MODEL_FILE)

EMBED_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
DB_DIR = "vectorstore"
DOCS_DIR = "docs"

CHUNK_SIZE = 800
CHUNK_OVERLAP = 100
# ---------------------------------------- #


def download_file(url, dest):
    """Download a file with a progress bar."""
    response = requests.get(url, stream=True)
    total = int(response.headers.get("content-length", 0))
    os.makedirs(os.path.dirname(dest), exist_ok=True)

    with open(dest, "wb") as file, tqdm(
        desc=f"⬇ Downloading {os.path.basename(dest)}",
        total=total,
        unit="B",
        unit_scale=True,
        unit_divisor=1024,
    ) as bar:
        for data in response.iter_content(chunk_size=1024):
            size = file.write(data)
            bar.update(size)


def ensure_model():
    """Ensure local model exists, otherwise download."""
    if not os.path.exists(MODEL_PATH):
        print("📥 Model not found. Downloading...")
        url = f"https://huggingface.co/{MODEL_REPO}/resolve/main/{MODEL_FILE}"
        download_file(url, MODEL_PATH)
    else:
        print("✅ Model already exists.")


def load_documents():
    """Load .txt, .pdf, .docx documents from DOCS_DIR."""
    docs = []
    for file in os.listdir(DOCS_DIR):
        path = os.path.join(DOCS_DIR, file)
        if file.lower().endswith(".txt"):
            with open(path, "r", encoding="utf-8") as f:
                docs.append(Document(page_content=f.read(), metadata={"source": file}))
        elif file.lower().endswith(".pdf"):
            reader = PdfReader(path)
            text = "\n".join([page.extract_text() for page in reader.pages if page.extract_text()])
            docs.append(Document(page_content=text, metadata={"source": file}))
        elif file.lower().endswith(".docx"):
            text = docx2txt.process(path)
            docs.append(Document(page_content=text, metadata={"source": file}))
    return docs


def needs_rebuild(index_file):
    """Check if FAISS index needs rebuilding based on document modification times."""
    if not os.path.exists(index_file):
        return True

    index_mtime = os.path.getmtime(index_file)

    for file in os.listdir(DOCS_DIR):
        path = os.path.join(DOCS_DIR, file)
        if os.path.getmtime(path) > index_mtime:
            return True
    return False


def build_or_load_vectorstore():
    """Load FAISS index if exists & up-to-date, else build it."""
    os.makedirs(DB_DIR, exist_ok=True)
    index_file = os.path.join(DB_DIR, "faiss_index")

    if not needs_rebuild(index_file):
        print("✅ Loading cached FAISS index...")
        return FAISS.load_local(
            index_file,
            HuggingFaceEmbeddings(model_name=EMBED_MODEL_NAME),
            allow_dangerous_deserialization=True
        )

    print("📄 Loading and chunking documents...")
    docs = load_documents()
    splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    chunks = splitter.split_documents(docs)

    print(f"✂ Created {len(chunks)} chunks. Embedding...")
    embeddings = HuggingFaceEmbeddings(model_name=EMBED_MODEL_NAME)
    vectordb = FAISS.from_documents(chunks, embeddings)

    vectordb.save_local(index_file)
    print("💾 FAISS index saved.")
    return vectordb


def main():
    ensure_model()

    print("⚡ Loading local LLaMA model...")
    llm = Llama(model_path=MODEL_PATH, n_ctx=2048, n_threads=6, verbose=False)

    vectordb = build_or_load_vectorstore()

    print("\n💬 Local RAG Chat Ready (type 'quit' to exit)")
    while True:
        query = input("\nYou: ")
        if query.lower() in ["quit", "exit"]:
            break

        results = vectordb.similarity_search(query, k=3)
        context = "\n".join([doc.page_content for doc in results])

        prompt = f"Context:\n{context}\n\nQuestion: {query}\nAnswer:"

        output = llm(prompt, max_tokens=256, stop=["Q:", "\n\n"], echo=False)
        print("\nAssistant:", output["choices"][0]["text"].strip())


if __name__ == "__main__":
    if not os.path.exists(DOCS_DIR):
        os.makedirs(DOCS_DIR)
        print(f"📂 Put your .txt, .pdf, or .docx files inside '{DOCS_DIR}' and re-run the script.")
        sys.exit()

    main()
