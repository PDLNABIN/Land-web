import glob
import os
import uuid
from pathlib import Path

import chromadb
from chromadb.api.types import Documents, EmbeddingFunction, Embeddings
from dotenv import load_dotenv
from google import genai
from google.genai import types
from sentence_transformers import SentenceTransformer
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR / ".env")

DATA_PATH = Path(os.getenv("LAND_DATA_PATH", BASE_DIR.parent))
VECTOR_STORE_DIR = Path(os.getenv("VECTOR_STORE_DIR", BASE_DIR / "vector_store"))
FINGERPRINT_PATH = VECTOR_STORE_DIR / "fingerprint.txt"

GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-3.5-flash-lite")
EMBEDDING_MODEL = "jangedoo/all-MiniLM-L6-v3-nepali"
hf_embedder = SentenceTransformer(EMBEDDING_MODEL)

class NepaliEmbeddingFunction(EmbeddingFunction):
    def __init__(self):
        super().__init__()
        
    def __call__(self, input: Documents) -> Embeddings:
        # ChromaDB expects a list of lists of floats
        return hf_embedder.encode(input).tolist()

CHUNK_SIZE = int(os.getenv("RAG_CHUNK_SIZE", "1000"))
CHUNK_OVERLAP = int(os.getenv("RAG_CHUNK_OVERLAP", "200"))
TOP_K = int(os.getenv("RAG_TOP_K", "5"))

api_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
_client = None

def get_client():
    global _client
    if _client is None:
        if not api_key:
            raise RuntimeError("Set GOOGLE_API_KEY or GEMINI_API_KEY for Google AI Studio.")
        _client = genai.Client(api_key=api_key)
    return _client

def load_documents():
    """Loads all markdown files from the configured data path."""
    md_files = glob.glob(str(DATA_PATH / "*.md"))
    documents = []
    for file_path in md_files:
        loader = TextLoader(file_path, encoding="utf-8")
        documents.extend(loader.load())
    return documents

def split_documents(documents):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )
    return text_splitter.split_documents(documents)

def _documents_fingerprint():
    files = sorted(DATA_PATH.glob("*.md"))
    file_stats = [
        f"{path.resolve()}:{path.stat().st_mtime_ns}:{path.stat().st_size}"
        for path in files
    ]
    return "|".join(
        [
            EMBEDDING_MODEL,
            str(CHUNK_SIZE),
            str(CHUNK_OVERLAP),
            *file_stats,
        ]
    )

def _load_vector_store(fingerprint: str):
    if not FINGERPRINT_PATH.exists():
        return None
        
    with FINGERPRINT_PATH.open("r", encoding="utf-8") as file:
        saved_fingerprint = file.read().strip()
        
    if saved_fingerprint != fingerprint:
        return None

    # Connect to existing ChromaDB
    chroma_client = chromadb.PersistentClient(path=str(VECTOR_STORE_DIR))
    collection = chroma_client.get_collection(
        name="land_docs",
        embedding_function=NepaliEmbeddingFunction()
    )
    return collection

def _build_vector_store(fingerprint: str):
    documents = load_documents()
    chunks = split_documents(documents)
    if not chunks:
        raise RuntimeError(f"No markdown documents found in {DATA_PATH}.")

    VECTOR_STORE_DIR.mkdir(parents=True, exist_ok=True)
    chroma_client = chromadb.PersistentClient(path=str(VECTOR_STORE_DIR))
    
    # Reset collection if it already exists to avoid duplicates when fingerprint changes
    try:
        chroma_client.delete_collection(name="land_docs")
    except Exception:
        pass
        
    collection = chroma_client.create_collection(
        name="land_docs",
        embedding_function=NepaliEmbeddingFunction(),
        metadata={"hnsw:space": "cosine"}
    )
    
    texts = [chunk.page_content for chunk in chunks]
    metadatas = [chunk.metadata for chunk in chunks]
    ids = [str(uuid.uuid4()) for _ in chunks]

    # Batch insert to avoid exceeding potential payload limits
    BATCH_SIZE = 100
    for i in range(0, len(texts), BATCH_SIZE):
        collection.add(
            documents=texts[i:i+BATCH_SIZE],
            metadatas=metadatas[i:i+BATCH_SIZE],
            ids=ids[i:i+BATCH_SIZE]
        )

    with FINGERPRINT_PATH.open("w", encoding="utf-8") as file:
        file.write(fingerprint)

    return collection

def get_vector_store():
    fingerprint = _documents_fingerprint()
    existing = _load_vector_store(fingerprint)
    if existing is not None:
        return existing
    return _build_vector_store(fingerprint)

def extract_location(text: str) -> str:
    prompt = (
        "Extract the core location or place name from this text. "
        "Return ONLY the place name in Nepali script (e.g., 'मनोहरा', 'शुभकामना'). "
        "Remove suffixes like 'मा', 'को', 'तिर'. "
        "If no specific place is mentioned, return 'NONE'.\n\n"
        f"Text: {text}"
    )
    response = get_client().models.generate_content(
        model=GEMINI_MODEL,
        contents=prompt
    )
    return getattr(response, "text", str(response)).strip()

def extract_location(text: str) -> str:
    prompt = (
        "You are parsing a query for a Nepal land valuation database. "
        "Extract the core location or place name from this text. "
        "Return ONLY the place name in Nepali script (e.g., 'असन', 'मनोहरा', 'शुभकामना'). "
        "Remove suffixes like 'मा', 'को', 'तिर', 'क्षेत्र'. "
        "If no specific place is mentioned, return 'NONE'.\n\n"
        f"Text: {text}"
    )
    response = get_client().models.generate_content(
        model=GEMINI_MODEL,
        contents=prompt
    )
    return getattr(response, "text", str(response)).strip()

def retrieve_relevant_documents(location_keyword: str, top_k: int = 50):
    collection = get_vector_store()
    relevant = []
    
    if not location_keyword or location_keyword == "NONE":
        return relevant

    # Use the isolated location name to search the database
    try:
        # First try strict keyword matching
        results = collection.query(
            query_texts=[location_keyword],
            n_results=top_k,
            where_document={"$contains": location_keyword}
        )
        
        # Fallback to semantic search if strict matching yields no results
        if not results["documents"] or not results["documents"][0]:
            results = collection.query(
                query_texts=[location_keyword],
                n_results=top_k
            )
            
        if results["documents"] and results["documents"][0]:
            documents = results["documents"][0]
            metadatas = results["metadatas"][0] if results["metadatas"] else [{}] * len(documents)
            for doc, meta in zip(documents, metadatas):
                relevant.append({
                    "page_content": doc,
                    "metadata": meta or {},
                })
    except Exception as e:
        print(f"Search error: {e}")
            
    return relevant

def get_context(location_keyword: str) -> str:
    relevant_docs = retrieve_relevant_documents(location_keyword, top_k=50) # Fetch up to 50 matching chunks
    context_blocks = []
    for doc in relevant_docs:
        source = doc["metadata"].get("source", "unknown source")
        context_blocks.append(f"Source: {source}\n{doc['page_content']}")
    return "\n\n".join(context_blocks)

def ask_question(question: str) -> str:
    print(f"Original question: {question}")
    
    # Extract just the location in Nepali to query the vector DB
    location_keyword = extract_location(question)
    print(f"Extracted Nepali location for database search: {location_keyword}")
    
    context = get_context(location_keyword)

    system_prompt = (
        "You are a precise assistant for Nepal land valuations. "
        "Use ONLY the retrieved context below to answer the user's specific question. "
        "CRITICAL INSTRUCTION: Read the user's FULL question carefully. Filter the context to extract ONLY the specific information they asked for. "
        "DO NOT output data for other locations, roads, or wards that the user did not ask about, even if they appear in the context. "
        "DO NOT include bureaucratic fluff from the source documents such as serial numbers (e.g., 'सि.नं. ५२'), section numbers, internal codes, or legal jargon. Provide the clean, natural answer. "
        "For example, if the user asks for 'kacchi road in ason', ONLY provide the valuation for the kacchi road in Asan. Ignore Hanumandhoka, New Road, or pitched roads in Asan. "
        "However, you MUST format the relevant information beautifully (e.g., using a Markdown table or a clear bullet point) so it doesn't look like a raw number. "
        "IMPORTANT: You MUST cite the source document of your information at the end of your answer (e.g., '*स्रोत: chabahil.md*'). The source file name is provided in the context blocks. "
        "If no information is found for the user's specific query, say that you do not have information on that specific detail. "
        "CRITICAL INSTRUCTION: You MUST output your final answer entirely in Nepali language, regardless of the input language.\n\n"
        f"Context retrieved for '{location_keyword}':\n{context}\n\n"
    )

    response = get_client().models.generate_content(
        model=GEMINI_MODEL,
        contents=question,
        config=types.GenerateContentConfig(system_instruction=system_prompt),
    )
    return getattr(response, "text", str(response))

if __name__ == "__main__":
    print(ask_question("kacchi road on ason area"))
