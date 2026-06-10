import os
import re
import pickle
import numpy as np
from pathlib import Path

import fitz                                          # PyMuPDF — PDF text extraction
from sentence_transformers import SentenceTransformer
import faiss

# ─────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────
EMBED_MODEL_NAME  = "all-MiniLM-L6-v2"   # 80 MB, free, fast
CHUNK_SIZE        = 5                     # sentences per chunk
CHUNK_OVERLAP     = 1                     # overlap between chunks
INDEX_CACHE_FILE  = "rag_index.pkl"       # saved to disk — skip re-indexing on restart
TOP_K_DEFAULT     = 3                     # chunks returned per query


# ─────────────────────────────────────────────────────────
# RAG RETRIEVER
# ─────────────────────────────────────────────────────────
class RAGRetriever:
    """
    Builds a FAISS vector index from a folder of medical PDFs.
    Provides dense semantic retrieval for any natural language query.

    Usage:
        rag = RAGRetriever(pdf_folder="medical_db")
        doc_name, doc_text, top_chunks = rag.retrieve("hantavirus transmission")
        context = rag.get_context("what is the treatment for HPS?", top_k=5)
    """

    def __init__(self, pdf_folder: str = "medical_db", force_rebuild: bool = False):
        self.pdf_folder = pdf_folder
        self.embed_model = SentenceTransformer(EMBED_MODEL_NAME)

        # Internal state — populated by _build_index()
        self._chunks      : list[str]  = []   # text of each chunk
        self._chunk_files : list[str]  = []   # which PDF each chunk came from
        self._chunk_texts : dict[str, str] = {}   # filename → full doc text
        self._index       = None               # FAISS index

        cache = Path(INDEX_CACHE_FILE)
        if not force_rebuild and cache.exists():
            print("📦 Loading RAG index from cache...")
            self._load_index(cache)
        else:
            print("🔨 Building RAG vector index...")
            self._build_index()
            self._save_index(cache)

        print(f"✅ RAG ready — {len(self._chunks)} chunks from "
              f"{len(self._chunk_texts)} PDFs")

    # ─────────────────────────────────────────────────────
    # PUBLIC API
    # ─────────────────────────────────────────────────────
    def retrieve(
        self,
        query   : str,
        top_k   : int = TOP_K_DEFAULT,
    ) -> tuple[str | None, str | None, list[str]]:
        """
        Returns (best_filename, full_doc_text, top_k_chunks).

        best_filename — the PDF that contributed the most top chunks.
        full_doc_text — complete text of that PDF (for compatibility
                        with the existing scan pipeline).
        top_k_chunks  — the most semantically relevant text snippets
                        (use these as context for Gemini Q&A).
        """
        if not self._chunks:
            return None, None, []

        top_chunks, top_files = self._search(query, top_k)

        if not top_files:
            return None, None, []

        # Pick the PDF that appears most often in the top results
        from collections import Counter
        best_file = Counter(top_files).most_common(1)[0][0]
        full_text = self._chunk_texts.get(best_file, "")

        return best_file, full_text, top_chunks

    def get_context(self, query: str, top_k: int = 5) -> str:
        """
        Returns the top_k most relevant chunks joined as a context string.
        Use this instead of doc_text[:3000] when calling Gemini.
        Semantically targeted context = better answers.
        """
        top_chunks, _ = self._search(query, top_k)
        return "\n\n---\n\n".join(top_chunks)

    def add_pdf(self, pdf_path: str) -> None:
        """
        Adds a single new PDF to the index at runtime.
        Useful if the user uploads a new document mid-session.
        """
        fname = os.path.basename(pdf_path)
        text  = self._extract_pdf_text(pdf_path)
        if not text.strip():
            print(f"⚠️  No text extracted from {fname}")
            return

        self._chunk_texts[fname] = text
        new_chunks = self._make_chunks(text)

        if not new_chunks:
            return

        vecs = self.embed_model.encode(new_chunks, show_progress_bar=False)
        vecs = np.array(vecs, dtype="float32")

        self._index.add(vecs)
        self._chunks.extend(new_chunks)
        self._chunk_files.extend([fname] * len(new_chunks))

        # Update cache
        self._save_index(Path(INDEX_CACHE_FILE))
        print(f"✅ Added {len(new_chunks)} chunks from {fname}")

    # ─────────────────────────────────────────────────────
    # INTERNAL — INDEX BUILD
    # ─────────────────────────────────────────────────────
    def _build_index(self) -> None:
        if not os.path.exists(self.pdf_folder):
            print(f"⚠️  PDF folder '{self.pdf_folder}' not found.")
            return

        pdf_files = [
            f for f in os.listdir(self.pdf_folder)
            if f.lower().endswith(".pdf")
        ]
        if not pdf_files:
            print(f"⚠️  No PDFs found in '{self.pdf_folder}'.")
            return

        all_chunks : list[str] = []
        all_files  : list[str] = []

        for fname in sorted(pdf_files):
            path = os.path.join(self.pdf_folder, fname)
            text = self._extract_pdf_text(path)
            if not text.strip():
                print(f"  ⚠️  Skipping {fname} — no text extracted")
                continue

            self._chunk_texts[fname] = text
            chunks = self._make_chunks(text)
            all_chunks.extend(chunks)
            all_files.extend([fname] * len(chunks))
            print(f"  📄 {fname} → {len(chunks)} chunks")

        if not all_chunks:
            print("⚠️  No chunks produced — index is empty.")
            return

        print(f"\n🧮 Encoding {len(all_chunks)} chunks with {EMBED_MODEL_NAME}...")
        vecs = self.embed_model.encode(
            all_chunks,
            batch_size        = 64,
            show_progress_bar = True,
            convert_to_numpy  = True,
        )
        vecs = np.array(vecs, dtype="float32")

        # L2-normalise → cosine similarity via inner product
        faiss.normalize_L2(vecs)
        dim           = vecs.shape[1]
        self._index   = faiss.IndexFlatIP(dim)   # Inner Product = cosine after normalisation
        self._index.add(vecs)
        self._chunks      = all_chunks
        self._chunk_files = all_files

    def _search(self, query: str, top_k: int) -> tuple[list[str], list[str]]:
        """Returns (top_k chunk texts, their source filenames)."""
        if self._index is None or self._index.ntotal == 0:
            return [], []

        q_vec = self.embed_model.encode([query], convert_to_numpy=True)
        q_vec = np.array(q_vec, dtype="float32")
        faiss.normalize_L2(q_vec)

        k = min(top_k, self._index.ntotal)
        _, ids = self._index.search(q_vec, k)

        top_chunks = [self._chunks[i]      for i in ids[0] if i >= 0]
        top_files  = [self._chunk_files[i] for i in ids[0] if i >= 0]
        return top_chunks, top_files

    # ─────────────────────────────────────────────────────
    # INTERNAL — PDF EXTRACTION & CHUNKING
    # ─────────────────────────────────────────────────────
    @staticmethod
    def _extract_pdf_text(path: str) -> str:
        """Extracts all text from a PDF using PyMuPDF."""
        try:
            doc  = fitz.open(path)
            text = "".join(page.get_text() for page in doc)
            return text
        except Exception as e:
            print(f"  ⚠️  Error reading {path}: {e}")
            return ""

    @staticmethod
    def _make_chunks(text: str) -> list[str]:
        """
        Splits document text into overlapping sentence-window chunks.
        Each chunk = CHUNK_SIZE consecutive sentences with CHUNK_OVERLAP
        sentences shared with the next chunk.

        Why overlap? So a query that matches a sentence near a chunk
        boundary still retrieves the surrounding context.
        """
        # Simple sentence split — keeps text clean
        sentences = [
            s.strip()
            for s in re.split(r"(?<=[.!?])\s+", text)
            if len(s.strip()) > 20   # skip headers, page numbers, etc.
        ]

        if not sentences:
            return []

        step   = max(1, CHUNK_SIZE - CHUNK_OVERLAP)
        chunks = []

        for i in range(0, len(sentences), step):
            window = sentences[i : i + CHUNK_SIZE]
            chunk  = " ".join(window).strip()
            if chunk:
                chunks.append(chunk)

        return chunks

    # ─────────────────────────────────────────────────────
    # INTERNAL — CACHE SAVE / LOAD
    # ─────────────────────────────────────────────────────
    def _save_index(self, path: Path) -> None:
        payload = {
            "chunks"      : self._chunks,
            "chunk_files" : self._chunk_files,
            "chunk_texts" : self._chunk_texts,
            "index_bytes" : faiss.serialize_index(self._index),
        }
        with open(path, "wb") as f:
            pickle.dump(payload, f)
        print(f"💾 Index cached → {path}")

    def _load_index(self, path: Path) -> None:
        with open(path, "rb") as f:
            payload = pickle.load(f)
        self._chunks      = payload["chunks"]
        self._chunk_files = payload["chunk_files"]
        self._chunk_texts = payload["chunk_texts"]
        self._index       = faiss.deserialize_index(payload["index_bytes"])


# ─────────────────────────────────────────────────────────
# QUICK TEST
# ─────────────────────────────────────────────────────────
if __name__ == "__main__":
    rag = RAGRetriever(pdf_folder="medical_db", force_rebuild=True)

    test_queries = [
        "what is the treatment for hantavirus?",
        "how does ebola spread?",
        "symptoms of diabetes mellitus",
        "COVID-19 vaccination",
    ]

    for q in test_queries:
        fname, _, chunks = rag.retrieve(q, top_k=3)
        print(f"\nQuery : {q}")
        print(f"Source: {fname}")
        print(f"Top chunk: {chunks[0][:200] if chunks else 'none'}...")
        print("─" * 60)