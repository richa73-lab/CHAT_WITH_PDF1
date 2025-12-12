import os
from typing import Dict, Any, List, Tuple
import re

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
import uvicorn
import PyPDF2
import numpy as np
from dotenv import load_dotenv
from google import genai  # from google-genai package


# ---------- Load environment ----------
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

if not GEMINI_API_KEY:
    raise RuntimeError("GEMINI_API_KEY is not set in .env")

# ---------- Gemini client ----------
client = genai.Client(api_key=GEMINI_API_KEY)

CHAT_MODEL = "gemini-2.5-flash"
EMBED_MODEL = "text-embedding-004"

# ---------- FastAPI app ----------
app = FastAPI()

# CORS so browser JS can call the API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],       # for local dev this is fine
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# In-memory store of chunks for the current PDF
pdf_chunks: List[Dict[str, Any]] = []


# ---------- Helpers ----------

def extract_text_from_pdf(file_obj) -> str:
    """Read all pages from a PDF file-like object and return plain text."""
    reader = PyPDF2.PdfReader(file_obj)
    text = ""
    for page in reader.pages:
        page_text = page.extract_text() or ""
        text += page_text + "\n"
    return text


def split_into_chunks(text: str, max_chars: int = 800) -> List[str]:
    """Very simple sentence-ish splitter into ~max_chars chunks."""
    sentences = text.split(". ")
    chunks = []
    current = ""

    for s in sentences:
        if len(current) + len(s) + 2 > max_chars:
            if current.strip():
                chunks.append(current.strip())
            current = s
        else:
            current += s + ". "

    if current.strip():
        chunks.append(current.strip())

    return chunks


def cosine_similarity(a: List[float], b: List[float]) -> float:
    """Cosine similarity between two vectors."""
    va = np.array(a, dtype=float)
    vb = np.array(b, dtype=float)
    denom = (np.linalg.norm(va) * np.linalg.norm(vb))
    if denom == 0:
        return 0.0
    return float(np.dot(va, vb) / denom)


def embed_text(text: str) -> list[float]:
    resp = client.models.embed_content(
        model=EMBED_MODEL,
        contents=text,
    )
    return list(resp.embeddings[0].values)


def is_verbose_request(question: str) -> bool:
    """
    Return True when user asked for more detailed/explained answer.
    Basic keyword matching - extend as needed.
    """
    q = question.lower()
    verbose_keywords = [
        "explain", "in depth", "detail", "detailed", "elaborate",
        "more", "more detail", "explain more", "full", "thorough"
    ]
    for kw in verbose_keywords:
        if kw in q:
            return True
    return False


def rerank_chunks_with_gemini(question: str, chunks: List[Dict[str, Any]], top_k: int = 4) -> List[Dict[str, Any]]:
    """
    Use the Gemini model as a reranker:
    - Provide the question + a numbered list of candidate chunks
    - Ask Gemini to return the top K most relevant chunk numbers (comma separated)
    - Return the corresponding chunks in ranked order.

    This uses a short call to the chat model for better reranking than raw cosine alone.
    """
    if not chunks:
        return []

    # Build candidate list (limit text length for prompt safety)
    candidates_text = ""
    for i, c in enumerate(chunks, start=1):
        snippet = c["text"]
        # keep snippet reasonably short in the reranker prompt
        if len(snippet) > 800:
            snippet = snippet[:800].rsplit(" ", 1)[0] + "..."
        candidates_text += f"{i}. {snippet}\n\n"

    rerank_prompt = f"""
You are a relevance rater. Given a user question and a list of text chunks extracted from a PDF, return the TOP {top_k} chunk numbers (in descending relevance order) as a comma-separated list with no extra text.

Question:
{question}

Chunks:
{candidates_text}

Only reply with numbers separated by commas, e.g. "3,1,7,2"
"""

    resp = client.models.generate_content(
        model=CHAT_MODEL,
        contents=rerank_prompt,
    )

    text = resp.text.strip()
    # try to extract numbers
    numbers = re.findall(r"\d+", text)
    ranked_indices: List[int] = []
    for n in numbers:
        try:
            idx = int(n) - 1  # prompt used 1-based indexing
            if 0 <= idx < len(chunks) and idx not in ranked_indices:
                ranked_indices.append(idx)
            if len(ranked_indices) >= top_k:
                break
        except Exception:
            continue

    # fallback: if LLM didn't give good output, use cosine ordering
    if not ranked_indices:
        # sort by cosine similarity between question embedding and chunk embedding
        q_vec = embed_text(question)
        scored = sorted(
            chunks,
            key=lambda c: cosine_similarity(q_vec, c["embedding"]),
            reverse=True,
        )
        return scored[:top_k]

    # Return the selected chunks in the rank order
    ranked = [chunks[i] for i in ranked_indices]
    return ranked


# ---------- API: Upload PDF ----------

@app.post("/upload")
async def upload_pdf(pdf: UploadFile = File(...)):
    """
    Receives a PDF file, extracts text, chunks it, embeds each chunk,
    and stores it in memory for later questions.
    """
    global pdf_chunks

    if not pdf.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Please upload a PDF file")

    # Read PDF bytes into memory
    text = extract_text_from_pdf(pdf.file)
    if not text.strip():
        raise HTTPException(status_code=400, detail="Could not extract text from PDF")

    chunks = split_into_chunks(text, max_chars=800)

    # Embed each chunk
    embedded_chunks = []
    for ch in chunks:
        vec = embed_text(ch)
        embedded_chunks.append({"text": ch, "embedding": vec})

    pdf_chunks = embedded_chunks

    return {"ok": True, "chunks": len(pdf_chunks)}


# ---------- API: Chat with PDF ----------

@app.post("/chat")
async def chat(body: Dict[str, Any]):
    """
    Receives: { "question": "...", "top_k": optional int (how many initial candidates to consider) }
    Returns:  { "answer": "..." }
    """
    global pdf_chunks

    question = (body.get("question", "") or "").strip()
    if not question:
        raise HTTPException(status_code=400, detail="Question is required")

    if not pdf_chunks:
        raise HTTPException(status_code=400, detail="No PDF indexed yet")

    # how many candidate chunks to consider initially (defaults)
    initial_k = int(body.get("initial_k", 8))  # retrieve top 8 by cosine first
    final_k = int(body.get("final_k", 4))     # rerank then keep top 4

    # 1. Embed the question (for cosine initial ranking)
    q_vec = embed_text(question)

    # 2. Initial ranking by cosine similarity
    initial_candidates = sorted(
        pdf_chunks,
        key=lambda c: cosine_similarity(q_vec, c["embedding"]),
        reverse=True
    )[:initial_k]

    # 3. Rerank candidates using Gemini (LLM) to get the most relevant ones
    reranked = rerank_chunks_with_gemini(question, initial_candidates, top_k=final_k)

    # 4. Build context text from reranked chunks
    context_text = "\n\n".join(
        [f"Chunk {i+1}:\n{c['text']}" for i, c in enumerate(reranked)]
    )

    # 5. Determine verbosity
    verbose = is_verbose_request(question)

    # 6. Build prompt for Gemini with clear instructions for conciseness vs depth
    if verbose:
        behavior_instruction = (
            "You are a calm, friendly tutor. Answer using the PDF context below. "
            "Provide a thorough, structured explanation with multiple points and examples where relevant."
        )
    else:
        behavior_instruction = (
            "You are a calm, friendly tutor. Answer using the PDF context below. "
            "Be concise: give 2-4 short bullet points or a short paragraph. "
            "If the user asks later for more detail, expand."
        )

    prompt = f"""{behavior_instruction}

PDF context:
{context_text}

Question: {question}

If the context above does not contain the answer, say: "I couldn't find a confident answer in the provided document." and offer to upload another document or ask for clarification.
"""

    # 7. Call Gemini chat model
    resp = client.models.generate_content(
        model=CHAT_MODEL,
        contents=prompt,
    )

    answer = resp.text
    # Optionally, include which chunks were used
    used_chunks = [c["text"][:200].replace("\n", " ") + ("..." if len(c["text"]) > 200 else "") for c in reranked]

    return {"answer": answer, "used_context_snippets": used_chunks}


# ---------- Serve frontend (public folder) ----------
app.mount("/", StaticFiles(directory="public", html=True), name="public")


# ---------- Run server ----------
if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)
