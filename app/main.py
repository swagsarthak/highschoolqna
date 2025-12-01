"""Minimal FastAPI frontend/back-end for multi-subject RAG (chemistry/physics)."""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import requests
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

SUBJECT_LLM = {
    "chemistry": "llama3",
    "physics": "llama3",
    "biology": "llama3",
    "math": "qwen2.5:1.5b",
}
ROOT = Path(__file__).resolve().parent.parent
SUBJECTS = {
    "chemistry": {
        "chunks": ROOT / "cleaning" / "chunks" / "chemistry" / "OrganicChemistry-SAMPLE_9ADraVJ_chunks.jsonl",
        "embeddings": ROOT / "cleaning" / "chunks" / "chemistry" / "OrganicChemistry-SAMPLE_9ADraVJ_embeddings_ollama.npy",
        "meta": ROOT / "cleaning" / "chunks" / "chemistry" / "OrganicChemistry-SAMPLE_9ADraVJ_embeddings_ollama_meta.jsonl",
        "title": "Chemistry",
        "tagline": "Organic Chemistry",
    },
    "physics": {
        # Update after generating physics chunks/embeddings
        "chunks": ROOT / "cleaning" / "chunks" / "physics" / "UniversityPhysics15e_chunks.jsonl",
        "embeddings": ROOT / "cleaning" / "chunks" / "physics" / "UniversityPhysics15e_embeddings_ollama.npy",
        "meta": ROOT / "cleaning" / "chunks" / "physics" / "UniversityPhysics15e_embeddings_ollama_meta.jsonl",
        "title": "Physics",
        "tagline": "University Physics",
    },
    "biology": {
        "chunks": ROOT / "cleaning" / "chunks" / "biology" / "bio_merged_chunks.jsonl",
        "embeddings": ROOT / "cleaning" / "chunks" / "biology" / "bio_merged_embeddings_ollama.npy",
        "meta": ROOT / "cleaning" / "chunks" / "biology" / "bio_merged_embeddings_ollama_meta.jsonl",
        "title": "Biology",
        "tagline": "11 std Biology",
    },
    "math": {
        "chunks": ROOT / "cleaning" / "chunks" / "math" / "math_unified_chunks.jsonl",
        "embeddings": ROOT / "cleaning" / "chunks" / "math" / "math_unified_embeddings_ollama.npy",
        "meta": ROOT / "cleaning" / "chunks" / "math" / "math_unified_embeddings_ollama_meta.jsonl",
        "title": "Math",
        "tagline": "Merged math course pack",
    },
}

DEFAULT_EMBED_MODEL = "mxbai-embed-large"
DEFAULT_LLM_MODEL = "llama3"
EMBED_URL = "http://localhost:11434/api/embed"
CHAT_URL = "http://localhost:11434/api/chat"


class QARequest(BaseModel):
    query: str = Field(..., description="User question")
    top_k: int = Field(15, ge=1, le=20, description="Number of contexts to use")
    embed_model: str = Field(DEFAULT_EMBED_MODEL, description="Ollama embedding model")
    llm_model: str = Field(DEFAULT_LLM_MODEL, description="Ollama chat model")
    subject: str = Field("chemistry", description="Subject key (chemistry, physics, ...)")
    refine: bool = Field(True, description="Polish the draft answer with a second pass for clarity")


class QAResponse(BaseModel):
    answer: str
    contexts: List[Dict[str, str]]


def load_chunks_from(path: Path) -> Dict[str, str]:
    if not path.exists():
        raise FileNotFoundError(f"Missing chunks file: {path}")
    with path.open("r", encoding="utf-8") as f:
        return {json.loads(line)["id"]: json.loads(line)["text"] for line in f}


def load_meta_ids_from(path: Path) -> List[str]:
    if not path.exists():
        raise FileNotFoundError(f"Missing meta file: {path}")
    ids: List[str] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            ids.append(json.loads(line)["id"])
    return ids


def get_paths(subject: str) -> Tuple[Path, Path, Path]:
    if subject not in SUBJECTS:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown subject '{subject}'. Available: {list(SUBJECTS.keys())}",
        )
    cfg = SUBJECTS[subject]
    return cfg["chunks"], cfg["embeddings"], cfg["meta"]


DATA_CACHE: Dict[str, Tuple[Dict[str, str], List[str], np.ndarray]] = {}


def get_data(subject: str) -> Tuple[Dict[str, str], List[str], np.ndarray]:
    if subject in DATA_CACHE:
        return DATA_CACHE[subject]

    chunks_path, emb_path, meta_path = get_paths(subject)
    if not (chunks_path.exists() and emb_path.exists() and meta_path.exists()):
        raise HTTPException(
            status_code=500,
            detail=f"Missing data for subject '{subject}'. Expected {chunks_path}, {emb_path}, {meta_path}",
        )

    chunks = load_chunks_from(chunks_path)
    meta_ids = load_meta_ids_from(meta_path)
    emb_matrix = np.load(emb_path)
    DATA_CACHE[subject] = (chunks, meta_ids, emb_matrix)
    return chunks, meta_ids, emb_matrix


def embed_text(model: str, text: str) -> np.ndarray:
    resp = requests.post(EMBED_URL, json={"model": model, "input": [text]})
    if not resp.ok:
        raise HTTPException(
            status_code=502, detail=f"Embedding failed ({resp.status_code}): {resp.text}"
        )
    data = resp.json()
    return np.array(data["embeddings"][0], dtype=np.float32)


def cosine_sim_matrix(query_vec: np.ndarray, mat: np.ndarray) -> np.ndarray:
    mat_norm = mat / np.linalg.norm(mat, axis=1, keepdims=True)
    q_norm = query_vec / np.linalg.norm(query_vec)
    return mat_norm @ q_norm


def top_k(scores: np.ndarray, ids: List[str], k: int) -> List[Tuple[str, float]]:
    idxs = np.argpartition(-scores, kth=min(k, len(scores) - 1))[:k]
    sorted_idxs = idxs[np.argsort(-scores[idxs])]
    return [(ids[i], float(scores[i])) for i in sorted_idxs]


STOPWORDS = {
    "the",
    "a",
    "an",
    "and",
    "or",
    "of",
    "in",
    "on",
    "for",
    "to",
    "with",
    "by",
    "at",
    "from",
    "as",
    "that",
    "this",
    "these",
    "those",
    "is",
    "are",
    "was",
    "were",
}


def rerank_with_overlap(
    query: str, results: List[Tuple[str, float]], chunks: Dict[str, str]
) -> List[Tuple[str, float]]:
    tokens = [t for t in re.findall(r"\w+", query.lower()) if len(t) > 2 and t not in STOPWORDS]
    if not tokens:
        return results
    scored: List[Tuple[str, float, float]] = []
    for cid, base in results:
        text = chunks.get(cid, "").lower()
        overlap = sum(1 for tok in tokens if tok in text)
        combined = base + 0.05 * overlap
        scored.append((cid, base, combined))
    scored.sort(key=lambda x: x[2], reverse=True)
    return [(cid, base) for cid, base, _ in scored]


def build_prompt(query: str, contexts: List[Tuple[str, str]]) -> str:
    context_block = "\n\n".join(
        f"Snippet {i+1}:\n{text[:1200]}" for i, (_cid, text) in enumerate(contexts)
    )
    return (
        "Answer the question using ONLY the provided context.\n"
        "- Be CONCISE but ACCURATE: extract the relevant information and state it clearly.\n"
        "- If the context contains the answer, give it directly in 1-4 sentences.\n"
        "- If the context does NOT contain information to answer the question, say: "
        "'This information is not covered in the available materials.'\n"
        "- Never mention snippet numbers, chunk IDs, or source identifiers.\n\n"
        f"Context:\n{context_block}\n\n"
        f"Question: {query}\n\nAnswer:"
    )


def call_llm(model: str, prompt: str) -> str:
    payload = {
        "model": model,
        "messages": [
            {
                "role": "system",
                "content": (
                    "You are a concise tutor. Give SHORT, direct answers using only the provided context. "
                    "Never mention chunk IDs, snippet numbers, page IDs, or any source identifiers."
                ),
            },
            {"role": "user", "content": prompt},
        ],
        "stream": False,
        "options": {"temperature": 0},
    }
    resp = requests.post(CHAT_URL, json=payload)
    if not resp.ok:
        raise HTTPException(
            status_code=502, detail=f"LLM request failed ({resp.status_code}): {resp.text}"
        )
    data = resp.json()
    return data["message"]["content"]


def refine_answer(model: str, draft: str, contexts: List[Dict[str, str]]) -> str:
    # Use numbered snippets instead of exposing internal chunk IDs
    snippets_text = "\n".join(f"[{i+1}] {c['snippet']}" for i, c in enumerate(contexts))
    prompt = (
        "Polish the draft into a SHORT, direct answer. Rules:\n"
        "- Be concise: 1-3 sentences for simple questions, up to a short paragraph for complex ones.\n"
        "- Do NOT mention snippet numbers, chunk IDs, page IDs, or any identifiers.\n"
        "- Do NOT add hedging phrases like 'based on the snippets' or 'according to the context'.\n"
        "- Just answer the question naturally as if you knew the answer.\n"
        "- If uncertain, say so briefly.\n\n"
        f"Draft:\n{draft}\n\n"
        f"Reference snippets:\n{snippets_text}"
    )
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": "You are a concise editor. Make answers short and direct. Never mention source IDs."},
            {"role": "user", "content": prompt},
        ],
        "stream": False,
        "options": {"temperature": 0},
    }
    resp = requests.post(CHAT_URL, json=payload)
    if not resp.ok:
        # Fall back to draft on error
        return draft
    data = resp.json()
    return data["message"]["content"]


def build_snippet(text: str, limit: int = 320) -> str:
    flat = text.replace("\n", " ")
    return (flat[:limit] + "..." ) if len(flat) > limit else flat


app = FastAPI(title="Organic Chemistry RAG", version="0.1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def serve_frontend():
    index_path = Path(__file__).resolve().parent / "static" / "index.html"
    return FileResponse(index_path)


@app.get("/api/subjects")
def subjects():
    items = []
    for key, cfg in SUBJECTS.items():
        items.append(
            {
                "key": key,
                "title": cfg.get("title", key.title()),
                "tagline": cfg.get("tagline", ""),
                "has_data": cfg["chunks"].exists() and cfg["embeddings"].exists() and cfg["meta"].exists(),
            }
        )
    return {"subjects": items}


@app.post("/api/qa", response_model=QAResponse)
def qa(req: QARequest):
    chunks, meta_ids, emb_matrix = get_data(req.subject)
    q_vec = embed_text(req.embed_model, req.query)
    scores = cosine_sim_matrix(q_vec, emb_matrix)
    results = rerank_with_overlap(req.query, top_k(scores, meta_ids, req.top_k), chunks)

    contexts: List[Tuple[str, str]] = []
    context_payload: List[Dict[str, str]] = []
    for cid, score in results:
        text = chunks.get(cid, "")
        contexts.append((cid, text))
        context_payload.append({"id": cid, "score": f"{score:.4f}", "snippet": build_snippet(text)})

    prompt = build_prompt(req.query, contexts)
    model = SUBJECT_LLM.get(req.subject, req.llm_model)
    print(f"[API QA] subject={req.subject}  llm_model={model}")
    answer = call_llm(model, prompt).strip()
    if req.refine:
        answer = refine_answer(req.llm_model, answer, context_payload).strip()
    return QAResponse(answer=answer, contexts=context_payload)