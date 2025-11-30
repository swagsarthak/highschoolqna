# Retrieval demo (Ollama embeddings)

Simple script to query the chunked book using locally generated Ollama embeddings.

Requirements
- `ollama` running locally (tested with `mxbai-embed-large`; pull it first: `ollama pull mxbai-embed-large`)
- Python deps: `numpy`, `requests`
- Data files (produced by cleaning pipeline):
  - Chemistry chunks: `cleaning/chunks/chemistry/OrganicChemistry-SAMPLE_9ADraVJ_chunks.jsonl`
  - Chemistry embeddings: `cleaning/chunks/chemistry/OrganicChemistry-SAMPLE_9ADraVJ_embeddings_ollama.npy`
  - Chemistry metadata: `cleaning/chunks/chemistry/OrganicChemistry-SAMPLE_9ADraVJ_embeddings_ollama_meta.jsonl`
  - Physics chunks: `cleaning/chunks/physics/UniversityPhysics15e_chunks.jsonl`
  - Physics embeddings: `cleaning/chunks/physics/UniversityPhysics15e_embeddings_ollama.npy`
  - Physics metadata: `cleaning/chunks/physics/UniversityPhysics15e_embeddings_ollama_meta.jsonl`

Run
```bash
# Example query (uses default model mxbai-embed-large and top-5)
python retrieval/retrieve.py "What is the structure of methane?" --subject chemistry

# Override model and top-k if needed
python retrieval/retrieve.py "Explain sp3 hybridization" --model mxbai-embed-large --top-k 8 --subject chemistry
```

Notes
- The retrieval uses cosine similarity between precomputed chunk embeddings and a query embedding generated via Ollama (same model).
- If you regenerate embeddings, rerun `cleaning/step6_ollama_embed_chunks.py` before using this script.
- Use `--subject` to target the right folder (`chemistry`, `physics`, etc.).

QA (RAG) with an LLM
--------------------
To get an answer (not just chunks), use `qa.py`, which retrieves top chunks and asks an Ollama LLM (default `llama3`) with the context:
```bash
python retrieval/qa.py "Explain sp3 hybridization" --top-k 5 \
  --embed-model mxbai-embed-large \
  --llm-model llama3 \
  --subject chemistry
```
Pull models first:
- Embedding: `ollama pull mxbai-embed-large`
- LLM: `ollama pull llama3` (or another chat-capable model)

Frontend (FastAPI)
------------------
A minimal FastAPI app is in `app/`. It serves a simple web UI and an `/api/qa` endpoint that uses the same Ollama embeddings/LLM.
Run:
```bash
uvicorn app.main:app --reload
```
Then open http://localhost:8000/ and ask questions. It expects the same embedding files produced by the cleaning pipeline and an Ollama chat model (default `llama3`).
