# Cleaning workspace

Use this folder to hold everything related to preparing books for RAG.

- `raw_pdf/` keep a copy of the original PDF you are cleaning
- `text_workdir/` intermediate artifacts (extracted text, chunking drafts, notes)
- `clean_text/` final cleaned text ready for indexing
- `chunks/` subject-specific chunk outputs for RAG ingestion (e.g., `chunks/chemistry`, `chunks/physics`, `chunks/biology`)
- `images/` extracted figures per subject (optional; used when you want to keep images)

Processing steps:
1. `step1_extract_text.py` pulls raw text with page markers into `text_workdir/`.
1b. `step1b_extract_images.py` (optional) saves page images to `cleaning/images/<subject>/<book>/` and writes JSONL metadata; can OCR if `pytesseract` + Pillow are installed. Pass `--filter` to also filter/deduplicate in one step.
2. `step2_trim_pages.py` removes front/back matter by page range into `clean_text/`.
3. `step3_normalize_text.py` strips headers/footers/page markers and normalizes whitespace while keeping paragraph separation.
4. `step4_chunk_text.py` splits normalized text into overlapping chunks for RAG.
5. `step5_embed_chunks.py` embeds chunks with `sentence-transformers/all-MiniLM-L6-v2` (fast, strong baseline). Override the model by passing a model name as the first CLI arg.
6. `step6_ollama_embed_chunks.py` embeds chunks with a local Ollama embedding model (default `mxbai-embed-large`). Override the model via `--model` or CLI arg.

Quick usage (text pipeline)
---------------------------
```bash
# 0) (one-time) install embedding deps
pip install sentence-transformers

# 1-5) run the pipeline end to end
python cleaning/step1_extract_text.py
# (optional) extract figures for a book (add --ocr for OCR; add --filter to drop logos/dedup in one go)
python cleaning/step1b_extract_images.py --pdf cleaning/raw_pdf/... --subject <subject> [--ocr] [--filter]
python cleaning/step2_trim_pages.py
python cleaning/step3_normalize_text.py
python cleaning/step4_chunk_text.py
python cleaning/step5_embed_chunks.py  # or pass another model as first arg
# (optional local embedding via Ollama; pull a model first)
# ollama pull mxbai-embed-large   # or another embedding model, e.g., nomic-embed-text
python cleaning/step6_ollama_embed_chunks.py  # or pass another Ollama model as first arg
```

Key outputs (for slides/report)
- Raw extracted: `cleaning/text_workdir/OrganicChemistry-SAMPLE_9ADraVJ_raw.txt`
- Trimmed: `cleaning/clean_text/OrganicChemistry-SAMPLE_9ADraVJ_clean.txt` (front/back matter removed)
- Normalized: `cleaning/clean_text/OrganicChemistry-SAMPLE_9ADraVJ_clean_normalized.txt`
- Chunks (chemistry): `cleaning/chunks/chemistry/OrganicChemistry-SAMPLE_9ADraVJ_chunks.jsonl` (current: 335 chunks, ~280 words, 40-word overlap)
- Embeddings (chemistry, sentence-transformers): `cleaning/chunks/chemistry/OrganicChemistry-SAMPLE_9ADraVJ_embeddings.npy` (+ `..._embeddings_meta.jsonl`)
- Embeddings (chemistry, Ollama): `cleaning/chunks/chemistry/OrganicChemistry-SAMPLE_9ADraVJ_embeddings_ollama.npy` (+ `..._embeddings_ollama_meta.jsonl`)
- Raw extracted (physics): `cleaning/text_workdir/UniversityPhysics15e_raw.txt`
- Trimmed (physics): `cleaning/clean_text/UniversityPhysics15e_clean.txt` (pages 30-1544 kept)
- Normalized (physics): `cleaning/clean_text/UniversityPhysics15e_clean_normalized.txt`
- Chunks (physics): `cleaning/chunks/physics/UniversityPhysics15e_chunks.jsonl` (~7514 chunks with current params)
- Embeddings (physics, Ollama): `cleaning/chunks/physics/UniversityPhysics15e_embeddings_ollama.npy` (+ `..._embeddings_ollama_meta.jsonl`)
