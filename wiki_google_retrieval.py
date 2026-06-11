import requests
import pandas as pd
import wikipediaapi
import trafilatura
import pickle
import faiss
import numpy as np

from tqdm import tqdm
from sentence_transformers import SentenceTransformer, util

# ============================================================
# CONFIG
# ============================================================

SERPER_API_KEY = "YOUR_SERPER_API_KEY"

EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

TOP_WIKI = 2
TOP_GOOGLE = 3

MAX_WEBPAGE_CHARS = 5000

# ============================================================
# LOAD EMBEDDING MODEL
# ============================================================

print("Loading embedding model...")

embedder = SentenceTransformer(EMBED_MODEL)

# ============================================================
# LOAD WIKIPEDIA INDEX
# ============================================================

print("Loading Wikipedia FAISS index...")

index = faiss.read_index("retrieval/wiki.index")

with open("retrieval/wiki_chunks.pkl", "rb") as f:
    wiki_chunks = pickle.load(f)

# ============================================================
# WIKIPEDIA RETRIEVAL
# ============================================================

def retrieve_wikipedia(claim, k=TOP_WIKI):

    embedding = embedder.encode(
        [claim],
        convert_to_numpy=True
    )

    distances, indices = index.search(embedding, k)

    results = []

    for idx, dist in zip(indices[0], distances[0]):

        chunk = wiki_chunks[idx]

        results.append({
            "source": "wikipedia",
            "title": chunk["title"],
            "url": f"https://en.wikipedia.org/wiki/{chunk['title'].replace(' ', '_')}",
            "text": chunk["text"],
            "score": float(dist)
        })

    return results

# ============================================================
# GOOGLE SEARCH
# ============================================================

def google_search(query, k=5):

    url = "https://google.serper.dev/search"

    payload = {
        "q": query,
        "num": k
    }

    headers = {
        "X-API-KEY": SERPER_API_KEY,
        "Content-Type": "application/json"
    }

    response = requests.post(
        url,
        json=payload,
        headers=headers
    )

    response.raise_for_status()

    data = response.json()

    results = []

    for item in data.get("organic", []):

        results.append({
            "title": item.get("title"),
            "url": item.get("link"),
            "snippet": item.get("snippet", "")
        })

    return results

# ============================================================
# WEBPAGE EXTRACTION
# ============================================================

def extract_webpage_text(url):

    try:

        downloaded = trafilatura.fetch_url(url)

        if downloaded is None:
            return None

        text = trafilatura.extract(
            downloaded,
            include_comments=False,
            include_tables=False
        )

        if text is None:
            return None

        return text[:MAX_WEBPAGE_CHARS]

    except Exception:

        return None

# ============================================================
# GOOGLE RETRIEVAL
# ============================================================

def retrieve_google(claim, k=TOP_GOOGLE):

    search_results = google_search(claim, k=k)

    retrieved = []

    for result in search_results:

        text = extract_webpage_text(result["url"])

        if text is None:
            continue

        retrieved.append({
            "source": "google",
            "title": result["title"],
            "url": result["url"],
            "text": text,
            "score": None
        })

    return retrieved

# ============================================================
# MERGE + RERANK
# ============================================================

def rerank_evidence(claim, evidence, top_k=5):

    texts = [e["text"] for e in evidence]

    claim_emb = embedder.encode(
        claim,
        convert_to_tensor=True
    )

    text_embs = embedder.encode(
        texts,
        convert_to_tensor=True
    )

    similarities = util.cos_sim(
        claim_emb,
        text_embs
    )[0]

    ranked = []

    for sim, e in zip(similarities, evidence):

        e["similarity"] = float(sim)
        ranked.append(e)

    ranked.sort(
        key=lambda x: x["similarity"],
        reverse=True
    )

    return ranked[:top_k]

# ============================================================
# MAIN PIPELINE
# ============================================================

def retrieve_evidence_for_claim(claim):

    wiki_results = retrieve_wikipedia(claim)

    google_results = retrieve_google(claim)

    combined = wiki_results + google_results

    reranked = rerank_evidence(
        claim,
        combined
    )

    return reranked

# ============================================================
# LOAD CLAIMS
# ============================================================

claims_df = pd.read_parquet(
    "claims/extracted_claims.parquet"
)

all_evidence = []

# ============================================================
# RUN RETRIEVAL
# ============================================================

for _, row in tqdm(claims_df.iterrows(),
                   total=len(claims_df)):

    claim = row["claim"]

    try:

        evidence_list = retrieve_evidence_for_claim(
            claim
        )

        for rank, ev in enumerate(evidence_list):

            all_evidence.append({
                "prompt_id": row["prompt_id"],
                "model": row["model"],
                "claim_id": row["claim_id"],
                "claim": claim,
                "rank": rank,
                "source": ev["source"],
                "title": ev["title"],
                "url": ev["url"],
                "evidence_text": ev["text"],
                "similarity": ev["similarity"]
            })

    except Exception as e:

        print(f"FAILED: {claim}")
        print(e)

# ============================================================
# SAVE
# ============================================================

evidence_df = pd.DataFrame(all_evidence)

evidence_df.to_parquet(
    "retrieval/evidence.parquet",
    index=False
)

print("Saved evidence.")