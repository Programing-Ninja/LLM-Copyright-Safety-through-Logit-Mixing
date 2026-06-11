import requests
import time
import json
import random

# -------------------------------
# Config
# -------------------------------
TOTAL_ENTITIES = 200
BATCH_SIZE = 50
SLEEP_BETWEEN_CALLS = 3

# -------------------------------
# Query function (robust)
# -------------------------------
def fetch_batch(limit=50, offset=0):
    url = "https://query.wikidata.org/sparql"

    query = f"""
    SELECT ?humanLabel WHERE {{
      ?human wdt:P31 wd:Q5.
      SERVICE wikibase:label {{ bd:serviceParam wikibase:language "en". }}
    }}
    LIMIT {limit}
    OFFSET {offset}
    """

    headers = {
        "User-Agent": "Mozilla/5.0 (compatible; DatasetBuilder/1.0)"
    }

    try:
        response = requests.get(
            url,
            params={"query": query, "format": "json"},
            headers=headers,
            timeout=20
        )

        if response.status_code != 200:
            print(f"[WARN] Bad status: {response.status_code}")
            return []

        if not response.text.strip():
            print("[WARN] Empty response")
            return []

        data = response.json()

        results = [
            item["humanLabel"]["value"]
            for item in data["results"]["bindings"]
        ]

        return results

    except Exception as e:
        print("[ERROR]", e)
        return []

# -------------------------------
# Main collection loop
# -------------------------------
def build_dataset(total=200):
    entities = set()
    offset = 0

    while len(entities) < total:
        print(f"Fetching batch (offset={offset})...")

        batch = fetch_batch(limit=BATCH_SIZE, offset=offset)

        if not batch:
            print("Retrying after delay...")
            time.sleep(5)
            continue

        entities.update(batch)
        offset += BATCH_SIZE

        print(f"Collected: {len(entities)}")

        time.sleep(SLEEP_BETWEEN_CALLS)

    return list(entities)[:total]

# -------------------------------
# Prompt builder
# -------------------------------
def make_prompt(entity):
    return f"""Write a factual biography about {entity}.
Include only factual information that you are confident in.

Biography:"""

# -------------------------------
# Run
# -------------------------------
if __name__ == "__main__":
    entities = build_dataset(TOTAL_ENTITIES)

    # Optional shuffle (better distribution)
    random.shuffle(entities)

    prompts = [make_prompt(e) for e in entities]

    dataset = {
        "entities": entities,
        "prompts": prompts
    }

    with open("biography_dataset.json", "w") as f:
        json.dump(dataset, f, indent=2)

    print("\n✅ Dataset saved to biography_dataset.json")
    print(f"Total entities: {len(entities)}")
    print("Sample:", entities[:5])