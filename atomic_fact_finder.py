import json
import pandas as pd
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from tqdm import tqdm

MODEL_NAME = "meta-llama/Llama-3.1-8B-Instruct"

PROMPT_TEMPLATE = """
You are an information extraction system.

Extract atomic, objectively verifiable factual claims from the text.

RULES:
- Extract ONLY factual claims.
- Ignore opinions, speculation, stylistic language, and vague statements.
- Each claim must express one independently verifiable fact.
- Preserve the original meaning.
- Do NOT split claims into trivial fragments.
- Rewrite claims into standalone sentences when necessary.
- Do NOT add information not present in the text.
- Avoid duplicate claims.

OUTPUT:
Return ONLY valid JSON.

Format:
{
  "claims": [
    "claim 1",
    "claim 2"
  ]
}

TEXT:
{TEXT}
"""

print("Loading model...")

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.float16,
    device_map="auto"
)

df = pd.read_parquet("outputs/model_outputs.parquet")

all_claims = []

for idx, row in tqdm(df.iterrows(), total=len(df)):

    prompt = PROMPT_TEMPLATE.replace("{TEXT}", row["response"])

    messages = [
        {"role": "user", "content": prompt}
    ]

    input_text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    inputs = tokenizer(
        input_text,
        return_tensors="pt"
    ).to(model.device)

    outputs = model.generate(
        **inputs,
        max_new_tokens=512,
        temperature=0.0,
        do_sample=False
    )

    generated = tokenizer.decode(
        outputs[0][inputs.input_ids.shape[1]:],
        skip_special_tokens=True
    )

    try:
        parsed = json.loads(generated)

        claims = parsed.get("claims", [])

        for i, claim in enumerate(claims):

            all_claims.append({
                "prompt_id": row["prompt_id"],
                "model": row["model"],
                "claim_id": i,
                "claim": claim
            })

    except Exception as e:

        print(f"Failed parsing row {idx}")
        print(generated)

claims_df = pd.DataFrame(all_claims)

claims_df.to_parquet(
    "claims/extracted_claims.parquet",
    index=False
)

print("Done.")