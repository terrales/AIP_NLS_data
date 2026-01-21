#!/usr/bin/env python3
"""
Measure repetitivity of the subject field from the original catalogue.
"""

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

import pandas as pd
import numpy as np
import torch
from tqdm import tqdm
from openai import OpenAI
from transformers import AutoModel, AutoTokenizer
import random
from concurrent.futures import ThreadPoolExecutor, as_completed

# Configuration
EMBEDDING_MODEL = "Qwen/Qwen3-Embedding-8B"
GPT_MODEL = "gpt-5.2"
SAMPLE_SIZE = 100
TOP_K = 5
RANDOM_SEED = 42

# File paths
BASE_DIR = "/disk/scratch/s1891075/AIP_NLS_data/yintao"
INPUT_FILE = os.path.join(BASE_DIR, "layer_3_topics_all.csv")
OUTPUT_CSV = os.path.join(BASE_DIR, "subject_repetitivity_results.csv")

# Set random seed
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

def load_qwen3_model():
    """Load Qwen3 embedding model."""
    print(f"Loading Qwen3 embedding model: {EMBEDDING_MODEL}")
    tokenizer = AutoTokenizer.from_pretrained(EMBEDDING_MODEL, trust_remote_code=True)
    model = AutoModel.from_pretrained(EMBEDDING_MODEL, trust_remote_code=True, torch_dtype=torch.float16)
    model = model.cuda()
    model.eval()
    return model, tokenizer

def embed_topics(topics, model, tokenizer, batch_size=64):
    """Embed topic names using Qwen3."""
    embeddings = []

    for i in tqdm(range(0, len(topics), batch_size), desc="Embedding subjects"):
        batch = topics[i:i+batch_size]
        inputs = tokenizer(batch, padding=True, truncation=True, max_length=512, return_tensors="pt")
        inputs = {k: v.cuda() for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model(**inputs)
            # Use mean pooling
            attention_mask = inputs['attention_mask']
            hidden_states = outputs.last_hidden_state
            mask_expanded = attention_mask.unsqueeze(-1).expand(hidden_states.size()).float()
            sum_embeddings = torch.sum(hidden_states * mask_expanded, 1)
            sum_mask = torch.clamp(mask_expanded.sum(1), min=1e-9)
            batch_embeddings = sum_embeddings / sum_mask
            batch_embeddings = torch.nn.functional.normalize(batch_embeddings, p=2, dim=1)
            embeddings.append(batch_embeddings.cpu().numpy())

    return np.vstack(embeddings)

def find_top_k_similar(embeddings, idx, k=5):
    """Find top k most similar topics (excluding self) using cosine similarity."""
    query = embeddings[idx:idx+1]  # Shape: (1, dim)

    # Cosine similarity (embeddings are already normalized)
    similarities = np.dot(embeddings, query.T).flatten()

    # Set self-similarity to -inf to exclude
    similarities[idx] = -np.inf

    # Get top k indices
    top_k_indices = np.argsort(similarities)[-k:][::-1]
    top_k_scores = similarities[top_k_indices]

    return top_k_indices, top_k_scores

def judge_repetitivity(topic, similar_topics, client):
    """Use GPT-5.2 to judge if any similar topics are semantically redundant."""
    similar_list = "\n".join([f"{i+1}. {t}" for i, t in enumerate(similar_topics)])

    prompt = f"""You are evaluating topic names from a library catalogue classification system.

Given the main topic and its 5 most similar topics (by embedding similarity), determine if ANY of the similar topics are semantically redundant with the main topic - meaning they represent essentially the same concept and should be merged into a single topic.

Main Topic: "{topic}"

Most Similar Topics:
{similar_list}

Is there at least ONE topic in the list above that is semantically redundant with the main topic (i.e., they should be merged)?

Answer with ONLY "YES" or "NO".
- YES: At least one similar topic is redundant and should be merged with the main topic
- NO: All similar topics are sufficiently distinct from the main topic"""

    try:
        response = client.chat.completions.create(
            model=GPT_MODEL,
            messages=[{"role": "user", "content": prompt}],
            max_completion_tokens=10,
            temperature=0
        )
        answer = response.choices[0].message.content.strip().upper()
        return "YES" in answer
    except Exception as e:
        print(f"Error calling GPT: {e}")
        return None

def main():
    print("="*60)
    print("Subject Field Repetitivity Measurement")
    print("="*60)

    # Initialize OpenAI client
    client = OpenAI()

    # Load Qwen3 model
    model, tokenizer = load_qwen3_model()

    # Load data
    print(f"\nLoading data from {INPUT_FILE}...")
    df = pd.read_csv(INPUT_FILE, low_memory=False)
    print(f"Total rows: {len(df)}")

    # Get unique subjects
    unique_subjects = df['subject'].dropna().unique().tolist()
    n_subjects = len(unique_subjects)
    print(f"Found {n_subjects} unique subjects")

    # Embed all subjects
    print("\nEmbedding subjects with Qwen3...")
    embeddings = embed_topics(unique_subjects, model, tokenizer)
    print(f"Embeddings shape: {embeddings.shape}")

    # Sample subjects
    n_sample = min(SAMPLE_SIZE, n_subjects)
    sampled_indices = random.sample(range(n_subjects), n_sample)
    print(f"Sampled {n_sample} subjects for evaluation")

    # Find similar subjects and judge redundancy
    redundant_count = 0
    results_detail = []

    def evaluate_subject(idx):
        subject = unique_subjects[idx]
        top_k_indices, top_k_scores = find_top_k_similar(embeddings, idx, k=TOP_K)
        similar_subjects = [unique_subjects[i] for i in top_k_indices]
        is_redundant = judge_repetitivity(subject, similar_subjects, client)
        return {
            "subject": subject,
            "similar_subjects": similar_subjects,
            "similarities": top_k_scores.tolist(),
            "is_redundant": is_redundant
        }

    print("\nEvaluating redundancy with GPT-5.2...")
    with ThreadPoolExecutor(max_workers=16) as executor:
        futures = {executor.submit(evaluate_subject, idx): idx for idx in sampled_indices}
        for future in tqdm(as_completed(futures), total=len(sampled_indices), desc="GPT-5.2 evaluation"):
            result = future.result()
            results_detail.append(result)
            if result["is_redundant"]:
                redundant_count += 1

    ratio = redundant_count / n_sample if n_sample > 0 else 0.0
    print(f"\nRedundancy ratio: {redundant_count}/{n_sample} = {ratio:.2%}")

    # Save results
    result_df = pd.DataFrame([{
        "setting": "Original Subject",
        "level": "subject",
        "n_topics": n_subjects,
        "n_sampled": n_sample,
        "n_redundant": redundant_count,
        "ratio": ratio
    }])
    result_df.to_csv(OUTPUT_CSV, index=False)
    print(f"\nResults saved to: {OUTPUT_CSV}")

    # Print result
    print("\n" + "="*60)
    print("RESULT")
    print("="*60)
    print(f"Original Subject Field:")
    print(f"  - Unique subjects: {n_subjects}")
    print(f"  - Sampled: {n_sample}")
    print(f"  - Redundant: {redundant_count}")
    print(f"  - Ratio: {ratio:.1%}")

    return ratio, n_subjects

if __name__ == "__main__":
    main()
