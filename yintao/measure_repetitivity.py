#!/usr/bin/env python3
"""
Measure repetitivity of topic modeling by finding similar topics
and using GPT-5.2 to judge if they are semantically redundant.
"""

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

import pandas as pd
import numpy as np
import pickle
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
FILES = {
    "K-Means + Qwen3": os.path.join(BASE_DIR, "layer_3_topics_kmeans.csv"),
    "HDBSCAN + MiniLM": os.path.join(BASE_DIR, "layer_3_topics_minilm_v2.csv"),
    "HDBSCAN + Qwen3": os.path.join(BASE_DIR, "layer_3_topics_all.csv"),
}
LEVELS = ["topic_0", "topic_1", "topic_2"]

OUTPUT_CSV = os.path.join(BASE_DIR, "repetitivity_results.csv")
OUTPUT_LATEX = os.path.join(BASE_DIR, "repetitivity_table.tex")

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

    for i in tqdm(range(0, len(topics), batch_size), desc="Embedding topics"):
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

def process_level(df, level, model, tokenizer, client):
    """Process a single level and return repetitivity stats."""
    print(f"\n{'='*60}")
    print(f"Processing {level}")
    print(f"{'='*60}")

    # Get unique topics
    unique_topics = df[level].dropna().unique().tolist()
    n_topics = len(unique_topics)
    print(f"Found {n_topics} unique topics")

    if n_topics < 2:
        print(f"Not enough topics for comparison, skipping...")
        return {"n_topics": n_topics, "n_sampled": 0, "n_redundant": 0, "ratio": 0.0}

    # Embed all topics
    print("Embedding topics with Qwen3...")
    embeddings = embed_topics(unique_topics, model, tokenizer)
    print(f"Embeddings shape: {embeddings.shape}")

    # Sample topics
    n_sample = min(SAMPLE_SIZE, n_topics)
    sampled_indices = random.sample(range(n_topics), n_sample)
    print(f"Sampled {n_sample} topics for evaluation")

    # Find similar topics and judge redundancy
    redundant_count = 0
    results_detail = []

    def evaluate_topic(idx):
        topic = unique_topics[idx]
        top_k_indices, top_k_scores = find_top_k_similar(embeddings, idx, k=TOP_K)
        similar_topics = [unique_topics[i] for i in top_k_indices]
        is_redundant = judge_repetitivity(topic, similar_topics, client)
        return {
            "topic": topic,
            "similar_topics": similar_topics,
            "similarities": top_k_scores.tolist(),
            "is_redundant": is_redundant
        }

    print("Evaluating redundancy with GPT-5.2...")
    with ThreadPoolExecutor(max_workers=16) as executor:
        futures = {executor.submit(evaluate_topic, idx): idx for idx in sampled_indices}
        for future in tqdm(as_completed(futures), total=len(sampled_indices), desc="GPT-5.2 evaluation"):
            result = future.result()
            results_detail.append(result)
            if result["is_redundant"]:
                redundant_count += 1

    ratio = redundant_count / n_sample if n_sample > 0 else 0.0
    print(f"Redundancy ratio: {redundant_count}/{n_sample} = {ratio:.2%}")

    return {
        "n_topics": n_topics,
        "n_sampled": n_sample,
        "n_redundant": redundant_count,
        "ratio": ratio,
        "details": results_detail
    }

def main():
    print("="*60)
    print("Topic Repetitivity Measurement")
    print("="*60)

    # Initialize OpenAI client
    client = OpenAI()

    # Load Qwen3 model
    model, tokenizer = load_qwen3_model()

    # Results storage
    all_results = []

    for setting_name, file_path in FILES.items():
        print(f"\n{'#'*60}")
        print(f"Processing: {setting_name}")
        print(f"File: {file_path}")
        print(f"{'#'*60}")

        # Load data
        df = pd.read_csv(file_path)
        print(f"Loaded {len(df)} records")

        for level in LEVELS:
            result = process_level(df, level, model, tokenizer, client)
            all_results.append({
                "setting": setting_name,
                "level": level,
                "n_topics": result["n_topics"],
                "n_sampled": result["n_sampled"],
                "n_redundant": result["n_redundant"],
                "ratio": result["ratio"]
            })

    # Save results to CSV
    results_df = pd.DataFrame(all_results)
    results_df.to_csv(OUTPUT_CSV, index=False)
    print(f"\nResults saved to: {OUTPUT_CSV}")

    # Display results
    print("\n" + "="*60)
    print("RESULTS SUMMARY")
    print("="*60)

    # Pivot table for display
    pivot_df = results_df.pivot(index='setting', columns='level', values='ratio')
    pivot_df = pivot_df[LEVELS]  # Ensure column order
    print("\nRepetitivity Ratios (proportion judged as redundant):")
    print(pivot_df.to_string())

    # Generate LaTeX table
    latex_content = generate_latex_table(results_df)
    with open(OUTPUT_LATEX, 'w') as f:
        f.write(latex_content)
    print(f"\nLaTeX table saved to: {OUTPUT_LATEX}")
    print("\nLaTeX Table:")
    print(latex_content)

    return results_df

def generate_latex_table(results_df):
    """Generate LaTeX table from results."""
    # Pivot for easier access
    pivot = results_df.pivot(index='setting', columns='level', values='ratio')
    pivot_n = results_df.pivot(index='setting', columns='level', values='n_topics')

    settings = ["K-Means + Qwen3", "HDBSCAN + MiniLM", "HDBSCAN + Qwen3"]

    latex = r"""\begin{table}[htbp]
\centering
\caption{Topic Repetitivity Analysis: Proportion of topics with semantically redundant neighbors}
\label{tab:repetitivity}
\begin{tabular}{lccc}
\toprule
\textbf{Setting} & \textbf{Topic Level 0} & \textbf{Topic Level 1} & \textbf{Topic Level 2} \\
\midrule
"""

    for setting in settings:
        row_values = []
        for level in LEVELS:
            ratio = pivot.loc[setting, level]
            n_topics = int(pivot_n.loc[setting, level])
            row_values.append(f"{ratio:.1%} ({n_topics})")

        latex += f"{setting} & {' & '.join(row_values)} \\\\\n"

    latex += r"""\bottomrule
\end{tabular}
\vspace{2mm}
\footnotesize{Note: Values show the proportion of sampled topics (up to 100) that have at least one semantically redundant neighbor among their top-5 most similar topics, as judged by GPT-5.2. Numbers in parentheses indicate the total unique topics at each level. Lower values indicate better topic distinctiveness.}
\end{table}
"""

    return latex

if __name__ == "__main__":
    main()
