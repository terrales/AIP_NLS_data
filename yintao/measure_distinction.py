#!/usr/bin/env python3
"""
Measure distinction of topic modeling by finding similar topics
and using GPT-5.2 to score the distinction level (1-5).
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
import re
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

OUTPUT_CSV = os.path.join(BASE_DIR, "distinction_results.csv")
OUTPUT_LATEX = os.path.join(BASE_DIR, "distinction_table.tex")

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

def score_distinction(topic, similar_topics, client):
    """Use GPT-5.2 to score the distinction level of the main topic (1-5)."""
    similar_list = "\n".join([f"{i+1}. {t}" for i, t in enumerate(similar_topics)])

    prompt = f"""You are evaluating topic names from a library catalogue classification system.

Given the main topic and its 5 most similar topics (by embedding similarity), score how distinctive the main topic is compared to its most similar neighbors.

Main Topic: "{topic}"

Most Similar Topics:
{similar_list}

Score the distinction level of the main topic on a scale of 1-5:
- 1: Identical concept - the main topic is essentially the same as at least one similar topic (should be merged)
- 2: Very similar - the main topic overlaps significantly with at least one similar topic
- 3: Moderately distinct - the main topic has some overlap but represents a different aspect
- 4: Quite distinct - the main topic is clearly different from all similar topics
- 5: Very distinctive - the main topic is completely unique and has no semantic overlap with any similar topic

Answer with ONLY a single number (1, 2, 3, 4, or 5)."""

    try:
        response = client.chat.completions.create(
            model=GPT_MODEL,
            messages=[{"role": "user", "content": prompt}],
            max_completion_tokens=10,
            temperature=0
        )
        answer = response.choices[0].message.content.strip()
        # Extract the number from the response
        match = re.search(r'[1-5]', answer)
        if match:
            return int(match.group())
        else:
            print(f"Could not parse score from: {answer}")
            return None
    except Exception as e:
        print(f"Error calling GPT: {e}")
        return None

def process_level(df, level, model, tokenizer, client):
    """Process a single level and return distinction stats."""
    print(f"\n{'='*60}")
    print(f"Processing {level}")
    print(f"{'='*60}")

    # Get unique topics
    unique_topics = df[level].dropna().unique().tolist()
    n_topics = len(unique_topics)
    print(f"Found {n_topics} unique topics")

    if n_topics < 2:
        print(f"Not enough topics for comparison, skipping...")
        return {"n_topics": n_topics, "n_sampled": 0, "avg_score": None, "scores": []}

    # Embed all topics
    print("Embedding topics with Qwen3...")
    embeddings = embed_topics(unique_topics, model, tokenizer)
    print(f"Embeddings shape: {embeddings.shape}")

    # Sample topics
    n_sample = min(SAMPLE_SIZE, n_topics)
    sampled_indices = random.sample(range(n_topics), n_sample)
    print(f"Sampled {n_sample} topics for evaluation")

    # Find similar topics and score distinction
    scores = []
    results_detail = []

    def evaluate_topic(idx):
        topic = unique_topics[idx]
        top_k_indices, top_k_scores = find_top_k_similar(embeddings, idx, k=TOP_K)
        similar_topics = [unique_topics[i] for i in top_k_indices]
        distinction_score = score_distinction(topic, similar_topics, client)
        return {
            "topic": topic,
            "similar_topics": similar_topics,
            "similarities": top_k_scores.tolist(),
            "distinction_score": distinction_score
        }

    print("Evaluating distinction with GPT-5.2...")
    with ThreadPoolExecutor(max_workers=16) as executor:
        futures = {executor.submit(evaluate_topic, idx): idx for idx in sampled_indices}
        for future in tqdm(as_completed(futures), total=len(sampled_indices), desc="GPT-5.2 evaluation"):
            result = future.result()
            results_detail.append(result)
            if result["distinction_score"] is not None:
                scores.append(result["distinction_score"])

    avg_score = np.mean(scores) if scores else None
    print(f"Average distinction score: {avg_score:.2f} (from {len(scores)} valid scores)")

    return {
        "n_topics": n_topics,
        "n_sampled": n_sample,
        "n_valid": len(scores),
        "avg_score": avg_score,
        "scores": scores,
        "details": results_detail
    }

def process_subject_field(file_path, model, tokenizer, client):
    """Process the subject field from the original file."""
    print(f"\n{'='*60}")
    print(f"Processing Original Subject Field")
    print(f"{'='*60}")

    # Load data
    df = pd.read_csv(file_path, low_memory=False)

    # Get unique subjects
    unique_subjects = df['subject'].dropna().unique().tolist()
    n_subjects = len(unique_subjects)
    print(f"Found {n_subjects} unique subjects")

    # Embed all subjects
    print("Embedding subjects with Qwen3...")
    embeddings = embed_topics(unique_subjects, model, tokenizer)
    print(f"Embeddings shape: {embeddings.shape}")

    # Sample subjects
    n_sample = min(SAMPLE_SIZE, n_subjects)
    sampled_indices = random.sample(range(n_subjects), n_sample)
    print(f"Sampled {n_sample} subjects for evaluation")

    # Find similar subjects and score distinction
    scores = []
    results_detail = []

    def evaluate_subject(idx):
        subject = unique_subjects[idx]
        top_k_indices, top_k_scores = find_top_k_similar(embeddings, idx, k=TOP_K)
        similar_subjects = [unique_subjects[i] for i in top_k_indices]
        distinction_score = score_distinction(subject, similar_subjects, client)
        return {
            "subject": subject,
            "similar_subjects": similar_subjects,
            "similarities": top_k_scores.tolist(),
            "distinction_score": distinction_score
        }

    print("Evaluating distinction with GPT-5.2...")
    with ThreadPoolExecutor(max_workers=16) as executor:
        futures = {executor.submit(evaluate_subject, idx): idx for idx in sampled_indices}
        for future in tqdm(as_completed(futures), total=len(sampled_indices), desc="GPT-5.2 evaluation"):
            result = future.result()
            results_detail.append(result)
            if result["distinction_score"] is not None:
                scores.append(result["distinction_score"])

    avg_score = np.mean(scores) if scores else None
    print(f"Average distinction score: {avg_score:.2f} (from {len(scores)} valid scores)")

    return {
        "n_topics": n_subjects,
        "n_sampled": n_sample,
        "n_valid": len(scores),
        "avg_score": avg_score,
        "scores": scores,
        "details": results_detail
    }

def main():
    print("="*60)
    print("Topic Distinction Measurement (Score 1-5)")
    print("="*60)

    # Initialize OpenAI client
    client = OpenAI()

    # Load Qwen3 model
    model, tokenizer = load_qwen3_model()

    # Results storage
    all_results = []

    # Process topic files
    for setting_name, file_path in FILES.items():
        print(f"\n{'#'*60}")
        print(f"Processing: {setting_name}")
        print(f"File: {file_path}")
        print(f"{'#'*60}")

        # Load data
        df = pd.read_csv(file_path, low_memory=False)
        print(f"Loaded {len(df)} records")

        for level in LEVELS:
            result = process_level(df, level, model, tokenizer, client)
            all_results.append({
                "setting": setting_name,
                "level": level,
                "n_topics": result["n_topics"],
                "n_sampled": result["n_sampled"],
                "n_valid": result["n_valid"],
                "avg_score": result["avg_score"]
            })

    # Process Original Subject field
    print(f"\n{'#'*60}")
    print(f"Processing: Original Subject")
    print(f"{'#'*60}")

    subject_result = process_subject_field(
        os.path.join(BASE_DIR, "layer_3_topics_all.csv"),
        model, tokenizer, client
    )
    all_results.append({
        "setting": "Original Subject",
        "level": "subject",
        "n_topics": subject_result["n_topics"],
        "n_sampled": subject_result["n_sampled"],
        "n_valid": subject_result["n_valid"],
        "avg_score": subject_result["avg_score"]
    })

    # Save results to CSV
    results_df = pd.DataFrame(all_results)
    results_df.to_csv(OUTPUT_CSV, index=False)
    print(f"\nResults saved to: {OUTPUT_CSV}")

    # Display results
    print("\n" + "="*60)
    print("RESULTS SUMMARY")
    print("="*60)
    print("\nAverage Distinction Scores (1=identical, 5=very distinctive):")
    print(results_df.to_string())

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
    # Separate subject and topic results
    topic_results = results_df[results_df['level'] != 'subject']
    subject_result = results_df[results_df['level'] == 'subject']

    # Pivot for easier access
    pivot = topic_results.pivot(index='setting', columns='level', values='avg_score')
    pivot_n = topic_results.pivot(index='setting', columns='level', values='n_topics')

    settings = ["K-Means + Qwen3", "HDBSCAN + MiniLM", "HDBSCAN + Qwen3"]

    latex = r"""\begin{table}[htbp]
\centering
\caption{Topic Distinction Analysis: Average distinction score (1=identical, 5=very distinctive)}
\label{tab:distinction}
\begin{tabular}{lccc}
\toprule
\textbf{Setting} & \textbf{Topic Level 0} & \textbf{Topic Level 1} & \textbf{Topic Level 2} \\
\midrule
"""

    # Add Original Subject row
    if not subject_result.empty:
        subj_score = subject_result['avg_score'].values[0]
        subj_n = int(subject_result['n_topics'].values[0])
        latex += f"Original Subject & {subj_score:.2f} ({subj_n:,}) & -- & -- \\\\\n"
        latex += r"\midrule" + "\n"

    for setting in settings:
        row_values = []
        for level in LEVELS:
            score = pivot.loc[setting, level]
            n_topics = int(pivot_n.loc[setting, level])
            if pd.notna(score):
                row_values.append(f"{score:.2f} ({n_topics:,})")
            else:
                row_values.append("--")

        latex += f"{setting} & {' & '.join(row_values)} \\\\\n"

    latex += r"""\bottomrule
\end{tabular}
\vspace{2mm}
\footnotesize{Note: Values show the average distinction score (1-5) of sampled topics (up to 100) compared to their top-5 most similar topics, as judged by GPT-5.2. Score meanings: 1=identical concept, 2=very similar, 3=moderately distinct, 4=quite distinct, 5=very distinctive. Numbers in parentheses indicate the total unique topics at each level. Higher scores indicate better topic distinctiveness.}
\end{table}
"""

    return latex

if __name__ == "__main__":
    main()
