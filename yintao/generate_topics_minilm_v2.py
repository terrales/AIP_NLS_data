"""
Generate 3-layered topics using MiniLM embeddings with HDBSCAN clustering.
Uses cached reduced embeddings from reduced_embeddings_minilm.pkl
Uses MiniLM (sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2) for ALL embedding steps

Pipeline:
1. Load cached reduced embeddings (skip UMAP)
2. Cluster with HDBSCAN to get topic_0 labels
3. Generate topic_0 names with LLM
4. Cluster topic_0 with HDBSCAN using MiniLM embedding
5. Generate topic_1 names with LLM
6. Cluster topic_1 with HDBSCAN using MiniLM embedding
7. Generate topic_2 names with LLM
8. Save all results

Purpose: Compare MiniLM vs Qwen3 embedding models (both using HDBSCAN)
"""

import pandas as pd
import pickle
import numpy as np
import cuml
import torch
from tqdm import tqdm
from openai import OpenAI
from concurrent.futures import ThreadPoolExecutor
from multiprocessing import cpu_count
import os
import requests
import traceback
from datetime import datetime

# Set GPU
os.environ['CUDA_VISIBLE_DEVICES'] = '1'

# Slack Configuration
SLACK_WEBHOOK_URL = 'https://hooks.slack.com/services/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'


def send_slack_notification(message, is_error=False):
    """Send a notification to Slack."""
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    emoji = "🔴" if is_error else "📊"

    payload = {
        'text': f"{emoji} *MiniLM+HDBSCAN Topic Generation* ({timestamp})\n{message}"
    }

    try:
        response = requests.post(SLACK_WEBHOOK_URL, json=payload, timeout=10)
        if response.status_code != 200:
            print(f"Slack notification failed: {response.status_code}")
    except Exception as e:
        print(f"Failed to send Slack notification: {e}")


def log_and_notify(message, is_error=False, slack=True):
    """Print message and optionally send to Slack."""
    print(message)
    if slack:
        send_slack_notification(message, is_error)


# Configuration
REDUCED_EMBEDDINGS_PATH = '/disk/scratch/s1891075/AIP_NLS_data/yintao/reduced_embeddings_minilm.pkl'
RAW_DATA_PATH = '/disk/scratch/s1891075/AIP_NLS_data/yintao/layer_3_topics_all.csv'
OUTPUT_DIR = '/disk/scratch/s1891075/AIP_NLS_data/yintao'

# Output paths
TOPIC_0_PATH = os.path.join(OUTPUT_DIR, 'topic_0_minilm_v2.parquet')
TOPIC_1_PATH = os.path.join(OUTPUT_DIR, 'topic_1_minilm_v2.parquet')
FINAL_OUTPUT_PATH = os.path.join(OUTPUT_DIR, 'layer_3_topics_minilm_v2.csv')

# MiniLM embedding model for ALL embedding steps
TOPIC_EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"


def assign_noise_pytorch(data, labels, noise_batch_size=1024, clean_batch_size=100000, device='cuda'):
    """Assigns noise points (-1) to the closest non-noise cluster using PyTorch."""
    if not isinstance(data, torch.Tensor):
        data = torch.tensor(data, dtype=torch.float32)
    if not isinstance(labels, torch.Tensor):
        labels_tensor = torch.tensor(labels, dtype=torch.long)
    else:
        labels_tensor = labels.clone()

    noise_mask = (labels_tensor == -1)
    if not noise_mask.any():
        return labels if isinstance(labels, np.ndarray) else labels.cpu().numpy()

    noise_indices = torch.nonzero(noise_mask).squeeze()
    clean_indices = torch.nonzero(~noise_mask).squeeze()

    noise_data = data[noise_indices]
    clean_data = data[clean_indices]
    clean_labels = labels_tensor[clean_indices]

    if noise_indices.dim() == 0:
        noise_data = noise_data.unsqueeze(0)
        noise_indices = noise_indices.unsqueeze(0)

    print(f"Assigning {len(noise_data)} noise points to {len(clean_data)} clean points...")
    print(f"Using noise_batch_size={noise_batch_size}, clean_batch_size={clean_batch_size}")

    new_labels = labels_tensor.clone()
    num_noise = noise_data.shape[0]
    num_clean = clean_data.shape[0]

    for i in tqdm(list(range(0, num_noise, noise_batch_size))):
        noise_end = min(i + noise_batch_size, num_noise)
        noise_batch = noise_data[i:noise_end].to(device)
        batch_len = noise_end - i

        min_dists = torch.full((batch_len,), float('inf'), device=device)
        min_indices = torch.zeros(batch_len, dtype=torch.long, device=device)

        for j in range(0, num_clean, clean_batch_size):
            clean_end = min(j + clean_batch_size, num_clean)
            clean_batch = clean_data[j:clean_end].to(device)
            clean_labels_batch = clean_labels[j:clean_end].to(device)

            dists = torch.cdist(noise_batch, clean_batch, p=2)
            batch_min_dists, batch_min_idx = torch.min(dists, dim=1)

            update_mask = batch_min_dists < min_dists
            min_dists[update_mask] = batch_min_dists[update_mask]
            min_indices[update_mask] = clean_labels_batch[batch_min_idx[update_mask]]

            del dists, clean_batch, clean_labels_batch
            torch.cuda.empty_cache()

        original_indices_for_batch = noise_indices[i:noise_end]
        new_labels[original_indices_for_batch] = min_indices.cpu()

        del noise_batch, min_dists, min_indices
        torch.cuda.empty_cache()

    return new_labels.cpu().numpy()


def generate_text(prompt: str, model: str = "gpt-4o") -> str:
    """Send a text prompt to the OpenAI API and return the generated response."""
    client = OpenAI()
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
    )
    return response.choices[0].message.content.strip()


def get_topic_name(topics_list: list) -> str:
    """Generate a topic name from a list of subtopics."""
    PROMPT = """
Please review the list of book topics below and distill them into a single summary topic consisting of just 1-3 words. Your output should be a definitive category or theme that best captures the entire collection.

The List: {list_of_topics}
"""
    prompt = PROMPT.format(list_of_topics='\n'.join(topics_list[:50]))
    return generate_text(prompt)


def cluster_and_name_topics_hdbscan(topic_table, n_neighbors, n_components, min_cluster_size, layer_name):
    """Cluster topics using HDBSCAN with MiniLM embedding and generate names."""
    from sentence_transformers import SentenceTransformer

    print(f"\n{'='*60}")
    print(f"Generating {layer_name} with HDBSCAN (MiniLM embedding)")
    print(f"{'='*60}")

    docs = topic_table['topic'].unique().tolist()
    print(f"Unique topics to cluster: {len(docs)}")

    # Embed topic names using MiniLM (not Qwen3!)
    print(f"Loading MiniLM embedding model ({TOPIC_EMBEDDING_MODEL})...")
    model = SentenceTransformer(TOPIC_EMBEDDING_MODEL)
    print("Embedding topic names with MiniLM...")
    embeddings = model.encode(docs, batch_size=256, show_progress_bar=True)
    print(f"Embeddings shape: {embeddings.shape}")

    # Reduce dimensions with UMAP for clustering
    print(f"Reducing dimensions with UMAP (n_neighbors={n_neighbors}, n_components={n_components})...")
    umap = cuml.manifold.UMAP(
        n_neighbors=n_neighbors,
        n_components=n_components,
        metric='cosine',
        min_dist=0.0,
        spread=1.0,
        random_state=42,
    )
    reduced_data = umap.fit_transform(embeddings)
    print(f"Reduced shape: {reduced_data.shape}")

    # Cluster with HDBSCAN
    print(f"Clustering with HDBSCAN (min_cluster_size={min_cluster_size})...")
    clusterer = cuml.cluster.hdbscan.HDBSCAN(
        min_cluster_size=min_cluster_size,
        metric='euclidean',
        prediction_data=True
    )
    clusterer.fit(reduced_data)

    labels = clusterer.labels_
    n_clusters = labels.max() + 1
    n_noise = (labels == -1).sum()
    print(f"Found {n_clusters} clusters, {n_noise} noise points")

    # Assign noise points
    labels_assigned = assign_noise_pytorch(reduced_data, labels)

    # Create topic -> cluster_id mapping
    topic_labels = [(topic, label) for topic, label in zip(docs, labels_assigned.tolist())]
    cluster_table = pd.DataFrame(topic_labels, columns=['topic', 'cluster_id'])

    # Generate names for each cluster using LLM
    print(f"Generating names for {n_clusters} clusters using GPT-4...")

    def process_cluster(args):
        cluster_id, table = args
        subset = table[table.cluster_id == cluster_id]
        sample_n = min(40, len(subset))
        if sample_n == 0:
            return {'name': 'Miscellaneous', 'cluster_id': cluster_id}
        sample_topics = subset.sample(n=sample_n, replace=True)['topic'].tolist()
        topic_name = get_topic_name(sample_topics)
        return {'name': topic_name, 'cluster_id': cluster_id}

    args_list = [(cid, cluster_table) for cid in cluster_table.cluster_id.unique().tolist()]

    num_threads = min(cpu_count(), 32)
    print(f"Using {num_threads} threads...")

    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        results = list(tqdm(executor.map(process_cluster, args_list), total=len(args_list)))

    # Create mapping
    id2name = {r['cluster_id']: r['name'] for r in results}
    topic_to_new_topic = {}
    for row in cluster_table.to_dict('records'):
        topic_to_new_topic[row['topic']] = id2name[row['cluster_id']]

    return topic_to_new_topic, cluster_table


def main():
    try:
        log_and_notify("🚀 Starting MiniLM + HDBSCAN topic generation (GPU1)")

        print("="*60)
        print("Step 1: Loading cached reduced embeddings and raw data")
        print("="*60)

        # Load raw data
        print(f"Loading raw data from {RAW_DATA_PATH}...")
        data = pd.read_csv(RAW_DATA_PATH, low_memory=False)
        print(f"Data shape: {data.shape}")

        # Load cached reduced embeddings (skip UMAP)
        print(f"Loading cached reduced embeddings from {REDUCED_EMBEDDINGS_PATH}...")
        with open(REDUCED_EMBEDDINGS_PATH, 'rb') as f:
            emb_data = pickle.load(f)

        reduced_embeddings = emb_data['reduced_embeddings']
        indices = emb_data['indices']
        print(f"Reduced embeddings shape: {reduced_embeddings.shape}")
        print(f"Indices: {len(indices)}")

        log_and_notify(f"✅ Loaded data: {data.shape[0]} rows, reduced embeddings: {reduced_embeddings.shape}")

        print("\n" + "="*60)
        print("Step 2: HDBSCAN Clustering for topic_0")
        print("="*60)

        log_and_notify("⏳ Starting HDBSCAN clustering for topic_0...")

        print("Running HDBSCAN clustering...")
        clusterer = cuml.cluster.hdbscan.HDBSCAN(
            min_cluster_size=50,
            metric='euclidean',
            prediction_data=True
        )
        clusterer.fit(reduced_embeddings)

        labels = clusterer.labels_
        n_clusters = labels.max() + 1
        n_noise = (labels == -1).sum()
        print(f"Found {n_clusters} clusters, {n_noise} noise points")

        log_and_notify(f"✅ HDBSCAN found {n_clusters} clusters, {n_noise} noise points")

        # Assign noise points
        print("Assigning noise points to nearest clusters...")
        labels_assigned = assign_noise_pytorch(
            reduced_embeddings, labels,
            noise_batch_size=512, clean_batch_size=50000
        )

        # Add labels to data
        idx_to_label = {idx: label for idx, label in zip(indices, labels_assigned.tolist())}
        data['topic_label_new'] = data.index.map(lambda x: idx_to_label.get(x, -1))

        print(f"Unique topic_0 labels: {data['topic_label_new'].nunique()}")

        # Generate topic_0 names
        print("\nGenerating topic_0 names...")
        log_and_notify(f"⏳ Generating names for {data['topic_label_new'].nunique()} topic_0 clusters...")

        def get_cluster_description(cluster_data):
            texts = []
            sample = cluster_data.sample(n=min(30, len(cluster_data)))
            for _, row in sample.iterrows():
                parts = []
                if pd.notna(row.get('title')):
                    parts.append(str(row['title']))
                if pd.notna(row.get('subject')):
                    parts.append(str(row['subject']))
                if parts:
                    texts.append(' - '.join(parts))
            return texts

        def process_cluster_topic0(args):
            cluster_id, df = args
            cluster_data = df[df['topic_label_new'] == cluster_id]
            if len(cluster_data) == 0:
                return {'name': 'Miscellaneous', 'cluster_id': cluster_id}

            texts = get_cluster_description(cluster_data)
            if not texts:
                return {'name': 'Miscellaneous', 'cluster_id': cluster_id}

            PROMPT = """
You are a semantic summarizer. 
Given a list of book infomation, infer their common theme, topic, or concept. Return a concise English phrase or a word (1–5 words) that best summarizes all of them.

{texts}

Topic name:"""
            prompt = PROMPT.format(texts='\n'.join(texts[:30]))
            try:
                name = generate_text(prompt)
                name = name.strip().strip('"').strip("'")
                if len(name) > 50:
                    name = name[:50]
            except Exception as e:
                print(f"Error generating name for cluster {cluster_id}: {e}")
                name = f"Topic_{cluster_id}"
            return {'name': name, 'cluster_id': cluster_id}

        unique_labels = sorted(data['topic_label_new'].unique().tolist())
        unique_labels = [l for l in unique_labels if l != -1]
        print(f"Generating names for {len(unique_labels)} clusters...")

        args_list = [(cid, data) for cid in unique_labels]
        num_threads = min(cpu_count(), 32)

        with ThreadPoolExecutor(max_workers=num_threads) as executor:
            results = list(tqdm(executor.map(process_cluster_topic0, args_list), total=len(args_list)))

        label_to_topic0 = {r['cluster_id']: r['name'] for r in results}
        data['topic_0'] = data['topic_label_new'].map(label_to_topic0)

        # Save topic_0 checkpoint
        print(f"Saving topic_0 results to {TOPIC_0_PATH}...")
        data.to_parquet(TOPIC_0_PATH)

        print(f"Unique topic_0: {data['topic_0'].nunique()}")
        log_and_notify(f"✅ topic_0 complete! {data['topic_0'].nunique()} unique topics")

        print("\n" + "="*60)
        print("Step 3: HDBSCAN for topic_1 using MiniLM embedding")
        print("="*60)

        log_and_notify("⏳ Starting topic_1 generation (HDBSCAN + MiniLM embedding)...")

        topic_0_table = pd.DataFrame({'topic': data['topic_0'].unique()})
        topic0_to_topic1, _ = cluster_and_name_topics_hdbscan(
            topic_0_table,
            n_neighbors=15,
            n_components=15,
            min_cluster_size=5,
            layer_name="topic_1"
        )

        data['topic_1'] = data['topic_0'].map(topic0_to_topic1)

        print(f"Saving topic_1 results to {TOPIC_1_PATH}...")
        data.to_parquet(TOPIC_1_PATH)

        print(f"Unique topic_1: {data['topic_1'].nunique()}")
        log_and_notify(f"✅ topic_1 complete! {data['topic_1'].nunique()} unique topics")

        print("\n" + "="*60)
        print("Step 4: HDBSCAN for topic_2 using MiniLM embedding")
        print("="*60)

        log_and_notify("⏳ Starting topic_2 generation (HDBSCAN + MiniLM embedding)...")

        topic_1_table = pd.DataFrame({'topic': data['topic_1'].unique()})
        topic1_to_topic2, _ = cluster_and_name_topics_hdbscan(
            topic_1_table,
            n_neighbors=8,
            n_components=8,
            min_cluster_size=4,
            layer_name="topic_2"
        )

        data['topic_2'] = data['topic_1'].map(topic1_to_topic2)

        print(f"Unique topic_2: {data['topic_2'].nunique()}")
        log_and_notify(f"✅ topic_2 complete! {data['topic_2'].nunique()} unique topics")

        print("\n" + "="*60)
        print("Step 5: Saving final results")
        print("="*60)

        print(f"Saving final results to {FINAL_OUTPUT_PATH}...")
        data.to_csv(FINAL_OUTPUT_PATH, index=False)

        summary = f"""🎉 *MiniLM + HDBSCAN Topic Generation Complete!*
• Total records: {len(data)}
• Unique topic_0: {data['topic_0'].nunique()}
• Unique topic_1: {data['topic_1'].nunique()}
• Unique topic_2: {data['topic_2'].nunique()}

*Configuration:*
• Clustering: HDBSCAN
• Base embedding: MiniLM (cached reduced embeddings)
• Topic name embedding: MiniLM ({TOPIC_EMBEDDING_MODEL})

*Output files:*
• Topic_0: {TOPIC_0_PATH}
• Topic_1: {TOPIC_1_PATH}
• Final: {FINAL_OUTPUT_PATH}"""
        print(summary)
        log_and_notify(summary)
        print("\nDone!")

    except Exception as e:
        error_msg = f"❌ Error: {str(e)}\n\nTraceback:\n{traceback.format_exc()}"
        log_and_notify(error_msg, is_error=True)
        raise


if __name__ == "__main__":
    main()
