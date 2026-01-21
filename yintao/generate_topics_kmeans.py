"""
Generate 3-layered topics using K-Means clustering with Qwen3 embeddings.
Uses embeddings from yintao/embeddings/ (Qwen3 8B embeddings)
Requires UMAP dimension reduction (no cached reduced embeddings)

Pipeline:
1. Load Qwen3 embeddings from 4 pickle files
2. Run UMAP dimension reduction
3. Cluster with K-Means (K=6000) to get topic_0 labels
4. Generate topic_0 names with LLM
5. Cluster topic_0 with K-Means (K=200) using Qwen3 embedding
6. Generate topic_1 names with LLM
7. Cluster topic_1 with K-Means (K=10) using Qwen3 embedding
8. Generate topic_2 names with LLM
9. Save all results

Purpose: Compare K-Means vs HDBSCAN (both using Qwen3 embeddings)
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
import glob
import requests
import traceback
from datetime import datetime

# Set GPU
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

# Slack Configuration
SLACK_WEBHOOK_URL = 'https://xxxxx'


def send_slack_notification(message, is_error=False):
    """Send a notification to Slack."""
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    emoji = "🔴" if is_error else "📊"

    payload = {
        'text': f"{emoji} *KMeans+Qwen3 Topic Generation* ({timestamp})\n{message}"
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
EMBEDDINGS_DIR = '/disk/scratch/s1891075/AIP_NLS_data/yintao/embeddings'
RAW_DATA_PATH = '/disk/scratch/s1891075/AIP_NLS_data/yintao/layer_3_topics_all.csv'
OUTPUT_DIR = '/disk/scratch/s1891075/AIP_NLS_data/yintao'

# Output paths
REDUCED_EMBEDDINGS_PATH = os.path.join(OUTPUT_DIR, 'reduced_embeddings_qwen3.pkl')
TOPIC_0_PATH = os.path.join(OUTPUT_DIR, 'topic_0_kmeans.parquet')
TOPIC_1_PATH = os.path.join(OUTPUT_DIR, 'topic_1_kmeans.parquet')
FINAL_OUTPUT_PATH = os.path.join(OUTPUT_DIR, 'layer_3_topics_kmeans.csv')

# KMeans parameters
K_TOPIC_0 = 6000
K_TOPIC_1 = 200
K_TOPIC_2 = 10

# Qwen3 embedding model for topic names
TOPIC_EMBEDDING_MODEL = "Qwen/Qwen3-Embedding-8B"


def load_qwen3_embeddings():
    """Load and flatten Qwen3 embeddings from 4 pickle files."""
    print(f"Loading Qwen3 embeddings from {EMBEDDINGS_DIR}...")

    files = sorted(glob.glob(os.path.join(EMBEDDINGS_DIR, 'embedding_list_*.pkl')))
    print(f"Found {len(files)} embedding files")

    all_embeddings = []
    all_indices = []
    current_idx = 0

    for f in tqdm(files, desc="Loading embedding files"):
        with open(f, 'rb') as fp:
            data = pickle.load(fp)

        # data[0] = metadata batches, data[1] = embedding batches
        for emb_batch in data[1]:
            batch_size = emb_batch.shape[0]
            all_embeddings.append(emb_batch)
            all_indices.extend(range(current_idx, current_idx + batch_size))
            current_idx += batch_size

    print("Concatenating embeddings...")
    embeddings = np.vstack(all_embeddings).astype(np.float32)
    print(f"Total embeddings shape: {embeddings.shape}")

    return embeddings, all_indices


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


def cluster_and_name_topics_kmeans(topic_table, n_clusters, n_neighbors, n_components, layer_name):
    """Cluster topics using K-Means with Qwen3 embedding and generate names."""
    from sentence_transformers import SentenceTransformer

    print(f"\n{'='*60}")
    print(f"Generating {layer_name} with K-Means (K={n_clusters})")
    print(f"{'='*60}")

    docs = topic_table['topic'].unique().tolist()
    print(f"Unique topics to cluster: {len(docs)}")

    # Embed topic names using Qwen3
    print(f"Loading Qwen3 embedding model ({TOPIC_EMBEDDING_MODEL})...")
    model = SentenceTransformer(TOPIC_EMBEDDING_MODEL)
    print("Embedding topic names with Qwen3...")
    embeddings = model.encode(docs, batch_size=64, show_progress_bar=True)
    print(f"Embeddings shape: {embeddings.shape}")

    # Reduce dimensions with UMAP for better clustering
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

    # Cluster with K-Means
    actual_k = min(n_clusters, len(docs))
    print(f"Clustering with K-Means (K={actual_k})...")
    kmeans = cuml.cluster.KMeans(n_clusters=actual_k, random_state=42, max_iter=300)
    labels = kmeans.fit_predict(reduced_data)

    n_clusters_found = len(np.unique(labels))
    print(f"Found {n_clusters_found} clusters")

    # Create topic -> cluster_id mapping
    topic_labels = [(topic, int(label)) for topic, label in zip(docs, labels)]
    cluster_table = pd.DataFrame(topic_labels, columns=['topic', 'cluster_id'])

    # Generate names for each cluster using LLM
    print(f"Generating names for {n_clusters_found} clusters using GPT-4...")

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
        log_and_notify("🚀 Starting K-Means + Qwen3 topic generation (GPU0)")

        print("="*60)
        print("Step 1: Loading raw data and Qwen3 embeddings")
        print("="*60)

        # Load raw data
        print(f"Loading raw data from {RAW_DATA_PATH}...")
        data = pd.read_csv(RAW_DATA_PATH, low_memory=False)
        print(f"Data shape: {data.shape}")

        # Load Qwen3 embeddings
        embeddings, indices = load_qwen3_embeddings()

        log_and_notify(f"✅ Loaded data: {data.shape[0]} rows, embeddings: {embeddings.shape}")

        print("\n" + "="*60)
        print("Step 2: UMAP Dimension Reduction")
        print("="*60)

        log_and_notify("⏳ Starting UMAP dimension reduction...")

        print("Running UMAP (n_neighbors=15, n_components=50)...")
        umap_model = cuml.manifold.UMAP(
            n_neighbors=15,
            n_components=50,
            metric='cosine',
            min_dist=0.0,
            spread=1.0,
            random_state=42,
        )
        reduced_embeddings = umap_model.fit_transform(embeddings)
        print(f"Reduced embeddings shape: {reduced_embeddings.shape}")

        # Save reduced embeddings checkpoint
        print(f"Saving reduced embeddings to {REDUCED_EMBEDDINGS_PATH}...")
        with open(REDUCED_EMBEDDINGS_PATH, 'wb') as f:
            pickle.dump({
                'reduced_embeddings': reduced_embeddings,
                'indices': indices,
                'umap_params': {
                    'n_neighbors': 15,
                    'n_components': 50,
                    'metric': 'cosine',
                }
            }, f)

        log_and_notify(f"✅ UMAP complete! Reduced shape: {reduced_embeddings.shape}")

        # Free memory
        del embeddings
        import gc
        gc.collect()

        print("\n" + "="*60)
        print(f"Step 3: K-Means Clustering (K={K_TOPIC_0}) for topic_0")
        print("="*60)

        log_and_notify(f"⏳ Starting K-Means clustering (K={K_TOPIC_0})...")

        print(f"Running K-Means (K={K_TOPIC_0})...")
        kmeans = cuml.cluster.KMeans(n_clusters=K_TOPIC_0, random_state=42, max_iter=300)
        labels = kmeans.fit_predict(reduced_embeddings)

        n_clusters = len(np.unique(labels))
        print(f"Assigned to {n_clusters} clusters")

        log_and_notify(f"✅ K-Means assigned data to {n_clusters} clusters")

        # Add labels to data
        idx_to_label = {idx: int(label) for idx, label in zip(indices, labels)}
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
        print(f"Step 4: K-Means (K={K_TOPIC_1}) for topic_1 using Qwen3 embedding")
        print("="*60)

        log_and_notify(f"⏳ Starting topic_1 generation (K={K_TOPIC_1}, Qwen3 embedding)...")

        topic_0_table = pd.DataFrame({'topic': data['topic_0'].unique()})
        topic0_to_topic1, _ = cluster_and_name_topics_kmeans(
            topic_0_table,
            n_clusters=K_TOPIC_1,
            n_neighbors=15,
            n_components=15,
            layer_name="topic_1"
        )

        data['topic_1'] = data['topic_0'].map(topic0_to_topic1)

        print(f"Saving topic_1 results to {TOPIC_1_PATH}...")
        data.to_parquet(TOPIC_1_PATH)

        print(f"Unique topic_1: {data['topic_1'].nunique()}")
        log_and_notify(f"✅ topic_1 complete! {data['topic_1'].nunique()} unique topics")

        print("\n" + "="*60)
        print(f"Step 5: K-Means (K={K_TOPIC_2}) for topic_2 using Qwen3 embedding")
        print("="*60)

        log_and_notify(f"⏳ Starting topic_2 generation (K={K_TOPIC_2}, Qwen3 embedding)...")

        topic_1_table = pd.DataFrame({'topic': data['topic_1'].unique()})
        topic1_to_topic2, _ = cluster_and_name_topics_kmeans(
            topic_1_table,
            n_clusters=K_TOPIC_2,
            n_neighbors=8,
            n_components=8,
            layer_name="topic_2"
        )

        data['topic_2'] = data['topic_1'].map(topic1_to_topic2)

        print(f"Unique topic_2: {data['topic_2'].nunique()}")
        log_and_notify(f"✅ topic_2 complete! {data['topic_2'].nunique()} unique topics")

        print("\n" + "="*60)
        print("Step 6: Saving final results")
        print("="*60)

        print(f"Saving final results to {FINAL_OUTPUT_PATH}...")
        data.to_csv(FINAL_OUTPUT_PATH, index=False)

        summary = f"""🎉 *K-Means + Qwen3 Topic Generation Complete!*
• Total records: {len(data)}
• Unique topic_0: {data['topic_0'].nunique()} (K={K_TOPIC_0})
• Unique topic_1: {data['topic_1'].nunique()} (K={K_TOPIC_1})
• Unique topic_2: {data['topic_2'].nunique()} (K={K_TOPIC_2})

*Configuration:*
• Clustering: K-Means
• Embedding model: Qwen3-Embedding-8B
• Topic name embedding: Qwen3-Embedding-8B

*Output files:*
• Reduced embeddings: {REDUCED_EMBEDDINGS_PATH}
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
