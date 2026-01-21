"""
Calculate Silhouette Scores for different topic modeling settings.
- Qwen3+HDBSCAN: embeddings from embeddings/ folder, topics from layer_3_topics_all.csv
- MiniLM+HDBSCAN: embeddings from embeddings_minilm/, topics from layer_3_topics_minilm_v2.csv
- Qwen3+KMeans: embeddings from embeddings/ folder, topics from layer_3_topics_kmeans.csv
"""

import numpy as np
import pandas as pd
import pickle
from sklearn.metrics import silhouette_score, davies_bouldin_score
from pathlib import Path
import warnings
import gc
warnings.filterwarnings('ignore')

# Configuration
SAMPLE_SIZE = 50000  # Sample size for silhouette calculation (full data is too large)
RANDOM_SEED = 42

def load_qwen3_embeddings_sampled(embeddings_dir, sample_indices):
    """Load Qwen3 embeddings only for sampled indices to save memory."""
    print("Loading Qwen3 embeddings for sampled indices...")

    # First pass: determine which file each index belongs to
    file_boundaries = []
    total = 0
    for i in range(4):
        filepath = embeddings_dir / f"embedding_list_{i}.pkl"
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
            embeddings_batches = data[1]
            file_count = sum(e.shape[0] for e in embeddings_batches)
            file_boundaries.append((total, total + file_count))
            total += file_count
        del data
        gc.collect()

    print(f"  Total embeddings across files: {total}")

    # Map sample indices to file indices
    sample_embeddings = []
    sample_indices_set = set(sample_indices)

    for file_idx in range(4):
        start, end = file_boundaries[file_idx]
        # Find which sample indices are in this file
        local_indices = [(idx, idx - start) for idx in sample_indices if start <= idx < end]

        if not local_indices:
            continue

        print(f"  Loading file {file_idx} for {len(local_indices)} samples...")
        filepath = embeddings_dir / f"embedding_list_{file_idx}.pkl"
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
            embeddings_batches = data[1]

            # Flatten embeddings
            flat_embeddings = np.vstack(embeddings_batches)

            # Extract needed embeddings
            for global_idx, local_idx in local_indices:
                sample_embeddings.append((global_idx, flat_embeddings[local_idx]))

        del data, flat_embeddings
        gc.collect()

    # Sort by original indices and extract embeddings
    sample_embeddings.sort(key=lambda x: x[0])
    embeddings = np.array([e[1] for e in sample_embeddings])
    print(f"  Loaded {len(embeddings)} embeddings with dimension {embeddings.shape[1]}")
    return embeddings

def load_qwen3_embeddings_full(embeddings_dir):
    """Load all Qwen3 embeddings (memory intensive)."""
    print("Loading all Qwen3 embeddings...")
    all_embeddings = []

    for i in range(4):
        filepath = embeddings_dir / f"embedding_list_{i}.pkl"
        print(f"  Loading {filepath.name}...")
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
            embeddings_batches = data[1]
            # Flatten batches
            for batch in embeddings_batches:
                all_embeddings.append(batch)
        del data
        gc.collect()

    embeddings = np.vstack(all_embeddings)
    print(f"  Loaded {len(embeddings)} embeddings with dimension {embeddings.shape[1]}")
    del all_embeddings
    gc.collect()
    return embeddings

def load_minilm_embeddings(embeddings_path):
    """Load MiniLM embeddings from pickle file."""
    print("Loading MiniLM embeddings...")
    with open(embeddings_path, 'rb') as f:
        data = pickle.load(f)
    embeddings = data['embeddings']
    print(f"  Loaded {len(embeddings)} embeddings with dimension {embeddings.shape[1]}")
    return embeddings

def load_topic_labels(csv_path, column='topic_0'):
    """Load topic labels from CSV file."""
    print(f"Loading topic labels from {csv_path.name}...")
    df = pd.read_csv(csv_path)
    labels = df[column].values
    n_clusters = len(np.unique(labels))
    print(f"  Loaded {len(labels)} labels with {n_clusters} unique clusters")
    return labels

def calculate_metrics_sampled(embeddings, labels, sample_size=SAMPLE_SIZE, seed=RANDOM_SEED):
    """Calculate silhouette score and Davies-Bouldin index using sampling."""
    np.random.seed(seed)

    # Filter out noise points (label == -1 for HDBSCAN)
    valid_mask = labels != -1
    valid_indices = np.where(valid_mask)[0]

    n_valid = len(valid_indices)
    print(f"  Valid samples (non-noise): {n_valid}")

    if n_valid < sample_size:
        sample_indices = valid_indices
    else:
        sample_indices = np.random.choice(valid_indices, size=sample_size, replace=False)

    sample_embeddings = embeddings[sample_indices]
    sample_labels = labels[sample_indices]

    # Check we have at least 2 clusters
    unique_labels = np.unique(sample_labels)
    if len(unique_labels) < 2:
        print("  Warning: Less than 2 clusters found in sample!")
        return None, None

    print(f"  Calculating metrics on {len(sample_indices)} samples ({len(unique_labels)} clusters in sample)...")

    silhouette = silhouette_score(sample_embeddings, sample_labels)
    print(f"  Silhouette Score: {silhouette:.4f}")

    db_index = davies_bouldin_score(sample_embeddings, sample_labels)
    print(f"  Davies-Bouldin Index: {db_index:.4f}")

    return silhouette, db_index

def main():
    base_dir = Path("/disk/scratch/s1891075/AIP_NLS_data/yintao")

    results = []
    np.random.seed(RANDOM_SEED)

    # Load all topic labels first (small files)
    print("Loading topic labels...")
    qwen3_hdbscan_labels = load_topic_labels(base_dir / "layer_3_topics_all.csv")
    minilm_hdbscan_labels = load_topic_labels(base_dir / "layer_3_topics_minilm_v2.csv")
    qwen3_kmeans_labels = load_topic_labels(base_dir / "layer_3_topics_kmeans.csv")

    # 1. Qwen3+HDBSCAN
    print("\n" + "="*60)
    print("1. Qwen3+HDBSCAN")
    print("="*60)

    # Load full Qwen3 embeddings (will reuse for KMeans)
    qwen3_embeddings = load_qwen3_embeddings_full(base_dir / "embeddings")

    silhouette_qwen3_hdbscan, db_qwen3_hdbscan = calculate_metrics_sampled(qwen3_embeddings, qwen3_hdbscan_labels)
    results.append({
        'Method': 'Qwen3+HDBSCAN',
        'Embedding': 'Qwen3-Embedding',
        'Clustering': 'HDBSCAN',
        'Silhouette Score': silhouette_qwen3_hdbscan,
        'Davies-Bouldin Index': db_qwen3_hdbscan
    })

    # 2. Qwen3+KMeans (use same embeddings)
    print("\n" + "="*60)
    print("2. Qwen3+KMeans")
    print("="*60)

    silhouette_qwen3_kmeans, db_qwen3_kmeans = calculate_metrics_sampled(qwen3_embeddings, qwen3_kmeans_labels)
    results.append({
        'Method': 'Qwen3+KMeans',
        'Embedding': 'Qwen3-Embedding',
        'Clustering': 'KMeans',
        'Silhouette Score': silhouette_qwen3_kmeans,
        'Davies-Bouldin Index': db_qwen3_kmeans
    })

    # Free Qwen3 embeddings memory before loading MiniLM
    del qwen3_embeddings
    gc.collect()

    # 3. MiniLM+HDBSCAN
    print("\n" + "="*60)
    print("3. MiniLM+HDBSCAN")
    print("="*60)
    minilm_embeddings = load_minilm_embeddings(base_dir / "embeddings_minilm" / "embeddings_minilm.pkl")

    silhouette_minilm_hdbscan, db_minilm_hdbscan = calculate_metrics_sampled(minilm_embeddings, minilm_hdbscan_labels)
    results.append({
        'Method': 'MiniLM+HDBSCAN',
        'Embedding': 'MiniLM',
        'Clustering': 'HDBSCAN',
        'Silhouette Score': silhouette_minilm_hdbscan,
        'Davies-Bouldin Index': db_minilm_hdbscan
    })

    # Free MiniLM embeddings memory
    del minilm_embeddings
    gc.collect()

    # 4. Original Subject Field (with Qwen3 embeddings)
    print("\n" + "="*60)
    print("4. Original Subject Field")
    print("="*60)

    # Load subject labels and encode as numeric
    print("Loading subject labels...")
    df = pd.read_csv(base_dir / "layer_3_topics_all.csv", low_memory=False)
    subject_mask = df['subject'].notna()
    valid_indices = np.where(subject_mask)[0]
    print(f"  Valid subjects: {len(valid_indices)} / {len(df)}")

    # Encode subject strings as numeric labels
    from sklearn.preprocessing import LabelEncoder
    le = LabelEncoder()
    subject_labels_valid = le.fit_transform(df.loc[subject_mask, 'subject'].values)
    print(f"  Unique subjects: {len(le.classes_)}")

    # Reload Qwen3 embeddings for subject calculation
    qwen3_embeddings = load_qwen3_embeddings_full(base_dir / "embeddings")

    # Extract only embeddings for valid subject indices
    qwen3_embeddings_valid = qwen3_embeddings[valid_indices]
    del qwen3_embeddings
    gc.collect()

    silhouette_subject, db_subject = calculate_metrics_sampled(qwen3_embeddings_valid, subject_labels_valid)
    results.append({
        'Method': 'Original Subject',
        'Embedding': 'Qwen3-Embedding',
        'Clustering': 'Human-Curated',
        'Silhouette Score': silhouette_subject,
        'Davies-Bouldin Index': db_subject
    })

    del qwen3_embeddings_valid
    gc.collect()

    # Create results DataFrame
    df_results = pd.DataFrame(results)

    # Save to CSV
    csv_path = base_dir / "silhouette_scores.csv"
    df_results.to_csv(csv_path, index=False)
    print(f"\n\nResults saved to {csv_path}")

    # Generate LaTeX table
    latex_table = r"""\begin{table}[h]
\centering
\begin{tabular}{llccc}
\toprule
\textbf{Method} & \textbf{Embedding} & \textbf{Clustering} & \textbf{Silhouette Score} & \textbf{Davies-Bouldin Index} \\
\midrule
"""
    for _, row in df_results.iterrows():
        sil_str = f"{row['Silhouette Score']:.4f}" if row['Silhouette Score'] is not None else "N/A"
        db_str = f"{row['Davies-Bouldin Index']:.4f}" if row['Davies-Bouldin Index'] is not None else "N/A"
        latex_table += f"{row['Method']} & {row['Embedding']} & {row['Clustering']} & {sil_str} & {db_str} \\\\\n"

    latex_table += r"""\bottomrule
\end{tabular}
\caption{Clustering quality metrics for different topic modeling methods using Layer 0 topic clusters. Calculated on a random sample of 50,000 data points. Lower Davies-Bouldin Index indicates better clustering.}
\end{table}"""

    # Save LaTeX table
    latex_path = base_dir / "silhouette_scores.tex"
    with open(latex_path, 'w') as f:
        f.write(latex_table)
    print(f"LaTeX table saved to {latex_path}")

    # Print summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(df_results.to_string(index=False))
    print("\n" + latex_table)

if __name__ == "__main__":
    main()
