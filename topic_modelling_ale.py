import pickle
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import umap.umap_ as umap
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import warnings
warnings.filterwarnings('ignore')

# ===========================
# STEP 1: LOAD YOUR DATA
# ===========================

# Load the main dataframe
df = pd.read_csv('../nls-catalogue-published-material/data_cleaned.csv')  # or pd.read_pickle if pickle
print(f"Loaded {len(df)} records")
print(f"Columns: {df.columns.tolist()}")

# Load the embeddings
with open('../nls-catalogue-published-material/subject_embedding.pickle', 'rb') as f:
    subject_embeddings = pickle.load(f)  # Dict: {subject_value: embedding_vector}

with open('../nls-catalogue-published-material/type_embedding.pickle', 'rb') as f:
    type_embeddings = pickle.load(f)  # Dict: {type_value: embedding_vector}

print(f"\nSubject embeddings: {len(subject_embeddings)} unique values")
print(f"Type embeddings: {len(type_embeddings)} unique values")

# ===========================
# STEP 2: CREATE EFFICIENT EMBEDDING MATRICES
# ===========================

def create_embedding_matrix(df, embeddings_dict, column_name):
    """
    Create embedding matrix without storing in DataFrame
    Returns: embedding matrix, valid indices, and unique value mapping
    """
    # Get unique values and their indices
    unique_values = df[column_name].unique()
    
    # Create embedding matrix only for unique values
    embedding_list = []
    valid_values = []
    
    for val in unique_values:
        if val in embeddings_dict:
            embedding_list.append(embeddings_dict[val])
            valid_values.append(val)
    
    # Convert to numpy array
    embedding_matrix = np.array(embedding_list)
    
    # Create value to index mapping
    value_to_idx = {val: idx for idx, val in enumerate(valid_values)}
    
    # Create index array for the full dataframe
    df_indices = df[column_name].map(value_to_idx).values
    valid_mask = ~pd.isna(df_indices)
    df_indices = df_indices[valid_mask].astype(int)
    
    print(f"{column_name}: {len(embedding_list)} unique embeddings, shape {embedding_matrix.shape}")
    print(f"  Valid rows: {valid_mask.sum()}/{len(df)}")
    
    return embedding_matrix, df_indices, valid_mask, value_to_idx

# Create embedding matrices (memory efficient)
subject_matrix, subject_indices, subject_mask, subject_map = \
    create_embedding_matrix(df, subject_embeddings, 'subject')

type_matrix, type_indices, type_mask, type_map = \
    create_embedding_matrix(df, type_embeddings, 'type')

# Filter dataframe to only rows with valid embeddings
valid_mask = subject_mask & type_mask
df = df[valid_mask].reset_index(drop=True)

# Update indices after filtering
subject_indices = df['subject'].map(subject_map).values
type_indices = df['type'].map(type_map).values

print(f"\nDataframe after filtering: {len(df)} records")
print(f"Memory saved: not storing {len(df) * subject_matrix.shape[1] * 8 / 1e9:.2f} GB in DataFrame")

# ===========================
# STEP 3: ANALYZE SUBJECTS
# ===========================

print("\n" + "="*50)
print("ANALYZING SUBJECT EMBEDDINGS")
print("="*50)

# Use the pre-computed unique embedding matrix
# Map each dataframe row to its embedding via indices
X_subjects = subject_matrix[subject_indices]
print(f"Subject embedding shape: {X_subjects.shape}")
print(f"Using {len(subject_matrix)} unique embeddings for {len(df)} rows")

# Reduce dimensions for clustering (optional but speeds things up)
print("\nReducing dimensions with PCA...")
pca = PCA(n_components=min(500, X_subjects.shape[0]))
X_subjects_pca = pca.fit_transform(X_subjects)
print(f"Explained variance: {pca.explained_variance_ratio_.sum():.2%}")

# Find optimal number of clusters using elbow method
""""
print("\nFinding optimal number of clusters...")
inertias = []
K_range = range(5, 105, 10)
for k in K_range:
    print(f"  Testing k={k}...")
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(X_subjects_pca)
    inertias.append(kmeans.inertia_)

plt.figure(figsize=(10, 6))
plt.plot(K_range, inertias, 'bo-')
plt.xlabel('Number of Clusters (k)')
plt.ylabel('Inertia')
plt.title('Elbow Method - Subject Clusters')
plt.grid(True)
plt.savefig('subject_elbow_curve.png', dpi=300, bbox_inches='tight')
plt.show()
"""
# Cluster the subjects (adjust n_clusters based on elbow curve)
n_subject_clusters = 25  # ADJUST THIS based on elbow curve
print(f"\nClustering into {n_subject_clusters} clusters...")
kmeans = KMeans(n_clusters=n_subject_clusters, random_state=42, n_init=10)
df['subject_cluster'] = kmeans.fit_predict(X_subjects_pca)

# Reduce to 2D for visualization
"""
print("Reducing to 2D with UMAP...")
reducer = umap.UMAP(n_components=2, n_neighbors=15, min_dist=0.1, 
                    metric='cosine', random_state=42)
X_subjects_2d = reducer.fit_transform(X_subjects)
df['subject_x'] = X_subjects_2d[:, 0]
df['subject_y'] = X_subjects_2d[:, 1]
"""
# Analyze topics in each cluster
print("\nTop subjects per cluster:")
for cluster_id in range(n_subject_clusters):
    cluster_df = df[df['subject_cluster'] == cluster_id]
    top_subjects = cluster_df['subject'].value_counts().head(5)
    print(f"\nCluster {cluster_id} (n={len(cluster_df)}):")
    for subj, count in top_subjects.items():
        print(f"  {subj}: {count}")
"""
# Visualize subject clusters
plt.figure(figsize=(14, 10))
scatter = plt.scatter(df['subject_x'], df['subject_y'], 
                     c=df['subject_cluster'], cmap='tab20', 
                     alpha=0.6, s=30)
plt.colorbar(scatter, label='Cluster')
plt.title('Subject Embeddings - Cluster Visualization')
plt.xlabel('UMAP 1')
plt.ylabel('UMAP 2')
plt.savefig('subject_clusters.png', dpi=300, bbox_inches='tight')
plt.show()
"""

# ===========================
# STEP 4: ANALYZE TYPES
# ===========================

print("\n" + "="*50)
print("ANALYZING TYPE EMBEDDINGS")
print("="*50)

# Use the pre-computed unique embedding matrix
X_types = type_matrix[type_indices]
print(f"Type embedding shape: {X_types.shape}")
print(f"Using {len(type_matrix)} unique embeddings for {len(df)} rows")

# Cluster types (usually fewer clusters for types)
n_type_clusters = 8  # ADJUST THIS
print(f"\nClustering into {n_type_clusters} clusters...")
kmeans_type = KMeans(n_clusters=n_type_clusters, random_state=42, n_init=10)
df['type_cluster'] = kmeans_type.fit_predict(X_types)

# Reduce to 2D
print("Reducing to 2D with UMAP...")
reducer_type = umap.UMAP(n_components=2, n_neighbors=15, min_dist=0.1,
                        metric='cosine', random_state=42)
X_types_2d = reducer_type.fit_transform(X_types)
df['type_x'] = X_types_2d[:, 0]
df['type_y'] = X_types_2d[:, 1]

# Analyze type clusters
print("\nTypes per cluster:")
for cluster_id in range(n_type_clusters):
    cluster_df = df[df['type_cluster'] == cluster_id]
    top_types = cluster_df['type'].value_counts().head(5)
    print(f"\nCluster {cluster_id} (n={len(cluster_df)}):")
    for typ, count in top_types.items():
        print(f"  {typ}: {count}")

# Visualize type clusters
plt.figure(figsize=(14, 10))
scatter = plt.scatter(df['type_x'], df['type_y'], 
                     c=df['type_cluster'], cmap='tab10', 
                     alpha=0.6, s=30)
plt.colorbar(scatter, label='Cluster')
plt.title('Type Embeddings - Cluster Visualization')
plt.xlabel('UMAP 1')
plt.ylabel('UMAP 2')
plt.savefig('type_clusters.png', dpi=300, bbox_inches='tight')
plt.show()

# ===========================
# STEP 5: TEMPORAL ANALYSIS
# ===========================

print("\n" + "="*50)
print("TEMPORAL ANALYSIS")
print("="*50)

# Convert date column to datetime (adjust column name as needed)
date_column = 'date'  # CHANGE THIS to your actual date column name
df[date_column] = pd.to_datetime(df[date_column], errors='coerce')
df = df.dropna(subset=[date_column])
df['year'] = df[date_column].dt.year

print(f"Date range: {df['year'].min()} to {df['year'].max()}")

# Cluster distribution over time
temporal_dist = df.groupby(['year', 'subject_cluster']).size().unstack(fill_value=0)

plt.figure(figsize=(16, 8))
temporal_dist.plot(kind='area', stacked=True, alpha=0.7, 
                   colormap='tab20', ax=plt.gca())
plt.xlabel('Year')
plt.ylabel('Number of Records')
plt.title('Subject Topic Evolution Over Time')
plt.legend(title='Cluster', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.savefig('temporal_evolution.png', dpi=300, bbox_inches='tight')
plt.show()

# Show top growing and declining topics
recent_years = df[df['year'] >= df['year'].max() - 10]
old_years = df[df['year'] <= df['year'].min() + 10]

recent_dist = recent_years['subject_cluster'].value_counts()
old_dist = old_years['subject_cluster'].value_counts()

print("\nEmerging topics (most common in recent decade):")
for cluster in recent_dist.head(5).index:
    top_subj = recent_years[recent_years['subject_cluster'] == cluster]['subject'].value_counts().head(3)
    print(f"\nCluster {cluster}:")
    print(top_subj.to_string())

# ===========================
# STEP 6: LANGUAGE ANALYSIS
# ===========================

print("\n" + "="*50)
print("LANGUAGE ANALYSIS")
print("="*50)

language_column = 'language'  # CHANGE THIS to your actual language column
top_languages = df[language_column].value_counts().head(5)
print(f"\nTop 5 languages:")
print(top_languages)

# Plot cluster distribution by language
fig, axes = plt.subplots(2, 3, figsize=(18, 12))
axes = axes.flatten()

for idx, (lang, count) in enumerate(top_languages.items()):
    if idx >= 6:
        break
    
    lang_df = df[df[language_column] == lang]
    ax = axes[idx]
    
    scatter = ax.scatter(lang_df['subject_x'], lang_df['subject_y'],
                        c=lang_df['subject_cluster'], cmap='tab20',
                        alpha=0.6, s=20)
    ax.set_title(f'{lang} (n={count:,})')
    ax.set_xlabel('UMAP 1')
    ax.set_ylabel('UMAP 2')

# Remove empty subplots
for idx in range(len(top_languages), 6):
    fig.delaxes(axes[idx])

plt.tight_layout()
plt.savefig('language_comparison.png', dpi=300, bbox_inches='tight')
plt.show()

# Language-specific topic analysis
print("\nCluster distribution by language:")
lang_cluster_dist = pd.crosstab(df[language_column], df['subject_cluster'], 
                                normalize='index') * 100

# Show top 3 clusters per language
for lang in top_languages.head(5).index:
    top_clusters = lang_cluster_dist.loc[lang].nlargest(3)
    print(f"\n{lang} - Top clusters:")
    for cluster, pct in top_clusters.items():
        print(f"  Cluster {cluster}: {pct:.1f}%")

# Heatmap of language vs cluster
plt.figure(figsize=(14, 8))
sns.heatmap(lang_cluster_dist.loc[top_languages.head(10).index], 
            cmap='YlOrRd', annot=False, fmt='.1f', cbar_kws={'label': 'Percentage'})
plt.title('Subject Cluster Distribution by Language (%)')
plt.xlabel('Cluster')
plt.ylabel('Language')
plt.tight_layout()
plt.savefig('language_cluster_heatmap.png', dpi=300, bbox_inches='tight')
plt.show()

# ===========================
# STEP 7: CROSS-ANALYSIS
# ===========================

print("\n" + "="*50)
print("CROSS-ANALYSIS: SUBJECT vs TYPE")
print("="*50)

# Contingency table
subject_type_ct = pd.crosstab(df['subject_cluster'], df['type_cluster'])
print("\nSubject-Type cluster contingency table:")
print(subject_type_ct)

# Heatmap
plt.figure(figsize=(12, 10))
sns.heatmap(subject_type_ct, annot=True, fmt='d', cmap='Blues')
plt.title('Subject Cluster vs Type Cluster')
plt.xlabel('Type Cluster')
plt.ylabel('Subject Cluster')
plt.savefig('subject_type_heatmap.png', dpi=300, bbox_inches='tight')
plt.show()

# ===========================
# STEP 8: SAVE RESULTS
# ===========================

print("\n" + "="*50)
print("SAVING RESULTS")
print("="*50)

# Save full dataframe with clusters and coordinates
output_cols = ['title', 'subject', 'type', language_column, 'year',
               'subject_cluster', 'type_cluster',
               'subject_x', 'subject_y', 'type_x', 'type_y']

# Only keep columns that exist
output_cols = [col for col in output_cols if col in df.columns]
df[output_cols].to_csv('catalogue_with_clusters.csv', index=False)
print("Saved: catalogue_with_clusters.csv")

# Save cluster summaries
subject_cluster_summary = []
for cluster_id in range(n_subject_clusters):
    cluster_df = df[df['subject_cluster'] == cluster_id]
    top_subjects = cluster_df['subject'].value_counts().head(10)
    
    subject_cluster_summary.append({
        'cluster_id': cluster_id,
        'size': len(cluster_df),
        'top_subjects': ', '.join([f"{s} ({c})" for s, c in top_subjects.items()])
    })

summary_df = pd.DataFrame(subject_cluster_summary)
summary_df.to_csv('subject_cluster_summary.csv', index=False)
print("Saved: subject_cluster_summary.csv")

print("\n" + "="*50)
print("ANALYSIS COMPLETE!")
print("="*50)
print(f"\nTotal records analyzed: {len(df):,}")
print(f"Subject clusters: {n_subject_clusters}")
print(f"Type clusters: {n_type_clusters}")
print(f"Date range: {df['year'].min()} - {df['year'].max()}")
print(f"Languages: {df[language_column].nunique()}")
print("\nGenerated files:")
print("  - subject_elbow_curve.png")
print("  - subject_clusters.png")
print("  - type_clusters.png")
print("  - temporal_evolution.png")
print("  - language_comparison.png")
print("  - language_cluster_heatmap.png")
print("  - subject_type_heatmap.png")
print("  - catalogue_with_clusters.csv")
print("  - subject_cluster_summary.csv")