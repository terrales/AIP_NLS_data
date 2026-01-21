"""
Create embeddings using sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2.
This is a smaller, faster multilingual model.
"""
import pandas as pd
import pickle
import numpy as np
import argparse
import os
from tqdm import tqdm
from sentence_transformers import SentenceTransformer

# Set environment
os.environ.setdefault('TMPDIR', '/disk/scratch/s1891075/tmp')


def create_text_for_embedding(row: pd.Series) -> str:
    """Create text representation for embedding from a row."""
    parts = []
    if pd.notna(row.get('title')):
        parts.append(f"Title: {row['title']}")
    if pd.notna(row.get('creator')):
        parts.append(f"Authors: {row['creator']}")
    if pd.notna(row.get('subject')):
        parts.append(f"Subject: {row['subject']}")
    if pd.notna(row.get('description')):
        parts.append(f"Description: {row['description']}")
    if pd.notna(row.get('language')):
        parts.append(f"Language: {row['language']}")
    return "\n".join(parts) if parts else ""


def main():
    parser = argparse.ArgumentParser(description="Create embeddings using MiniLM")
    parser.add_argument("--input", type=str, required=True, help="Input CSV file path")
    parser.add_argument("--output", type=str, required=True, help="Output pickle file path")
    parser.add_argument("--batch-size", type=int, default=512, help="Batch size for embedding")
    parser.add_argument("--checkpoint-size", type=int, default=500000, help="Save checkpoint every N samples")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of rows")
    parser.add_argument("--model", type=str, default="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
                        help="Model name")
    args = parser.parse_args()

    print(f"Loading CSV from {args.input}...")
    df = pd.read_csv(args.input, low_memory=False)

    if args.limit:
        df = df.head(args.limit)
        print(f"Limited to {args.limit} rows")

    print(f"Loaded {len(df)} rows")

    # Create text representations
    print("Creating text representations...")
    texts = df.apply(create_text_for_embedding, axis=1).tolist()

    # Filter out empty texts and track indices
    valid_data = [(i, t) for i, t in enumerate(texts) if t.strip()]
    valid_indices = [d[0] for d in valid_data]
    valid_texts = [d[1] for d in valid_data]
    print(f"Valid texts: {len(valid_texts)} / {len(texts)}")

    # Load model
    print(f"Loading model {args.model}...")
    model = SentenceTransformer(args.model)
    print(f"Model loaded. Embedding dimension: {model.get_sentence_embedding_dimension()}")

    # Create checkpoint directory
    output_dir = os.path.dirname(args.output) if os.path.dirname(args.output) else "."
    checkpoint_dir = os.path.join(output_dir, "checkpoints_minilm")
    os.makedirs(checkpoint_dir, exist_ok=True)
    print(f"Checkpoint directory: {checkpoint_dir}")

    print(f"Starting embedding generation for {len(valid_texts)} texts...")

    # Process in batches with checkpointing
    all_embeddings = []
    current_indices = []
    checkpoint_num = 0

    for i in tqdm(range(0, len(valid_texts), args.batch_size), desc="Creating embeddings"):
        batch_texts = valid_texts[i:i + args.batch_size]
        batch_indices = valid_indices[i:i + args.batch_size]

        # Embed batch
        batch_embeddings = model.encode(batch_texts, show_progress_bar=False)
        all_embeddings.extend(batch_embeddings)
        current_indices.extend(batch_indices)

        # Save checkpoint if needed
        if len(all_embeddings) >= args.checkpoint_size:
            checkpoint_path = os.path.join(checkpoint_dir, f"checkpoint_{checkpoint_num:04d}.pkl")
            with open(checkpoint_path, "wb") as f:
                pickle.dump({
                    "indices": current_indices,
                    "embeddings": np.array(all_embeddings, dtype=np.float32)
                }, f)
            print(f"\nSaved checkpoint {checkpoint_num} with {len(all_embeddings)} embeddings")
            checkpoint_num += 1
            all_embeddings = []
            current_indices = []

    # Save remaining embeddings
    if all_embeddings:
        checkpoint_path = os.path.join(checkpoint_dir, f"checkpoint_{checkpoint_num:04d}.pkl")
        with open(checkpoint_path, "wb") as f:
            pickle.dump({
                "indices": current_indices,
                "embeddings": np.array(all_embeddings, dtype=np.float32)
            }, f)
        print(f"\nSaved final checkpoint {checkpoint_num} with {len(all_embeddings)} embeddings")
        checkpoint_num += 1

    # Merge checkpoints
    print(f"\nMerging {checkpoint_num} checkpoints...")
    import glob
    checkpoint_files = sorted(glob.glob(os.path.join(checkpoint_dir, "checkpoint_*.pkl")))

    all_indices = []
    all_embeddings = []

    for cp_file in tqdm(checkpoint_files, desc="Loading checkpoints"):
        with open(cp_file, "rb") as f:
            data = pickle.load(f)
            all_indices.extend(data["indices"])
            all_embeddings.append(data["embeddings"])

    print("Concatenating embeddings...")
    merged_embeddings = np.vstack(all_embeddings)

    result = {
        "indices": all_indices,
        "embeddings": merged_embeddings,
        "model": args.model,
        "total_rows": len(df),
    }

    print(f"Saving merged embeddings to {args.output}...")
    with open(args.output, "wb") as f:
        pickle.dump(result, f)

    print(f"\nFinal output: {len(all_indices)} embeddings saved")
    print(f"Embedding dimension: {merged_embeddings.shape[1]}")
    print("Done!")


if __name__ == "__main__":
    main()
