"""
Create embeddings using vLLM offline inference (no HTTP server).
Optimized for maximum GPU utilization with large batches.
Saves checkpoints to manage memory usage.
"""
import pandas as pd
import pickle
import numpy as np
import argparse
import os
import glob
from tqdm import tqdm

# Set environment before importing vllm
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


def save_checkpoint(checkpoint_dir, checkpoint_num, indices, embeddings):
    """Save a checkpoint to disk."""
    checkpoint_path = os.path.join(checkpoint_dir, f"checkpoint_{checkpoint_num:04d}.pkl")
    with open(checkpoint_path, "wb") as f:
        pickle.dump({"indices": indices, "embeddings": np.array(embeddings)}, f)
    print(f"\n  Saved checkpoint {checkpoint_num} with {len(embeddings)} embeddings")
    return checkpoint_path


def merge_checkpoints(checkpoint_dir, output_path, model_name, total_rows):
    """Merge all checkpoints into final output file."""
    checkpoint_files = sorted(glob.glob(os.path.join(checkpoint_dir, "checkpoint_*.pkl")))
    print(f"\nMerging {len(checkpoint_files)} checkpoints...")

    all_indices = []
    all_embeddings = []

    for cp_file in tqdm(checkpoint_files, desc="Loading checkpoints"):
        with open(cp_file, "rb") as f:
            data = pickle.load(f)
            all_indices.extend(data["indices"])
            all_embeddings.append(data["embeddings"])

    # Concatenate all embeddings
    print("Concatenating embeddings...")
    merged_embeddings = np.vstack(all_embeddings)

    result = {
        "indices": all_indices,
        "embeddings": merged_embeddings,
        "model": model_name,
        "total_rows": total_rows,
    }

    print(f"Saving merged result to {output_path}...")
    with open(output_path, "wb") as f:
        pickle.dump(result, f)

    return len(all_indices)


def main():
    parser = argparse.ArgumentParser(description="Create embeddings using vLLM offline inference")
    parser.add_argument("--input", type=str, required=True, help="Input CSV file path")
    parser.add_argument("--output", type=str, required=True, help="Output pickle file path")
    parser.add_argument("--batch-size", type=int, default=20000, help="Batch size for embedding")
    parser.add_argument("--checkpoint-size", type=int, default=200000, help="Save checkpoint every N samples")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of rows")
    parser.add_argument("--model", type=str, default="tencent/KaLM-Embedding-Gemma3-12B-2511",
                        help="Model name")
    parser.add_argument("--tensor-parallel", type=int, default=2, help="Tensor parallel size")
    parser.add_argument("--max-num-seqs", type=int, default=2048, help="Max concurrent sequences")
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

    # Create checkpoint directory
    output_dir = os.path.dirname(args.output) if os.path.dirname(args.output) else "."
    checkpoint_dir = os.path.join(output_dir, "checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)
    print(f"Checkpoint directory: {checkpoint_dir}")

    # Import vLLM after setting up
    print(f"Loading model {args.model}...")
    print(f"  tensor_parallel={args.tensor_parallel}")
    print(f"  max_num_seqs={args.max_num_seqs}")
    print(f"  batch_size={args.batch_size}")
    print(f"  checkpoint_size={args.checkpoint_size}")

    from vllm import LLM

    llm = LLM(
        model=args.model,
        tensor_parallel_size=args.tensor_parallel,
        trust_remote_code=True,
        max_model_len=4096,
        gpu_memory_utilization=0.95,
        max_num_seqs=args.max_num_seqs,
        disable_log_stats=True,
    )

    print(f"Model loaded. Starting embedding generation for {len(valid_texts)} texts...")

    # Process in large batches with progress bar and checkpointing
    current_embeddings = []
    current_indices = []
    checkpoint_num = 0
    total_processed = 0

    with tqdm(total=len(valid_texts), desc="Creating embeddings", unit="text") as pbar:
        for i in range(0, len(valid_texts), args.batch_size):
            batch_texts = valid_texts[i:i + args.batch_size]
            batch_indices = valid_indices[i:i + args.batch_size]

            # Submit batch - vLLM handles internal parallelism
            outputs = llm.embed(batch_texts)

            # Extract embeddings
            batch_embeddings = [output.outputs.embedding for output in outputs]
            current_embeddings.extend(batch_embeddings)
            current_indices.extend(batch_indices)
            total_processed += len(batch_texts)

            pbar.update(len(batch_texts))

            # Save checkpoint if we've accumulated enough samples
            if len(current_embeddings) >= args.checkpoint_size:
                save_checkpoint(checkpoint_dir, checkpoint_num, current_indices, current_embeddings)
                checkpoint_num += 1
                # Clear memory
                current_embeddings = []
                current_indices = []

    # Save any remaining embeddings as final checkpoint
    if current_embeddings:
        save_checkpoint(checkpoint_dir, checkpoint_num, current_indices, current_embeddings)
        checkpoint_num += 1

    print(f"\nGenerated {total_processed} embeddings in {checkpoint_num} checkpoints")

    # Merge all checkpoints into final output
    total_merged = merge_checkpoints(checkpoint_dir, args.output, args.model, len(df))

    print(f"\nFinal output: {total_merged} embeddings saved to {args.output}")
    print("Done!")


if __name__ == "__main__":
    main()
