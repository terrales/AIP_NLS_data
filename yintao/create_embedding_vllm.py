import pandas as pd
import aiohttp
import asyncio
import pickle
from tqdm.asyncio import tqdm
import numpy as np
import argparse
import os


VLLM_URL = "http://localhost:8001/v1/embeddings"
MODEL_NAME = "tencent/KaLM-Embedding-Gemma3-12B-2511"


async def get_embeddings_batch(session: aiohttp.ClientSession, batch_idx: int,
                                texts: list[str], semaphore: asyncio.Semaphore,
                                max_retries: int = 10) -> tuple[int, list[list[float]], str]:
    """Get embeddings for a batch of texts from vLLM server."""
    async with semaphore:
        for attempt in range(max_retries):
            try:
                async with session.post(
                    VLLM_URL,
                    json={"input": texts, "model": MODEL_NAME},
                    timeout=aiohttp.ClientTimeout(total=600)
                ) as response:
                    response.raise_for_status()
                    data = await response.json()
                    embeddings = sorted(data["data"], key=lambda x: x["index"])
                    return batch_idx, [e["embedding"] for e in embeddings], None
            except Exception as e:
                if attempt == max_retries - 1:
                    return batch_idx, None, str(e)
                await asyncio.sleep(0.5)
        return batch_idx, None, "Max retries exceeded"


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


async def process_all_batches(batches: list, batch_index_ranges: list, valid_indices: list,
                               max_concurrent: int, max_retries: int) -> tuple[list, list, list]:
    """Process all batches with high concurrency."""
    all_embeddings = [None] * len(batches)
    semaphore = asyncio.Semaphore(max_concurrent)

    # Use a single connector with high limits
    connector = aiohttp.TCPConnector(
        limit=max_concurrent * 2,
        limit_per_host=max_concurrent * 2,
        keepalive_timeout=300
    )

    async with aiohttp.ClientSession(connector=connector) as session:
        # Create all tasks at once to flood the server
        tasks = [
            get_embeddings_batch(session, batch_idx, texts, semaphore, max_retries)
            for batch_idx, texts in batches
        ]

        # Process with progress bar
        failed_batches = []
        with tqdm(total=len(batches), desc="Creating embeddings",
                  unit="batch", dynamic_ncols=True, smoothing=0.01) as pbar:
            for coro in asyncio.as_completed(tasks):
                batch_idx, embeddings, error = await coro
                if error:
                    print(f"\nBatch {batch_idx} failed: {error}")
                    failed_batches.append(batch_idx)
                else:
                    all_embeddings[batch_idx] = embeddings
                pbar.update(1)

    # Flatten embeddings and collect corresponding indices (preserving order)
    flat_embeddings = []
    final_indices = []
    still_failed = []
    for batch_idx, batch_embeddings in enumerate(all_embeddings):
        if batch_embeddings is not None:
            flat_embeddings.extend(batch_embeddings)
            start, end = batch_index_ranges[batch_idx]
            final_indices.extend(valid_indices[start:end])
        else:
            still_failed.append(batch_idx)

    return flat_embeddings, final_indices, still_failed


async def main_async(args):
    print(f"Loading CSV from {args.input}...")
    df = pd.read_csv(args.input, low_memory=False)

    if args.limit:
        df = df.head(args.limit)
        print(f"Limited to {args.limit} rows for testing")

    print(f"Loaded {len(df)} rows")

    # Create text representations
    print("Creating text representations...")
    texts = df.apply(create_text_for_embedding, axis=1).tolist()

    # Filter out empty texts
    valid_indices = [i for i, t in enumerate(texts) if t.strip()]
    valid_texts = [texts[i] for i in valid_indices]
    print(f"Valid texts: {len(valid_texts)} / {len(texts)}")

    # Create batches with their corresponding valid_indices ranges
    batches = []
    batch_index_ranges = []
    for i in range(0, len(valid_texts), args.batch_size):
        batch_texts = valid_texts[i:i + args.batch_size]
        batch_idx = len(batches)
        batches.append((batch_idx, batch_texts))
        batch_index_ranges.append((i, i + len(batch_texts)))

    print(f"Created {len(batches)} batches of size {args.batch_size}")
    print(f"Using {args.max_concurrent} max concurrent requests (flooding server)")

    # Process all batches
    flat_embeddings, final_indices, still_failed = await process_all_batches(
        batches, batch_index_ranges, valid_indices,
        args.max_concurrent, args.max_retries
    )

    print(f"\nSuccessfully created {len(flat_embeddings)} embeddings")
    if still_failed:
        print(f"Failed batches: {len(still_failed)}")
        print(f"Failed batch indices: {sorted(still_failed)[:20]}...")

    # Create output directory if needed
    os.makedirs(os.path.dirname(args.output) if os.path.dirname(args.output) else ".", exist_ok=True)

    # Create result dictionary
    result = {
        "indices": final_indices,
        "embeddings": np.array(flat_embeddings) if flat_embeddings else np.array([]),
        "model": MODEL_NAME,
        "total_rows": len(df),
        "failed_batches": sorted(still_failed) if still_failed else []
    }

    # Save results
    print(f"Saving embeddings to {args.output}...")
    with open(args.output, "wb") as f:
        pickle.dump(result, f)

    print("Done!")


def main():
    parser = argparse.ArgumentParser(description="Create embeddings using vLLM server (async flood)")
    parser.add_argument("--input", type=str, required=True,
                        help="Input CSV file path")
    parser.add_argument("--output", type=str, required=True,
                        help="Output pickle file path")
    parser.add_argument("--batch-size", type=int, default=128,
                        help="Texts per batch (smaller = more requests)")
    parser.add_argument("--max-concurrent", type=int, default=256,
                        help="Maximum concurrent requests to flood server")
    parser.add_argument("--limit", type=int, default=None,
                        help="Limit number of rows to process (for testing)")
    parser.add_argument("--max-retries", type=int, default=10,
                        help="Maximum retries per batch")
    args = parser.parse_args()

    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
