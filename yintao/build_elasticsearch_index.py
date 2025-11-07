PARTITION_ID = 3

from elasticsearch import Elasticsearch, helpers                                
import pickle
import numpy as np
from tqdm import tqdm
import re
from typing import List
from copy import deepcopy

def extract_years(text: str | None) -> List[str]:
    """
    Extracts all 4-digit year-like numbers from a given text using regex.

    Args:
        text (str): The input string.

    Returns:
        List[str]: A list of all matched 4-digit years as strings.
    """
    if text is None:
        return []
    pattern = r"\b\d{4}\b"  # match exactly 4 digits as a whole word
    return re.findall(pattern, text)


client = Elasticsearch('http://localhost:9200')

files = [f'/exports/eddie/scratch/s1891075/AIP_NLS_data/embeddings/embedding_list_{PARTITION_ID}.pkl']
data = []
for file in files:
    with open(file, 'rb') as f:
        data.append(pickle.load(f))


metas = []
for file in tqdm(data):
    for batch in file[0]:
        for row in batch[0]:
            metas.append(row)

ebd_batches = []
for file in data:
    for batch in file[1]:
        ebd_batches.append(batch)
ebd = np.concatenate(ebd_batches, axis=0)

for meta in tqdm(metas):
    year = extract_years(meta['date'])
    meta['year_0'] = year

cnt_0 = 0
cnt_gt1 = 0
gt1s = []
future = 0
for meta in tqdm(metas):
    l = len(meta['year_0'])
    if l == 0:
        cnt_0 += 1
    else:
        if int(meta['year_0'][0]) > 2025:
            future += 1
    if l > 1:
        cnt_gt1 += 1
        gt1s.append(meta)
    if l == 1 and int(meta['year_0'][0]) <= 2025:
        meta['year_clean'] = int(meta['year_0'][0])
    else:
        meta['year_clean'] = None
print(cnt_0, cnt_gt1, future)

index_name = "nls_embedding_4"

# Define index with dense_vector support
settings = {
    "settings": {"number_of_shards": 64},
    "mappings": {
        "properties": {
            "title": {"type": "text"},
            'creator': {"type": "text"},
            'type': {"type": "text"},
            'publisher': {"type": "text"},
            'date': {"type": "text"},
            'language': {"type": "text"},
            'description': {"type": "text"},
            'subject': {"type": "text"},
            'relation': {"type": "text"},
            'rights': {"type": "text"},
            'identifier': {"type": "text"},
            'coverage': {"type": "text"},
            'format': {"type": "text"},
            'year_clean': {"type": "integer"},
            "embedding": {"type": "dense_vector", "dims": 4096, "index": True, "similarity": "cosine"}
        }
    }
}

if not client.indices.exists(index=index_name):
    client.indices.create(index=index_name, body=settings)

meta_list = []
for idx, meta in enumerate(tqdm(metas)):
    mm = deepcopy(meta)
    mm['embedding'] = ebd[idx].tolist()
    del mm['year_0']
    meta_list.append({
        '_index': index_name,
        '_source': mm
    })
    if len(meta_list) > 1000:
        try:
            helpers.bulk(client, meta_list)
        except Exception as e:
            print(e)
        meta_list = []

helpers.bulk(client, meta_list)
print('Done!')
