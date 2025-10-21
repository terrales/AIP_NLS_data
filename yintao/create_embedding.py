import torch
import torch.distributed as dist
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from sentence_transformers import SentenceTransformer
import os
import pandas as pd
from tqdm import tqdm
import pickle

dist.init_process_group()

# Load the model
model = SentenceTransformer("Qwen/Qwen3-Embedding-8B").half().to(dist.get_rank())

def read_xml_files(folder_path):
    """Read all XML files in a folder and return a concatenated DataFrame."""
    # List all XML files in the folder
    xml_files = [f for f in os.listdir(folder_path) if f.endswith('.xml')]
    
    # Read each XML file and append to a list
    df_list = []
    for xml_file in xml_files:
        file_path = os.path.join(folder_path, xml_file)
        df = pd.read_xml(file_path)
        df_list.append(df)
    
    # Concatenate all DataFrames in the list
    concatenated_df = pd.concat(df_list, ignore_index=True)
    
    return concatenated_df

print('begin data loading...')
data = read_xml_files('/disk/scratch/s1891075/AIP_data/nls-catalogue-published-material/nls-catalogue-published-material_dc')
dataset = data.to_dict(orient='records', index=True)
print('data loaded!')


def collate_fn(batch):
    texts = []
    items = []
    for row in batch:
        texts.append(f"Titile: {row['title']}\nAuthors: {row['creator']}\nSubject: {row['subject']}\nRelation: {row['relation']}\nLangeuage: {row['language']}")
        items.append(batch)
    return {'texts': texts, 'items': items}

sampler = DistributedSampler(dataset, shuffle=False, drop_last=False)
print(f'RANK {dist.get_rank()} data size: {len(sampler)}')

loader = DataLoader(dataset, batch_size=16, shuffle=False, sampler=sampler, collate_fn=collate_fn)
print(f'RANK {dist.get_rank()} batch size: {len(loader)}')

embedding_list = []
item_list = []

for batch in tqdm(loader, disable=dist.get_rank() != 0):
    embedding = model.encode(batch['texts'])
    embedding_list.append(embedding)
    item_list.append(batch['items'])

ebd_obj = pickle.dumps((item_list, embedding_list))
with open(f'embedding_list_{dist.get_rank()}.pkl', 'wb') as f:
    f.write(ebd_obj)
