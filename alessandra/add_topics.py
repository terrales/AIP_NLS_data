

import pandas as pd
import json
from collections import Counter, defaultdict
import math

# Step 1: Load the sample CSV with topic modeling
print("Loading sample CSV with topic modeling...")
sample_df = pd.read_csv('./layer_3_topics_all.csv')
print(f"Sample contains {len(sample_df)} records with topics")
"""
# Step 2: Build subject-to-topic mapping from sample
print("\nBuilding subject-to-topic mapping...")
subject_to_topics = {}
for _, row in sample_df.iterrows():
    subject = row.get('subject')
    topic_0 = row.get('topic_0')
    topic_1 = row.get('topic_1')
    topic_2 = row.get('topic_2')
    if pd.notna(subject) and subject and pd.notna(topic_0):
        subject = str(subject).strip()
        # Normalise by removing trailing period
        if subject.endswith('.'):
            subject = subject[:-1].strip()
        if subject not in subject_to_topics:
            subject_to_topics[subject] = {
                'topic_0': topic_0,
                'topic_1': topic_1,
                'topic_2:': topic_2
            }
"""
# Build topic hierarchy from scratch: topic_2 -> topic_1 -> topic_0 -> subjects (with counts)
from collections import defaultdict

topic_hierarchy = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
subject_topic_counts = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(int))))
missing_subject_counts = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))

for _, row in sample_df.iterrows():
    subject = row.get('subject')
    topic_0 = row.get('topic_0')
    topic_1 = row.get('topic_1')
    topic_2 = row.get('topic_2')
    # Only require topics to be present, subject can be missing (NaN)
    if pd.notna(topic_0) and pd.notna(topic_1) and pd.notna(topic_2):
        if pd.notna(subject) and subject:
            subject_clean = str(subject).strip()
            if subject_clean.endswith('.'):
                subject_clean = subject_clean[:-1].strip()
            # Count subject occurrence
            subject_topic_counts[topic_2][topic_1][topic_0][subject_clean] += 1
        else:
            # Count missing subject
            missing_subject_counts[topic_2][topic_1][topic_0] += 1

# Prepare the nested topic hierarchy with subject names and values (counts)
topic_hierarchy_json = {}
for t2, t1s in subject_topic_counts.items():
    topic_hierarchy_json[t2] = {}
    for t1, t0s in t1s.items():
        topic_hierarchy_json[t2][t1] = {}
        for t0, subj_counts in t0s.items():
            subjects_list = [
                {'name': subj, 'value': count}
                for subj, count in subj_counts.items()
            ]
            total = sum(subj_counts.values()) + missing_subject_counts[t2][t1][t0]
            missing = missing_subject_counts[t2][t1][t0]
            percent_missing = (missing / total * 100) if total else 0
            topic_hierarchy_json[t2][t1][t0] = {
                'subjects': subjects_list,
                'missing_subjects': missing,
                'total_subjects': total,
                'percent_missing': percent_missing
            }

with open('topic_hierarchy_results.json', 'w') as f:
    json.dump(topic_hierarchy_json, f, indent=2)
print("\nSaved topic hierarchy results to topic_hierarchy_results.json.")


""""
# Step 3: Load the original CSV data
print("\nLoading original CSV data...")
csv_df = pd.read_csv('./data_cleaned.csv')
print(f"CSV contains {len(csv_df)} records")

# Step 4: Build the JSON structure from the CSV
print("\nBuilding JSON structure from CSV...")
years = sorted(csv_df['year'].dropna().astype(int).unique())
languages = sorted(csv_df['language'].dropna().unique())

series_list = []
enriched_count = 0
total_subjects_processed = 0


for lang in languages:
    lang_df = csv_df[csv_df['language'] == lang]
    values = []
    for year in years:
        year_df = lang_df[lang_df['year'] == year]
        count = len(year_df)
        type_counts = year_df['type'].value_counts().to_dict() if count else {}
        subject_counts = year_df['subject'].value_counts().to_dict() if count else {}

        # Topic aggregation using both subject and title
        topic_0_counts = Counter()
        topic_1_counts = Counter()
        topic_2_counts = Counter()
        subjects_matched = 0
        # Track which records have been matched to avoid double-counting
        matched_indices = set()
        # First, match by subject
        for idx, row in year_df.iterrows():
            matched = False
            subject = row.get('subject')
            title = row.get('title')
            for key in [subject, title]:
                if pd.notna(key) and key:
                    key_clean = str(key).strip()
                    if key_clean in subject_to_topics:
                        topic_info = subject_to_topics[key_clean]
                        if topic_info['topic_0']:
                            topic_0_counts[topic_info['topic_0']] += 1
                        if topic_info['topic_1']:
                            topic_1_counts[topic_info['topic_1']] += 1
                        if topic_info['topic_2:']:
                            topic_2_counts[topic_info['topic_2:']] += 1
                        matched = True
                        # Only count once per record
                        break
            if matched:
                subjects_matched += 1
                matched_indices.add(idx)

        entry = {
            'year': year,
            'count': count,
            'type': type_counts,
            'subject': subject_counts,
            'topic_0': dict(topic_0_counts) if topic_0_counts else {},
            'topic_1': dict(topic_1_counts) if topic_1_counts else {},
            'topic_2': dict(topic_2_counts) if topic_2_counts else {},
            'topics_coverage': {
                'total_items': count,
                'items_with_topics': subjects_matched,
                'coverage_percent': round(subjects_matched / count * 100, 1) if count > 0 else 0
            }
        }
        if topic_0_counts:
            enriched_count += 1
        values.append(entry)
    series_list.append({'language': lang, 'values': values})

json_data = {'series': series_list}

# Step 5: Save enriched JSON
# Step 5: Save enriched JSON

# Utility function to ensure all floats that are NaN are converted to None
import numpy as np
def deep_clean_nan(data):
    # Recursively converts float('nan') values in a dict/list to None and numpy types to native Python types for valid JSON.
    if isinstance(data, dict):
        return {k: deep_clean_nan(v) for k, v in data.items()}
    elif isinstance(data, list):
        return [deep_clean_nan(elem) for elem in data]
    elif isinstance(data, float) and math.isnan(data):
        return None
    elif isinstance(data, (np.integer,)):
        return int(data)
    elif isinstance(data, (np.floating,)):
        return float(data)
    elif isinstance(data, (np.ndarray,)):
        return data.tolist()
    else:
        return data

print("\nSaving enriched JSON...")
# Clean the json_data before saving
json_data_cleaned = deep_clean_nan(json_data)
with open('enriched_frequency_data.json', 'w') as f:
    json.dump(json_data_cleaned, f, indent=2)

# Optional: Create a separate mapping file for reference
print("Saving subject-to-topic mapping...")
with open('subject_topic_reference.json', 'w') as f:
    json.dump(subject_to_topics, f, indent=2)

# Summary statistics
print("\n" + "="*60)
print("ENRICHMENT SUMMARY")
print("="*60)
print(f"Total series in JSON: {len(json_data['series'])}")
print(f"Year entries enriched with topics: {enriched_count}")
print(f"Total subjects processed: {total_subjects_processed}")
print(f"Unique subjects with topic mappings: {len(subject_to_topics)}")

# Calculate overall coverage
total_items = sum(
    entry.get('count', 0)
    for series in json_data['series']
    for entry in series['values']
)
total_with_topics = sum(
    entry.get('topics_coverage', {}).get('items_with_topics', 0)
    for series in json_data['series']
    for entry in series['values']
)
print(f"\nOverall coverage: {total_with_topics}/{total_items} items ({(total_with_topics/total_items*100) if total_items else 0:.1f}%)")

print("\n" + "="*60)
print("FILES CREATED")
print("="*60)
print("  - enriched_frequency_data.json (your enriched JSON)")
print("  - subject_topic_reference.json (subject→topic lookup)")
"""