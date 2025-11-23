import pandas as pd
import json
import random
import copy
import os
import math # Import math for isnan check

# --- Configuration ---
CLEANED_DATA_PATH = 'alessandra/data_cleaned.csv'
BASE_FREQUENCY_PATH = 'alessandra/language_timeseries_detailed.json' 
# NEW: We will create a folder to hold the sharded files
OUTPUT_SAMPLES_DIR = 'alessandra/samples_shards' 

# --- Setup Output Directory ---
os.makedirs(OUTPUT_SAMPLES_DIR, exist_ok=True)
print(f"Created output directory: {OUTPUT_SAMPLES_DIR}")


# --- Data Preparation ---

# Load and clean DataFrame
df = pd.read_csv(CLEANED_DATA_PATH)
# Initial dropna to remove rows missing key lookup/grouping data
df = df.dropna(subset=['date', 'type_main', 'language'])
df['language'] = df['language'].astype(str)
df['date'] = df['date'].astype(int)

# Build a lookup for the dataframe by language and year
# We will iterate over the language groups to build the individual files
df_grouped_by_lang = df.groupby('language')

# Utility function to clean strings before they are written to JSON
def clean_string(text):
    """Replaces double quotes within strings to prevent JSON parsing errors."""
    if isinstance(text, str):
        # Replacing internal quotes with single quotes is a robust way to avoid JSON breakages
        return text.replace('"', "'") 
    return text

# Utility function for sampling
def safe_sample(group, n=3):
    """Safely samples rows from a DataFrame group and extracts metadata."""
    if len(group) == 0:
        return []
    
    sample_fields = ['title', 'creator', 'publisher']
    # Filter columns to keep based on what's available in the group
    cols_to_keep = [c for c in sample_fields if c in group.columns]
    
    samples_df = group.sample(n=min(n, len(group)), random_state=42)[cols_to_keep]
    records = samples_df.to_dict(orient='records')
    
    # Apply cleaning to all relevant string fields and explicitly check for NaNs in records
    for record in records:
        for key in ['title', 'creator', 'publisher']: # Added publisher here for robustness
            if key in record and pd.isna(record[key]):
                record[key] = None # Replace pandas NaN with Python None
            elif key in record:
                record[key] = clean_string(record[key])
    
    return records

# Utility function to ensure all floats that are NaN are converted to None
# This is a final safety net for any numerical fields that might have slipped through
def deep_clean_nan(data):
    """Recursively converts float('nan') values in a dict/list to None for valid JSON."""
    if isinstance(data, dict):
        return {k: deep_clean_nan(v) for k, v in data.items()}
    elif isinstance(data, list):
        return [deep_clean_nan(elem) for elem in data]
    # Check for NaN float value
    elif isinstance(data, float) and math.isnan(data):
        return None
    else:
        return data


# --- Load Base Frequency Data ---
# Load this file to get the master list of years and types/subjects for all languages
try:
    with open(BASE_FREQUENCY_PATH, 'r', encoding='utf-8') as f:
        base_data = json.load(f)
except FileNotFoundError:
    print(f"Error: Base frequency file not found at {BASE_FREQUENCY_PATH}")
    exit(1)
except json.JSONDecodeError:
    print(f"Error: Base frequency file at {BASE_FREQUENCY_PATH} is not valid JSON.")
    exit(1)


# --- Process and Generate Sharded Samples Files ---

# Iterate over each language's data in the base structure
for lang_entry in base_data['series']:
    lang = str(lang_entry['language'])
    
    # 1. Initialize the lookup structure for this specific language only
    lang_samples_lookup = {}
    
    # Get the subset of the original dataframe for this language
    try:
        lang_df = df_grouped_by_lang.get_group(lang)
    except KeyError:
        print(f"Warning: No data found in CSV for language '{lang}'. Skipping.")
        continue # Skip to next language
        
    # Group the language-specific data by year for efficient lookup
    lang_df_grouped_by_year = lang_df.groupby('date')

    # 2. Iterate over every year in the base frequency data
    for year_entry in lang_entry['values']:
        year = int(year_entry['year'])
        year_key = str(year) # Use string keys for JSON compatibility
        
        # Get all rows for this language and year
        try:
            year_df = lang_df_grouped_by_year.get_group(year)
        except KeyError:
            # If no data for this year, use an empty DataFrame
            year_df = pd.DataFrame(columns=df.columns) 
        
        # --- BUILD SAMPLES FOR LOOKUP ---
        
        # a. Sample 1-5 books for the year (general sample)
        year_samples = safe_sample(year_df)
        
        # b. Sample 1-5 books for each type 
        type_counts = year_entry.get('type', {})
        # Only sample if 'type_main' column exists in the current year's data
        type_samples = {
            t: safe_sample(year_df[year_df['type_main'] == t]) 
            for t in type_counts if 'type_main' in year_df.columns
        }
        
        # c. Sample 1-5 books for each subject
        subject_counts = year_entry.get('subject', {})
        # Only sample if 'subject' column exists in the current year's data
        subject_samples = {
            s: safe_sample(year_df[year_df['subject'] == s]) 
            for s in subject_counts if 'subject' in year_df.columns
        }
        
        # Store samples in the language-specific lookup structure
        lang_samples_lookup[year_key] = {
            'year': year_samples,
            'type': type_samples,
            'subject': subject_samples
        }
    
    # 3. Final Cleaning and Write the language-specific file
    
    # --- VALIDATION STEP: Ensure no NaNs remain ---
    # Apply the recursive cleaning function just before dumping
    cleaned_lang_samples_lookup = deep_clean_nan(lang_samples_lookup)
    
    # Filename is sanitized to be safe for filesystems (e.g., replace spaces)
    safe_lang_name = lang.replace(' ', '_').replace('/', '_')
    output_filename = os.path.join(OUTPUT_SAMPLES_DIR, f'samples_{safe_lang_name}.json')
    
    try:
        with open(output_filename, 'w', encoding='utf-8') as f:
            # None values in Python are converted to 'null' in JSON, which is valid.
            json.dump(cleaned_lang_samples_lookup, f, ensure_ascii=False, indent=2)
            
        print(f"✅ Wrote samples for '{lang}' to: {output_filename}")
    except Exception as e:
        print(f"🛑 Error writing file for '{lang}': {e}")


print("\n--- Generation Complete ---")
print(f"Total sharded files generated in: {OUTPUT_SAMPLES_DIR}")