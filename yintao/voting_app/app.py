"""
Topic Voting App - Compare different topic clustering/summarization settings
Uses Bradley-Terry model for Elo-style ratings with bootstrap confidence intervals (like LMArena)
"""
import os
import json
import random
import sqlite3
import math
from datetime import datetime
from flask import Flask, render_template, request, jsonify, g
import pandas as pd
import numpy as np

app = Flask(__name__)
app.config['DATABASE'] = os.path.join(os.path.dirname(__file__), 'votes.db')

# Load settings configuration
SETTINGS_FILE = os.path.join(os.path.dirname(__file__), 'settings.json')
DATA_CACHE = {}

# Elo constants
BASE_ELO = 1000
ELO_SCALE = 400
BOOTSTRAP_ROUNDS = 1000


def get_db():
    """Get database connection."""
    if 'db' not in g:
        g.db = sqlite3.connect(app.config['DATABASE'])
        g.db.row_factory = sqlite3.Row
    return g.db


@app.teardown_appcontext
def close_db(error):
    """Close database connection."""
    db = g.pop('db', None)
    if db is not None:
        db.close()


def init_db():
    """Initialize the database."""
    db = get_db()
    db.execute('''
        CREATE TABLE IF NOT EXISTS votes (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            book_index INTEGER NOT NULL,
            setting_a TEXT NOT NULL,
            setting_b TEXT NOT NULL,
            winner TEXT NOT NULL,
            position_a TEXT NOT NULL,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    db.execute('''
        CREATE TABLE IF NOT EXISTS settings (
            id TEXT PRIMARY KEY,
            name TEXT NOT NULL,
            description TEXT,
            data_file TEXT NOT NULL,
            topic_columns TEXT NOT NULL,
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    db.commit()


def load_settings():
    """Load settings from JSON file."""
    if os.path.exists(SETTINGS_FILE):
        with open(SETTINGS_FILE, 'r') as f:
            return json.load(f)
    return {"settings": []}


def save_settings(settings_data):
    """Save settings to JSON file."""
    with open(SETTINGS_FILE, 'w') as f:
        json.dump(settings_data, f, indent=2)


def load_data_for_setting(setting):
    """Load data for a specific setting."""
    setting_id = setting['id']
    if setting_id not in DATA_CACHE:
        data_file = setting['data_file']
        if data_file.endswith('.parquet'):
            df = pd.read_parquet(data_file)
        else:
            df = pd.read_csv(data_file, low_memory=False)
        DATA_CACHE[setting_id] = df
    return DATA_CACHE[setting_id]


def get_book_topics(setting, book_index):
    """Get topics for a book from a specific setting."""
    df = load_data_for_setting(setting)
    if book_index >= len(df):
        return None

    row = df.iloc[book_index]
    topic_cols = setting['topic_columns']

    topics = {}
    for col in topic_cols:
        if col in df.columns:
            val = row[col]
            topics[col] = val if pd.notna(val) else "N/A"

    return topics


def get_book_info(setting, book_index):
    """Get book metadata."""
    df = load_data_for_setting(setting)
    if book_index >= len(df):
        return None

    row = df.iloc[book_index]
    info = {}
    # Include all metadata columns except topic columns and index
    exclude_cols = {'Unnamed: 0', 'topic_label', 'topic_0', 'topic_1', 'topic_2'}
    for col in df.columns:
        if col not in exclude_cols and not col.startswith('topic'):
            val = row[col]
            if pd.notna(val) and str(val).strip():
                info[col] = str(val)
    return info


def compute_elo_ratings(votes_list, setting_ids):
    """
    Compute Elo ratings using the Bradley-Terry model with maximum likelihood estimation.
    Returns dict of setting_id -> elo_rating
    """
    if not votes_list:
        return {sid: BASE_ELO for sid in setting_ids}

    # Initialize ratings
    ratings = {sid: BASE_ELO for sid in setting_ids}

    # Iterative update (simplified Bradley-Terry via online Elo updates)
    K = 32  # Learning rate

    for vote in votes_list:
        setting_a = vote['setting_a']
        setting_b = vote['setting_b']
        winner = vote['winner']

        if setting_a not in ratings or setting_b not in ratings:
            continue

        ra = ratings[setting_a]
        rb = ratings[setting_b]

        # Expected scores
        ea = 1.0 / (1.0 + 10 ** ((rb - ra) / ELO_SCALE))
        eb = 1.0 - ea

        # Actual scores
        if winner == 'a':
            sa, sb = 1.0, 0.0
        elif winner == 'b':
            sa, sb = 0.0, 1.0
        else:  # tie
            sa, sb = 0.5, 0.5

        # Update ratings
        ratings[setting_a] = ra + K * (sa - ea)
        ratings[setting_b] = rb + K * (sb - eb)

    return ratings


def compute_elo_with_confidence(votes_list, setting_ids, n_bootstrap=BOOTSTRAP_ROUNDS):
    """
    Compute Elo ratings with bootstrap confidence intervals.
    Returns dict with elo, ci_lower, ci_upper for each setting.
    """
    if not votes_list or len(votes_list) < 2:
        return {sid: {'elo': BASE_ELO, 'ci_lower': BASE_ELO, 'ci_upper': BASE_ELO}
                for sid in setting_ids}

    # Compute point estimate
    point_ratings = compute_elo_ratings(votes_list, setting_ids)

    # Set random seed based on vote count for reproducible results
    # This ensures CI stays stable until new votes come in
    np.random.seed(len(votes_list))

    # Bootstrap for confidence intervals
    bootstrap_ratings = {sid: [] for sid in setting_ids}
    votes_array = list(votes_list)

    for _ in range(n_bootstrap):
        # Resample with replacement
        sample = [votes_array[i] for i in np.random.randint(0, len(votes_array), len(votes_array))]
        ratings = compute_elo_ratings(sample, setting_ids)

        for sid in setting_ids:
            if sid in ratings:
                bootstrap_ratings[sid].append(ratings[sid])

    # Calculate confidence intervals (95%)
    results = {}
    for sid in setting_ids:
        if bootstrap_ratings[sid]:
            sorted_ratings = sorted(bootstrap_ratings[sid])
            ci_lower = sorted_ratings[int(0.025 * len(sorted_ratings))]
            ci_upper = sorted_ratings[int(0.975 * len(sorted_ratings))]
            results[sid] = {
                'elo': round(point_ratings.get(sid, BASE_ELO), 1),
                'ci_lower': round(ci_lower, 1),
                'ci_upper': round(ci_upper, 1)
            }
        else:
            results[sid] = {
                'elo': BASE_ELO,
                'ci_lower': BASE_ELO,
                'ci_upper': BASE_ELO
            }

    return results


@app.route('/')
def index():
    """Main voting page."""
    return render_template('index.html')


@app.route('/results')
def results_page():
    """Results/Leaderboard page."""
    return render_template('results.html')


@app.route('/api/settings', methods=['GET'])
def get_settings_list():
    """Get list of all settings."""
    settings_data = load_settings()
    return jsonify(settings_data['settings'])


@app.route('/api/settings', methods=['POST'])
def add_setting():
    """Add a new setting."""
    data = request.json
    settings_data = load_settings()

    # Check if setting ID already exists
    for s in settings_data['settings']:
        if s['id'] == data['id']:
            return jsonify({'error': 'Setting ID already exists'}), 400

    new_setting = {
        'id': data['id'],
        'name': data['name'],
        'description': data.get('description', ''),
        'data_file': data['data_file'],
        'topic_columns': data['topic_columns']
    }
    settings_data['settings'].append(new_setting)
    save_settings(settings_data)

    return jsonify({'success': True, 'setting': new_setting})


@app.route('/api/comparison', methods=['GET'])
def get_comparison():
    """Get a random book with two random settings for comparison."""
    settings_data = load_settings()
    settings = settings_data.get('settings', [])

    if len(settings) < 2:
        return jsonify({'error': 'Need at least 2 settings to compare'}), 400

    # Select two random settings
    setting_a, setting_b = random.sample(settings, 2)

    # Load data to get valid indices
    df_a = load_data_for_setting(setting_a)
    df_b = load_data_for_setting(setting_b)

    # Use the minimum length to ensure index is valid for both
    max_index = min(len(df_a), len(df_b))
    book_index = random.randint(0, max_index - 1)

    # Get book info and topics
    book_info = get_book_info(setting_a, book_index)
    topics_a = get_book_topics(setting_a, book_index)
    topics_b = get_book_topics(setting_b, book_index)

    # Randomize positions (left/right)
    swap_positions = random.choice([True, False])

    if swap_positions:
        left_setting = setting_b
        right_setting = setting_a
        left_topics = topics_b
        right_topics = topics_a
        position_a = 'right'
    else:
        left_setting = setting_a
        right_setting = setting_b
        left_topics = topics_a
        right_topics = topics_b
        position_a = 'left'

    return jsonify({
        'book_index': book_index,
        'book_info': book_info,
        'left': {
            'setting_id': left_setting['id'],
            'setting_name': left_setting['name'],
            'topics': left_topics
        },
        'right': {
            'setting_id': right_setting['id'],
            'setting_name': right_setting['name'],
            'topics': right_topics
        },
        'setting_a_id': setting_a['id'],
        'setting_b_id': setting_b['id'],
        'position_a': position_a
    })


@app.route('/api/vote', methods=['POST'])
def submit_vote():
    """Submit a vote."""
    data = request.json

    book_index = data['book_index']
    setting_a = data['setting_a']
    setting_b = data['setting_b']
    winner = data['winner']  # 'a', 'b', or 'tie'
    position_a = data['position_a']

    db = get_db()
    db.execute('''
        INSERT INTO votes (book_index, setting_a, setting_b, winner, position_a)
        VALUES (?, ?, ?, ?, ?)
    ''', (book_index, setting_a, setting_b, winner, position_a))
    db.commit()

    return jsonify({'success': True})


@app.route('/api/results', methods=['GET'])
def get_results():
    """Get voting results with Elo ratings and confidence intervals."""
    settings_data = load_settings()
    settings = {s['id']: s for s in settings_data.get('settings', [])}
    setting_ids = list(settings.keys())

    db = get_db()
    votes = db.execute('SELECT * FROM votes').fetchall()

    # Convert to list of dicts
    votes_list = [dict(v) for v in votes]

    # Calculate Elo with confidence intervals
    elo_results = compute_elo_with_confidence(votes_list, setting_ids)

    # Calculate basic stats
    stats = {}
    for setting_id in settings:
        stats[setting_id] = {
            'id': setting_id,
            'name': settings[setting_id]['name'],
            'description': settings[setting_id].get('description', ''),
            'wins': 0,
            'losses': 0,
            'ties': 0,
            'total': 0
        }

    matchups = {}

    for vote in votes_list:
        setting_a = vote['setting_a']
        setting_b = vote['setting_b']
        winner = vote['winner']

        if setting_a not in stats or setting_b not in stats:
            continue

        stats[setting_a]['total'] += 1
        stats[setting_b]['total'] += 1

        matchup_key = tuple(sorted([setting_a, setting_b]))
        if matchup_key not in matchups:
            matchups[matchup_key] = {setting_a: 0, setting_b: 0, 'ties': 0}

        if winner == 'a':
            stats[setting_a]['wins'] += 1
            stats[setting_b]['losses'] += 1
            matchups[matchup_key][setting_a] += 1
        elif winner == 'b':
            stats[setting_b]['wins'] += 1
            stats[setting_a]['losses'] += 1
            matchups[matchup_key][setting_b] += 1
        else:
            stats[setting_a]['ties'] += 1
            stats[setting_b]['ties'] += 1
            matchups[matchup_key]['ties'] += 1

    # Merge Elo results with stats
    rankings = []
    for setting_id in stats:
        entry = stats[setting_id].copy()
        elo_data = elo_results.get(setting_id, {'elo': BASE_ELO, 'ci_lower': BASE_ELO, 'ci_upper': BASE_ELO})
        entry['elo'] = elo_data['elo']
        entry['ci_lower'] = elo_data['ci_lower']
        entry['ci_upper'] = elo_data['ci_upper']
        entry['ci_range'] = f"+{round(elo_data['ci_upper'] - elo_data['elo'], 1)}/-{round(elo_data['elo'] - elo_data['ci_lower'], 1)}"

        # Calculate win rate
        total = entry['total']
        if total > 0:
            win_rate = (entry['wins'] + 0.5 * entry['ties']) / total
            entry['win_rate'] = round(win_rate * 100, 1)
        else:
            entry['win_rate'] = 50.0

        rankings.append(entry)

    # Sort by Elo rating
    rankings.sort(key=lambda x: x['elo'], reverse=True)

    # Format matchups
    matchup_results = []
    for (s1, s2), counts in matchups.items():
        matchup_results.append({
            'setting_1': {'id': s1, 'name': settings[s1]['name'], 'wins': counts[s1]},
            'setting_2': {'id': s2, 'name': settings[s2]['name'], 'wins': counts[s2]},
            'ties': counts['ties'],
            'total': counts[s1] + counts[s2] + counts['ties']
        })

    return jsonify({
        'rankings': rankings,
        'matchups': matchup_results,
        'total_votes': len(votes_list),
        'bootstrap_rounds': BOOTSTRAP_ROUNDS
    })


@app.route('/admin')
def admin():
    """Admin page for managing settings."""
    return render_template('admin.html')


if __name__ == '__main__':
    with app.app_context():
        init_db()
    app.run(host='0.0.0.0', port=5000, debug=True)
