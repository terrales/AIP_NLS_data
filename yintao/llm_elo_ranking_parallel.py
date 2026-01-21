"""
Parallel LLM-based Elo Rating System for Topic Level 0 Comparison
Uses async/parallel requests to GPT-5.2 for faster ranking
Compares only topic_0 level with subject field from original data
"""
import os
import json
import random
import argparse
import asyncio
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Any
from datetime import datetime
from openai import AsyncOpenAI
import time

# Elo constants
BASE_ELO = 1000
ELO_SCALE = 400
K_FACTOR = 32
BOOTSTRAP_ROUNDS = 1000

# Parallel processing settings
MAX_CONCURRENT_REQUESTS = 50  # Number of parallel API calls


class ParallelEloRankingSystem:
    """Parallel LLM-based Elo ranking system for topic_0 comparison."""

    def __init__(self, settings_config: Dict[str, Any], api_key: str = None,
                 model: str = "gpt-5.2", max_concurrent: int = MAX_CONCURRENT_REQUESTS):
        """
        Initialize the parallel ranking system.

        Args:
            settings_config: Dictionary containing settings to compare
            api_key: OpenAI API key (if None, reads from OPENAI_API_KEY env var)
            model: OpenAI model to use (default: gpt-5.2)
            max_concurrent: Maximum number of concurrent API requests
        """
        self.settings = settings_config['settings']
        self.setting_ids = [s['id'] for s in self.settings]
        self.data_cache = {}
        self.votes = []
        self.max_concurrent = max_concurrent

        # Initialize OpenAI async client
        api_key = api_key or os.getenv('OPENAI_API_KEY')
        if not api_key:
            raise ValueError("OpenAI API key not found. Set OPENAI_API_KEY environment variable.")

        self.client = AsyncOpenAI(api_key=api_key)
        self.model = model

        # Semaphore for rate limiting
        self.semaphore = asyncio.Semaphore(max_concurrent)

        # Load all data files
        self._load_all_data()

    def _load_all_data(self):
        """Load data for all settings."""
        print("Loading data files...")
        for setting in self.settings:
            setting_id = setting['id']
            data_file = setting['data_file']

            if data_file.endswith('.parquet'):
                df = pd.read_parquet(data_file)
            else:
                df = pd.read_csv(data_file, low_memory=False)

            self.data_cache[setting_id] = df
            print(f"  Loaded {setting_id}: {len(df)} rows")

    def get_book_topic_0(self, setting_id: str, book_index: int) -> str:
        """Get topic_0 for a book from a specific setting."""
        df = self.data_cache[setting_id]
        if book_index >= len(df):
            return None

        row = df.iloc[book_index]

        # Get topic_0
        if 'topic_0' in df.columns:
            val = row['topic_0']
            return val if pd.notna(val) else "N/A"
        return "N/A"

    def get_book_info(self, setting_id: str, book_index: int) -> Dict[str, str]:
        """Get book metadata including subject."""
        df = self.data_cache[setting_id]
        if book_index >= len(df):
            return None

        row = df.iloc[book_index]
        info = {}

        # Include key metadata columns
        metadata_cols = ['title', 'creator', 'subject', 'description']
        for col in metadata_cols:
            if col in df.columns:
                val = row[col]
                if pd.notna(val) and str(val).strip():
                    info[col] = str(val)

        return info

    def format_comparison_prompt(self, book_info: Dict, topic_0_a: str, topic_0_b: str,
                                 setting_a_name: str, setting_b_name: str) -> str:
        """Format a prompt for LLM to compare two topic_0 assignments."""

        # Format book information
        book_text = "Book Information:\n"
        for key, val in book_info.items():
            book_text += f"  {key}: {val}\n"

        # Format topic_0 from both settings
        topics_text = f"""
Setting A ({setting_a_name}) - Level 0 Topic:
  {topic_0_a}

Setting B ({setting_b_name}) - Level 0 Topic:
  {topic_0_b}
"""

        prompt = f"""You are evaluating two different topic modeling approaches for categorizing library books.
Each approach assigns a high-level topic category (Level 0) to books.

{book_text}
{topics_text}

Please evaluate which setting provides a better Level 0 topic assignment for this book. Consider:
1. Relevance: Does the topic accurately reflect the book's main subject matter?
2. Specificity: Is the topic appropriately specific for a high-level category?
3. Usefulness: Would this topic help users discover or browse for this book?
4. Consistency: If a subject field exists, does the topic align well with it?

Based on these criteria, which setting is better?

Respond with ONLY one of these three options:
- "A" if Setting A is better
- "B" if Setting B is better
- "TIE" if both are roughly equal in quality

Your response:"""

        return prompt

    async def ask_llm_judge(self, book_info: Dict, topic_0_a: str, topic_0_b: str,
                           setting_a_name: str, setting_b_name: str) -> str:
        """Ask LLM to judge which topic_0 assignment is better (async)."""

        prompt = self.format_comparison_prompt(book_info, topic_0_a, topic_0_b,
                                               setting_a_name, setting_b_name)

        async with self.semaphore:  # Rate limiting
            try:
                # Use max_completion_tokens for GPT-5.x models
                params = {
                    "model": self.model,
                    "messages": [
                        {"role": "system", "content": "You are an expert librarian and information scientist."},
                        {"role": "user", "content": prompt}
                    ],
                    "temperature": 0.3
                }

                if 'gpt-5' in self.model.lower():
                    params["max_completion_tokens"] = 10
                else:
                    params["max_tokens"] = 10

                response = await self.client.chat.completions.create(**params)

                answer = response.choices[0].message.content.strip().upper()

                # Parse the response
                if 'A' in answer and 'B' not in answer:
                    return 'a'
                elif 'B' in answer and 'A' not in answer:
                    return 'b'
                else:
                    return 'tie'

            except Exception as e:
                print(f"Error calling LLM: {e}")
                return 'tie'

    async def run_single_vote(self) -> Dict[str, Any]:
        """Run a single voting round asynchronously."""

        # Select two random settings
        setting_a_id, setting_b_id = random.sample(self.setting_ids, 2)
        setting_a = next(s for s in self.settings if s['id'] == setting_a_id)
        setting_b = next(s for s in self.settings if s['id'] == setting_b_id)

        # Get valid book index (must exist in both settings)
        max_index = min(len(self.data_cache[setting_a_id]),
                       len(self.data_cache[setting_b_id]))
        book_index = random.randint(0, max_index - 1)

        # Get book info and topic_0
        book_info = self.get_book_info(setting_a_id, book_index)
        topic_0_a = self.get_book_topic_0(setting_a_id, book_index)
        topic_0_b = self.get_book_topic_0(setting_b_id, book_index)

        # Ask LLM to judge
        winner = await self.ask_llm_judge(book_info, topic_0_a, topic_0_b,
                                          setting_a['name'], setting_b['name'])

        # Return the vote
        vote = {
            'book_index': book_index,
            'setting_a': setting_a_id,
            'setting_b': setting_b_id,
            'winner': winner,
            'timestamp': datetime.now().isoformat()
        }

        return vote

    async def run_parallel_votes(self, num_votes: int, batch_size: int = 100) -> List[Dict]:
        """Run multiple votes in parallel batches."""
        all_votes = []

        for batch_start in range(0, num_votes, batch_size):
            batch_end = min(batch_start + batch_size, num_votes)
            batch_count = batch_end - batch_start

            # Create tasks for this batch
            tasks = [self.run_single_vote() for _ in range(batch_count)]

            # Run batch in parallel
            batch_votes = await asyncio.gather(*tasks)
            all_votes.extend(batch_votes)

            # Update progress
            print(f"Completed {batch_end}/{num_votes} votes...")

            # Calculate and display current ratings every 100 votes
            if batch_end % 100 == 0:
                current_ratings = self.compute_elo_ratings(all_votes)
                sorted_settings = sorted(current_ratings.items(),
                                       key=lambda x: x[1], reverse=True)
                print(f"  Current leaders: ", end="")
                for sid, rating in sorted_settings:
                    setting_name = next(s['name'] for s in self.settings if s['id'] == sid)
                    print(f"{setting_name}: {rating:.1f}  ", end="")
                print()

        return all_votes

    def compute_elo_ratings(self, votes_list: List[Dict] = None) -> Dict[str, float]:
        """
        Compute Elo ratings using Bradley-Terry model with iterative updates.
        """
        if votes_list is None:
            votes_list = self.votes

        if not votes_list:
            return {sid: BASE_ELO for sid in self.setting_ids}

        # Initialize ratings
        ratings = {sid: BASE_ELO for sid in self.setting_ids}

        # Iterative Elo updates
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
            ratings[setting_a] = ra + K_FACTOR * (sa - ea)
            ratings[setting_b] = rb + K_FACTOR * (sb - eb)

        return ratings

    def compute_elo_with_confidence(self, n_bootstrap: int = BOOTSTRAP_ROUNDS) -> Dict[str, Dict]:
        """Compute Elo ratings with bootstrap confidence intervals."""
        if not self.votes or len(self.votes) < 2:
            return {sid: {'elo': BASE_ELO, 'ci_lower': BASE_ELO, 'ci_upper': BASE_ELO}
                   for sid in self.setting_ids}

        # Compute point estimate
        point_ratings = self.compute_elo_ratings()

        # Set random seed for reproducibility
        np.random.seed(len(self.votes))

        # Bootstrap for confidence intervals
        bootstrap_ratings = {sid: [] for sid in self.setting_ids}
        votes_array = list(self.votes)

        for _ in range(n_bootstrap):
            sample_indices = np.random.randint(0, len(votes_array), len(votes_array))
            sample = [votes_array[i] for i in sample_indices]
            ratings = self.compute_elo_ratings(sample)

            for sid in self.setting_ids:
                if sid in ratings:
                    bootstrap_ratings[sid].append(ratings[sid])

        # Calculate 95% confidence intervals
        results = {}
        for sid in self.setting_ids:
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

    def get_statistics(self) -> Dict[str, Any]:
        """Calculate detailed statistics for each setting."""
        stats = {}
        for setting_id in self.setting_ids:
            setting = next(s for s in self.settings if s['id'] == setting_id)
            stats[setting_id] = {
                'id': setting_id,
                'name': setting['name'],
                'description': setting.get('description', ''),
                'wins': 0,
                'losses': 0,
                'ties': 0,
                'total': 0
            }

        # Count wins/losses/ties
        for vote in self.votes:
            setting_a = vote['setting_a']
            setting_b = vote['setting_b']
            winner = vote['winner']

            if setting_a not in stats or setting_b not in stats:
                continue

            stats[setting_a]['total'] += 1
            stats[setting_b]['total'] += 1

            if winner == 'a':
                stats[setting_a]['wins'] += 1
                stats[setting_b]['losses'] += 1
            elif winner == 'b':
                stats[setting_b]['wins'] += 1
                stats[setting_a]['losses'] += 1
            else:
                stats[setting_a]['ties'] += 1
                stats[setting_b]['ties'] += 1

        return stats

    async def run_tournament(self, num_votes: int, batch_size: int = 100) -> Dict[str, Any]:
        """
        Run a full tournament with parallel voting.

        Args:
            num_votes: Number of voting rounds to run
            batch_size: Number of votes to run in parallel per batch

        Returns:
            Dictionary with rankings and statistics
        """
        print(f"\n{'='*60}")
        print(f"Parallel LLM Elo Ranking Tournament (Topic Level 0)")
        print(f"Model: {self.model}")
        print(f"Settings to compare: {len(self.settings)}")
        print(f"Number of votes: {num_votes}")
        print(f"Max concurrent requests: {self.max_concurrent}")
        print(f"Batch size: {batch_size}")
        print(f"{'='*60}\n")

        start_time = time.time()

        # Run voting in parallel batches
        self.votes = await self.run_parallel_votes(num_votes, batch_size)

        elapsed_time = time.time() - start_time
        votes_per_second = num_votes / elapsed_time

        print(f"\n{'='*60}")
        print(f"Voting completed!")
        print(f"Total time: {elapsed_time/60:.1f} minutes")
        print(f"Rate: {votes_per_second:.1f} votes/second")
        print(f"{'='*60}\n")

        # Calculate final results
        print(f"Calculating final Elo ratings with confidence intervals...")
        elo_results = self.compute_elo_with_confidence()
        stats = self.get_statistics()

        # Merge results
        rankings = []
        for setting_id in self.setting_ids:
            entry = stats[setting_id].copy()
            elo_data = elo_results[setting_id]
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

        return {
            'rankings': rankings,
            'total_votes': len(self.votes),
            'votes': self.votes,
            'model': self.model,
            'timestamp': datetime.now().isoformat(),
            'elapsed_time': elapsed_time,
            'votes_per_second': votes_per_second
        }

    def save_results(self, results: Dict[str, Any], output_file: str):
        """Save results to JSON file."""
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to: {output_file}")

    def print_results(self, results: Dict[str, Any]):
        """Print results in a readable format."""
        print(f"\n{'='*80}")
        print(f"FINAL RANKINGS (Topic Level 0 Comparison)")
        print(f"{'='*80}\n")

        print(f"{'Rank':<6} {'Setting':<30} {'Elo':<10} {'CI Range':<15} {'W-L-T':<15} {'Win%':<8}")
        print(f"{'-'*80}")

        for i, entry in enumerate(results['rankings'], 1):
            wlt = f"{entry['wins']}-{entry['losses']}-{entry['ties']}"
            print(f"{i:<6} {entry['name']:<30} {entry['elo']:<10.1f} {entry['ci_range']:<15} "
                  f"{wlt:<15} {entry['win_rate']:<8.1f}%")

        print(f"\n{'='*80}")
        print(f"Total votes: {results['total_votes']}")
        print(f"Model used: {results['model']}")
        print(f"Time: {results['elapsed_time']/60:.1f} minutes ({results['votes_per_second']:.1f} votes/sec)")
        print(f"{'='*80}\n")


async def main_async(args):
    """Async main function."""
    # Load configuration
    print(f"Loading configuration from: {args.config}")
    with open(args.config, 'r') as f:
        settings_config = json.load(f)

    # Initialize ranking system
    ranking_system = ParallelEloRankingSystem(
        settings_config=settings_config,
        api_key=args.api_key,
        model=args.model,
        max_concurrent=args.max_concurrent
    )

    # Run tournament
    results = await ranking_system.run_tournament(
        num_votes=args.num_votes,
        batch_size=args.batch_size
    )

    # Print and save results
    ranking_system.print_results(results)
    ranking_system.save_results(results, args.output)


def main():
    parser = argparse.ArgumentParser(
        description='Parallel LLM-based Elo ranking system for topic_0 comparison'
    )
    parser.add_argument(
        '--config',
        type=str,
        required=True,
        help='Path to settings configuration JSON file'
    )
    parser.add_argument(
        '--num-votes',
        type=int,
        required=True,
        help='Number of voting rounds to run'
    )
    parser.add_argument(
        '--model',
        type=str,
        default='gpt-5.2',
        help='OpenAI model to use (default: gpt-5.2)'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='topic_0_comparison_results.json',
        help='Output file for results (default: topic_0_comparison_results.json)'
    )
    parser.add_argument(
        '--api-key',
        type=str,
        help='OpenAI API key (if not set via OPENAI_API_KEY env var)'
    )
    parser.add_argument(
        '--max-concurrent',
        type=int,
        default=50,
        help='Maximum concurrent API requests (default: 50)'
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        default=100,
        help='Votes per parallel batch (default: 100)'
    )

    args = parser.parse_args()

    # Run async main
    asyncio.run(main_async(args))


if __name__ == '__main__':
    main()
