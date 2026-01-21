#!/usr/bin/env python3
"""
Visualization tool for LLM-based Elo ranking results
Generates comprehensive charts and analysis from ranking JSON files
"""
import json
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 10


class RankingVisualizer:
    """Visualize LLM Elo ranking results."""

    def __init__(self, results_file: str):
        """
        Initialize visualizer with results file.

        Args:
            results_file: Path to JSON results file
        """
        print(f"Loading results from: {results_file}")
        with open(results_file, 'r') as f:
            self.results = json.load(f)

        self.rankings = self.results['rankings']
        self.votes = self.results['votes']
        self.total_votes = self.results['total_votes']
        self.model = self.results['model']
        self.timestamp = self.results.get('timestamp', 'Unknown')

        # Create DataFrame for easier analysis
        self.rankings_df = pd.DataFrame(self.rankings)
        self.votes_df = pd.DataFrame(self.votes)

        print(f"Loaded {len(self.rankings)} settings with {self.total_votes} total votes")

    def plot_elo_rankings(self, ax=None):
        """Plot Elo rankings with confidence intervals."""
        if ax is None:
            fig, ax = plt.subplots(figsize=(12, 6))

        # Sort by Elo (descending)
        df = self.rankings_df.sort_values('elo', ascending=True)

        # Calculate error bars
        errors_lower = df['elo'] - df['ci_lower']
        errors_upper = df['ci_upper'] - df['elo']
        errors = [errors_lower.values, errors_upper.values]

        # Create horizontal bar chart
        y_pos = np.arange(len(df))
        bars = ax.barh(y_pos, df['elo'], xerr=errors, capsize=5,
                      color=sns.color_palette("viridis", len(df)), alpha=0.8,
                      edgecolor='black', linewidth=1.5)

        # Customize
        ax.set_yticks(y_pos)
        ax.set_yticklabels(df['name'])
        ax.set_xlabel('Elo Rating', fontsize=12, fontweight='bold')
        ax.set_title('Elo Rankings with 95% Confidence Intervals',
                    fontsize=14, fontweight='bold', pad=20)
        ax.axvline(x=1000, color='red', linestyle='--', linewidth=2,
                  alpha=0.5, label='Base Elo (1000)')

        # Add value labels
        for i, (idx, row) in enumerate(df.iterrows()):
            ax.text(row['elo'] + 5, i, f"{row['elo']:.1f}",
                   va='center', fontweight='bold')

        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()

        return ax

    def plot_win_stats(self, ax=None):
        """Plot win/loss/tie statistics."""
        if ax is None:
            fig, ax = plt.subplots(figsize=(12, 6))

        df = self.rankings_df.sort_values('elo', ascending=False)

        # Prepare data for stacked bar chart
        x = np.arange(len(df))
        width = 0.6

        # Create stacked bars
        p1 = ax.bar(x, df['wins'], width, label='Wins',
                   color='#2ecc71', edgecolor='black', linewidth=1)
        p2 = ax.bar(x, df['ties'], width, bottom=df['wins'],
                   label='Ties', color='#f39c12', edgecolor='black', linewidth=1)
        p3 = ax.bar(x, df['losses'], width,
                   bottom=df['wins'] + df['ties'],
                   label='Losses', color='#e74c3c', edgecolor='black', linewidth=1)

        # Customize
        ax.set_xlabel('Settings', fontsize=12, fontweight='bold')
        ax.set_ylabel('Number of Votes', fontsize=12, fontweight='bold')
        ax.set_title('Win/Loss/Tie Statistics',
                    fontsize=14, fontweight='bold', pad=20)
        ax.set_xticks(x)
        ax.set_xticklabels(df['name'], rotation=45, ha='right')
        ax.legend(loc='upper right')
        ax.grid(True, alpha=0.3, axis='y')

        # Add total labels on top
        for i, (idx, row) in enumerate(df.iterrows()):
            total = row['total']
            ax.text(i, total + 1, str(total), ha='center',
                   va='bottom', fontweight='bold')

        plt.tight_layout()
        return ax

    def plot_win_rates(self, ax=None):
        """Plot win rates as percentage."""
        if ax is None:
            fig, ax = plt.subplots(figsize=(12, 6))

        df = self.rankings_df.sort_values('win_rate', ascending=True)

        # Create horizontal bar chart
        y_pos = np.arange(len(df))
        colors = ['#e74c3c' if x < 50 else '#2ecc71' for x in df['win_rate']]

        bars = ax.barh(y_pos, df['win_rate'], color=colors, alpha=0.8,
                      edgecolor='black', linewidth=1.5)

        # Customize
        ax.set_yticks(y_pos)
        ax.set_yticklabels(df['name'])
        ax.set_xlabel('Win Rate (%)', fontsize=12, fontweight='bold')
        ax.set_title('Win Rate Comparison (Wins + 0.5*Ties)',
                    fontsize=14, fontweight='bold', pad=20)
        ax.axvline(x=50, color='black', linestyle='--', linewidth=2,
                  alpha=0.5, label='50% (baseline)')
        ax.set_xlim(0, 100)

        # Add value labels
        for i, (idx, row) in enumerate(df.iterrows()):
            ax.text(row['win_rate'] + 1, i, f"{row['win_rate']:.1f}%",
                   va='center', fontweight='bold')

        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()

        return ax

    def plot_matchup_matrix(self, ax=None):
        """Plot head-to-head matchup matrix."""
        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 8))

        # Create matchup matrix
        setting_ids = [r['id'] for r in self.rankings]
        setting_names = [r['name'] for r in self.rankings]
        n = len(setting_ids)

        matrix = np.zeros((n, n))

        for vote in self.votes:
            setting_a = vote['setting_a']
            setting_b = vote['setting_b']
            winner = vote['winner']

            if setting_a in setting_ids and setting_b in setting_ids:
                idx_a = setting_ids.index(setting_a)
                idx_b = setting_ids.index(setting_b)

                if winner == 'a':
                    matrix[idx_a, idx_b] += 1
                elif winner == 'b':
                    matrix[idx_b, idx_a] += 1
                else:  # tie
                    matrix[idx_a, idx_b] += 0.5
                    matrix[idx_b, idx_a] += 0.5

        # Plot heatmap
        sns.heatmap(matrix, annot=True, fmt='.0f', cmap='RdYlGn',
                   xticklabels=setting_names, yticklabels=setting_names,
                   ax=ax, cbar_kws={'label': 'Wins'}, linewidths=1,
                   linecolor='black', square=True)

        ax.set_title('Head-to-Head Matchup Matrix\n(Row vs Column)',
                    fontsize=14, fontweight='bold', pad=20)
        ax.set_xlabel('Opponent', fontsize=12, fontweight='bold')
        ax.set_ylabel('Setting', fontsize=12, fontweight='bold')

        plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
        plt.setp(ax.get_yticklabels(), rotation=0)
        plt.tight_layout()

        return ax

    def plot_elo_evolution(self, ax=None):
        """Plot Elo rating evolution over time."""
        if ax is None:
            fig, ax = plt.subplots(figsize=(12, 6))

        # Simulate Elo evolution by replaying votes
        setting_ids = [r['id'] for r in self.rankings]
        elo_history = {sid: [1000] for sid in setting_ids}  # Start at base Elo

        BASE_ELO = 1000
        ELO_SCALE = 400
        K_FACTOR = 32

        current_elos = {sid: BASE_ELO for sid in setting_ids}

        for vote in self.votes:
            setting_a = vote['setting_a']
            setting_b = vote['setting_b']
            winner = vote['winner']

            if setting_a in current_elos and setting_b in current_elos:
                ra = current_elos[setting_a]
                rb = current_elos[setting_b]

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
                current_elos[setting_a] = ra + K_FACTOR * (sa - ea)
                current_elos[setting_b] = rb + K_FACTOR * (sb - eb)

                # Record history
                elo_history[setting_a].append(current_elos[setting_a])
                elo_history[setting_b].append(current_elos[setting_b])

                # For settings not involved, keep same rating
                for sid in setting_ids:
                    if sid != setting_a and sid != setting_b:
                        elo_history[sid].append(current_elos[sid])

        # Plot evolution
        colors = sns.color_palette("husl", len(setting_ids))
        for i, sid in enumerate(setting_ids):
            setting_name = next(r['name'] for r in self.rankings if r['id'] == sid)
            ax.plot(elo_history[sid], label=setting_name,
                   color=colors[i], linewidth=2, alpha=0.8)

        ax.axhline(y=1000, color='red', linestyle='--', linewidth=1.5,
                  alpha=0.5, label='Base Elo')
        ax.set_xlabel('Vote Number', fontsize=12, fontweight='bold')
        ax.set_ylabel('Elo Rating', fontsize=12, fontweight='bold')
        ax.set_title('Elo Rating Evolution Over Time',
                    fontsize=14, fontweight='bold', pad=20)
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3)
        plt.tight_layout()

        return ax

    def plot_vote_distribution(self, ax=None):
        """Plot distribution of vote outcomes."""
        if ax is None:
            fig, ax = plt.subplots(figsize=(8, 6))

        # Count outcomes
        outcomes = self.votes_df['winner'].value_counts()
        labels = {'a': 'Setting A Wins', 'b': 'Setting B Wins', 'tie': 'Ties'}
        colors = {'a': '#2ecc71', 'b': '#3498db', 'tie': '#f39c12'}

        # Create pie chart
        outcome_labels = [labels.get(k, k) for k in outcomes.index]
        outcome_colors = [colors.get(k, '#95a5a6') for k in outcomes.index]

        wedges, texts, autotexts = ax.pie(outcomes.values, labels=outcome_labels,
                                          colors=outcome_colors, autopct='%1.1f%%',
                                          startangle=90, textprops={'fontsize': 11})

        # Make percentage text bold
        for autotext in autotexts:
            autotext.set_color('white')
            autotext.set_fontweight('bold')
            autotext.set_fontsize(12)

        ax.set_title('Vote Outcome Distribution',
                    fontsize=14, fontweight='bold', pad=20)

        plt.tight_layout()
        return ax

    def create_summary_stats(self, ax=None):
        """Create a text summary of statistics."""
        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 6))

        ax.axis('off')

        # Prepare summary text
        summary_lines = [
            "RANKING SUMMARY",
            "=" * 60,
            f"Model Used: {self.model}",
            f"Total Votes: {self.total_votes}",
            f"Timestamp: {self.timestamp}",
            f"Number of Settings: {len(self.rankings)}",
            "",
            "TOP 3 SETTINGS:",
            "-" * 60,
        ]

        # Add top 3 settings
        top_3 = self.rankings_df.nlargest(3, 'elo')
        for i, (idx, row) in enumerate(top_3.iterrows(), 1):
            summary_lines.extend([
                f"{i}. {row['name']}",
                f"   Elo: {row['elo']:.1f} (±{row['ci_upper'] - row['elo']:.1f})",
                f"   Win Rate: {row['win_rate']:.1f}%",
                f"   Record: {row['wins']}-{row['losses']}-{row['ties']}",
                ""
            ])

        # Calculate additional stats
        elo_range = self.rankings_df['elo'].max() - self.rankings_df['elo'].min()
        summary_lines.extend([
            "STATISTICS:",
            "-" * 60,
            f"Elo Range: {elo_range:.1f}",
            f"Mean Elo: {self.rankings_df['elo'].mean():.1f}",
            f"Median Win Rate: {self.rankings_df['win_rate'].median():.1f}%",
        ])

        # Display text
        text = '\n'.join(summary_lines)
        ax.text(0.1, 0.95, text, transform=ax.transAxes,
               fontsize=11, verticalalignment='top',
               fontfamily='monospace',
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))

        plt.tight_layout()
        return ax

    def generate_all_plots(self, output_dir: str = "ranking_visualizations"):
        """Generate all visualization plots and save to directory."""
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)

        print(f"\nGenerating visualizations in: {output_path}")

        # 1. Elo Rankings
        print("  Creating Elo rankings plot...")
        fig, ax = plt.subplots(figsize=(12, 6))
        self.plot_elo_rankings(ax)
        fig.savefig(output_path / "01_elo_rankings.png", dpi=300, bbox_inches='tight')
        plt.close(fig)

        # 2. Win Stats
        print("  Creating win statistics plot...")
        fig, ax = plt.subplots(figsize=(12, 6))
        self.plot_win_stats(ax)
        fig.savefig(output_path / "02_win_stats.png", dpi=300, bbox_inches='tight')
        plt.close(fig)

        # 3. Win Rates
        print("  Creating win rates plot...")
        fig, ax = plt.subplots(figsize=(12, 6))
        self.plot_win_rates(ax)
        fig.savefig(output_path / "03_win_rates.png", dpi=300, bbox_inches='tight')
        plt.close(fig)

        # 4. Matchup Matrix
        print("  Creating matchup matrix...")
        fig, ax = plt.subplots(figsize=(10, 8))
        self.plot_matchup_matrix(ax)
        fig.savefig(output_path / "04_matchup_matrix.png", dpi=300, bbox_inches='tight')
        plt.close(fig)

        # 5. Elo Evolution
        print("  Creating Elo evolution plot...")
        fig, ax = plt.subplots(figsize=(12, 6))
        self.plot_elo_evolution(ax)
        fig.savefig(output_path / "05_elo_evolution.png", dpi=300, bbox_inches='tight')
        plt.close(fig)

        # 6. Vote Distribution
        print("  Creating vote distribution plot...")
        fig, ax = plt.subplots(figsize=(8, 6))
        self.plot_vote_distribution(ax)
        fig.savefig(output_path / "06_vote_distribution.png", dpi=300, bbox_inches='tight')
        plt.close(fig)

        # 7. Summary Stats
        print("  Creating summary statistics...")
        fig, ax = plt.subplots(figsize=(10, 6))
        self.create_summary_stats(ax)
        fig.savefig(output_path / "07_summary_stats.png", dpi=300, bbox_inches='tight')
        plt.close(fig)

        # 8. Create comprehensive dashboard
        print("  Creating comprehensive dashboard...")
        self.create_dashboard(output_path / "00_dashboard.png")

        print(f"\n✓ All visualizations saved to: {output_path}")

        # Generate HTML report
        self.generate_html_report(output_path)

    def create_dashboard(self, output_file: str):
        """Create a comprehensive dashboard with multiple subplots."""
        fig = plt.figure(figsize=(20, 12))
        gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

        # Elo Rankings
        ax1 = fig.add_subplot(gs[0, :2])
        self.plot_elo_rankings(ax1)

        # Win Stats
        ax2 = fig.add_subplot(gs[1, :2])
        self.plot_win_stats(ax2)

        # Win Rates
        ax3 = fig.add_subplot(gs[2, :2])
        self.plot_win_rates(ax3)

        # Matchup Matrix
        ax4 = fig.add_subplot(gs[0:2, 2])
        self.plot_matchup_matrix(ax4)

        # Vote Distribution
        ax5 = fig.add_subplot(gs[2, 2])
        self.plot_vote_distribution(ax5)

        fig.suptitle(f'LLM Elo Ranking Dashboard - {self.model}',
                    fontsize=18, fontweight='bold', y=0.995)

        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close(fig)

    def generate_html_report(self, output_dir: Path):
        """Generate an HTML report with all visualizations."""
        html_content = f"""
<!DOCTYPE html>
<html>
<head>
    <title>LLM Ranking Results - Visualization Report</title>
    <style>
        body {{
            font-family: Arial, sans-serif;
            max-width: 1400px;
            margin: 0 auto;
            padding: 20px;
            background-color: #f5f5f5;
        }}
        h1, h2 {{
            color: #2c3e50;
        }}
        .header {{
            background-color: #3498db;
            color: white;
            padding: 20px;
            border-radius: 5px;
            margin-bottom: 30px;
        }}
        .section {{
            background-color: white;
            padding: 20px;
            margin-bottom: 30px;
            border-radius: 5px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        img {{
            max-width: 100%;
            height: auto;
            border: 1px solid #ddd;
            border-radius: 4px;
            padding: 5px;
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
            margin-top: 20px;
        }}
        th, td {{
            padding: 12px;
            text-align: left;
            border-bottom: 1px solid #ddd;
        }}
        th {{
            background-color: #3498db;
            color: white;
        }}
        tr:hover {{
            background-color: #f5f5f5;
        }}
        .metric {{
            display: inline-block;
            margin: 10px 20px;
            font-size: 18px;
        }}
        .metric-value {{
            font-weight: bold;
            color: #3498db;
            font-size: 24px;
        }}
    </style>
</head>
<body>
    <div class="header">
        <h1>LLM Elo Ranking Results</h1>
        <div class="metric">
            <div>Model: <span class="metric-value">{self.model}</span></div>
        </div>
        <div class="metric">
            <div>Total Votes: <span class="metric-value">{self.total_votes}</span></div>
        </div>
        <div class="metric">
            <div>Settings: <span class="metric-value">{len(self.rankings)}</span></div>
        </div>
        <div class="metric">
            <div>Timestamp: <span class="metric-value">{self.timestamp}</span></div>
        </div>
    </div>

    <div class="section">
        <h2>Dashboard Overview</h2>
        <img src="00_dashboard.png" alt="Dashboard">
    </div>

    <div class="section">
        <h2>Rankings Table</h2>
        <table>
            <thead>
                <tr>
                    <th>Rank</th>
                    <th>Setting</th>
                    <th>Elo</th>
                    <th>CI Range</th>
                    <th>Wins</th>
                    <th>Losses</th>
                    <th>Ties</th>
                    <th>Win Rate</th>
                </tr>
            </thead>
            <tbody>
"""

        # Add table rows
        for i, (idx, row) in enumerate(self.rankings_df.sort_values('elo', ascending=False).iterrows(), 1):
            html_content += f"""
                <tr>
                    <td>{i}</td>
                    <td><strong>{row['name']}</strong><br><small>{row['description']}</small></td>
                    <td>{row['elo']:.1f}</td>
                    <td>{row['ci_range']}</td>
                    <td>{row['wins']}</td>
                    <td>{row['losses']}</td>
                    <td>{row['ties']}</td>
                    <td>{row['win_rate']:.1f}%</td>
                </tr>
"""

        html_content += """
            </tbody>
        </table>
    </div>

    <div class="section">
        <h2>Elo Rankings with Confidence Intervals</h2>
        <img src="01_elo_rankings.png" alt="Elo Rankings">
    </div>

    <div class="section">
        <h2>Win/Loss/Tie Statistics</h2>
        <img src="02_win_stats.png" alt="Win Stats">
    </div>

    <div class="section">
        <h2>Win Rate Comparison</h2>
        <img src="03_win_rates.png" alt="Win Rates">
    </div>

    <div class="section">
        <h2>Head-to-Head Matchup Matrix</h2>
        <img src="04_matchup_matrix.png" alt="Matchup Matrix">
        <p><small>Each cell shows how many times the row setting beat the column setting.</small></p>
    </div>

    <div class="section">
        <h2>Elo Rating Evolution</h2>
        <img src="05_elo_evolution.png" alt="Elo Evolution">
    </div>

    <div class="section">
        <h2>Vote Outcome Distribution</h2>
        <img src="06_vote_distribution.png" alt="Vote Distribution">
    </div>

    <div class="section">
        <h2>Summary Statistics</h2>
        <img src="07_summary_stats.png" alt="Summary Stats">
    </div>

</body>
</html>
"""

        # Save HTML file
        html_file = output_dir / "report.html"
        with open(html_file, 'w') as f:
            f.write(html_content)

        print(f"  ✓ HTML report saved to: {html_file}")


def main():
    parser = argparse.ArgumentParser(
        description='Visualize LLM Elo ranking results'
    )
    parser.add_argument(
        'results_file',
        type=str,
        help='Path to JSON results file from llm_elo_ranking.py'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='ranking_visualizations',
        help='Output directory for visualizations (default: ranking_visualizations)'
    )
    parser.add_argument(
        '--show',
        action='store_true',
        help='Display plots interactively (in addition to saving)'
    )

    args = parser.parse_args()

    # Create visualizer
    visualizer = RankingVisualizer(args.results_file)

    # Generate all plots
    visualizer.generate_all_plots(args.output_dir)

    print(f"\n{'='*70}")
    print(f"Visualization complete!")
    print(f"{'='*70}")
    print(f"View the HTML report: {args.output_dir}/report.html")
    print(f"{'='*70}\n")

    # Optionally show plots interactively
    if args.show:
        print("Displaying plots interactively...")
        fig = plt.figure(figsize=(20, 12))
        gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

        ax1 = fig.add_subplot(gs[0, :2])
        visualizer.plot_elo_rankings(ax1)

        ax2 = fig.add_subplot(gs[1, :2])
        visualizer.plot_win_stats(ax2)

        ax3 = fig.add_subplot(gs[2, :2])
        visualizer.plot_win_rates(ax3)

        ax4 = fig.add_subplot(gs[0:2, 2])
        visualizer.plot_matchup_matrix(ax4)

        ax5 = fig.add_subplot(gs[2, 2])
        visualizer.plot_vote_distribution(ax5)

        plt.show()


if __name__ == '__main__':
    main()
