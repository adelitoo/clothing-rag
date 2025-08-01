import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, Any

sns.set_theme(style="whitegrid")

def save_dataframe_to_csv(df: pd.DataFrame, file_name: str, reports_dir: Path):
    reports_dir.mkdir(parents=True, exist_ok=True)
    output_path = reports_dir / file_name
    df.to_csv(output_path, index=False)
    print(f"📊 Report saved successfully to: {output_path}")

def plot_performance_heatmap(summary_df: pd.DataFrame, title: str, output_filename: str, reports_dir: Path):
    reports_dir.mkdir(parents=True, exist_ok=True)
    heatmap_data = summary_df.set_index('Metric')
    
    _, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(
        heatmap_data, 
        annot=True,          
        fmt=".4f",           
        cmap='viridis',      
        linewidths=.5,       
        ax=ax,
        cbar_kws={'label': 'Performance Score'}
    )
    
    ax.set_title(title, size=16, pad=20)
    ax.set_ylabel('')
    ax.set_xlabel('System', size=12)
    plt.xticks(rotation=0)
    plt.yticks(rotation=0)
    
    output_path = reports_dir / output_filename
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"📈 Heatmap saved successfully to: {output_path}")
    plt.show()

def plot_grouped_bar_chart(summary_df: pd.DataFrame, systems_config: Dict, title: str, output_filename: str, reports_dir: Path):
    reports_dir.mkdir(parents=True, exist_ok=True)
    
    plot_df = summary_df.set_index('Metric')
    system_names = [sys_info['name'] for sys_info in systems_config.values()]
    colors = [sys_info['color'] for sys_info in systems_config.values()]
    
    ax = plot_df[system_names].plot(
        kind='bar',
        color=colors,
        figsize=(12, 8),
        rot=0
    )
    
    ax.set_title(title, fontsize=18, pad=20)
    ax.set_xlabel('')
    ax.set_ylabel('Score', fontsize=14)
    ax.set_ylim(0, max(1.0, plot_df[system_names].max().max() * 1.1))
    ax.legend(title='System', fontsize=12)
    
    for container in ax.containers:
        ax.bar_label(container, fmt='%.3f', fontsize=10, padding=3)
    
    plt.tight_layout()
    output_path = reports_dir / output_filename
    plt.savefig(output_path, dpi=300)
    print(f"📈 Bar chart saved successfully to: {output_path}")
    plt.show()

def plot_score_distribution(results_df: pd.DataFrame, systems_config: Dict, title: str, output_filename: str, reports_dir: Path):
    reports_dir.mkdir(parents=True, exist_ok=True)
    
    _, ax = plt.subplots(figsize=(12, 7))
    
    palette = {sys_info['name']: sys_info['color'] for sys_info in systems_config.values()}

    sns.kdeplot(data=results_df, x='score', hue='model', fill=True, common_norm=False,
                palette=palette, ax=ax)

    for _, system_info in systems_config.items():
        system_name = system_info['name']
        avg_score = results_df[results_df['model'] == system_name]['score'].mean()
        ax.axvline(avg_score, color=system_info['color'], linestyle='--', linewidth=2, 
                   label=f'{system_name} Avg: {avg_score:.4f}')

    ax.set_title(title, fontsize=16, pad=20)
    ax.set_xlabel('Score', fontsize=12)
    ax.set_ylabel('Density', fontsize=12)
    ax.legend()
    ax.set_xlim(0, 1)
    
    output_path = reports_dir / output_filename
    plt.savefig(output_path)
    print(f"📈 Distribution plot saved as {output_path}")
    plt.show()
    
def plot_llm_pie_chart(summary_df: pd.DataFrame, title: str, output_filename: str, reports_dir: Path):
    reports_dir.mkdir(parents=True, exist_ok=True)
    preferences = summary_df['llm_preference'].value_counts(normalize=True).reindex(['A', 'B', 'Tie']).fillna(0) * 100

    _, ax = plt.subplots(figsize=(8, 6))
    preferences.plot(
        kind='pie',
        ax=ax,
        autopct='%1.1f%%',
        labels=['System A Wins', 'System B Wins', 'Tie'],
        colors=['#4CAF50', '#F44336', '#FFC107'],
        wedgeprops=dict(width=0.4)
    )
    ax.set_title(title)
    ax.set_ylabel('')

    output_path = reports_dir / output_filename
    plt.savefig(output_path, bbox_inches='tight')
    print(f"📈 Pie chart saved to: {output_path}")
    plt.show()