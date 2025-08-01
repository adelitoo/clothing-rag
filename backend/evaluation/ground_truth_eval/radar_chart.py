
import numpy as np
import matplotlib.pyplot as plt
import os

labels = ["nDCG", "MAP", "MRR"]
system_a_scores = [0.6198, 0.4962, 0.5785] 
system_b_scores = [0.8948, 0.8409, 0.8662]  

def create_radar_chart(labels, system_a_scores, system_b_scores):
    num_vars = len(labels)

    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()

    system_a_scores += system_a_scores[:1]
    system_b_scores += system_b_scores[:1]
    angles += angles[:1]

    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))

    ax.plot(angles, system_a_scores, color='#F44336', linewidth=2, linestyle='solid', label='System A (LLM)')
    ax.fill(angles, system_a_scores, color='#F44336', alpha=0.25)

    ax.plot(angles, system_b_scores, color='#4CAF50', linewidth=2, linestyle='solid', label='System B (Baseline)')
    ax.fill(angles, system_b_scores, color='#4CAF50', alpha=0.25)

    ax.set_ylim(0, 1)

    ax.set_thetagrids(np.degrees(angles[:-1]), labels)

    plt.title('Overall System Performance Comparison', size=20, color='black', y=1.1)
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))

    if not os.path.exists("evaluation_reports"):
        os.makedirs("evaluation_reports")
    
    plot_path = os.path.join("evaluation_reports", "evaluation_summary_radar.png")
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    
    print(f"✅ Radar chart saved successfully to: {plot_path}")
    plt.show()


if __name__ == "__main__":
    create_radar_chart(labels, system_a_scores, system_b_scores)
