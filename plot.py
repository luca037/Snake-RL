import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import pandas as pd
import numpy as np

### Global settings and vars ###
plt.style.use('seaborn-v0_8-bright')
INPUT_CSV = "./output/csv/stats.csv"

CERBERUS_EFF = "./output/csv/cerberus_efficiency.csv"
BASELINE_EFF = "./output/csv/baseline.csv"


def efficency_plot(outfile):
    try:
        df_c = pd.read_csv(CERBERUS_EFF)
        df_b = pd.read_csv(BASELINE_EFF)
    except FileNotFoundError:
        print("ERROR: File not found.")
        return
    
    cerberus_y = []
    baseline_y = []
    
    for i in range(1, 98):
        score_c = (df_c['score'] == i)
        
        if not score_c.sum():
            cerberus_y.append(0)
        else:
            steps_c = df_c[score_c]['steps']
            avg_steps_c = steps_c.sum() / score_c.sum()
            cerberus_y.append(avg_steps_c)
    
        score_b = (df_b['score'] == i)
        
        if not score_b.sum():
            baseline_y.append(0)
        else:
            steps_b = df_b[score_b]['steps']
            avg_steps_b = steps_b.sum() / score_b.sum()
            baseline_y.append(avg_steps_b)
    
    plt.tight_layout()
    plt.xlabel("Score", fontsize=12)
    plt.ylabel("Average number of steps", fontsize=12)
    plt.xticks(np.arange(0, 100, 10))
    plt.xlim(-1, 98)
    plt.grid(alpha=0.4)
    plt.plot(range(1, 98), cerberus_y, color='green', label='CerberusAgent')
    plt.plot(range(1, 98), baseline_y, color='purple', label='Baseline')
    plt.legend(fontsize=12)
    plt.savefig(outfile)
    plt.close()


def score_histogram_plot(outfile):
    try:
        df = pd.read_csv(INPUT_CSV)
    except FileNotFoundError:
        print("ERROR: File not found.")
        return

    # Define bins.
    bins = np.arange(0, 99)

    # Create the histogram and capture the 'patches'.
    n, bins, patches = plt.hist(df['score'], bins=bins, density=True, edgecolor='black', alpha=1, align='left')

    # Apply Custom Coloring.
    for i, patch in enumerate(patches):
        if i <= 24:
            patch.set_facecolor('#ff9999')  # Red/Salmon for 0-24
        elif i <= 37:
            patch.set_facecolor('#ffcc99')  # Orange/Gold for 25-37
        else:
            patch.set_facecolor('#99ff99')  # Green for 38+

    # Formatting.
    plt.xlabel('Score', fontsize=12)
    plt.ylabel('Probability', fontsize=12)
    
    # Add a custom legend to explain the colors.
    legend_elements = [
        Patch(facecolor='#ff9999', edgecolor='black', alpha=1, label='Head 1'),
        Patch(facecolor='#ffcc99', edgecolor='black', alpha=1, label='Head 2'),
        Patch(facecolor='#99ff99', edgecolor='black', alpha=1, label='Head 3')
    ]
    plt.legend(handles=legend_elements, loc='upper right')

    plt.xticks(np.arange(0, 100, 5))
    plt.xlim(-1, 98)
    plt.grid(axis='y', linestyle='--', alpha=0.4)
    
    # 6. Save and show

    plt.tight_layout()
    plt.savefig(outfile)
    plt.close()


def monitor_training_plot(outfile):
    try:
        df = pd.read_csv(INPUT_CSV)
    except Exception as e:
        # Avoid crashing if the file is being written to at the exact moment.
        print(f"ERROR: File {INPUT_CSV} not found.\n({e})")
        return

    # Extract data columns
    # Using .get() ensures that if a column is missing, 
    # it doesn't crash, just returns None.
    scores = df.get('score')
    mean_scores_100 = df.get('mean_score_100')
    tot_avg_score = df.get('mean_score')
    mean_reward_100 = df.get('mean_reward_100')

    avg_q = df.get('avg_q')
    losses = df.get('max_loss')
    epsilon = df.get('epsilon')
    
    time_steps = range(len(df))

    fig = plt.figure(figsize=(10, 6))
    
    # 1. Top Left: SCORES and REWARDS.
    ax1 = fig.add_subplot(2, 2, 1)
    if scores is not None:
        ax1.plot(time_steps, scores, label='Raw Score', color='blue', alpha=0.3)
    if mean_scores_100 is not None:
        ax1.plot(time_steps, mean_scores_100, label='Mean Score (100)', color='blue', linewidth=2)
    if tot_avg_score is not None:
        ax1.plot(time_steps, tot_avg_score, label='Total Avg', color='green', linestyle='--')
    if mean_reward_100 is not None:
        ax1.plot(time_steps, mean_reward_100, label='Mean Reward (100)', color='cyan', linewidth=2)
        
    ax1.set_title('Game Scores')
    ax1.set_ylabel('Score')
    ax1.legend(loc='upper left')

    # 2. Top Right: AVG Q VALUE.
    ax2 = fig.add_subplot(2, 2, 2)
    if avg_q is not None:
        ax2.plot(time_steps, avg_q, label='Q(s, a)', color='orange')
    ax2.set_title('Average action value (Q)')
    ax2.set_ylabel('Q(s, a)')
    ax2.legend(loc='upper left')

    # 3. Bottom Left: LOSS.
    ax3 = fig.add_subplot(2, 2, 3)
    if losses is not None:
        ax3.plot(time_steps, losses, label='Mean Loss (100)', color='red')
    ax3.set_title('Training Loss')
    ax3.set_ylabel('Loss')
    ax3.set_xlabel('Episode')

    # 4. Bottom Right: EPSILON.
    ax4 = fig.add_subplot(2, 2, 4)
    if epsilon is not None:
        ax4.plot(time_steps, epsilon, label='Epsilon', color='purple')
    ax4.set_title('Epsilon Decay')
    ax4.set_ylabel('Epsilon')
    ax4.set_xlabel('Episode')

    # Clean up layout and display.
    plt.tight_layout()
    plt.savefig(outfile)
    plt.close(fig)
    

if __name__ == '__main__':
    monitor_training_plot("./output/plots/monitor_training_plot.pdf")
    score_histogram_plot("./output/plots/score_histogram_plot.pdf")
    efficency_plot("./output/plots/efficiency_plot.pdf")

