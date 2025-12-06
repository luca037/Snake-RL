import matplotlib.pyplot as plt
import pandas as pd
from IPython import display
import time

INPUT_CSV = "./output/csv/avg_score2.csv"

# Optional: Make the plots look nicer
plt.style.use('ggplot') 
plt.ion()

def plot_training():
    try:
        df = pd.read_csv(INPUT_CSV)
    except Exception as e:
        # Avoid crashing if the file is being written to at the exact moment
        print(f"Waiting for data... ({e})")
        time.sleep(1)
        return

    # Extract data columns
    # Using .get() ensures that if a column is missing, it doesn't crash, just returns None
    scores = df.get('score')
    mean_scores_100 = df.get('mean_score_100')
    tot_avg_score = df.get('mean_score')
    
    rewards = df.get('reward')
    mean_rewards_100 = df.get('mean_reward_100')
    
    losses = df.get('mean_loss_100')
    epsilon = df.get('epsilon')
    
    time_steps = range(len(df))

    # --- PLOTTING LOGIC ---
    display.clear_output(wait=True)
    
    # Create a figure with a 2x2 grid size
    fig = plt.figure(figsize=(10, 6))
    
    # 1. Top Left: SCORES
    ax1 = fig.add_subplot(2, 2, 1)
    if scores is not None:
        ax1.plot(time_steps, scores, label='Raw Score', color='blue', alpha=0.3)
    if mean_scores_100 is not None:
        ax1.plot(time_steps, mean_scores_100, label='Mean Score (100)', color='blue', linewidth=2)
    if tot_avg_score is not None:
        ax1.plot(time_steps, tot_avg_score, label='Total Avg', color='green', linestyle='--')
    ax1.set_title('Game Scores')
    ax1.set_ylabel('Score')
    ax1.legend(loc='upper left')

    # 2. Top Right: REWARDS
    ax2 = fig.add_subplot(2, 2, 2)
    if rewards is not None:
        ax2.plot(time_steps, rewards, label='Raw Reward', color='orange', alpha=0.3)
    if mean_rewards_100 is not None:
        ax2.plot(time_steps, mean_rewards_100, label='Mean Reward (100)', color='darkorange', linewidth=2)
    ax2.set_title('Rewards')
    ax2.set_ylabel('Reward')
    ax2.legend(loc='upper left')

    # 3. Bottom Left: LOSS
    ax3 = fig.add_subplot(2, 2, 3)
    if losses is not None:
        ax3.plot(time_steps, losses, label='Mean Loss (100)', color='red')
    ax3.set_title('Training Loss')
    ax3.set_ylabel('Loss')
    ax3.set_xlabel('Episode')

    # 4. Bottom Right: EPSILON
    ax4 = fig.add_subplot(2, 2, 4)
    if epsilon is not None:
        ax4.plot(time_steps, epsilon, label='Epsilon', color='purple')
    ax4.set_title('Epsilon Decay')
    ax4.set_ylabel('Epsilon')
    ax4.set_xlabel('Episode')

    # Clean up layout and display
    plt.tight_layout()
    display.display(plt.gcf())
    
    # Close the figure to prevent memory leaks in the loop
    return fig

if __name__ == '__main__':
    while True:
        fig = plot_training()
        plt.pause(5) # Update every 5 seconds
        plt.close(fig)
