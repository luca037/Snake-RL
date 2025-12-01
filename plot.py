import matplotlib.pyplot as plt
import pandas as pd

INPUT_CSV = "./output/csv/avg_score2.csv"

df = pd.read_csv(INPUT_CSV)

avg_scores = df['avgScore']
score = df['score']
total_score = df['total_score']
mean_score_20 = df['mean_score_20']
time_steps = [y for y in range(len(avg_scores))]

plt.plot(time_steps, avg_scores, label='Avg Score', color='green')
plt.plot(time_steps, mean_score_20, label='Avg Score (last 20 games)', color='blue')
#plt.plot(time_steps, score, label='Score curruent game', color='purple')
#plt.plot(time_steps, total_score, label='Total score', color='orange')
plt.tight_layout()
plt.legend()
plt.savefig("plot.png")
