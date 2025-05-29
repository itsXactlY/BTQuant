import polars as pl
import matplotlib.pyplot as plt
import seaborn as sns
import json

with open("backtest_results.json") as f:
    raw_data = json.load(f)

df = pl.DataFrame(raw_data)
df = df.filter(pl.col("final_value") > 0)
df = df.sort("final_value")
total_value = df["final_value"].sum()
df = df.with_columns([
    (pl.col("final_value") / total_value * 100).alias("percentage")
])

coins = df["coin"].to_list()
values = df["final_value"].to_list()
percentages = df["percentage"].to_list()

sns.set_theme(style="whitegrid")
fig_height = max(6, len(coins) * 0.25)
fig, ax = plt.subplots(figsize=(18, fig_height))
colors = sns.color_palette("turbo", len(coins))
bars = ax.barh(coins, values, color=colors)

for i, (v, p) in enumerate(zip(values, percentages)):
    ax.text(v, i, f"{v:,.2f} ({p:.1f}%)", va='center', ha='left', fontsize=8, color='black')

ax.set_title("Final Value Distribution per Coin", fontsize=12, weight='bold')
ax.set_xlabel("Final Value", fontsize=10)
ax.set_ylabel("Coin", fontsize=10)

plt.tight_layout()
plt.grid(True, axis='x', linestyle='--', alpha=0.4)
plt.show()
