
import json
import matplotlib.pyplot as plt

# Load experiment results
with open("hw2_results.json", "r") as f:
    results = json.load(f)

fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

partitions = ["IID", "label_skew"]
titles = ["IID Partition", "Label-Skew Partition"]
colors = ["#2563eb", "#16a34a", "#dc2626"]

for ax, partition, title in zip(axes, partitions, titles):
    for H, color in zip([1, 5, 20], colors):
        result = next(
            r for r in results
            if r["partition"] == partition and r["H"] == H
        )

        rounds = [entry["round"] for entry in result["log"]]
        accuracy = [entry["val_acc"] for entry in result["log"]]

        ax.plot(
            rounds, accuracy,
            marker="o",
            markersize=3,
            linewidth=1.8,
            color=color,
            label=f"H = {H}"
        )

    ax.set_title(title)
    ax.set_xlabel("Communication Rounds")
    ax.set_ylabel("Validation Accuracy (%)")
    ax.grid(True, alpha=0.3)
    ax.legend()

plt.tight_layout()
plt.savefig("Q1c_accuracy.pdf", bbox_inches="tight")
plt.savefig("Q1c_accuracy.png", dpi=300, bbox_inches="tight")
plt.show()
