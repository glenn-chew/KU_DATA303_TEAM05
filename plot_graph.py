import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("fid_results1.csv")

colors = {
    "no_ada": "red",
    "std_ada": "blue", 
    "dropout": "green",
    "progressive": "purple"
}

labels = {
    "no_ada": "No ADA",
    "std_ada": "Standard ADA",
    "dropout": "ADA + Dropout",
    "progressive": "ADA + Progressive Dropout"
}

# Graph 1 — No ADA vs ADA only
# fig, ax = plt.subplots(figsize=(10, 5))
# for model in ["no_ada", "std_ada"]:
#     subset = df[df["model"] == model]
#     ax.plot(subset["kimgs"], subset["fid"],
#             marker="o", label=labels[model], 
#             color=colors[model], linewidth=2)
# ax.set_xlabel("kimgs")
# ax.set_ylabel("FID Score (lower is better)")
# ax.set_title("Effect of ADA on Limited Data Training")
# ax.legend()
# ax.grid(True, alpha=0.3)
# plt.tight_layout()
# plt.savefig("fid_ada_vs_noada.png", dpi=150)
# plt.show()

# Graph 2 — No ADA vs ADA vs Dropout
fig, ax = plt.subplots(figsize=(10, 5))
for model in ["no_ada", "std_ada", "dropout", "progressive"]:
    subset = df[df["model"] == model]
    ax.plot(subset["kimgs"], subset["fid"],
            marker="o", label=labels[model],
            color=colors[model], linewidth=2)
ax.set_xlabel("kimgs")
ax.set_ylabel("FID Score (lower is better)")
ax.set_title("ADA vs ADA + Dropout Comparison")
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("fid_all_comparison.png", dpi=150)
plt.show()