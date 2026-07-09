import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# read attention weights
blocks = []
current = []

with open("attention_weights.csv") as f:
    next(f)  # skip header
    for line in f:
        line = line.strip()
        if line == "---":
            if current:
                blocks.append(np.array(current))
                current = []
        else:
            current.append([float(x) for x in line.split(",")])

if current:
    blocks.append(np.array(current))

print(f"total attention blocks captured: {len(blocks)}")

# plot all blocks as heatmaps
n = len(blocks)
cols = 4
rows = (n + cols - 1) // cols

fig, axes = plt.subplots(rows, cols, figsize=(16, rows * 4))
axes = axes.flatten()

labels = ["enc_l1_h1", "enc_l1_h2", "enc_l2_h1", "enc_l2_h2",
          "dec_l1_masked_h1", "dec_l1_masked_h2",
          "dec_l1_cross_h1", "dec_l1_cross_h2",
          "dec_l2_masked_h1", "dec_l2_masked_h2",
          "dec_l2_cross_h1", "dec_l2_cross_h2"]

for idx, (block, ax) in enumerate(zip(blocks, axes)):
    im = ax.imshow(block, cmap="Blues", vmin=0, vmax=1)
    label = labels[idx] if idx < len(labels) else f"block_{idx}"
    ax.set_title(label, fontsize=9)
    ax.set_xlabel("key position")
    ax.set_ylabel("query position")
    plt.colorbar(im, ax=ax)

# hide unused axes
for idx in range(len(blocks), len(axes)):
    axes[idx].set_visible(False)

plt.suptitle("attention heatmaps — transformer in c", fontsize=14)
plt.tight_layout()
plt.savefig("attention_heatmaps.png", dpi=150)
plt.close()
print("saved attention_heatmaps.png")