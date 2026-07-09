import numpy as np
import matplotlib.pyplot as plt

# read loss curve
epochs = []
losses = []

with open("loss_curve.csv") as f:
    next(f)  # skip header
    for line in f:
        e, l = line.strip().split(",")
        epochs.append(int(e))
        losses.append(float(l))

plt.figure(figsize=(10, 5))
plt.plot(epochs, losses, color='royalblue', linewidth=1.5)
plt.axhline(y=2.302, color='red', linestyle='--', 
            label='random baseline (log(10) = 2.302)')
plt.xlabel('epoch')
plt.ylabel('cross entropy loss')
plt.title('copy task training — transformer in c')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('loss_curve.png', dpi=150)
plt.close()
print("saved loss_curve.png")