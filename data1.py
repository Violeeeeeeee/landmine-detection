import numpy as np
import matplotlib.pyplot as plt

file_path = "datasets/giuriati_2/20170621_deg0_HHVV.npy"

dataset = np.load(file_path, allow_pickle=True).item()
data = dataset['data']
gt = dataset['ground_truth']

fig, axes = plt.subplots(1, 2, figsize=(15, 6))

# 1. B-scan (vertical)
mid_line = data.shape[0] // 2
b_scan = data[mid_line, :, :] # shape: [170, 440]

vmax_b = np.percentile(np.abs(b_scan), 95)
axes[0].imshow(b_scan, cmap='gray', aspect='auto', vmin=-vmax_b, vmax=vmax_b)
axes[0].set_title(f"B-scan (line {mid_line}), point: {gt[mid_line]}")
axes[0].set_xlabel("trace (distance X)")
axes[0].set_ylabel("depth (Z)")

# 2. C-scan (horizontal - from above)
mid_depth = data.shape[1] // 4
c_scan = data[:, mid_depth, :] # shape: [66, 440]

vmax_c = np.percentile(np.abs(c_scan), 95)
axes[1].imshow(c_scan, cmap='gray', aspect='auto', vmin=-vmax_c, vmax=vmax_c)
axes[1].set_title(f"C-scan (depth {mid_depth}) - look from above")
axes[1].set_xlabel("trace (distance X)")
axes[1].set_ylabel("lines (Y)")

plt.tight_layout()
plt.savefig("polimi_dataset_exploration.png", dpi=150)
print("[*] plot saved as 'polimi_dataset_exploration.png'")
