import numpy as np
import matplotlib.pyplot as plt

file_path = "datasets/giuriati_2/20170621_deg0_HHVV.npy"

print("Завантаження датасету...")
dataset = np.load(file_path, allow_pickle=True).item()

data = dataset['data']
gt = dataset['ground_truth']

print("-" * 40)
print(f"Вміст словника (ключі): {list(dataset.keys())}")
print(f"Розмірність матриці даних (Shape): {np.shape(data)}")
print(f"Розмірність розмітки (Ground Truth): {np.shape(gt)}")
print(f"Мінімальне значення: {data.min():.4f}")
print(f"Максимальне значення: {data.max():.4f}")
print("-" * 40)

# У їхньому main.ipynb вони роблять хитрий трюк з осями:
# переносять найменший вимір (який відповідає за канали поляризації HH/VV) у самий кінець масиву.
vis_data = np.moveaxis(data, np.argmin(data.shape), -1)
print(f"Формат після moveaxis (для візуалізації): {vis_data.shape}")

fig, axes = plt.subplots(1, 2, figsize=(15, 6))

# Беремо перший канал поляризації (наприклад, HH)
ch1_data = vis_data[..., 0]

# 1. B-scan (Вертикальний зріз). Беремо лінію посередині куба.
mid_line = ch1_data.shape[0] // 2
b_scan = ch1_data[mid_line, :, :]

vmax_b = np.percentile(np.abs(b_scan), 95)
axes[0].imshow(b_scan.T, cmap='gray', aspect='auto', vmin=-vmax_b, vmax=vmax_b)
axes[0].set_title(f"B-scan (Лінія {mid_line})")
axes[0].set_xlabel("Відстань")
axes[0].set_ylabel("Глибина")

# 2. C-scan (Вигляд зверху). Беремо зріз на певній глибині.
# Оскільки ми не знаємо точно яка вісь є глибиною, припустимо що остання
mid_depth = ch1_data.shape[-1] // 4
c_scan = ch1_data[:, :, mid_depth]

vmax_c = np.percentile(np.abs(c_scan), 95)
axes[1].imshow(c_scan, cmap='gray', aspect='auto', vmin=-vmax_c, vmax=vmax_c)
axes[1].set_title(f"C-scan (Глибина {mid_depth})")
axes[1].set_xlabel("Відстань X")
axes[1].set_ylabel("Лінії Y")

plt.tight_layout()
plt.savefig("polimi_dataset_exploration.png", dpi=150)
print("[*] Графіки збережено у файл 'polimi_dataset_exploration.png'!")
