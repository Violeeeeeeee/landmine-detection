import numpy as np

in_path = 'datasets/giuriati_2/20170621_deg0_HHVV.npy'

print("loading dataset...")
dataset = np.load(in_path, allow_pickle=True).item()

data = dataset['data']
gt = np.asarray(dataset['ground_truth'])

print("-" * 40)
print(f"matrix shape (data): {data.shape}")
print(f"label shape (ground_truth): {gt.shape}")
print("-" * 40)

print(f"quantity of B-scans (Y - parallel lines): {data.shape[0]}")
print(f"depth (Z - time/depth): {data.shape[1]}")
print(f"trace (X - inline/distance): {data.shape[2]}")
