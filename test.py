import os
import gc
import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import roc_curve, roc_auc_score
import matplotlib.pyplot as plt
from tqdm import tqdm

import net_torch as net
from python_patch_extractor import PatchExtractor

# variables
in_path = 'giuriati_2/20170621_deg0_HHVV.npy'
out_path = 'cnn_article'
architecture = 'Auto3D2'
ny = 3
patch_size = 64
patch_stride = 4
n_bsc = 5
batch_size = 128
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

field, campaign = in_path.split('/')
campaign, extension = campaign.split('.')
out_name = f"{field}_{campaign}_{architecture}_patch{patch_size}_stride{patch_stride}_bsc{n_bsc}_ny{ny}"
chkpt_path = os.path.join(out_path, f"{out_name}.pth")

# load data
print("Loading dataset...")
dataset = np.load('./datasets/' + str(in_path), allow_pickle=True).item()
data = dataset['data']
gt = np.asarray(dataset['ground_truth'])
del dataset
gc.collect()

test_idx = np.arange(data.shape[0])
train_idx = np.where(gt == 0)[0][:n_bsc]
test_idx = np.delete(test_idx, train_idx)

testset = data[test_idx]
gt = gt[test_idx]
del data
gc.collect()

testset = np.moveaxis(testset, np.argmin(testset.shape), -1)

# extract patches
print("Extracting patches...")
patch_size_tuple = (patch_size, patch_size, ny)
patch_stride_tuple = (patch_stride, patch_stride, 1)
pe = PatchExtractor(patch_size_tuple, stride=patch_stride_tuple)

patches = pe.extract(testset)
patchesIdx = patches.shape
del testset
gc.collect()

# convert to pytorch format and load into dataloader
test_patches = patches.reshape((-1,) + patch_size_tuple).transpose(0, 3, 1, 2)
test_dataset = TensorDataset(torch.tensor(test_patches, dtype=torch.float32))
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# free the massive numpy arrays before inference starts!
del patches, test_patches
gc.collect()

# load model
# map the old keras string name to our new pytorch class
if architecture == 'Auto3D2':
    model_class = net.Autoencoder2
elif architecture == 'Auto3D1':
    model_class = net.Autoencoder1
elif architecture == 'Auto3D3':
    model_class = net.Autoencoder3
else:
    raise ValueError(f"Unknown architecture: {architecture}")

# instantiate the model
model = model_class(in_channels=ny, out_channels=ny).to(device)
model.load_state_dict(torch.load(chkpt_path))
model.eval()

# inference
mseFeat_list = []
print("Running inference...")
with torch.no_grad():
    for (batch_x,) in tqdm(test_loader):
        batch_x = batch_x.to(device)
        enc_orig = model.encode(batch_x)
        dec_hat = model.decode(enc_orig)
        enc_hat = model.encode(dec_hat)

        # calculate mse feature per batch
        mse = ((enc_orig - enc_hat)**2).mean(dim=(1, 2, 3))
        mseFeat_list.append(mse.cpu().numpy())

mseFeat = np.concatenate(mseFeat_list)
del test_loader, test_dataset, mseFeat_list
gc.collect()

# Memory-Mapped Reconstruction
N = len(mseFeat)
flattened_shape = (N,) + patch_size_tuple

print("Broadcasting to disk (memmap) to save RAM...")
# this creates a temporary file on your disk instead of using ram
temp_file = 'temp_mse_patches.dat'
mseFeat_patches = np.memmap(temp_file, dtype='float32', mode='w+', shape=flattened_shape)

# iterate to broadcast the scalar mse value to the 64x64x3 patch space
for i in tqdm(range(N), desc="Writing patches"):
    mseFeat_patches[i] = mseFeat[i]

mseFeat_patches.flush() # ensure everything is written to disk

print("Reconstructing volume...")
mseFeat_vol = pe.reconstruct(mseFeat_patches.reshape(patchesIdx))

# clean up the temporary file
del mseFeat_patches
os.remove(temp_file)
gc.collect()

# evaluation
print("Evaluating...")
mse_mask_max = np.max(mseFeat_vol, axis=(0, 1))
fpr_max, tpr_max, thresholds_max = roc_curve(gt, mse_mask_max)
roc_auc_max = roc_auc_score(gt, mse_mask_max)
print('Best AUC = %0.2f' % roc_auc_max)


# visualization
print("Generating plots...")

# plot ROC curve
plt.figure(figsize=(8, 6))
plt.plot(fpr_max, tpr_max, color='red', lw=2, label=f'{architecture} (AUC = {roc_auc_max:.2f})')
plt.plot([0, 1], [0, 1], color='gray', lw=1, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC)')
plt.legend(loc="lower right")
plt.grid(True)
plt.savefig(f'roc_curve_{architecture}.png', dpi=150)
plt.show()

# plot anomaly score vs ground truth (1d representation)
plt.figure(figsize=(14, 5))
normalized_anomaly = mse_mask_max / (np.max(mse_mask_max) + 1e-8)

plt.plot(normalized_anomaly, label='Max Anomaly Score (Normalized)', color='blue', linewidth=2)
plt.plot(gt, label='Ground Truth (1 = Mine, 0 = Clear)', color='black', linestyle='dashed', linewidth=2)
plt.xlabel('B-scan Index (y)')
plt.ylabel('Score / Label')
plt.title('Anomaly Detection Score across B-scans')
plt.legend()
plt.grid(True)
plt.savefig(f'anomaly_vs_gt_{architecture}.png', dpi=150)
plt.show()

# plot 2d anomaly mask for a single B-scan
mine_indices = np.where(gt == 1)[0]
if len(mine_indices) > 0:
    target_y = mine_indices[0]

    plt.figure(figsize=(10, 5))
    if len(mseFeat_vol.shape) == 3:
        if mseFeat_vol.shape[-1] == len(gt):
            mask_slice = mseFeat_vol[:, :, target_y]
        else:
            mask_slice = mseFeat_vol[target_y, :, :]

        vmax = np.percentile(mask_slice, 95)

        plt.imshow(mask_slice, cmap='hot', aspect='auto', vmax=vmax)
        plt.colorbar(label='MSE Anomaly Error')
        plt.title(f'2D Anomaly Mask for B-scan {target_y} (Ground Truth: Mine)')
        plt.xlabel('Traces (X)')
        plt.ylabel('Depth (Z)')
        plt.savefig(f'anomaly_mask_bscan_{target_y}.png', dpi=150)
        plt.show()
else:
    print("У тестовому наборі немає мін для візуалізації маски")

print("Plots saved successfully!")
