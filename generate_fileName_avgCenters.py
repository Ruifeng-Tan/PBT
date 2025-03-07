# %% [markdown]
# # README
# This notebook is used to compute the alignment center for each sample. The alignment center is the average vector from all embeddings whose cycle life is ±$\alpha$ for the source sample.

# %%
import json
import numpy as np
from tqdm import tqdm
import pickle
import os
from sklearn.manifold import TSNE
from sklearn.preprocessing import MinMaxScaler
import matplotlib
import torch
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

# %%
pretrained_model_path = '/hpc2hdd/home/rtan474/checkpoints/BatteryLifeLLMv9_Trial_P_sl1_lr5e-05_dm128_nh8_el2_dl2_df256_llmLayers32_LoraTrue_lradjconstant_datasetMIX_large_alignFalse_DGFalse_lossMSE_wd0.0_wlFalse_woDKPrFalse_pretrainedFalse_tl16-Llama31I-8B_dropOut0.15_new' # the path to the saved model checkpoints that produce these embeddings. In this notebook, the path will be used as the saving path to save the avg centers.
total_label_llm_out = np.load(f'{pretrained_model_path}/total_label_llm_out.npy')
total_labels = np.load(f'{pretrained_model_path}/total_labels.npy')
total_file_names = json.load(open(f'{pretrained_model_path}/total_file_names.json'))
total_indices = [i for i in range(len(total_label_llm_out))]


# %%
unique_labels, unique_indices = np.unique(total_labels, return_index=True)
unique_file_names = [total_file_names[i] for i in unique_indices]
unique_label_llm_out = total_label_llm_out[unique_indices]
unique_indices = np.arange(len(unique_label_llm_out))
print(len(unique_labels))
print(len(unique_file_names), unique_file_names)

# %%
alpha = 0.125 # allow alpha MAPE
upper_allowed_cycle_life = unique_labels * (1+alpha)
lower_allowed_cycle_life = unique_labels * (1-alpha)

total_avg_centers = [] # alignment center for each training sample
for i, label in tqdm(enumerate(unique_labels)):
    upper, lower = upper_allowed_cycle_life[i], lower_allowed_cycle_life[i]
    selected_indices = unique_indices[np.logical_and(unique_labels>=lower, unique_labels<=upper)]
    selected_vectors = unique_label_llm_out[selected_indices]
    avg_center = np.mean(selected_vectors, axis=0) # [D]
    total_avg_centers.append(avg_center)
total_avg_centers = np.array(total_avg_centers)
file_name_avgCenters_dict = dict(zip(total_file_names, total_avg_centers))
with open(f'{pretrained_model_path}/fileName_avgCenters_dict.pkl', 'wb') as f:
    pickle.dump(file_name_avgCenters_dict, f)
print(total_avg_centers.shape)

tsne = TSNE(n_components=2, random_state=2024)
transformed_features = tsne.fit_transform(total_avg_centers) # [N,2]
colormap =  matplotlib.colormaps['coolwarm']
norm = matplotlib.colors.Normalize(vmin=min(unique_labels), vmax=max(unique_labels))
total_colors = MinMaxScaler().fit_transform(np.array(unique_labels).reshape(-1, 1))
total_colors = total_colors.reshape(-1)

fig = plt.figure(figsize=(5,5))
plt.scatter(transformed_features[:,0], transformed_features[:,1], c=colormap(total_colors))
fig.colorbar(matplotlib.cm.ScalarMappable(norm=norm, cmap=colormap),label='Cycle life',ax=plt.gca())
plt.title('label_llm_out')
plt.savefig('./avgcenter.jpg', dpi=600)
plt.show()
