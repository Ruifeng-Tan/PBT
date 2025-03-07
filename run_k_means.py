# %% [markdown]
# # README
# This notebook is used to obtain the embeddings of the domain-knowledge prompt. Then, we can use the embeddings to cluster the samples with K-means.

# %%
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import matplotlib.cm as cm
import pickle
import transformers
from matplotlib.lines import Line2D
import seaborn as sns
import json
import torch
from data_provider.data_split_recorder import split_recorder
from Prompts.Mapping_helper import Mapping_helper
from transformers import GPT2Config, GPT2Tokenizer, GPT2Model, AutoTokenizer, AutoModel, AutoConfig, Phi3Config

# %%
cell_names = [i.split('.pkl')[0] for i in split_recorder.MIX_large_train_files]
cell_names

# %% [markdown]
# ## Tokenize and embedding

# %%
def create_causal_mask(B, seq_len):
    '''
    return:
        casual mask: [B, L, L]. 0 indicates masked.
    '''
    # Create a lower triangular matrix of shape (seq_len, seq_len)
    mask = torch.tril(torch.ones(seq_len, seq_len))  # (L, L)
    mask = mask.unsqueeze(0).expand(B, -1, -1)
    return mask

# %%
# loader the tokenizer and model
# models--meta-llama--Llama-3.1-70B-Instruct
# /data/LLMs/models--meta-llama--Llama-3.1-70B-Instruct/snapshots/945c8663693130f8be2ee66210e062158b2a9693
# '/data/LLMs/models--meta-llama--Llama-3.1-8B-Instruct/snapshots/0e9e39f249a16976918f6564b8830bc894c89659'
LLM_path = '/hpc2hdd/home/rtan474/jhaidata/LLMs/models--meta-llama--Llama-3.1-70B-Instruct/snapshots/945c8663693130f8be2ee66210e062158b2a9693'
set_n_clusters = 5
llama_config = AutoConfig.from_pretrained(LLM_path)
language_model = AutoModel.from_pretrained(
            LLM_path,
            # 'huggyllama/llama-7b',
            trust_remote_code=True,
            local_files_only=True,
            config=llama_config,
            load_in_4bit=True
        )
tokenizer = AutoTokenizer.from_pretrained(
                LLM_path,
                # 'huggyllama/llama-7b',
                trust_remote_code=True,
                local_files_only=True, 
                pad_token='<|endoftext|>'
            )
tokenizer.padding_side = 'right' # set the padding side

# %%
def get_features_from_cellNames(cell_names):
    cellName_prompt = {}
    for cell_name in cell_names:
        bg_prompt = (
                    f"Task description: You are an expert in predicting battery cycle life. " 
                    f"The cycle life is the number of cycles until the battery's discharge capacity reaches 80% of its nominal capacity. "
                    f"The discharge capacity is calculated under the described operating condition. "
                    f"Please directly output the cycle life of the battery based on the provided data. "
                    )
        helper = Mapping_helper(prompt_type='PROTOCOL', cell_name=cell_name)
        tmp_prompt = bg_prompt + helper.do_mapping()

        # Llama-instruct
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": tmp_prompt}
        ]

        tmp_prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False
        )

        res = tokenizer(tmp_prompt, return_tensors="pt")
        input_ids, attention_mask = res['input_ids'][:,1:], res['attention_mask'][:,1:]
        llama_enc_out = language_model.get_input_embeddings()(input_ids) # [1, L', d_llm]
        
        cache_position = torch.arange(0, llama_enc_out.shape[1], device=llama_enc_out.device)
        position_ids = cache_position.unsqueeze(0)
        DLP_attention_mask = attention_mask.unsqueeze(1) # [B, 1, L]
        DLP_attention_mask = DLP_attention_mask.expand(-1, DLP_attention_mask.shape[-1], -1) # [B, L, L]
        DLP_attention_mask = DLP_attention_mask.unsqueeze(1) # [B, 1, L, L]
        
        casual_mask = create_causal_mask(1, llama_enc_out.shape[1])
        casual_mask = casual_mask.unsqueeze(1) # [B, 1, L, L]

        DLP_attention_mask = torch.where(casual_mask.to(DLP_attention_mask.device)==1, DLP_attention_mask, torch.zeros_like(DLP_attention_mask))
        DLP_attention_mask = DLP_attention_mask==1 # set False to mask

        hidden_states = llama_enc_out
        for i, layer in enumerate(language_model.layers):
            res = layer(hidden_states=hidden_states, position_ids=position_ids, attention_mask=DLP_attention_mask, cache_position=cache_position)
            hidden_states = res[0]

        features = hidden_states[:,-1,:].detach().cpu().numpy().reshape(1, -1)
        
        cellName_prompt[cell_name] = features
    return cellName_prompt

cellName_prompt = get_features_from_cellNames(cell_names)

# get the features from validation and testing sets
val_cell_names = [i.split('.pkl')[0] for i in split_recorder.MIX_large_val_files]
test_cell_names = [i.split('.pkl')[0] for i in split_recorder.MIX_large_test_files]
val_cellName_prompt = get_features_from_cellNames(val_cell_names)
test_cellName_prompt = get_features_from_cellNames(test_cell_names)



# %% [markdown]
# ## Visualize the embeddings

# %%
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
total_features = []
names = []
for name, feature in cellName_prompt.items():
    name = name.split('_')[0]
    names.append(name)
    total_features.append(feature)

val_total_features = []
val_names = []
for name, feature in val_cellName_prompt.items():
    name = name.split('_')[0]
    val_names.append(name)
    val_total_features.append(feature)

test_total_features = []
test_names = []
for name, feature in test_cellName_prompt.items():
    name = name.split('_')[0]
    test_names.append(name)
    test_total_features.append(feature)

concated_features = np.concat(total_features+val_total_features+test_total_features, axis=0)

train_split = len(cellName_prompt)
val_split = train_split + len(val_cellName_prompt)

total_features = np.concat(total_features, axis=0)
# Perform TSNE to project the features to 2D
tsne = TSNE(n_components=2, random_state=42)
X_embedded = tsne.fit_transform(concated_features)

# Visualize the TSNE projection
plt.figure(figsize=(5, 4))
plt.scatter(X_embedded[:train_split, 0], X_embedded[:train_split, 1], c='blue', alpha=0.7, edgecolors='k', s=50)
plt.scatter(X_embedded[train_split:val_split, 0], X_embedded[train_split:val_split, 1], c='red', alpha=0.7, edgecolors='k', s=50)
plt.scatter(X_embedded[val_split:, 0], X_embedded[val_split:, 1], c='green', alpha=0.7, edgecolors='k', s=50)
# # Add names to points (optional, if the dataset is small)
# for i, name in enumerate(names):
#     plt.text(X_embedded[i, 0] + 0.2, X_embedded[i, 1], name, fontsize=8)

plt.title("TSNE Visualization of Features")
plt.xlabel("TSNE Dimension 1")
plt.ylabel("TSNE Dimension 2")
plt.grid(True)
plt.savefig('./figures/t-SNE_train-val-test.jpg', dpi=600)
plt.show()

# %% [markdown]
# ## Cluster and visualize
# We cluster the samples using K-Means and visualize the clusters

# %%
# Use the K that has the highest score
k_means = KMeans(n_clusters=set_n_clusters, random_state=10)
cluster_labels = k_means.fit_predict(total_features)

# Perform t-SNE on the embeddings (total_features)
tsne = TSNE(n_components=2, random_state=42)  # Reduce to 2D for visualization
tsne_results = tsne.fit_transform(total_features)
dataset_names = names
print(len(dataset_names))
plt.figure(figsize=(10, 8))
# Define a list of different shapes for the clusters
dataset_shapes = ['o', 's', '^', 'D', 'v', 'p', '*', 'X', '+', '3', '<', '>', '4', 'H']
unique_datasets = list(set(dataset_names))
dataset_shape_map = dict(zip(unique_datasets, dataset_shapes))

# Define a list of different colors for the datasets
# We'll use a set to get unique dataset names, and then assign a color to each dataset
unique_cluster_labels = list(set(cluster_labels))
colors = sns.color_palette("Set1", len(unique_cluster_labels))

# Map each unique dataset name to a color
cluster_color_map = {name: colors[i] for i, name in enumerate(unique_cluster_labels)}

# Plot each point with its corresponding cluster label and dataset shape
appear_datasetNames = []
for i in range(len(total_features)):
    if dataset_names[i] not in appear_datasetNames:
        plt.scatter(tsne_results[i, 0], tsne_results[i, 1],
                    c=[cluster_color_map[cluster_labels[i]]],  # Color by cluster label
                    marker=dataset_shape_map[dataset_names[i]],  # Shape by dataset name
                    cmap='Set1', s=100, edgecolor='k', label=dataset_names[i])
    else:
        plt.scatter(tsne_results[i, 0], tsne_results[i, 1],
                    c=[cluster_color_map[cluster_labels[i]]],  # Color by cluster label
                    marker=dataset_shape_map[dataset_names[i]],  # Shape by dataset name
                    cmap='Set1', s=100, edgecolor='k')
    appear_datasetNames.append(dataset_names[i])

# Add labels and title
plt.title("t-SNE Visualization of K-Means Clusters with Dataset Shapes", fontsize=16)
plt.xlabel("t-SNE Component 1", fontsize=14)
plt.ylabel("t-SNE Component 2", fontsize=14)

# Custom legend for cluster shapes only (no color distinction)
legend_shapes = [Line2D([0], [0], marker=shape, color='w', markerfacecolor='k', markersize=15) for shape in dataset_shapes]
# plt.legend(dataset_shapes, unique_datasets, title="Datasets", loc='upper right')
plt.savefig('./figures/t-SNE_clustering.jpg', dpi=600)
plt.show()


# %% [markdown]
# ## Save the results

# %%
## Export the clustering results
cluster_labels = [int(i) for i in cluster_labels]
cellNames_ClusterLabels = dict(zip(cell_names, cluster_labels))



# %%
## Export the domain-knowledge prompt embeddings of the samples
save_path = '/hpc2hdd/home/rtan474/python_works/Battery-LLM/dataset/'
if 'models--meta-llama--Llama-3.1-70B-Instruct' in LLM_path:
    with open(f'{save_path}cellNames_ClusterLabels_70b.json', 'w') as f:
        json.dump(cellNames_ClusterLabels, f)
    with open(f'{save_path}training_DKP_embed_70b.pkl', 'wb') as f:
        pickle.dump(cellName_prompt, f)

    with open(f'{save_path}validation_DKP_embed_70b.pkl', 'wb') as f:
        pickle.dump(val_cellName_prompt, f)

    with open(f'{save_path}testing_DKP_embed_70b.pkl', 'wb') as f:
        pickle.dump(test_cellName_prompt, f)
else:
    with open(f'{save_path}cellNames_ClusterLabels.json', 'w') as f:
        json.dump(cellNames_ClusterLabels, f)
    with open(f'{save_path}training_DKP_embed.pkl', 'wb') as f:
        pickle.dump(cellName_prompt, f)

    with open(f'{save_path}validation_DKP_embed.pkl', 'wb') as f:
        pickle.dump(val_cellName_prompt, f)

    with open(f'{save_path}testing_DKP_embed.pkl', 'wb') as f:
        pickle.dump(test_cellName_prompt, f)


