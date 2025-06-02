import os
import numpy as np
import pickle
import json
import pandas as pd
import matplotlib.pyplot as plt
import warnings
from sympy.physics.control.control_plots import matplotlib
from tqdm import tqdm
import seaborn as sns
from sklearn.linear_model import LinearRegression

warnings.filterwarnings('ignore')

def draw_third_b():
    path = '/data/trf/python_works/Battery-LLM/dataset/third_fig_plot_data/'
    files_path = os.listdir(path)
    files = [i for i in files_path if i.endswith('.csv')]
    cycles = []
    total_soh = []
    file_name_list = []
    for file in files:# load the cell data
        file_name = file.split('_third_fig_data')[0]
        cell_df = pd.read_csv(path + file)
        soh = cell_df['SOH'].values
        cycle = max(cell_df['Cycle number'].values)
        cycles.append(cycle)
        total_soh.append(soh)
        file_name_list.append(file_name)

    # Plot results
    plt.subplot(1, 2, 1)
    plt.xlabel('Cycle number', fontsize='15')
    plt.ylabel('SOH', fontsize='15')
    plt.grid(alpha=.3)

    cycle_min = min(cycles)
    cycle_max = max(cycles)
    norm = matplotlib.colors.Normalize(vmin=cycle_min, vmax=cycle_max)
    colormap = sns.color_palette("coolwarm_r", as_cmap=True)
    cb = fig.colorbar(matplotlib.cm.ScalarMappable(norm=norm, cmap=colormap), ax=plt.gca())
    cb.set_label('Cycle life', fontsize='15')

    for soh, cycle, file_name in zip(total_soh, cycles, file_name_list):
        color = colormap(norm(cycle))
        plt.plot(range(1, cycle + 1), soh, 'k-', c=color, linewidth=1)

    plt.title('SOH-cycle number', fontsize='15')

def draw_third_a():
    path = '/data/trf/python_works/Battery-LLM/dataset/'
    files_path = os.listdir(path)
    json_files = [i for i in files_path if i.endswith('labels.json')]
    cycles_length = []
    type = []
    zn_sample = 0
    na_sample = 0
    li_sample = 0
    calb_sample = 0
    for file in tqdm(json_files):
        
        with open(path + file, 'rb') as f:
            cell = json.load(f)
            for key, value in cell.items():
                if 'ZN-coin_labels' in file:
                    zn_sample += 1
                    type.append('Zinc-ion')
                    cycles_length.append(value)
                elif 'NA-coin_labels' in file:
                    continue
                elif 'MICH_EXP_labels' in file:
                    continue
                elif 'MICH_labels' in file:
                    continue
                elif 'CALB' in file:
                    calb_sample += 1
                    type.append('CALB')
                    cycles_length.append(value)
                else:
                    li_sample += 1
                    type.append('Lithium-ion')
                    cycles_length.append(value)
    data = {
        'cycles_length': cycles_length,
        'battery_type': type
    }
    df = pd.DataFrame(data)
    plt.subplot(1, 2, 2)
    sns.histplot(data=df, x="cycles_length", hue="battery_type", multiple="stack", binwidth=100)
    plt.xlabel('Cycle Life', fontsize='15')
    plt.ylabel('Cell Number', fontsize='15')
    plt.title('Battery Life Histogram', fontsize='15')

fig = plt.figure(figsize=(12,5))
draw_third_a()
draw_third_b()
fig.tight_layout()
plt.savefig('./figures/third_fig.jpg', dpi=600)
plt.savefig('./figures/third_fig.pdf', dpi=600)