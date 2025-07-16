import os
import pickle
from tqdm import tqdm
import pandas as pd

path = '/data/trf/python_works/BatteryLife/dataset/'
datasets = ['CALCE', 'HNEI', 'MATR', 'UL_PUR', 'SNL', 'MICH_EXP', 'MICH', 'RWTH', 'HUST', 'Tongji', 'Stanford', 'XJTU', 'ISU_ILCC', 'CALB', 'ZN-coin', 'NA-ion']
for dataset in tqdm(datasets):
    file_path = os.path.join(path, dataset)
    files = [i for i in os.listdir(file_path) if i.endswith('.pkl')][0]
    with open(os.path.join(file_path, files), 'rb') as f:
        cell_data = pickle.load(f)
    
    length = len(cell_data['cycle_data'])
    cell = cell_data['cycle_data']
    nominal_capacity = cell_data['nominal_capacity_in_Ah']
    df = pd.DataFrame()

    time_s = []

    for i in range(0, length):
        cycle_df = pd.DataFrame()
        cycle_data_len = len(cell_data['cycle_data'][i])
        cycle_data = cell_data['cycle_data'][i]
        cycle_df['Time (s)'] = cycle_data['time_in_s']
        cycle_df['Voltage (V)'] = cycle_data['voltage_in_V']
        cycle_df['Current (A)'] = cycle_data['current_in_A']
        cycle_df['Charge capacity (Ah)'] = cycle_data['charge_capacity_in_Ah']
        cycle_df['Discharge capacity (Ah)'] = cycle_data['discharge_capacity_in_Ah']
        cycle_df['Cycle_number'] = cycle_data['cycle_number']

        df = pd.concat([df, cycle_df], ignore_index=True)
    
    df.to_csv(f'./Cycling_Data/{dataset}_Cycling_Data.csv', index=False)
