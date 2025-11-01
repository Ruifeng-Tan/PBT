import pickle

data_path = '/data/trf/python_works/BatteryLife/dataset/HNEI/HNEI_18650_NMC_LCO_25C_0-100_0.5-1.5C_e.pkl'
with open(data_path, 'rb') as f:
    data = pickle.load(f)
    cycling_data = data['cycle_data'][0]
    cycling_data_capacity = cycling_data['discharge_capacity_in_Ah']
    print(cycling_data_capacity)