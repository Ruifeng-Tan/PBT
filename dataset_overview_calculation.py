from data_provider.data_split_recorder import split_recorder
import os
import json
def get_agingCondition_battery_num(file_names, aging_conditions):
    label_path = '/data/trf/python_works/PBT_BatteryLife/dataset/Life labels'
    label_files_path = os.listdir(label_path)
    label_json_files = [i for i in label_files_path if i.endswith('.json')]
    cell_names = []
    label_file_names = [i.replace('--', '-#') if 'Tongji' in i else i for i in file_names]
    for file in label_json_files:
        if file.startswith('Stanford_labels'):
            continue
        with open(os.path.join(label_path, file), 'r') as f:
            label_data = json.load(f)
        # print(file, len(label_data))
        for key, value in label_data.items():
            if value <= 100:
                # print(key, value)
                continue
            cell_names.append(key)  
    cell_names = [i for i in cell_names if i in label_file_names]
    cell_names = [i.replace('-#', '--') if 'Tongji' in i else i for i in cell_names]
    unique_aging_conditions = []
    for cell_name in cell_names:
        agingConditionID = aging_conditions[cell_name]
        if agingConditionID not in unique_aging_conditions:
            unique_aging_conditions.append(agingConditionID)

    return len(cell_names), len(unique_aging_conditions)

label_path = '/data/trf/python_works/PBT_BatteryLife/dataset/Life labels'
label_files_path = os.listdir(label_path)
label_json_files = [i for i in label_files_path if i.endswith('.json')]
label_names = []
recorder_file_names = split_recorder.MIX_all_2024_test_files + split_recorder.MIX_all_2024_train_files + split_recorder.MIX_all_2024_val_files
print(len(recorder_file_names))
BatLiNet_file_names = split_recorder.CALCE_train_files + split_recorder.CALCE_val_files + split_recorder.CALCE_test_files + split_recorder.MATR_train_files + split_recorder.MATR_val_files + split_recorder.MATR_test_files + split_recorder.HUST_train_files + split_recorder.HUST_val_files + split_recorder.HUST_test_files + split_recorder.HNEI_train_files + split_recorder.HNEI_val_files + split_recorder.HNEI_test_files + split_recorder.RWTH_train_files + split_recorder.RWTH_val_files + split_recorder.RWTH_test_files + split_recorder.SNL_train_files + split_recorder.SNL_val_files + split_recorder.SNL_test_files + split_recorder.UL_PUR_train_files + split_recorder.UL_PUR_val_files + split_recorder.UL_PUR_test_files

with open('./gate_data/name2agingConditionID.json') as file:
    aging_conditions = json.load(file)

print('PBT')
print(get_agingCondition_battery_num(recorder_file_names, aging_conditions))
print('BatLiNet')
print(get_agingCondition_battery_num(BatLiNet_file_names, aging_conditions))







