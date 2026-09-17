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
# The MIX_large train/val/test split follows data_provider/data_loader.py (dataset == 'MIX_large')
recorder_file_names = split_recorder.MIX_large_train_files + split_recorder.MIX_large_val_files + split_recorder.MIX_large_test_files
BatLiNet_file_names = split_recorder.CALCE_train_files + split_recorder.CALCE_val_files + split_recorder.CALCE_test_files + split_recorder.MATR_train_files + split_recorder.MATR_val_files + split_recorder.MATR_test_files + split_recorder.HUST_train_files + split_recorder.HUST_val_files + split_recorder.HUST_test_files + split_recorder.HNEI_train_files + split_recorder.HNEI_val_files + split_recorder.HNEI_test_files + split_recorder.RWTH_train_files + split_recorder.RWTH_val_files + split_recorder.RWTH_test_files + split_recorder.SNL_train_files + split_recorder.SNL_val_files + split_recorder.SNL_test_files + split_recorder.UL_PUR_train_files + split_recorder.UL_PUR_val_files + split_recorder.UL_PUR_test_files

with open('./gate_data/name2agingConditionID.json') as file:
    aging_conditions = json.load(file)

print('PBT')
print(get_agingCondition_battery_num(recorder_file_names, aging_conditions))
print('HNEI')
HNEI_file_names = split_recorder.HNEI_train_files + split_recorder.HNEI_val_files + split_recorder.HNEI_test_files
print(get_agingCondition_battery_num(HNEI_file_names, aging_conditions))
print('MATR')
MATR_file_names = split_recorder.MATR_train_files + split_recorder.MATR_val_files + split_recorder.MATR_test_files
print(get_agingCondition_battery_num(MATR_file_names, aging_conditions))
print('MICH')
MICH_file_names = split_recorder.MICH_train_files + split_recorder.MICH_val_files + split_recorder.MICH_test_files
print(get_agingCondition_battery_num(MICH_file_names, aging_conditions))
print('XJTU')
XJTU_file_names = split_recorder.XJTU_train_files + split_recorder.XJTU_val_files + split_recorder.XJTU_test_files
print(get_agingCondition_battery_num(XJTU_file_names, aging_conditions))
print('Stanford')
Stanford_file_names = split_recorder.Stanford_train_files + split_recorder.Stanford_val_files + split_recorder.Stanford_test_files
print(get_agingCondition_battery_num(Stanford_file_names, aging_conditions))
print('RWTH')
RWTH_file_names = split_recorder.RWTH_train_files + split_recorder.RWTH_val_files + split_recorder.RWTH_test_files
print(get_agingCondition_battery_num(RWTH_file_names, aging_conditions))
print('MICH_EXP')
MICH_EXP_file_names = split_recorder.MICH_EXP_train_files + split_recorder.MICH_EXP_val_files + split_recorder.MICH_EXP_test_files
print(get_agingCondition_battery_num(MICH_EXP_file_names, aging_conditions))
print('Tongji')
Tongji_file_names = split_recorder.Tongji_train_files + split_recorder.Tongji_val_files + split_recorder.Tongji_test_files
print(get_agingCondition_battery_num(Tongji_file_names, aging_conditions))
print('HUST')
HUST_file_names = split_recorder.HUST_train_files + split_recorder.HUST_val_files + split_recorder.HUST_test_files
print(get_agingCondition_battery_num(HUST_file_names, aging_conditions))
print('SNL')
SNL_file_names = split_recorder.SNL_train_files + split_recorder.SNL_val_files + split_recorder.SNL_test_files
print(get_agingCondition_battery_num(SNL_file_names, aging_conditions))
print('ISU_ILCC')
ISU_ILCC_file_names = split_recorder.ISU_ILCC_train_files + split_recorder.ISU_ILCC_val_files + split_recorder.ISU_ILCC_test_files
print(get_agingCondition_battery_num(ISU_ILCC_file_names, aging_conditions))
print('CALCE')
CALCE_file_names = split_recorder.CALCE_train_files + split_recorder.CALCE_val_files + split_recorder.CALCE_test_files
print(get_agingCondition_battery_num(CALCE_file_names, aging_conditions))
print('UL_PUR')
UL_PUR_file_names = split_recorder.UL_PUR_train_files + split_recorder.UL_PUR_val_files + split_recorder.UL_PUR_test_files
print(get_agingCondition_battery_num(UL_PUR_file_names, aging_conditions))
print('CALB')
CALB_file_names = split_recorder.CALB_train_files + split_recorder.CALB_val_files + split_recorder.CALB_test_files
print(get_agingCondition_battery_num(CALB_file_names, aging_conditions))
print('ZN-coin')
ZN_coin_file_names = split_recorder.ZNcoin_train_files + split_recorder.ZNcoin_val_files + split_recorder.ZNcoin_test_files
print(get_agingCondition_battery_num(ZN_coin_file_names, aging_conditions))
print('NA-ion')
NA_ion_file_names = split_recorder.NAion_2021_train_files + split_recorder.NAion_2021_val_files + split_recorder.NAion_2021_test_files
print(get_agingCondition_battery_num(NA_ion_file_names, aging_conditions))
print('--------------')
print('BatLiNet')
print(get_agingCondition_battery_num(BatLiNet_file_names, aging_conditions))









