import pickle
import matplotlib.pyplot as plt
from tqdm import tqdm
import seaborn as sns
import numpy as np

def draw_discharge_sequence(fig):
    with open('/data/trf/python_works/BatteryLife/dataset/ISU_ILCC/ISU-ILCC_G49C1.pkl', 'rb') as f:
        MATR_data = pickle.load(f)

    length = len(MATR_data['cycle_data'])
    plus_time = 0
    total_current = []
    total_voltage = []
    total_time = []
    nominal_capacity=MATR_data['nominal_capacity_in_Ah']
    for i in (range(length)):
        cycle_data = MATR_data['cycle_data'][i]
        current= np.array(cycle_data['current_in_A'])
        voltage = np.array(cycle_data['voltage_in_V'])
        time = np.array(cycle_data['time_in_s'])

        current_records_in_C = current/nominal_capacity
        cutoff_voltage_indices = np.nonzero(current_records_in_C>=0.01) # This includes constant-voltage charge data, 49th cycle of MATR_b1c18 has some abnormal voltage records
        charge_end_index = cutoff_voltage_indices[0][-1] # after charge_end_index, there are rest after charge, discharge, and rest after discharge data

        discharge_voltages = voltage[charge_end_index:]
        discharge_currents = current[charge_end_index:]
        discharge_times = time[charge_end_index:]
        
        discharge_voltages = discharge_voltages[np.abs(discharge_currents)>0.01]
        discharge_currents = discharge_currents[np.abs(discharge_currents)>0.01]
        discharge_times = discharge_times[np.abs(discharge_currents)>0.01]
        

        total_time = total_time + list(discharge_times)
        total_current = total_current + list(discharge_currents)
        total_voltage = total_voltage + list(discharge_voltages)
        if i == 4:# draw first 5 cycles
            break
    
    ax1 = plt.subplot(2, 1, 2)
    color = sns.color_palette()[0]
    ax1.scatter(total_time, total_voltage)
    ax1.set_xlabel('Time(s)')
    ax1.set_ylabel('Voltage(V)', color=color)
    ax1.tick_params('y', colors=color)

    ax2 = ax1.twinx()
    color = sns.color_palette()[3]
    ax2.set_ylim(-0.5, 0.5)
    ax2.plot(total_time, total_current, '-', color=color)
    ax2.set_ylabel('Current(A)', color=color)
    ax2.tick_params('y', colors=color)
    plt.title('Voltage-Current vs time Profile')

def draw_charge_sequence(fig):
    with open('/data/trf/python_works/BatteryLife/dataset/ISU_ILCC/ISU-ILCC_G49C1.pkl', 'rb') as f:
        MATR_data = pickle.load(f)

    length = len(MATR_data['cycle_data'])
    plus_time = 0
    total_current = []
    total_voltage = []
    total_time = []
    nominal_capacity=MATR_data['nominal_capacity_in_Ah']
    for i in (range(length)):
        cycle_data = MATR_data['cycle_data'][i]
        current= np.array(cycle_data['current_in_A'])
        voltage = np.array(cycle_data['voltage_in_V'])
        charge_capacity_records = np.array(cycle_data['charge_capacity_in_Ah'])
        discharge_capacity_records = np.array(cycle_data['discharge_capacity_in_Ah'])
        print(charge_capacity_records)
        time = np.array(cycle_data['time_in_s'])

        current_records_in_C = current/nominal_capacity
        cutoff_voltage_indices = np.nonzero(current_records_in_C>=0.01) # This includes constant-voltage charge data, 49th cycle of MATR_b1c18 has some abnormal voltage records
        charge_end_index = cutoff_voltage_indices[0][-1] # after charge_end_index, there are rest after charge, discharge, and rest after discharge data
        
        charge_voltages = voltage[:charge_end_index]
        charge_currents = current[:charge_end_index]
        charge_times = time[:charge_end_index]
        

        total_time = total_time + list(charge_times)
        total_current = total_current + list(charge_currents)
        total_voltage = total_voltage + list(charge_voltages)
        if i == 4:# draw first 5 cycles
            break
    
    ax1 = plt.subplot(2, 1, 1)
    color = sns.color_palette()[0]
    ax1.scatter(total_time, total_voltage)
    ax1.set_xlabel('Time(s)')
    ax1.set_ylabel('Voltage(V)', color=color)
    ax1.tick_params('y', colors=color)

    ax2 = ax1.twinx()
    color = sns.color_palette()[3]
    ax2.set_ylim(-0.5, 0.5)
    ax2.plot(total_time, total_current, '-', color=color)
    ax2.set_ylabel('Current(A)', color=color)
    ax2.tick_params('y', colors=color)
    plt.title('Voltage-Current vs time Profile')


def draw_full_sequence(fig):
    with open('/data/trf/python_works/BatteryLife/dataset/ISU_ILCC/ISU-ILCC_G49C1.pkl', 'rb') as f:
        MATR_data = pickle.load(f)

    length = len(MATR_data['cycle_data'])
    total_current = []
    total_voltage = []
    total_time = []
    for i in (range(length)):
        cycle_data = MATR_data['cycle_data'][i]
        current = cycle_data['current_in_A']
        voltage = cycle_data['voltage_in_V']
        time = cycle_data['time_in_s']
        print(max(time)-min(time))
        total_time = total_time + time
        total_current = total_current + current
        total_voltage = total_voltage + voltage
        if i == 4:  # draw first 5 cycles
            break

    ax1 = plt.subplot(2, 1, 2)
    color = sns.color_palette()[0]
    ax1.plot(total_time, total_voltage, '-', color=color)
    ax1.set_xlabel('Time(s)')
    ax1.set_ylabel('Voltage(V)', color=color)
    ax1.tick_params('y', colors=color)

    ax2 = ax1.twinx()
    color = sns.color_palette()[3]
    ax2.set_ylim(-3.6, 2.5)
    ax2.plot(total_time, total_current, '-', color=color)
    ax2.set_ylabel('Current(A)', color=color)
    ax2.tick_params('y', colors=color)

fig = plt.figure(figsize=(12, 6))# set the size of the figure
draw_charge_sequence(fig)
draw_discharge_sequence(fig)
fig.tight_layout()
plt.savefig('./figures/ISU_ILCC_check.jpg', dpi=600)
plt.savefig('./figures/ISU_ILCC_check.pdf', dpi=600)
# plt.show()