import pickle
import matplotlib.pyplot as plt
from tqdm import tqdm
import seaborn as sns

def draw_MATR_sequence(fig):
    with open('/data/trf/python_works/Battery-LLM/dataset/MATR/MATR_b1c0.pkl', 'rb') as f:
        MATR_data = pickle.load(f)

    length = len(MATR_data['cycle_data'])
    plus_time = 0
    total_current = []
    total_voltage = []
    total_time = []
    for i in (range(length)):
        cycle_data = MATR_data['cycle_data'][i]
        current= cycle_data['current_in_A']
        voltage = cycle_data['voltage_in_V']
        time = cycle_data['time_in_s']
        time = [time + plus_time for time in time]
        plus_time = max(time)

        total_time = total_time + time
        total_current = total_current + current
        total_voltage = total_voltage + voltage
        if i == 4:# draw first 5 cycles
            break

    ax1 = plt.subplot(2, 1, 1)
    color = sns.color_palette()[0]
    ax1.plot(total_time, total_voltage, '-', color=color)
    ax1.set_xlabel('Time(s)')
    ax1.set_ylabel('Voltage(V)', color=color)
    ax1.tick_params('y', colors=color)

    ax2 = ax1.twinx()
    color = sns.color_palette()[3]
    ax2.set_ylim(-4.2, 7)
    ax2.plot(total_time, total_current, '-', color=color)
    ax2.set_ylabel('Current(A)', color=color)
    ax2.tick_params('y', colors=color)
    plt.title('Voltage-Current vs time Profile')


def draw_Tongji_sequence(fig):
    with open('/data/trf/python_works/Battery-LLM/dataset/Tongji/Tongji1_CY25-05_1-#1.pkl', 'rb') as f:
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
draw_MATR_sequence(fig)
draw_Tongji_sequence(fig)
fig.tight_layout()
plt.savefig('./figures/first_fig.jpg', dpi=600)
plt.savefig('./figures/first_fig.pdf', dpi=600)
# plt.show()