import numpy as np
import matplotlib.pyplot as plt
from datetime import date, datetime
import config


def progress_bar(current, total, reward, bar_length=20):
    fraction = current / total
    arrow = int(fraction * bar_length - 1) * '-' + '>'
    padding = int(bar_length - len(arrow)) * ' '
    ending = '\n' if current == total else '\r'
    print(f'Progress: [{arrow}{padding}] {int(fraction * 100)}%, cumulative reward: {reward}', end=ending)


def save_to_file(data, file_path):
    if not isinstance(file_path, str):
        return ""
    np.savetxt(file_path + '.csv', data, delimiter=',')


def load_from_file(file_path):
    return np.loadtxt(file_path, delimiter=',')


def make_plot(x_data, y_data, x_label, y_label, title):
    plt.plot(x_data, y_data)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)
    plt.show()


def split_on_axes(tensor):
    return [np.array([experience[field_index] for experience in tensor])
    for field_index in range(len(tensor[0]))]

def file_name(function_name : str, learning_rate, object, discount_factor = config.DISCOUNT_FACTOR, batch_size = config.BATCH_SIZE):
    date_time = datetime.now()
    dt_string = date_time.strftime("%d-%m_%H-%M")
    return f"{object}\\{function_name}_lr{learning_rate}_gamma{discount_factor}_batch{batch_size}_{dt_string}.csv"

