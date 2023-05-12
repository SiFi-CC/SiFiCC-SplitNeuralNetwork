import numpy as np


def match_sipm(list_sipm_positions):
    for i in range(len(list_sipm_positions)):
        print(np.sqrt((list_sipm_positions[0].x - list_sipm_positions[i].x) ** 2 + (
                    list_sipm_positions[0].x - list_sipm_positions[i].x) ** 2))

    return
