import numpy as np
import math

# data_listに対してsize個での移動平均を取る
def valid_convolve(data_list, moving_average_window=50):
    data_list = np.array(data_list)
    b = np.ones(moving_average_window) / moving_average_window
    data_list_moving_mean = np.convolve(data_list, b, mode="same")

    n_conv = math.ceil(moving_average_window / 2)

    # 補正部分
    data_list_moving_mean[0] *= moving_average_window / n_conv
    for i in range(1, n_conv):
        data_list_moving_mean[i] *= moving_average_window / (i + n_conv)
        data_list_moving_mean[-i] *= moving_average_window / (i + n_conv - (moving_average_window % 2))
    # size%2は奇数偶数での違いに対応するため

    return data_list_moving_mean
