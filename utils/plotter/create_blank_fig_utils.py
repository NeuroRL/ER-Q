import matplotlib.pyplot as plt
import os

def create_blank_fig(save_dir):
    os.makedirs(save_dir, exist_ok=True)

    plt.figure()
    plt.savefig(f"{save_dir}/blank.png")