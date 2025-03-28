import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from utils.common.dict_mappings import SAVE_IMAGE_EXT

def visual_task_plot(df, plot_setting, sim_info, figure_type):
    env = plot_setting["env"]
    env.reset()
    fig, ax, _ = env.render(mode="create_animation")
    height = env._height
    width = env._width
    count_table = np.zeros((height, width, env.action_space))

    for state, action in zip(df["states"], df["actions"]):
        state_index = np.argmax(state)
        row = state_index // width
        col = state_index % width
        count_table[row, col, action] += 1

    height = env._height
    width = env._width

    st_x, st_y = env.available_positions
    hx_x, hx_y = env.renderer_pos(st_x, st_y)
    xs = hx_x.repeat(env.action_space)
    ys = hx_y.repeat(env.action_space)

    w = 0.4
    aus, avs = env.available_actions
    aus = np.tile(aus * w, height * width)
    avs = np.tile(avs * w, height * width)

    im = ax.quiver(
        xs, ys, aus, avs, count_table, cmap="Reds",
        scale=1, angles="xy", scale_units="xy",
        width=0.005, headwidth=3, headaxislength=7,
        norm=Normalize(vmin=0, vmax=np.max(count_table))
    )

    ax.set_aspect("equal")
    ax.set_xlim(env.xlim)
    ax.set_ylim(env.ylim)

    fig.colorbar(im, ax=ax, orientation='vertical')
    for ext in SAVE_IMAGE_EXT["ext"]:
        plt.savefig(plot_setting["save_figures_dir"] + f"visual_task.{ext}")
    plt.clf()
    plt.close(fig)
    env._fig = None
