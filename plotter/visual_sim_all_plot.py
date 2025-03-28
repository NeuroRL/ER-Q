import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from utils.common.dict_mappings import SAVE_IMAGE_EXT

def visual_sim_all_plot(df, plot_setting, sim_info, figure_type):
    # TODO: 外部から指定できるようにする
    n_epi_post_learning_list = [0, 1, 10, 50, 100, 200] # 終端何エピソード分のデータを表示するかを指定。0を指定すると全てのデータが使用される。
    total_required_sims = sim_info["commom_info"]["total_required_sims"]
    env = plot_setting["env"]
    height = env._height
    width = env._width
    grouped = df.groupby('sim_id')

    for n_epi_post_learning in n_epi_post_learning_list:
        env.reset()
        fig, ax, _ = env.render(mode="create_animation")
        count_table = np.zeros((height, width, env.action_space))

        for sim_id, group in grouped:
            state_sequence_post_learning = group["states"][-n_epi_post_learning:]
            action_sequence_post_learning = group["actions"][-n_epi_post_learning:]
            for state_sequence, action_sequence in zip(state_sequence_post_learning, action_sequence_post_learning):
                for state, action in zip(state_sequence, action_sequence):
                    state_index = np.argmax(state)
                    row = state_index // width
                    col = state_index % width
                    count_table[row, col, action] += 1

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

        fig.colorbar(im, ax=ax)
        for ext in SAVE_IMAGE_EXT["ext"]:
            plt.savefig(plot_setting["save_figures_dir"] + f"visual_sim_all-{total_required_sims}_sim-{n_epi_post_learning}_terminal_epis.{ext}")
        plt.clf()
        plt.close(fig)
        env._fig = None
