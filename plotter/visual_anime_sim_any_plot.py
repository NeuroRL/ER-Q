import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from tqdm import tqdm

from utils.common.dict_mappings import VISUAL_EPISODE_INDEX_RANGE_DICT


def visual_anime_sim_any_plot(df, plot_setting, sim_info, figure_type):
    env_kind = plot_setting["env"].kind
    bias_table_epi_indices_range = VISUAL_EPISODE_INDEX_RANGE_DICT[env_kind]
    bias_table_epi_indices = []
    for start_epi, end_epi in bias_table_epi_indices_range:
        bias_table_epi_indices.extend(range(start_epi, end_epi + 1))
    grouped = df.groupby('sim_id')

    # 動画作成の準備
    env = plot_setting["env"]
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

    v_width = env._render_width
    v_height = env._render_height
    fourcc = cv2.VideoWriter_fourcc(*"H264")
    out = cv2.VideoWriter(f"{plot_setting["save_figures_dir"]}/create_animation/sim_animation.mp4", fourcc, 10.0, (v_width, v_height))

    for sim_id, group in grouped:
        for bias_table_epi_index in tqdm(bias_table_epi_indices):
            bias_table_epi = group['bias_table'].iloc[bias_table_epi_index-1]
            state_sequence = group['states'].iloc[bias_table_epi_index-1]
            hidden = group['hidden'].iloc[bias_table_epi_index-1]

            # T-mazeの場合は隠れ状態を指定してリセットを行う
            if hidden is None:
                env.reset()
            else:
                env.reset(hidden=hidden)

            if len(bias_table_epi) == 0:  # LSTMQnetの場合はcontinue
                continue

            # 最終step時にbias_tableを計算しないため、動画出力時は1step前のbias_tableを用いて最終stepを出力
            bias_table_epi.append(bias_table_epi[-1])

            for index_step in range(len(bias_table_epi)):
                bias_table = bias_table_epi[index_step]

                fig, ax, agt = env.render(mode="create_animation")
                agt.set_color("b")

                im = ax.quiver(
                    xs, ys, aus, avs, bias_table, cmap="Greens",
                    scale=1, angles="xy", scale_units="xy",
                    width=0.005, headwidth=3, headaxislength=7,
                    norm=Normalize(vmin=0, vmax=np.max(bias_table) if np.max(bias_table) > 0 else 1)
                )

                ax.set_aspect("equal")
                ax.set_xlim(env.xlim)
                ax.set_ylim(env.ylim)
                ax.set_title(f"episode: {bias_table_epi_index}, step: {index_step}", fontsize=20)

                cbar = fig.colorbar(im, ax=ax)
                cbar.ax.tick_params(labelsize=15)

                current_state_step = state_sequence[index_step]
                current_state_index = np.argmax(current_state_step)
                agt_x = current_state_index // width
                agt_y = current_state_index % width
                x, y = env.renderer_pos(*(agt_y, agt_x))

                agt.set_offsets([x, y])
                ax.draw_artist(agt)
                ax.draw_artist(im)
                fig.canvas.draw()
                fig.canvas.flush_events()

                im = np.frombuffer(fig.canvas.renderer.tostring_rgb(), dtype=np.uint8).reshape((v_height, v_width, 3))
                im = cv2.cvtColor(im, cv2.COLOR_RGB2BGR)
                out.write(im)

                plt.clf()
                plt.close(fig)
                env._fig = None

    out.release()
    plt.clf()
    plt.close()
