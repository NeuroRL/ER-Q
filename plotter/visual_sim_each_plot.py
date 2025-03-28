import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize

from utils.common.dict_mappings import VISUAL_EPISODE_INDEX_DICT, VISUAL_PERCENT_STEP_DICT, SAVE_IMAGE_EXT
from utils.plotter.env_info_utils import extract_env_type_from_envkind


def visual_sim_each_plot(df, plot_setting, sim_info, figure_type):
    visual_count_sim_each_plot(df, plot_setting, sim_info)
    visual_bias_sim_each_plot(df, plot_setting, sim_info)


def visual_count_sim_each_plot(df, plot_setting, sim_info):
    env_kind = plot_setting["env"].kind
    env_kind = extract_env_type_from_envkind(env_kind)
    count_table_epi_indices = VISUAL_EPISODE_INDEX_DICT[env_kind]
    percent_step_list = VISUAL_PERCENT_STEP_DICT[env_kind]
    grouped = df.groupby('sim_id')

    for sim_id, group in grouped:
        for count_table_epi_index in tqdm(count_table_epi_indices):
            state_sequence = group['states'].iloc[count_table_epi_index-1]
            action_sequence = group['actions'].iloc[count_table_epi_index-1]
            hidden = group['hidden'].iloc[count_table_epi_index-1]

            for percent_step in percent_step_list:
                index_step = int((percent_step / 100) * (len(state_sequence) - 1))
                state_sequence_step = state_sequence[:index_step+1]
                action_sequence_step = action_sequence[:index_step+1]

                env = plot_setting["env"]
                # T-mazeの場合は隠れ状態を指定してリセットを行う
                if hidden is None:
                    env.reset()
                else:
                    env.reset(hidden=hidden)
                fig, ax, agt = env.render(mode="create_animation")
                agt.set_color("b")
                height = env._height
                width = env._width
                count_table = np.zeros((height, width, env.action_space))

                for state, action in zip(state_sequence_step, action_sequence_step):
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
                ax.set_title(f"episode: {count_table_epi_index}", fontsize=20)

                cbar = fig.colorbar(im, ax=ax)
                cbar.ax.tick_params(labelsize=15)

                current_state_step = state_sequence[index_step]
                current_state_index = np.argmax(current_state_step)
                agt_x = current_state_index // width
                agt_y = current_state_index % width
                x, y = env.renderer_pos(*(agt_y, agt_x))

                agt.set_offsets([x, y])
                ax.draw_artist(agt)
                fig.canvas.blit(ax.bbox)
                fig.canvas.flush_events()
                for ext in SAVE_IMAGE_EXT["ext"]:
                    plt.savefig(f"{plot_setting["save_figures_dir"]}/visual_count_table/sim_id{sim_id}_epi{count_table_epi_index}_{percent_step}perstep.{ext}")
                plt.clf()
                plt.close(fig)
                env._fig = None


def visual_bias_sim_each_plot(df, plot_setting, sim_info):
    env_kind = plot_setting["env"].kind
    env_kind = extract_env_type_from_envkind(env_kind)
    bias_table_epi_indices = VISUAL_EPISODE_INDEX_DICT[env_kind]
    percent_step_list = VISUAL_PERCENT_STEP_DICT[env_kind]
    grouped = df.groupby('sim_id')

    for sim_id, group in grouped:
        for bias_table_epi_index in tqdm(bias_table_epi_indices):
            bias_table_epi = group['bias_table'].iloc[bias_table_epi_index-1]
            state_sequence = group['states'].iloc[bias_table_epi_index-1]
            hidden = group['hidden'].iloc[bias_table_epi_index-1]

            if len(bias_table_epi) == 0:  # LSTMQnetの場合はcontinue
                continue

            for percent_step in percent_step_list:
                index_step = int((percent_step / 100) * (len(bias_table_epi) - 1))
                bias_table = bias_table_epi[index_step]

                env = plot_setting["env"]
                # T-mazeの場合は隠れ状態を指定してリセットを行う
                if hidden is None:
                    env.reset()
                else:
                    env.reset(hidden=hidden)
                fig, ax, agt = env.render(mode="create_animation")
                agt.set_color("b")
                height = env._height
                width = env._width

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
                    xs, ys, aus, avs, bias_table, cmap="Greens",
                    scale=1, angles="xy", scale_units="xy",
                    width=0.005, headwidth=3, headaxislength=7,
                    norm=Normalize(vmin=0, vmax=np.max(bias_table))
                )

                ax.set_aspect("equal")
                ax.set_xlim(env.xlim)
                ax.set_ylim(env.ylim)
                ax.set_title(f"episode: {bias_table_epi_index}", fontsize=20)

                cbar = fig.colorbar(im, ax=ax)
                cbar.ax.tick_params(labelsize=15)

                current_state_step = state_sequence[index_step]
                current_state_index = np.argmax(current_state_step)
                agt_x = current_state_index // width
                agt_y = current_state_index % width
                x, y = env.renderer_pos(*(agt_y, agt_x))

                agt.set_offsets([x, y])
                ax.draw_artist(agt)
                fig.canvas.blit(ax.bbox)
                fig.canvas.flush_events()
                for ext in SAVE_IMAGE_EXT["ext"]:
                    plt.savefig(f"{plot_setting["save_figures_dir"]}/visual_bias_table/sim_id{sim_id}_epi{bias_table_epi_index}_{percent_step}perstep.{ext}")
                plt.clf()
                plt.close(fig)
                env._fig = None
