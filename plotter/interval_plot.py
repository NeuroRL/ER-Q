import matplotlib.pyplot as plt
from utils.create_legend_label_utils import create_interval_plot_legend_label_dict_per_sim_id
from utils.plotter.plotter_setting_utils import set_pre_plotter_settings, set_post_plotter_settings
from utils.common.dict_mappings import SAVE_IMAGE_EXT

def interval_plot(df, plot_setting, sim_info, figure_type):
    # sim_idをkey、ラベル名をvalueに持つ辞書の作成
    legend_label_dict_per_sim_id = create_interval_plot_legend_label_dict_per_sim_id(sim_info)

    metrics_list = ["return"]  # TODO: metrics_listは外部から取得できるようにする
    sims = sim_info['commom_info']["total_required_sims"]

    for metrics in metrics_list:
        # plot設定の指定
        title = ""
        xlabel=f'{metrics} interval'
        ylabel = "freqency"
        ylim_lower = None
        ylim_upper = None
        set_pre_plotter_settings()

        sim_id_grouped = df.groupby('sim_id')
        for sim_id, sim_id_group in sim_id_grouped:
            sim_id_group_data = []
            each_sim_id_grouped = sim_id_group.groupby('each_sim_id')

            legend_label = legend_label_dict_per_sim_id[sim_id]

            for each_sim_id, each_sim_id_group in each_sim_id_grouped:
                #  獲得報酬が0ではないエピソードを抽出
                non_zero_return_episodes = each_sim_id_group.loc[each_sim_id_group[metrics] != 0.0, "episodes"]  # TODO: Suboptimaの報酬スケールによって頻度の設定を変えられるようにする

                # 抽出したエピソードの差（インターバル）を計算
                non_zero_return_episodes_diff = non_zero_return_episodes.diff()
                # NOTE:
                #   先頭データは直前のデータが存在しないためdiffが計算できずにNaNになる。
                #   初期データに対して計算したいdiffはそのデータ自身なのでNanを自分自身で置き換える
                # TODO: 置き換える最小の値が1になるようにする（最初の報酬獲得が0Episode目の場合、そこだけdiffが0になってしまう）（最小値が0ではなく、「自分自身＋１」で埋めるべきなのかもしれない（先頭が0で始まっていることがよくないはずなので））
                #   下記に示すように修正できるはず
                #       non_zero_return_episodes_diff_filled = non_zero_return_episodes_diff.fillna( min(1, non_zero_return_episodes.iloc[0]))
                non_zero_return_episodes_diff_filled = non_zero_return_episodes_diff.fillna(non_zero_return_episodes.iloc[0])

                # 各simの初期30データのみ（全データ取得してしまうとER-Qの後半のインターバルがほとんど１になってしまうため）
                filtered_diff = non_zero_return_episodes_diff_filled[:30]
                intervals_list = filtered_diff.astype(int).tolist()

                sim_id_group_data += intervals_list
            plt.hist(sim_id_group_data, bins=60, edgecolor='black', alpha=0.3, label=legend_label)
        set_post_plotter_settings(title=title, xlabel=xlabel, ylabel=ylabel, ylim_lower=ylim_lower, ylim_upper=ylim_upper)
        plt.legend()
        for ext in SAVE_IMAGE_EXT["ext"]:
            plt.savefig(plot_setting["save_figures_dir"] + f"interval_plot_{metrics}.{ext}")
        plt.close()
