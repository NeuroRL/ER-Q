import matplotlib.pyplot as plt
from utils.create_legend_label_utils import create_interval_plot_legend_label_dict_per_sim_id
from utils.plotter.plotter_setting_utils import set_pre_plotter_settings, set_post_plotter_settings
from utils.common.dict_mappings import SAVE_IMAGE_EXT

def terminal_return_hist_plot(df, plot_setting, sim_info, figure_type):
    # sim_idをkey、ラベル名をvalueに持つ辞書の作成
    legend_label_dict_per_sim_id = create_interval_plot_legend_label_dict_per_sim_id(sim_info)

    metrics_list = ["return"]  # TODO: metrics_listは外部から取得できるようにする
    sims = sim_info['commom_info']["total_required_sims"]

    for metrics in metrics_list:
        # plot設定の指定
        title = ""
        xlabel = metrics
        ylabel = "freqency"
        ylim_lower = None
        ylim_upper = None
        set_pre_plotter_settings()

        sim_id_grouped = df.groupby('sim_id')
        for sim_id, sim_id_group in sim_id_grouped:
            each_sim_id_grouped = sim_id_group.groupby('each_sim_id')

            legend_label = legend_label_dict_per_sim_id[sim_id]

            sim_id_group_data = []
            for each_sim_id, each_sim_id_group in each_sim_id_grouped:
                #  獲得報酬が0以上、すなわち全てのデータを抽出
                non_zero_returns = each_sim_id_group.loc[each_sim_id_group[metrics] > 0.0, metrics]

                # すべての非ゼロ `return` 値をリストに追加
                sim_id_group_data += non_zero_returns.tolist()

            # `return` 値のヒストグラムをプロット
            plt.hist(sim_id_group_data, bins=60, edgecolor='black', alpha=0.3, label=legend_label)

        set_post_plotter_settings(title=title, xlabel=xlabel, ylabel=ylabel, ylim_lower=ylim_lower, ylim_upper=ylim_upper)
        # 余白を調整して図の中心を右寄せ
        plt.subplots_adjust(left=0.2, right=0.9)  # 左余白を広くして右寄せ
        plt.legend()
        for ext in SAVE_IMAGE_EXT["ext"]:
            plt.savefig(plot_setting["save_figures_dir"] + f"return_histogram_{metrics}.{ext}")
        plt.close()
