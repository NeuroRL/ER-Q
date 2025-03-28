import matplotlib.pyplot as plt
from utils.create_legend_label_utils import create_baseline_plot_legend_label_dict_per_sim_id
from utils.common.dict_mappings import SAVE_IMAGE_EXT
from utils.plotter.plotter_setting_utils import (
    calc_ylim_for_plotter,
    set_pre_plotter_settings,
    set_post_plotter_settings,
)
from utils.plotter.convert_for_plot_utils import convert_metrics_for_plot_ylabel

def baseline_plot(df, plot_setting, sim_info, figure_type):
    # TODO: each_sim_infoから取得するのはかなりよくない（0番目を参照することで意図しないバイアスがかかる可能性がある）ので、何か別の方法を模索する
    # sim_infoに複数のenv typeが指定されないという前提が守られているのであれば問題ない気はするが、これは守られていなさそう
    env = sim_info["each_sim_info_list"][0]["env"]
    envname = env.name

    legend_label_dict_per_sim_id = create_baseline_plot_legend_label_dict_per_sim_id(sim_info)

    metrics_list = ["return", "steps", "return_mean", "steps_mean"]  # ToDo: metrics_listは外部から取得できるようにする
    sims = sim_info['commom_info']["total_required_sims"]

    for metrics in metrics_list:
        grouped = df.groupby('sim_id')
        # plot前の設定を指定
        set_pre_plotter_settings()

        for sim_id, group in grouped:
            plt.plot(group.index, group[metrics], label=legend_label_dict_per_sim_id[sim_id])

        # plot後の設定を指定
        ylim_lower, ylim_upper = calc_ylim_for_plotter(figure_type=figure_type, envname=envname, metrics=metrics)
        ylabel_for_plot = convert_metrics_for_plot_ylabel(metrics)
        set_post_plotter_settings(title="", xlabel='Episodes', ylabel=ylabel_for_plot, ylim_lower=ylim_lower, ylim_upper=ylim_upper)
        plt.legend()
        for ext in SAVE_IMAGE_EXT["ext"]:
            plt.savefig(plot_setting["save_figures_dir"] + f"baseline_plot_{metrics}.{ext}")
