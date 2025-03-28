import matplotlib.pyplot as plt
from utils.create_legend_label_utils import create_sim_id_set_dict_per_model_name, create_param_plot_legend_label_dict_per_sim_id
from utils.plotter.plotter_setting_utils import set_pre_plotter_settings, set_post_plotter_settings, calc_ylim_for_plotter
from utils.plotter.convert_for_plot_utils import convert_metrics_for_plot_ylabel
from utils.common.dict_mappings import SAVE_IMAGE_EXT

def model_plot(df, plot_setting, sim_info, figure_type):
    # バイアスポリシー名（fig分離粒度）をkey、sim_idをの集合をvalueに持つ辞書の作成
    sim_id_set_dict_per_model_name = create_sim_id_set_dict_per_model_name(sim_info)
    # sim_idをkey、ラベル名をvalueに持つ辞書の作成
    legend_label_dict_per_sim_id = create_param_plot_legend_label_dict_per_sim_id(sim_info)

    metrics_list = ["return", "steps", "return_mean", "steps_mean"]  # ToDo: metrics_listは外部から取得できるようにする
    sims = sim_info['commom_info']["total_required_sims"]
    # TODO: each_sim_infoから取得するのはかなりよくない（0番目を参照することで意図しないバイアスがかかる可能性がある）ので、何か別の方法を模索する
    # sim_infoに複数のenv typeが指定されないという前提が守られているのであれば問題ない気はするが、これは守られていなさそう
    env = sim_info["each_sim_info_list"][0]["env"]
    envname = env.name

    for metrics in metrics_list:
        grouped = df.groupby('sim_id')

        for model_name, sim_id_set in sim_id_set_dict_per_model_name.items():
            # plot設定の指定
            title = ""
            xlabel='Episodes'
            ylim_lower, ylim_upper = calc_ylim_for_plotter(figure_type=figure_type, envname=envname, metrics=metrics)
            set_pre_plotter_settings()
            for sim_id in sim_id_set:
                # idを指定してグループ化されたデータを取得
                group = grouped.get_group(sim_id)
                legend_label = legend_label_dict_per_sim_id[sim_id]
                plt.plot(group.index, group[metrics], label=legend_label)
            ylabel_for_plot = convert_metrics_for_plot_ylabel(metrics)
            set_post_plotter_settings(title=title, xlabel=xlabel, ylabel=ylabel_for_plot, ylim_lower=ylim_lower, ylim_upper=ylim_upper)
            plt.legend()
            for ext in SAVE_IMAGE_EXT["ext"]:
                plt.savefig(plot_setting["save_figures_dir"] + f"model_plot_{metrics}-{model_name}.{ext}")

