import re
import matplotlib.pyplot as plt
from utils.plotter.plotter_setting_utils import set_pre_plotter_settings, set_post_plotter_settings
from utils.create_legend_label_utils import (
    create_envsize_legend_label_dict_per_sim_id,
    create_legend_label_list_for_env_plot,
    create_model_name_for_plot,
    create_sim_id_set_dict_per_model_name,
)
from utils.plotter.convert_for_plot_utils import (
    convert_env_name_for_plot,
    convert_metrics_for_plot_ylabel
)
from utils.common.dict_mappings import SAVE_IMAGE_EXT

# envsizeの文字列から数値を抽出する関数
def extract_hw_numbers(envsize_string):
    # 正規表現で H と W の後の数値を抽出
    match = re.match(r"W(\d+)_H(\d+)", envsize_string)
    if match:
        w_val, h_val = map(int, match.groups())
        return w_val, h_val
    return float("inf"), float("inf") # 不正な形式の場合、最後に並べる（TODO: 不正な型式の場合の対処方法は後ほど考える）


def envsize_plot(df, plot_setting, sim_info, figure_type):
    metrics_list = ["return", "steps"]  # TODO: metrics_listは外部から取得できるようにする
    summary_statistic_list = ["mean_val_over_epi"]
    sims = sim_info['commom_info']["total_required_sims"]

    legend_label_list = create_legend_label_list_for_env_plot(sim_info)
    # Keyがlabel（algname + bias policy name）、valueがenvsize_id_dictの辞書を初期化
    envsize_id_dict_per_label = {label: {} for label in legend_label_list}

    # TODO: 各種ロジックを外部関数に切り出す
    for each_sim_info in sim_info['each_sim_info_list']:
        env = each_sim_info["env"]
        envname = env.name
        envname_for_plot = convert_env_name_for_plot(envname=envname)

        # envsize_plot_dicの初期化
        # TODO: 外部の関数に委任すルべきかも...？
        envsize_plot_dict = {
            "env_size": f"W{env._width}_H{env._height}"
        }
        for summary_statistic in summary_statistic_list:
            envsize_plot_dict[summary_statistic] = None # DataFrameから取得したデータを格納するためのKey

        # infoから必要情報を取得
        sim_id = each_sim_info["sim_id"]
        # ラベル名を作成
        label_for_each_sim_info = create_model_name_for_plot(each_sim_info)

        if label_for_each_sim_info in envsize_id_dict_per_label:
            envsize_id_dict_per_label[label_for_each_sim_info][sim_id] = envsize_plot_dict
        else:
            print(f"{label_for_each_sim_info}は、envname_plotでは現在想定していない値です。アルゴリズム名とバイアスポリシーの設定を確認してください。")
    # metrics&代表値毎にplot作成
    for metrics in metrics_list:
        grouped = df.groupby('sim_id')
        for sim_id, group in grouped:
            # メトリクスに対応したデータの取得
            for summary_statistic in summary_statistic_list:
                metrics_value = None
                if summary_statistic == "mean_val_over_epi":
                    metrics_value = group[metrics].mean()
                elif summary_statistic == "last_val_over_epi":
                    metrics_value = group.iloc[-1][metrics]
                else:
                    raise Exception("envsize plotで使用している代表値は想定していない設定です。") # TODO: 後でカスタム例外作成する
                # 対応するアルゴリズムの辞書にデータを挿入
                for label, envsize_id_dict in envsize_id_dict_per_label.items():
                    if sim_id in envsize_id_dict:
                        envsize_id_dict[sim_id][summary_statistic] = metrics_value

        # アルゴリズムごとの辞書をパラメータ値でソート
        sorted_envsize_id_dict_per_label = {}
        for label, envsize_id_dict in envsize_id_dict_per_label.items():
            # H と W の数値でソート
            sorted_envsize_id_dict = dict(
                sorted(
                    envsize_id_dict.items(),
                    key=lambda item: extract_hw_numbers(item[1]["env_size"])
                )
            )
            sorted_envsize_id_dict_per_label[label] = sorted_envsize_id_dict

        # パラメータ値をリストに変換
        sorted_envsize_val_list_per_label = {}
        for label, sorted_envsize_id_dict in sorted_envsize_id_dict_per_label.items():
            sorted_envsize_val_list_per_label[label] = [val["env_size"] for val in sorted_envsize_id_dict.values()]


        # plotするデータ情報の整形
        for summary_statistic in summary_statistic_list:
            # plot設定の指定
            title = ""
            xlabel = "environment size"
            # TODO: 動的に指定できるように修正
            if "return" in metrics:
                ylim_lower = -0.1
                ylim_upper = 2.1
            else:
                ylim_lower = None
                ylim_upper = None
            set_pre_plotter_settings()

            # plot対象の代表値をlistに変換
            sorted_metrics_val_list_per_label = {}
            for label, sorted_envsize_id_dict in sorted_envsize_id_dict_per_label.items():
                sorted_metrics_val_list_per_label[label] = [val[summary_statistic] for val in sorted_envsize_id_dict.values()]

            # アルゴリズム毎にデータをplot
            for label in legend_label_list:
                sorted_envsize_val_list = sorted_envsize_val_list_per_label[label]
                sorted_metrics_val_list = sorted_metrics_val_list_per_label[label]
                plt.plot(sorted_envsize_val_list, sorted_metrics_val_list, marker='o', label=label)
            ylabel_for_plot = convert_metrics_for_plot_ylabel(metrics)
            set_post_plotter_settings(title=title, xlabel=xlabel, ylabel=ylabel_for_plot, ylim_lower=ylim_lower, ylim_upper=ylim_upper)
            plt.legend()
            for ext in SAVE_IMAGE_EXT["ext"]:
                plt.savefig(plot_setting["save_figures_dir"] + f"envsize_plot_{metrics}-{summary_statistic}.{ext}")
            plt.close()


def envsize_model_plot(df, plot_setting, sim_info, figure_type):
    # バイアスポリシー名（fig分離粒度）をkey、sim_idの集合をvalueに持つ辞書の作成
    sim_id_set_dict_per_model_name = create_sim_id_set_dict_per_model_name(sim_info)
    # sim_idをkey、ラベル名をvalueに持つ辞書の作成
    legend_label_dict_per_sim_id = create_envsize_legend_label_dict_per_sim_id(sim_info)

    metrics_list = ["return_mean", "steps_mean"]  # ToDo: metrics_listは外部から取得できるようにする
    sims = sim_info['commom_info']["total_required_sims"]

    for metrics in metrics_list:
        grouped = df.groupby('sim_id')

        for model_name, sim_id_set in sim_id_set_dict_per_model_name.items():
            # plot設定の指定
            title = ""
            xlabel = 'Episodes'
            # TODO: 動的に指定できるように修正
            if "return" in metrics:
                ylim_lower = -0.1
                ylim_upper = 2.1
            else:
                ylim_lower = None
                ylim_upper = None
            set_pre_plotter_settings()
            # ラベルでソートするために一旦"key: label名、value: sim_id" の辞書に変換
            # NOTE:
            #   modelでグループ化されていない状況でこれをやるとlabelが衝突する。
            #   しかし、modelでグループ化れさていればlabelは一意に定まるので問題ない
            legend_dict_divided_by_model = {legend_label_dict_per_sim_id[sim_id]: sim_id for sim_id in sim_id_set}
            sorted_legend_dict_divided_by_model = dict(
                sorted(legend_dict_divided_by_model.items(),
                key=lambda item: extract_hw_numbers(item[0]))
            )
            for legend_label, sim_id in sorted_legend_dict_divided_by_model.items():
                # idを指定してグループ化されたデータを取得
                group = grouped.get_group(sim_id)
                plt.plot(group.index, group[metrics], label=legend_label)
            ylabel_for_plot = convert_metrics_for_plot_ylabel(metrics)
            set_post_plotter_settings(title=title, xlabel=xlabel, ylabel=ylabel_for_plot, ylim_lower=ylim_lower, ylim_upper=ylim_upper)
            plt.legend()
            for ext in SAVE_IMAGE_EXT["ext"]:
                plt.savefig(plot_setting["save_figures_dir"] + f"model_plot_{metrics}-{model_name}.{ext}")
            plt.close()
