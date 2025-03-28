import matplotlib.pyplot as plt
from utils.create_legend_label_utils import create_legend_label_list_for_cmp_err_plot, create_legend_label_for_cmp_err_plot
from utils.common.dict_mappings import SAVE_IMAGE_EXT
from utils.plotter.plotter_setting_utils import set_pre_plotter_settings, set_post_plotter_settings, calc_ylim_for_plotter
from utils.plotter.convert_for_plot_utils import convert_metrics_for_plot_ylabel


def cmp_err_plot(df, plot_setting, sim_info, figure_type):
    metrics_list = ["return", "steps"]  # ToDo: metrics_listは外部から取得できるようにする
    summary_statistic_list = ["mean_val_over_epi", "last_val_over_epi"]
    sims = sim_info['commom_info']["total_required_sims"]
    param_name = "err_ratio" # TODO: ハードコーディングしないようにしたい...。
    env = sim_info["each_sim_info_list"][0]["env"]
    envname = env.name

    # 非共通パラメータplot時にER-Qをベースラインとして表示するために必要な変数の初期化
    min_param_val = 10**18
    max_param_val = -10**18

    legend_label_list = create_legend_label_list_for_cmp_err_plot(sim_info)
    # Keyがlabel（algname + bias policy name）、valueがparam_id_dictの辞書を初期化
    param_id_dict_per_label = {label: {} for label in legend_label_list}

    # TODO: 各種ロジックを外部関数に切り出す
    for each_sim_info in sim_info['each_sim_info_list']:
        # param_plot_dicの初期化
        param_plot_dict = {
            "param_val": each_sim_info["er_ratio"]
        }
        for summary_statistic in summary_statistic_list:
            param_plot_dict[summary_statistic] = None # DataFrameから取得したデータを格納するためのKey

        # infoから必要情報を取得
        sim_id = each_sim_info["sim_id"]
        # ラベル名を作成
        label_for_each_sim_info = create_legend_label_for_cmp_err_plot(each_sim_info)

        if label_for_each_sim_info in param_id_dict_per_label:
            param_id_dict_per_label[label_for_each_sim_info][sim_id] = param_plot_dict
        else:
            raise Exception(f"{param_id_dict_per_label}は、param_plotでは現在想定していない値です。アルゴリズム名とバイアスポリシーの設定を確認してください。")
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
                    raise Exception("param plotで使用している代表値は想定していない設定です。") # TODO: 後でカスタム例外作成する
                # 対応するアルゴリズムの辞書にデータを挿入
                for label, param_id_dict in param_id_dict_per_label.items():
                    if sim_id in param_id_dict:
                        param_id_dict[sim_id][summary_statistic] = metrics_value

        # アルゴリズムごとの辞書をパラメータ値でソート
        sorted_param_id_dict_per_label = {}
        for label, param_id_dict in param_id_dict_per_label.items():
            sorted_param_id_dict = dict(sorted(param_id_dict.items(), key=lambda item: item[1]["param_val"]))
            sorted_param_id_dict_per_label[label] = sorted_param_id_dict

        print(f"{sorted_param_id_dict_per_label=}")
        # パラメータ値をリストに変換
        sorted_param_val_list_per_label = {}
        for label, sorted_param_id_dict in sorted_param_id_dict_per_label.items():
            sorted_param_val_list_per_label[label] = [val["param_val"] for val in sorted_param_id_dict.values()]
            # 非共通パラメータplot時にER-Qをベースラインとして表示するために必要な変数の更新
            min_param_val = min(min_param_val, min(sorted_param_val_list_per_label[label]))
            max_param_val = max(max_param_val, max(sorted_param_val_list_per_label[label]))


        # plotするデータ情報の整形
        for summary_statistic in summary_statistic_list:
            # plot前の設定を指定
            set_pre_plotter_settings()

            # plot対象の代表値をlistに変換
            sorted_metrics_val_list_per_label = {}
            for label, sorted_param_id_dict in sorted_param_id_dict_per_label.items():
                sorted_metrics_val_list_per_label[label] = [val[summary_statistic] for val in sorted_param_id_dict.values()]

            # アルゴリズム毎にデータをplot
            for label in legend_label_list:
                sorted_param_val_list = sorted_param_val_list_per_label[label]
                sorted_metrics_val_list = sorted_metrics_val_list_per_label[label]
                if "LSTM" in label:
                    # 共通パラメータの場合は通常通りに表示
                    if param_name in {"alpha", "beta", "gamma", "batch"}: # TODO: 関数に切り出す
                        plt.plot(sorted_param_val_list, sorted_metrics_val_list, marker='o', label="LSTM-Q")
                    # 非共通パラメータの場合は横線で表示
                    else:
                        plt.plot([min_param_val, max_param_val], [sorted_metrics_val_list, sorted_metrics_val_list], '--', label=label)
                else:
                    plt.plot(sorted_param_val_list, sorted_metrics_val_list, marker='o', label=label)

            # plot後の設定を指定
            title = ""
            xlabel = param_name
            ylim_lower, ylim_upper = calc_ylim_for_plotter(figure_type=figure_type, envname=envname, metrics=metrics)
            ylabel_for_plot = convert_metrics_for_plot_ylabel(metrics)
            set_post_plotter_settings(title=title, xlabel=xlabel, ylabel=ylabel_for_plot, ylim_lower=ylim_lower, ylim_upper=ylim_upper)
            plt.legend()
            for ext in SAVE_IMAGE_EXT["ext"]:
                plt.savefig(plot_setting["save_figures_dir"] + f"param_plot_{metrics}-{summary_statistic}.{ext}")

