from utils.common.create_setting_from_json_utils import (
    load_default_dict_and_cmp_dict,
    create_intermediate_reprosentation_from_json
)
from utils.common.convert_json_to_dict_utils import (
    create_load_path_dict_from_json
)
from utils.plotter.data_creater_for_plot import (
    create_df_for_metrics_kind_figures,
    create_df_for_visual_kind_figures
)
from utils.create_settings_utils import (
    generate_df_and_plot_settings_for_visual_figures,
    generate_df_and_plot_settings_for_metrics_figures,
)
from plotter.cmp_err_plot import cmp_err_plot
from plotter.envsize_plot import envsize_plot, envsize_model_plot
from plotter.model_plot import model_plot
from plotter.param_plot import param_plot
from plotter.baseline_plot import baseline_plot
from plotter.interval_plot import interval_plot
from plotter.terminal_return_hist_plot import terminal_return_hist_plot
from plotter.steps_limit_plot import steps_limit_plot, steps_limit_model_plot
from plotter.visual_plot import visual_plot
from plotter.visual_task_plot import visual_task_plot
from plotter.visual_sim_each_plot import visual_sim_each_plot
from plotter.visual_sim_all_plot import visual_sim_all_plot
from plotter.visual_anime_sim_any_plot import visual_anime_sim_any_plot
from utils.common.create_path_utils import judge_is_animdata_for_plotter
from utils.plotter.validate_sims_utils import create_remaining_need_sims_dict
from utils.common.custom_exception_utils import InvalidJSONConfigException


# TODO: is_paper_fig を引数ではなく、もっといい感じのところから持ってこれるように修正
def plot_figures(default_config_dict, cmp_config_dict_list, is_paper_fig):
    # main.jsonに指定したplotしたい全てのfigを作成するために不足しているsim情報を抽出
    remaining_need_sims_dict = create_remaining_need_sims_dict(default_config_dict, cmp_config_dict_list)
    
    # 不足しているsimがある場合は実行を終了する
    # TODO: print結果をもう少し見やすくする
    if remaining_need_sims_dict:
        for data_path, remaining_needed_sims in remaining_need_sims_dict.items():
            print("-------------------------")
            print(f"データが足りない実験設定: {data_path}")
            print(f"足りないsimulation数: {remaining_needed_sims}")
        print("============================")
        raise InvalidJSONConfigException(f"{len(remaining_need_sims_dict)}個分の実験設定において、fig作成に必要なデータが不足しているようです。出力内容を確認してください。")
        print("============================")
    else:
        print("============================")
        print("指定したfigを作成するために必要なデータはすべて揃っています。fig作成を開始します。")
        print("============================")
    
        
    
    # main.jsonに複数指定した"cmp"対象をfor_each
    for cmp_config_dict in cmp_config_dict_list:
        # 1枚のplotに必要な設定がまとまっている "df_setting&plot_setting" のList作成する
        # TODO: 一旦共通部分としてmatch文の外に書き出しておく。必要性が生じたら中で個別処理を行う。
        param_env_alg_dict_list, info = create_intermediate_reprosentation_from_json(default_config_dict, cmp_config_dict)

        # アニメーションデータが必要かどうかを判定する
        is_animdata = judge_is_animdata_for_plotter(cmp_config_dict)

        
        figure_type = cmp_config_dict["figures_type"]
        # figure_type に応じた関数をそれぞれ変数に代入
        match figure_type:
            case "baseline_figures":
                generate_df_and_plot_settings = generate_df_and_plot_settings_for_metrics_figures
                create_df = create_df_for_metrics_kind_figures
                plot = baseline_plot
            case "cmp_err_figures":
                generate_df_and_plot_settings = generate_df_and_plot_settings_for_metrics_figures
                create_df = create_df_for_metrics_kind_figures
                plot = cmp_err_plot
            case "envsize_figures":
                generate_df_and_plot_settings = generate_df_and_plot_settings_for_metrics_figures
                create_df = create_df_for_metrics_kind_figures
                plot = envsize_plot
            case "envsize_model_figures":
                generate_df_and_plot_settings = generate_df_and_plot_settings_for_metrics_figures
                create_df = create_df_for_metrics_kind_figures
                plot = envsize_model_plot
            case "param_figures":
                generate_df_and_plot_settings = generate_df_and_plot_settings_for_metrics_figures
                create_df = create_df_for_metrics_kind_figures
                plot = param_plot
            case "model_figures":
                generate_df_and_plot_settings = generate_df_and_plot_settings_for_metrics_figures
                create_df = create_df_for_metrics_kind_figures
                plot = model_plot
            case "steps_limit_figures":
                generate_df_and_plot_settings = generate_df_and_plot_settings_for_metrics_figures
                create_df = create_df_for_metrics_kind_figures
                plot = steps_limit_plot
            case "steps_limit_model_figures":
                generate_df_and_plot_settings = generate_df_and_plot_settings_for_metrics_figures
                create_df = create_df_for_metrics_kind_figures
                plot = steps_limit_model_plot
            case "interval_figures":
                generate_df_and_plot_settings = generate_df_and_plot_settings_for_metrics_figures
                create_df = create_df_for_metrics_kind_figures
                plot = interval_plot
            case "terminal_return_figures":
                generate_df_and_plot_settings = generate_df_and_plot_settings_for_metrics_figures
                create_df = create_df_for_metrics_kind_figures
                plot = terminal_return_hist_plot
            case "visual_figures":
                generate_df_and_plot_settings = generate_df_and_plot_settings_for_visual_figures
                create_df = create_df_for_visual_kind_figures
                plot = visual_plot
            case "visual_task_figures":
                generate_df_and_plot_settings = generate_df_and_plot_settings_for_visual_figures
                create_df = create_df_for_visual_kind_figures
                plot = visual_task_plot
            case "visual_sim_each_figures":
                generate_df_and_plot_settings = generate_df_and_plot_settings_for_visual_figures
                create_df = create_df_for_visual_kind_figures
                plot = visual_sim_each_plot
            case "visual_sim_all_figures":
                generate_df_and_plot_settings = generate_df_and_plot_settings_for_visual_figures
                create_df = create_df_for_visual_kind_figures
                plot = visual_sim_all_plot
            case "visual_anime_sim_any_figures":
                generate_df_and_plot_settings = generate_df_and_plot_settings_for_visual_figures
                create_df = create_df_for_visual_kind_figures
                plot = visual_anime_sim_any_plot
            # TODO: "figure_type"の指定が異常な際の例外処理の追加
        
        # figure_typeごとに代入した各関数を実行
        df_setting_list, plot_setting_list, sim_info_list = generate_df_and_plot_settings(param_env_alg_dict_list, is_animdata, is_paper_fig, info=info, figure_type=figure_type)
        for df_setting, plot_setting, sim_info in zip(df_setting_list, plot_setting_list, sim_info_list):
            df = create_df(df_setting, plot_setting, sim_info, figure_type=figure_type)
            plot(df, plot_setting, sim_info, figure_type)


if __name__ == "__main__":
    config_dir = "config/plot"
    load_path_json_filename = "main.json"

    # main.jsonからload_path_dictを取得
    load_path_dict = create_load_path_dict_from_json(config_dir=config_dir, json_filename=load_path_json_filename)
    # load_path_dict から default_config_dict&cmp_config_dic を作成
    default_config_dict, cmp_config_dict_list = load_default_dict_and_cmp_dict(config_dir=config_dir, load_path_dict=load_path_dict)
    plot_figures(default_config_dict=default_config_dict, cmp_config_dict_list=cmp_config_dict_list, is_paper_fig=False)
