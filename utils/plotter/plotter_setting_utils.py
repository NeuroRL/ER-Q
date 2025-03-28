import matplotlib.pyplot as plt

from utils.plotter.env_info_utils import calc_theorematical_min_goas_steps

def set_pre_plotter_settings():
    plt.rcParams['font.size'] = 24

    plt.figure(figsize=(10, 7))

def set_post_plotter_settings(title, xlabel, ylabel, ylim_lower, ylim_upper):
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.ylim(ylim_lower, ylim_upper)


# 環境のタイプやサイズに合わせてylimを細かく指定するための関数。（Noneの場合はデータに合うように指定される）
def calc_ylim_for_plotter(figure_type, envname, metrics=None):
    visual_type_set = {"visual_figures", "visual_task_figures", "visual_sim_each_figures", "visual_sim_all_figures"}
    param_and_model_type_set = {"baseline_figures", "cmp_err_figures", "envsize_figures", "envsize_model_figures", "param_figures", "model_figures", "steps_limit_figures", "steps_limit_model_figures"}

    if figure_type in visual_type_set:
        # visual系ではylimは特に関係ないのでNoneに指定する
        ylim_lower = None
        ylim_upper = None
    elif figure_type in param_and_model_type_set:
        # param, model系ではmetricsや環境サイズによって指定値が変わるのでハンドリングする
        if metrics in {"return", "return_mean"}:
            ylim_lower = -0.1
            ylim_upper = 2.1
        elif metrics in {"steps", "steps_mean"}:
            # 理論最小ステップ数は環境の種類とサイズ数によって変化する
            theorematical_min_goas_steps = calc_theorematical_min_goas_steps(envname)

            # ジャストで指定すると見辛くなる為、微小量を加算
            plot_epsilon = 1.0 # TODO: 微小量もスケールに合わせて動的に調整できるようにしたい
            ylim_lower = theorematical_min_goas_steps - plot_epsilon
            ylim_upper = None
        else:
            raise ValueError("想定されていないメトリクスが指定されています。設定を確認してください。")
    else:
        raise ValueError("想定していないfigure_typeです。plotterのJSON設定を確認してください。")
    return ylim_lower, ylim_upper
