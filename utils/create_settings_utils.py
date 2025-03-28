import os
import copy
from collections import defaultdict, namedtuple

from utils.common.create_path_utils import (
    create_model_param_path,
    create_data_path
)
from utils.common.custom_exception_utils import InvalidJSONConfigException
from utils.common.dict_mappings import PLOTNAME_TO_ALGNAME_DICT


def generate_df_and_plot_settings_for_metrics_figures(param_env_alg_dict_list, is_animdata, is_paper_fig, info, figure_type):
    """
    param_env_alg_dict_list: 各シミュレーションの設定辞書のリスト
    is_animdata: アニメーション用データか否か
    is_paper_fig: 論文用図として出力するか否か
    info: plot作成に必要なその他の情報。例えば、"total_required_sims", "cmp_param_names", "cmp_env_kinds_set" など。
    figure_type: 作成する図の種類。下記のいずれかを指定
        - "baseline_figures"
        - "envsize_figures" or "envsize_model_figures"
        - "param_figures" or "model_figures"
        - "steps_limit_figures" or "steps_limit_model_figures"
        - "interval_figures" or "terminal_return_figures"
    
    戻り値:
        df_setting_list, plot_setting_list, sim_info_list
    """

    # --- 入力設定のバリデーション ---
    # TODO: バリデーションを外部関数に切り出す
    if figure_type in {"envsize_figures", "envsize_model_figures"}:
        # 環境比較図の場合：cmp_paramの指定は不要、かつ比較する環境は1種類のみとする
        if len(info.get("cmp_param_names", [])) > 0:
            raise ValueError("settings: cmp_param_names is not allowed in envsize_figures or envsize_model_figures.")
        if len(info.get("cmp_env_kinds_set", [])) != 1:
            raise ValueError("settings: exactly one cmp_env_kind is required in envsize_figures or envsize_model_figures.")
        cmp_env_kind = list(info["cmp_env_kinds_set"])[0]
    elif figure_type in {"param_figures", "model_figures"}:
        # パラメータ比較図の場合：cmp_paramは必ず1行指定
        if len(info.get("cmp_param_names", [])) != 1:
            raise ValueError("settings: exactly one cmp_param_names is required in param_figures.")
        cmp_paramname = info["cmp_param_names"][0]
    elif figure_type in {"steps_limit_figures", "steps_limit_model_figures"}:
        # steps_limit比較図の場合：cmp_paramは指定せず、環境は1種類のみ
        if len(info.get("cmp_param_names", [])) > 0:
            raise ValueError("settings: cmp_param_names is not allowed in steps_limit_figures or steps_limit_model_figures.")
        if len(info.get("cmp_env_kinds_set", [])) > 1:
            raise ValueError("settings: only one cmp_env_kind is allowed in steps_limit_figures or steps_limit_model_figures.")
        # cmp_env_kindを取得（存在しない場合もあるので注意）
        cmp_env_kind = list(info.get("cmp_env_kinds_set", []))[0] if info.get("cmp_env_kinds_set", []) else None
    elif figure_type in {"interval_figures", "terminal_return_figures"}:
        # interval比較図の場合：cmp_paramは指定せず、cmp_env_kindは使わない（またはNoneとする）
        if len(info.get("cmp_param_names", [])) > 0:
            raise ValueError("settings: cmp_param_names is not allowed in interval_figures or terminal_return_figures.")
        if len(info.get("cmp_env_kinds_set", [])) > 1:
            raise ValueError("settings: only one cmp_env_kind is allowed in interval_figures or terminal_return_figures.")
    elif figure_type in {"cmp_err_figures"}:
        # err（er_ratio）を横軸にとるfigの場合、each_sim_infoにあるerr情報を使用するため、cmp_paramが設定されている場合は設定過剰なので例外を投げる
        if len(info.get("cmp_param_names", [])) > 0:
            raise ValueError("settings: cmp_param_names is not allowed in cmp_err_figures")
    elif figure_type == "baseline_figures":
        # baselineの場合、特にバリデーションは無し
        pass
    else:
        raise ValueError(f"Unknown figure_type: {figure_type}")

    # --- 各種設定をグルーピングするための辞書 ---
    df_setting_list = []
    plot_setting_list = []
    sim_info_list = []

    grouping_dict_with_save_path = defaultdict(list)
    grouping_dict_with_sim_id_list = defaultdict(list)
    each_sim_info_per_sim_id_dict = {}

    # --- シミュレーション毎のループ ---
    for sim_id, param_env_alg_dict in enumerate(param_env_alg_dict_list):
        param_dict = param_env_alg_dict["param_dict"]
        env_namedtuple = param_env_alg_dict["env_namedtuple"]
        alg_name = param_env_alg_dict["alg_name"]

        # バイアスポリシー設定
        bias_policy_setting_dict = {
            "scale": param_dict["max_bias_policy_weight"],
            "variance": param_dict["var_bias_policy_weight"],
        }

        # err設定
        er_ratio = param_dict["er_ratio"]

        # 環境情報の展開
        env = env_namedtuple.env
        epis = env_namedtuple.epis

        # 図作成ごとに必要な情報の設定
        each_sim_info = {
            "sim_id": sim_id,
            "alg_name": alg_name,
            "env": env,
            "bias_policy_setting_dict": bias_policy_setting_dict,
            "er_ratio": er_ratio
        }
        # cmp_paramval_listの設定はfigure_type毎に異なる
        if figure_type in {"param_figures", "model_figures"}:
            each_sim_info["cmp_paramval_list"] = [param_dict[cmp_paramname]]
        else:
            each_sim_info["cmp_paramval_list"] = None

        # 各シミュレーションの情報を辞書に格納
        each_sim_info_per_sim_id_dict[sim_id] = each_sim_info

        # データの保存先（読み込みパス）の作成
        data_dir, fname = create_data_path(env, epis, param_dict, alg_name, is_animdata, is_save=False)
        reg_exp_data_path = f"./data/{data_dir}/{fname}"

        # 論文用の場合のプレフィックス調整
        save_figures_dir_prefix = "./img" if not is_paper_fig else "./paper_img"
        save_csv_dir_prefix = "./csv" if not is_paper_fig else "./paper_csv"

        # figure_type毎に保存先パスのディレクトリ名を設定
        # TODO: 外部関数化＆できそうならパラメータ化
        if figure_type == "baseline_figures":
            envname = f"{env.name}"
            save_figures_dir = f"{save_figures_dir_prefix}/baseline/{envname}/epi{epis}/"
            save_csv_dir = f"{save_csv_dir_prefix}/baseline/{envname}/epi{epis}/"
        elif figure_type in {"envsize_figures", "envsize_model_figures"}:
            envkind = f"{env.kind}"
            save_figures_dir = f"{save_figures_dir_prefix}/env/epi{epis}/{envkind}/"
            save_csv_dir = f"{save_csv_dir_prefix}/env/epi{epis}/{envkind}/"
        elif figure_type in {"param_figures", "model_figures"}:
            envname = f"{env.name}"
            save_figures_dir = f"{save_figures_dir_prefix}/param/{envname}/epi{epis}/{cmp_paramname}/"
            save_csv_dir = f"{save_csv_dir_prefix}/param/{envname}/epi{epis}/{cmp_paramname}/"
        elif figure_type in {"steps_limit_figures", "steps_limit_model_figures"}:
            envkind = f"{env.kind}"
            save_figures_dir = f"{save_figures_dir_prefix}/steps_limit/epi{epis}/{envkind}/"
            save_csv_dir = f"{save_csv_dir_prefix}/steps_limit/epi{epis}/{envkind}/"
        elif figure_type in {"interval_figures", "terminal_return_figures"}:
            envname = f"{env.name}"
            save_figures_dir = f"{save_figures_dir_prefix}/interval/epi{epis}/{envname}/"
            save_csv_dir = f"{save_csv_dir_prefix}/interval/epi{epis}/{envname}/"
        elif figure_type in {"cmp_err_figures"}:
            envname = f"{env.name}"
            save_figures_dir = f"{save_figures_dir_prefix}/param/{envname}/epi{epis}/err/"
            save_csv_dir = f"{save_csv_dir_prefix}/param/{envname}/epi{epis}/err/"
        else:
            # ここには来ないはず
            raise ValueError(f"Unhandled figure_type: {figure_type}")

        # グルーピング用のキーとして保存先ディレクトリを利用
        grouping_dict_with_save_path[save_figures_dir].append(reg_exp_data_path)
        grouping_dict_with_sim_id_list[save_figures_dir].append(sim_id)

    # --- グループごとにdf_setting, plot_setting, sim_infoを構築 ---
    for save_figures_dir, reg_exp_data_path_list in grouping_dict_with_save_path.items():
        sim_info = {
            "commom_info": {},
            "each_sim_info_list": []
        }
        df_setting = {}
        plot_setting = {}

        # 各種共通情報の設定：figure_type毎に異なる情報を付与
        if figure_type in {"param_figures", "model_figures"}:
            sim_info["commom_info"]["cmp_paramname_list"] = [cmp_paramname]
        else:
            sim_info["commom_info"]["cmp_paramname_list"] = None

        if figure_type in {"envsize_figures", "envsize_model_figures", "steps_limit_figures", "steps_limit_model_figures"}:
            sim_info["commom_info"]["cmp_env_kind"] = [cmp_env_kind]
        elif figure_type in {"interval_figures", "terminal_return_figures"}:
            sim_info["commom_info"]["cmp_env_kind"] = None

        sim_info["commom_info"]["total_required_sims"] = info["total_required_sims"]

        # 各グループに含まれるsimの情報をまとめる
        needed_sim_id_list_grouped_by_save_dir = grouping_dict_with_sim_id_list[save_figures_dir]
        sim_info["each_sim_info_list"] = [each_sim_info_per_sim_id_dict[sim_id] for sim_id in needed_sim_id_list_grouped_by_save_dir]

        # df_setting, plot_settingの各種情報
        df_setting["reg_exp_data_path_list"] = reg_exp_data_path_list
        plot_setting["save_figures_dir"] = save_figures_dir
        plot_setting["save_csv_dir"] = save_csv_dir

        df_setting_list.append(df_setting)
        plot_setting_list.append(plot_setting)
        sim_info_list.append(sim_info)

    return df_setting_list, plot_setting_list, sim_info_list


### max_bias_policy_weight & var_bias_policy_weight figures用の関数
# 直積情報から、df_settings と plot_settings (save_path, data_path, other_info) を作成する関数
# TODO: each_sim_infoを使用するように修正
def generate_df_and_plot_settings_for_visual_figures(param_env_alg_dict_list, is_animdata, is_paper_fig, info, figure_type):
    df_setting_list = []
    plot_setting_list = []
    sim_info_list = []

    # ToDo: dfに各種情報ぶち込めばsim_info必要なくなるはずなので、時間があったら考慮する
    # param plotするために必要な辞書(paramとかが入ってる)である sim_info を まとめる辞書
    # sim_info_list: jsonレベルの階層
    # sim_info: 一枚のplot作成に必要な情報
    # each_sim_info: 一枚のplotで必要な各種実験設定の情報(visual_figureでは使用しないので空のlist))
    sim_info = {
        "commom_info": {},
        "each_sim_info_list": []
    }
    sim_info["commom_info"]["total_required_sims"] = info["total_required_sims"]

    # 直積の param設定, env設定, alg_name を一つずつ展開
    for param_env_alg_dict in param_env_alg_dict_list:
        param_dict = param_env_alg_dict["param_dict"]
        env_namedtuple = param_env_alg_dict["env_namedtuple"]
        alg_name = param_env_alg_dict["alg_name"]

        # envの設定を展開
        env = env_namedtuple.env
        epis = env_namedtuple.epis
        envname = f"{env.name}"

        df_setting = {}
        plot_setting = {}
        
        # PATHの作成
        data_dir, fname = create_data_path(env, epis, param_dict, alg_name, is_animdata, is_save=False)

        # df_setting の作成 (data_path)
        # TODO: 先頭のベタ書き直す
        # collectorのsaveで'./data/'の部分がベタが記されており、その影響がここに派生している
        # collectorから該当記述をなくすだけで良さそうなのでそんなに大変じゃなさそう？
        reg_exp_data_path = f"./data/{data_dir}/{fname}"
        df_setting["reg_exp_data_path_list"] = [reg_exp_data_path]
        

        # 論文用figの場合は保存場所を変更する
        save_figures_dir_prefix = "./img"
        if is_paper_fig:
            save_figures_dir_prefix = "./paper_img"
        # 論文用csvの場合は保存場所を変更する
        save_csv_dir_prefix = "./csv"
        if is_paper_fig:
            save_csv_dir_prefix = "./paper_csv"

        # plot_setting の作成 (save_path)
        model_param = create_model_param_path(param_dict, alg_name)
        save_figures_dir = f"{save_figures_dir_prefix}/visual/{envname}/epi{epis}/{model_param}/"
        save_csv_dir = f"{save_csv_dir_prefix}/visual/{envname}/epi{epis}/{model_param}/"
        plot_setting["save_figures_dir"] = save_figures_dir
        plot_setting["save_csv_dir"] = save_csv_dir
        plot_setting["env"] = env

        # ディレクトリがなければ作成
        os.makedirs(save_figures_dir, exist_ok=True)
        os.makedirs(save_csv_dir, exist_ok=True)

        df_setting_list.append(df_setting)
        plot_setting_list.append(plot_setting)
        sim_info_list.append(sim_info)

    return df_setting_list, plot_setting_list, sim_info_list


"""
    その他
"""
# paramをdictからnamedtupleに変換する関数
def param_converter_dict2namedtuple(param_dict, alg_name):
    # namedtupleクラスの作成
    lstm_param_keys = ["alpha", "beta", "gamma", "batch"]
    er_param_keys = list(param_dict.keys())
    LSTMParamSettings = namedtuple("LSTMParamSettings", lstm_param_keys)
    ERParamSettings = namedtuple("ERParamSettings", er_param_keys)

    if alg_name == "LSTMQnet":
        ## LSTM-namedtupleインスタンスの作成
        param_namedtuple = LSTMParamSettings(
            alpha=param_dict["alpha"],
            beta=param_dict["beta"], 
            gamma=param_dict["gamma"], 
            batch=param_dict["batch"]
        )
    else:
        ## ER-namedtupleインスタンスの作成
        param_namedtuple = ERParamSettings(
            alpha=param_dict["alpha"],
            beta=param_dict["beta"],
            gamma=param_dict["gamma"],
            buffer_size=param_dict["buffer_size"],
            er_ratio = param_dict["er_ratio"],
            er_deactivate_epis = param_dict["er_deactivate_epis"],
            retrieval_size=param_dict["retrieval_size"],
            trajectory_length=param_dict["trajectory_length"],
            bias_definition=param_dict["bias_definition"],
            bias_decay_rate=param_dict["bias_decay_rate"],
            bias_var_power=param_dict["bias_var_power"],
            max_bias_policy_weight=param_dict["max_bias_policy_weight"],
            var_bias_policy_weight=param_dict["var_bias_policy_weight"],
            batch=param_dict["batch"]
        )
    
    return param_namedtuple

# 直積処理の関係等で、param_dictの値が正常か判定する関数 (現状では bpw が足して1になるか？くらいしか見るものはなさそうだが...)
def is_correct_param_dict(param_dict):
    # Validateに必要なパラメータの展開
    max_bias_policy_weight = param_dict["max_bias_policy_weight"]
    var_bias_policy_weight = param_dict["var_bias_policy_weight"]

    # Validate: "bpw" が足して1になるか？
    if max_bias_policy_weight + var_bias_policy_weight != 1:
        return False
    
    return True

# 不適切な環境設定を検出するための関数
def is_correct_env_namedtuple(env_namedtuple):
    env = env_namedtuple.env
    
    envname = env.name
    width = env._width
    height = env._height
    
    # TODO: case分に変更
    if "SubOptima" in envname:
        # 正方形かどうか
        if width != height:
            return False
        # 理論上の最小値。報酬源の間は3マスで固定する想定
        if width < 7 and height < 7:
            return False
    elif "HiddenExploreHex" in envname:
        # 正方形かどうか
        if width != height:
            return False
        # 理論上の最小値。とりあえず負の数にならなければいいことにする。要件にあわせて必要があれば変更する
        if width < 1 and height < 1:
            return False
    elif "FlaggedExtendedTmazeHidden" in envname:
        # TODO: 環境の形状に対する条件はこれ以外にもあり得るので、必要が発生したら変更する
        # T字の細くなっている部分が1マスになっているか？
        if width != height+2:
            return False
        # TODO: 理論上の最小値を設定する 
    else:
        raise InvalidJSONConfigException("'envs'の設定に問題がないことを確認してください。")
    
    return True
