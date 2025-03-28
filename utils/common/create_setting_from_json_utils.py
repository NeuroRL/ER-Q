import json
import copy
import itertools

from utils.common.create_setting_from_cartesian import (
	create_param_setting_dict_list_from_cartesian,
	create_env_setting_dict_list_from_cartesian
)

from utils.common.param_setting_utils import create_ignore_cmp_param_setting_set
from utils.plotter.convert_for_plot_utils import convert_algname_from_json
from utils.create_settings_utils import (
	param_converter_dict2namedtuple,
    is_correct_env_namedtuple,
	is_correct_param_dict
)


# default_config_dict&cmp_config_dicの2つの辞書を返す関数
def load_default_dict_and_cmp_dict(config_dir, load_path_dict):
    #  ToDo: simの場合は"plot_type"毎にdefaultが異なる可能性がある。もしdefault.jsonの内容を分ける必要があったらこの中で処理を分岐させる
    default_json_path = load_path_dict["default"]
    cmp_json_path_list = load_path_dict["cmp"]

    cmp_config_dict_list = []
    cmp_config_dict = {}
    # default設定jsonのload
    with open(f"{config_dir}/{default_json_path}") as default_config_file:
        default_config_dict = json.load(default_config_file)

    # cmp設定jsonのload
    for cmp_json_path in cmp_json_path_list:
        with open(f"{config_dir}/{cmp_json_path}") as cmp_config_file:
            cmp_config_dict = json.load(cmp_config_file)
        cmp_config_dict_list.append(cmp_config_dict)
    return default_config_dict, cmp_config_dict_list

# TODO: json設定値のバリデーションする...？
# settings配下に配置されているjsonベースの実験設定辞書を基に実験のための中間表現 (parameterの直積, envの直積) を作成する関数
def create_intermediate_reprosentation_from_json(default_config_dict, cmp_config_dict):
    info = {} # その他必要な情報を入れるための辞書
    
    # TODO: 可能なら "cmp_config_dict["total_required_sims"]" からではなく、"total_required_sims_dict" を参照するようにする
    info["total_required_sims"] = cmp_config_dict["total_required_sims"]

    ## 中間形式 (各種設定の直積の list) の作成
    cmp_param_dic = {}
    # cmp_param_dic を default設定で初期化
    cmp_param_dic["cmp_envs"] = copy.deepcopy(default_config_dict["envs"])
    cmp_param_dic["cmp_params"] = copy.deepcopy(default_config_dict["params"])

    # cmp_paramsの解凍
    for env_name, env_dic in cmp_config_dict["cmp_envs"].items():
        for env_param_key, env_param_val in env_dic.items():
            if env_param_val:
                # print(env_param_key, env_param_val)
                cmp_param_dic["cmp_envs"][env_name][env_param_key] = env_param_val

    info["cmp_param_names"] = []
    for param_key, param_val in cmp_config_dict["cmp_params"].items():
        if param_val:
            cmp_param_dic["cmp_params"][param_key] = param_val
            
            # 比較対象のパラメータとして考慮しないパラメータの集合を作成
            ignore_cmp_param_setting_set = create_ignore_cmp_param_setting_set()
            # 比較対象かつ、複数指定されているパラメータを "cmp_param_name" に保存する
            if len(param_val) > 1 and not(param_key in ignore_cmp_param_setting_set):
                info["cmp_param_names"].append(param_key)

    ## 直積を出すためにlistへ変換
    # env
    suboptima_keys = list(cmp_param_dic["cmp_envs"]["suboptima"].keys())
    suboptima_vals = list(cmp_param_dic["cmp_envs"]["suboptima"].values())
    tmaze_hidden_keys = list(cmp_param_dic["cmp_envs"]["tmaze_hidden"].keys())
    tmaze_hidden_vals = list(cmp_param_dic["cmp_envs"]["tmaze_hidden"].values())
    hex_hidden_keys = list(cmp_param_dic["cmp_envs"]["hex_hidden"].keys())
    hex_hidden_vals = list(cmp_param_dic["cmp_envs"]["hex_hidden"].values())
    # param
    param_keys = list(cmp_param_dic["cmp_params"].keys())
    param_vals = list(cmp_param_dic["cmp_params"].values())

    ## 直積を計算
    # env
    all_suboptima_vals = list(itertools.product(*suboptima_vals))
    all_tmaze_hidden_vals = list(itertools.product(*tmaze_hidden_vals))
    all_hex_hidden_vals = list(itertools.product(*hex_hidden_vals))
    # param
    all_param_vals = list(itertools.product(*param_vals))

    # 設定内容が間違っていないことを目視確認する用の print
    print(f"num of Suboptima settings: {len(all_suboptima_vals)}")
    print(f"{all_suboptima_vals=}", end="\n\n")
    print(f"num of Tmaze settings: {len(all_tmaze_hidden_vals)}")
    print(f"{all_tmaze_hidden_vals=}", end="\n\n")
    print(f"num of Hex settings: {len(all_hex_hidden_vals)}")
    print(f"{all_hex_hidden_vals=}", end="\n\n")
    print(f"num of Param settings: {len(all_param_vals)}")
    print(f"{all_param_vals=}", end="\n\n")

    # 中間形式 (直積) から simulation 実行のための list を作成
    # ToDo: 引数多すぎ問題どうにかする
    param_dict_list = create_param_setting_dict_list_from_cartesian(param_keys, all_param_vals)
    env_namedtuple_list = create_env_setting_dict_list_from_cartesian(suboptima_keys, all_suboptima_vals, tmaze_hidden_keys, all_tmaze_hidden_vals, hex_hidden_keys, all_hex_hidden_vals)

    alg_names_list = convert_algname_from_json(cmp_config_dict["cmp_alg_names"])
    # 設定内容が間違っていないことを目視確認する用の print
    print(f"num of algnames settings: {len(alg_names_list)}")
    print(f"{alg_names_list=}", end="\n\n")

    # param, env, alg を全部まとめた辞書のリストを作成
    param_env_alg_dict_list = []
    for alg_name in alg_names_list:
        for env_namedtuple in env_namedtuple_list:
            # env_namedtupleの正常判定
            if not is_correct_env_namedtuple(env_namedtuple):
                continue
            param_namedtuple_set = set() # param設定の重複を検知するための集合
            for param_dict in param_dict_list:
                # param_dictの正常判定
                if not is_correct_param_dict(param_dict):
                    continue
                param_env_alg_dict = {}
                # paramをdictからnamedtupleに変換
                param_namedtuple = param_converter_dict2namedtuple(param_dict, alg_name)
                # param設定の重複判定
                if param_namedtuple in param_namedtuple_set:
                    continue

                # param, env, alg を全部まとめた辞書の作成
                param_env_alg_dict["param_dict"] = param_dict
                param_env_alg_dict["env_namedtuple"] = env_namedtuple
                param_env_alg_dict["alg_name"] = alg_name
                # 集合、リストにそれぞれ追加
                param_env_alg_dict_list.append(param_env_alg_dict)
                param_namedtuple_set.add(param_namedtuple)

    # infoに必要な情報を入れていく
    info["cmp_env_names"] = []
    info["cmp_env_kinds_set"] = set()
    for env_namedtuple in env_namedtuple_list:
        info["cmp_env_names"].append(env_namedtuple.env.name)
        info["cmp_env_kinds_set"].add(env_namedtuple.env.kind)
    
    # 重複 & 異常設定を除いた後の、実質的な対象となる設定の表示
    print("==================")
    print(f"num of Settings after Validation: {len(param_env_alg_dict_list)}")
    print("------------------")
    print(f"{param_env_alg_dict_list=}")
    print("==================")

    return param_env_alg_dict_list, info
