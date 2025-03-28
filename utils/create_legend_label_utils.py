from utils.plotter.paramname_for_plot_utils import convert_paramname_for_plot
from utils.plotter.convert_for_plot_utils import (
    convert_bias_policy_name_for_plot,
    convert_algname_for_plot,
    convert_model_explanation_for_plot
)


# model_explanation（モデルの説明）をeach_sim_infoから作成する関数
def create_model_explanation_for_plot(each_sim_info, is_err_plot=False):
    # each_sim_info から必要情報を取得
    bias_policy_setting_dict = each_sim_info["bias_policy_setting_dict"]
    er_ratio_param_val = each_sim_info["er_ratio"]
    
    # 各種情報をplotに適した形式に変換
    bias_policy_name = convert_bias_policy_name_for_plot(bias_policy_setting_dict)
    model_explanation = convert_model_explanation_for_plot(er_ratio_param_val, bias_policy_name, is_err_plot=is_err_plot)
    
    return model_explanation


# model_name（algname + bias_policy_name）をeach_sim_infoから作成する関数
def create_model_name_for_plot(each_sim_info):
    model_name = ""
    
    # model_explanation（モデルの説明）をeach_sim_infoから作成する関数によって作成
    model_explanation = create_model_explanation_for_plot(each_sim_info=each_sim_info)
    
    # each_sim_info から必要情報を取得
    er_ratio_param_val = each_sim_info["er_ratio"]
    alg_name = each_sim_info["alg_name"]
    
    # 各種情報をplotに適した形式に変換
    converted_alg_name = convert_algname_for_plot([alg_name])[0]
    err_param_name = "ERR" # TODO: ハードコーディングしないようにしたい...。

    # LSTMの場合はアルゴリズム名のみ表示
    # "algname" によるデータの分離は検証時のみ使用する。論文用のデータはLSTMもerr=0として表現する。
    if converted_alg_name == "LSTM-Q":
        model_name = converted_alg_name
    else:
        model_name = f"{err_param_name}{er_ratio_param_val} ({model_explanation})"
        
    return model_name


# TODO: 他の箇所に合わせてeach_sim_infoとかを渡すべきか検討
#       paramname_listがsim_infoに含まれており、sim_infoを渡すのは過剰すぎる気がして一旦断念した。
# plotラベルのパラメータに関する部分を作成するための関数
def create_param_label_for_plot(paramname_list, paramval_list):
    param_label = ""
    
    param_label = " & ".join(f"{p_name}: {p_val}" for p_name, p_val in zip(paramname_list, paramval_list))
    
    return param_label


# plotラベルのパラメータに関する部分を作成するための関数
def create_envsize_label_for_plot(each_sim_info):
    env = each_sim_info["env"]
    envsize_label = f"W{env._width}_H{env._height}"
    
    return envsize_label


# plotラベルのsteps_limitに関する部分を作成するための関数
def create_steps_limit_label_for_plot(each_sim_info):
    env = each_sim_info["env"]
    steps_limit_label = f"{env._limit}"
    
    return steps_limit_label


####
# cmp_err_plot
####
# cmp_err_plot用のlegend_labelをeach_sim_infoから作成する関数
def create_legend_label_for_cmp_err_plot(each_sim_info):
    return create_model_explanation_for_plot(each_sim_info, is_err_plot=True)


# cmp_err_plot用のlegend_label_listをsim_infoから作成する関数
def create_legend_label_list_for_cmp_err_plot(sim_info):
    legend_labels = set()
    
    for each_sim_info in sim_info["each_sim_info_list"]:
        legend_label = create_legend_label_for_cmp_err_plot(each_sim_info)
        legend_labels.add(legend_label)
    return sorted(list(legend_labels))


####
# envsize_plot
####
# envsize_plot用のlegend_label_listをsim_infoから作成する関数（model_plotと基本的には一緒のはず）
def create_legend_label_list_for_env_plot(sim_info):
    legend_labels = set()
    
    for each_sim_info in sim_info["each_sim_info_list"]:
        legend_label = create_model_name_for_plot(each_sim_info)
        legend_labels.add(legend_label)
    return sorted(list(legend_labels))

####
# param_plot
####
# param_plot用のlegend_label_listをsim_infoから作成する関数
def create_legend_label_list_for_param_plot(sim_info):
    legend_labels = set()
    
    for each_sim_info in sim_info["each_sim_info_list"]:
        legend_label = create_model_name_for_plot(each_sim_info)
        legend_labels.add(legend_label)
    return sorted(list(legend_labels))


####
# model_plot
####
# TODO: legend_labelとは関係ない気もするので、機会があったらいい感じのファイルに切り出す
# バイアスポリシー名（fig分離粒度）をkey、sim_idをの集合をvalueに持つ辞書の作成
def create_sim_id_set_dict_per_model_name(sim_info):
    # 辞書の初期化
    sim_id_set_dict_per_model_name = {}
    
    for each_sim_info in sim_info["each_sim_info_list"]:
        # 必要情報の取得&作成
        sim_id = each_sim_info["sim_id"]
        model_name = create_model_name_for_plot(each_sim_info)
        
        # 既にkeyが存在する場合にはvalueである集合にsim_idを追加
        if model_name in sim_id_set_dict_per_model_name:
            sim_id_set_dict_per_model_name[model_name].add(sim_id)
        else: # 存在しない場合はvalueとなる集合を追加
            sim_id_set_dict_per_model_name[model_name] = {sim_id}
    
    return sim_id_set_dict_per_model_name


####
# steps_limit_plot
####
# steps_limit_plot用のlegend_label_listをsim_infoから作成する関数
def create_legend_label_list_for_steps_limit_plot(sim_info):
    legend_labels = set()
    
    for each_sim_info in sim_info["each_sim_info_list"]:
        legend_label = create_model_name_for_plot(each_sim_info)
        legend_labels.add(legend_label)
    return sorted(list(legend_labels))


# sim_idをkey、param_plot用のラベル名をvalueに持つ辞書の作成
def create_envsize_legend_label_dict_per_sim_id(sim_info):
    # 辞書の初期化
    legend_label_dict_per_sim_id = {}
    
    for each_sim_info in sim_info["each_sim_info_list"]:
        # 必要情報の取得&作成
        sim_id = each_sim_info["sim_id"]
        # plotラベルのパラメータに関する部分を作成する
        envsize_label = create_envsize_label_for_plot(each_sim_info)
        legend_label_dict_per_sim_id[sim_id] = envsize_label
    
    return legend_label_dict_per_sim_id


# sim_idをkey、param_plot用のラベル名をvalueに持つ辞書の作成
def create_param_plot_legend_label_dict_per_sim_id(sim_info):
    # 辞書の初期化
    legend_label_dict_per_sim_id = {}
    
    # パラメータ名をplot用の表記に変換
    paramname_list = convert_paramname_for_plot(sim_info["commom_info"]["cmp_paramname_list"])
    
    for each_sim_info in sim_info["each_sim_info_list"]:
        # 必要情報の取得&作成
        sim_id = each_sim_info["sim_id"]
        paramval_list = each_sim_info["cmp_paramval_list"]
        
        # plotラベルのパラメータに関する部分を作成する
        param_label = create_param_label_for_plot(paramname_list, paramval_list)
        
        legend_label_dict_per_sim_id[sim_id] = param_label
    
    return legend_label_dict_per_sim_id


# sim_idをkey、param_plot用のラベル名をvalueに持つ辞書の作成
def create_baseline_plot_legend_label_dict_per_sim_id(sim_info):
    # 辞書の初期化
    legend_label_dict_per_sim_id = {}
    
    for each_sim_info in sim_info["each_sim_info_list"]:
        # 必要情報の取得&作成
        sim_id = each_sim_info["sim_id"]
        model_name = create_model_name_for_plot(each_sim_info)
        
        legend_label_dict_per_sim_id[sim_id] = model_name
    
    return legend_label_dict_per_sim_id


# sim_idをkey、param_plot用のラベル名をvalueに持つ辞書の作成
def create_interval_plot_legend_label_dict_per_sim_id(sim_info):
    # 辞書の初期化
    legend_label_dict_per_sim_id = {}
    
    for each_sim_info in sim_info["each_sim_info_list"]:
        # 必要情報の取得&作成
        sim_id = each_sim_info["sim_id"]
        model_name = create_model_name_for_plot(each_sim_info)
        
        legend_label_dict_per_sim_id[sim_id] = model_name
    
    return legend_label_dict_per_sim_id
        
        
# sim_idをkey、param_plot用のラベル名をvalueに持つ辞書の作成
def create_steps_limit_legend_label_dict_per_sim_id(sim_info):
    # 辞書の初期化
    legend_label_dict_per_sim_id = {}
    
    for each_sim_info in sim_info["each_sim_info_list"]:
        # 必要情報の取得&作成
        sim_id = each_sim_info["sim_id"]
        # plotラベルのパラメータに関する部分を作成する
        steps_limit_label = create_steps_limit_label_for_plot(each_sim_info)
        legend_label_dict_per_sim_id[sim_id] = steps_limit_label
    
    return legend_label_dict_per_sim_id
