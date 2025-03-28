from utils.common.custom_exception_utils import InvalidJSONConfigException
from utils.common.trimming_trailing_zero import trimming_trailing_zero


def judge_is_animdata_for_plotter(cmp_config_dict):
    use_animdata_figure_type_set = {
        "visual_figures",
        "visual_task_figures",
        "visual_sim_each_figures",
        "visual_sim_all_figures",
        "visual_anime_sim_any_figures"
    }
    is_animdata = cmp_config_dict["figures_type"] in use_animdata_figure_type_set
    
    return is_animdata 


def judge_is_animdata_for_simulator(cmp_config_dict):
    is_animdata = cmp_config_dict["is_save_animdata"]
    
    return is_animdata 


def judge_is_animdata(cmp_config_dict):
    # TODO: plotter、simulatorのjsonの条件は必要十分になっているのかは議論の余地がありそう。
    is_animdata = None
    # plotterの場合
    if "figures_type" in cmp_config_dict:
        is_animdata = judge_is_animdata_for_plotter(cmp_config_dict)
    # simulatorの場合
    elif "is_save_animdata" in cmp_config_dict:
        is_animdata = judge_is_animdata_for_simulator(cmp_config_dict)
    else:
        raise InvalidJSONConfigException(
            "アニメーションデータが必要かどうかが判断できません。"
            "plotterであれば 'figures_type'、simulatorであれば 'is_save_animdata' を確認してください。"
        )
    
    return is_animdata


def create_model_param_path(param_dict, algname):
    # param の展開
    alpha = trimming_trailing_zero(param_dict["alpha"])
    beta = trimming_trailing_zero(param_dict["beta"])
    gamma = trimming_trailing_zero(param_dict["gamma"])
    buffer_size = trimming_trailing_zero(param_dict["buffer_size"])
    er_ratio = trimming_trailing_zero(param_dict["er_ratio"])
    er_deactivate_epis = trimming_trailing_zero(param_dict["er_deactivate_epis"])
    retrieval_size = trimming_trailing_zero(param_dict["retrieval_size"])
    trajectory_length = trimming_trailing_zero(param_dict["trajectory_length"])
    bias_decay_rate = trimming_trailing_zero(param_dict["bias_decay_rate"])
    bias_var_power = trimming_trailing_zero(param_dict["bias_var_power"])
    max_bias_policy_weight = trimming_trailing_zero(param_dict["max_bias_policy_weight"])
    var_bias_policy_weight = trimming_trailing_zero(param_dict["var_bias_policy_weight"])
    batch = trimming_trailing_zero(param_dict["batch"])

    model_param = ""
    # TODO: "CONVERT_PARAMNAME_DICT" を呼び出してべた書きをやめる
    match algname:
        case "LSTMQnet":
            model_param = f"alpha{alpha}_beta{beta}_gamma{gamma}_batch{batch}"
        case "M_AS_ER_LSTMQnet":
            model_param = f"bs{buffer_size}_tl{trajectory_length}_rs{retrieval_size}_alpha{alpha}_beta{beta}_gamma{gamma}_bdr{bias_decay_rate}_bvp{bias_var_power}_err{er_ratio}_ede{er_deactivate_epis}_mabpw{max_bias_policy_weight}_varbpw{var_bias_policy_weight}_batch{batch}"
        case "M_AS_ER_Only":
            model_param = f"bs{buffer_size}_tl{trajectory_length}_rs{retrieval_size}_alpha{alpha}_beta{beta}_gamma{gamma}_bdr{bias_decay_rate}_bvp{bias_var_power}_err{er_ratio}_ede{er_deactivate_epis}_mabpw{max_bias_policy_weight}_varbpw{var_bias_policy_weight}_batch{batch}"
        case _:
            raise InvalidJSONConfigException(f"'algname'の値が異常です。期待される値は 'LSTMQnet', 'M_AS_ER_LSTMQnet', 'M_AS_ER_Only'であるにも関わらず、入力された値は '{algname}' です。")
    
    return model_param


def create_data_path(env, num_epis, param_dict, algname, is_animdata, is_save):
    model_param = create_model_param_path(param_dict, algname)
    # ディレクトリ名の作成
    dirname = ""
    if is_animdata:
        dirname = f"{algname}/{env.name}/anim/{model_param}"
    else:
        dirname = f"{algname}/{env.name}/data/{model_param}"
    # ファイル名の作成
    filename = f"epi{num_epis}"
    if is_save:
        filename = f"{filename}"
    else:
        filename = f"{filename}_*.pickle"
    
    return dirname, filename

