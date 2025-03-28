from utils.plotter.env_info_utils import extract_env_info_from_envname
from utils.common.dict_mappings import (
    PLOTNAME_TO_ALGNAME_DICT, ALGNAME_TO_PLOTNAME_DICT, ENVNAME_CONVERT_DICT, CONVERT_METRICS_DICT
)
from utils.common.custom_exception_utils import InvalidJSONConfigException


# TODO: これだけfor_plotではなくcommonなので、どうにかする
# jsonに指定したアルゴリズム名 (ER-Q, ER-X, LSTM-Q) を program用の表記に変換する関数
def convert_algname_from_json(algname_list):
    new_algname_list = []
    # jsonに指定した簡略化したアルゴリズム名をprogramで使用する正式名称に直すための変換表
    algname_convert_dict = PLOTNAME_TO_ALGNAME_DICT
    # Validation：アルゴリズム名に重複設定がないか
    if len(algname_list) != len(set(algname_list)):
        raise ValueError("JSON設定の 'cmp_alg_names' に重複した設定があります。もう一度設定ファイルを確認し直してください。")
    
    algname_list = sorted(list(set(algname_list)), reverse=True)  # 重複を弾くために一度 Set に変換, LSTM-Qが0番目に来るように降順でソート

    # 変換表に則ってアルゴリズム名を変換
    for algname in algname_list:
        if algname in algname_convert_dict:
            new_algname_list.append(algname_convert_dict[algname])
        else:
            raise ValueError(f"JSON設定の 'cmp_alg_names' に想定されていない値が指定されています。想定される値は '{algname_convert_dict.keys()}' です。")
    
    return new_algname_list

def convert_algname_for_plot(algname_list):
    new_algname_list = []
    # programで使用する正式名称をplot用の簡略化したアルゴリズム名にへんかんする
    algname_convert_dict = ALGNAME_TO_PLOTNAME_DICT

    # 変換表に則ってアルゴリズム名を変換
    for algname in algname_list:
        if algname in algname_convert_dict:
            new_algname_list.append(algname_convert_dict[algname])
        else:
            raise ValueError(f"引数に想定されていない値が指定されています。想定される値は '{algname_convert_dict.keys()}' です。")
    
    return new_algname_list

def convert_bias_policy_name_for_plot(bias_policy_setting_dict):
    scale = bias_policy_setting_dict["scale"]
    variance = bias_policy_setting_dict["variance"]
    
    bias_policy_name = ""
    if scale > 0 and variance > 0:
        bias_policy_name = "Hybrid" # TODO: ここの名称は要相談。一旦適当に命名しておく。
    elif scale > 0:
        bias_policy_name = "Mean"
    elif variance > 0:
        bias_policy_name = "Variance"
    else:
        # TODO: もっと適切な例外処理に変更すべきかも...？
        raise InvalidJSONConfigException("bias policy ratio の設定に問題があります。bias policy ratioは必須の設定です。")
    return bias_policy_name

def convert_env_name_for_plot(envname):
    env_type, _ = extract_env_info_from_envname(envname)
    envname_convert_dict = ENVNAME_CONVERT_DICT
    return envname_convert_dict[env_type]


def convert_model_explanation_for_plot(er_ratio_param_val, bias_policy_name, is_err_plot=False):
    model_explanation = ""

    # cmp_err_plotの場合、横軸がERRになるため、ERRの値に応じてラベルを変更すると間違ったグルーピングがされるため、条件分岐を行う
    if is_err_plot:
        model_explanation = f"ER {bias_policy_name} + Q"
    else:
        if er_ratio_param_val >= 1:
            model_explanation = f"Only ER {bias_policy_name}, no Q"
        elif er_ratio_param_val <= 0:
            model_explanation = "Only Q, no ER"
        else:
            model_explanation = f"ER {bias_policy_name} + Q"
    
    return model_explanation

def convert_metrics_for_plot_ylabel(metrics):
    metrics_convert_dict = CONVERT_METRICS_DICT
    metrics_for_plot_ylabel = metrics_convert_dict[metrics]
    return metrics_for_plot_ylabel
