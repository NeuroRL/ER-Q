from utils.common.environment import (
    HiddenExploreHex,
    FlaggedExtendedTmazeHidden,
    SubOptima
)

from collections import namedtuple


# 中間形式 (直積) から param_dict の list を作成
def create_param_setting_dict_list_from_cartesian(param_keys=None, all_param_vals=None):
    ## params の復元
    param_dict_list = []
    # 直積で作成した実験設定から param_dict を復元しつつ foreach
    for param_seeing in all_param_vals:
        param_dict = {}
        for key, val in zip(param_keys, param_seeing):
            param_dict[key] = val
        param_dict_list.append(param_dict)

    return param_dict_list


# ToDo ここもっと効率化できそう？
# 中間形式 (直積) から Env設定 (namedtuple) の list を作成
# 重複判定は、各種valuesのリストを集合に変換することによって、この中で実施してしまっている。
def create_env_setting_dict_list_from_cartesian(suboptima_keys=None, all_suboptima_vals=None, tmaze_hidden_keys=None, all_tmaze_hidden_vals=None, hex_hidden_keys=None, all_hex_hidden_vals=None):
    ## envの復元
    env_namedtuple_list = []
    EnvSettings = namedtuple("EnvSettings", ["env", "epis"])
    # suboptima settings の辞書復元
    for suboptima_setting in set(all_suboptima_vals):
        env_setting_dict = {}
        for key, val in zip(suboptima_keys, suboptima_setting):
            env_setting_dict[key] = val
        # Env インスタンスの作成
        if env_setting_dict["available"]:
            epis = env_setting_dict["epis"]
            width = env_setting_dict["width"]
            height = env_setting_dict["height"]
            sd = env_setting_dict["sd"]
            rew_upper = env_setting_dict["rew_upper"]
            steps_limit = env_setting_dict["steps_limit"]
            is_render_agt = env_setting_dict["is_render_agt"]

            env = SubOptima(width=width, height=height, sd=sd, reward_upper=rew_upper, limit=steps_limit, state_type="onehot", is_render_agt=is_render_agt)
            # EnvSettings (namedtuple) をappend (env毎にepiを変化させたいため)
            env_namedtuple_list.append(EnvSettings(env, epis))
    # tmaze_hidden settings の辞書復元
    for tmaze_hidden_setting in set(all_tmaze_hidden_vals):
        env_setting_dict = {}
        for key, val in zip(tmaze_hidden_keys, tmaze_hidden_setting):
            env_setting_dict[key] = val
        # Env インスタンスの作成
        if env_setting_dict["available"]:
            epis = env_setting_dict["epis"]
            width = env_setting_dict["width"]
            height = env_setting_dict["height"]
            swap_timing = env_setting_dict["swap_timing"]
            steps_limit = env_setting_dict["steps_limit"]
            is_render_agt = env_setting_dict["is_render_agt"]

            env = FlaggedExtendedTmazeHidden(width=width, height=height, limit=steps_limit, state_type="onehot", swap=epis/swap_timing, is_render_agt=is_render_agt)
            # EnvSettings (namedtuple) をappend (env毎にepiを変化させたいため)
            env_namedtuple_list.append(EnvSettings(env, epis))
    for hex_hidden_setting in set(all_hex_hidden_vals):
        env_setting_dict = {}
        for key, val in zip(hex_hidden_keys, hex_hidden_setting):
            env_setting_dict[key] = val
        # Env インスタンスの作成
        if env_setting_dict["available"]:
            epis = env_setting_dict["epis"]
            width = env_setting_dict["width"]
            height = env_setting_dict["height"]
            swap_timing = env_setting_dict["swap_timing"]
            steps_limit = env_setting_dict["steps_limit"]
            is_render_agt = env_setting_dict["is_render_agt"]

            env = HiddenExploreHex(width=width, height=height, limit=steps_limit, state_type="onehot", swap=epis/swap_timing, is_render_agt=is_render_agt)
            # EnvSettings (namedtuple) をappend (env毎にepiを変化させたいため)
            env_namedtuple_list.append(EnvSettings(env, epis))
    
    return env_namedtuple_list
