from utils.common.dict_mappings import ENVNAME_CONVERT_DICT, THEORETICAL_MIN_GOAS_STEPS_DICT


def extract_env_info_from_envname(envname):
    envname_split_list = envname.split("_")
    # TODO: envnameの各種情報の順序は明確に決まっているわけではないので位置で指定せずに正規表現などを用いて抽出するように修正する
    env_type = envname_split_list[0]
    envsize_str = envname_split_list[2] + "_" + envname_split_list[3]
    return env_type, envsize_str


def extract_env_type_from_envkind(env_kind):
    envname_split_list = env_kind.split("_")
    # TODO: envnameの各種情報の順序は明確に決まっているわけではないので位置で指定せずに正規表現などを用いて抽出するように修正する
    env_type = envname_split_list[0]
    return env_type


def calc_theorematical_min_goas_steps(envname):
    env_type, envsize_str = extract_env_info_from_envname(envname)
    theorematical_min_goas_steps_dict = THEORETICAL_MIN_GOAS_STEPS_DICT
    return theorematical_min_goas_steps_dict[env_type][envsize_str]
