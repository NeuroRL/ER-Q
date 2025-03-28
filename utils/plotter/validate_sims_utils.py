from collections import defaultdict

from utils.common.validate_sims_utils import get_exist_sims
from utils.common.create_path_utils import judge_is_animdata
from utils.common.create_setting_from_json_utils import create_intermediate_reprosentation_from_json


def create_exist_and_total_required_sims_dict(default_config_dict, cmp_config_dict_list):
    # exist_sims_dict: {key: (path of data), val: (number of sims currently exist)}
    total_required_sims_dict = defaultdict(int)
    exist_sims_dict = {}
    
    # main.jsonに複数指定した"cmp"対象をfor_each
    for cmp_config_dict in cmp_config_dict_list:
        param_env_alg_dict_list, _ = create_intermediate_reprosentation_from_json(default_config_dict, cmp_config_dict)
        
        # データ参照先が'data'なのか'anim'なのか判定する
        is_animdata = judge_is_animdata(cmp_config_dict)
        
        for param_env_alg_dict in param_env_alg_dict_list:
            # 'param_env_alg_dict' で指定された実験設定の 'データパス' と '既存データ数' を取得
            reg_exp_data_path, exist_sims = get_exist_sims(param_env_alg_dict, is_animdata)
            
            # 参照データは同じだが 'total_required_sims' が異なる場合のためにより大きい設定値を取得
            # data_pathには 'figure_type' の情報が含まれないので単なる代入では情報が潰れてしまう
            total_required_sims = max(total_required_sims_dict[reg_exp_data_path], cmp_config_dict["total_required_sims"])
            
            exist_sims_dict[reg_exp_data_path] = exist_sims
            total_required_sims_dict[reg_exp_data_path] = total_required_sims
            
    return exist_sims_dict, total_required_sims_dict
            
    
def create_remaining_need_sims_dict(default_config_dict, cmp_config_dict_list):
    # 実施する実験実験が既に何simあるかの情報を取得
    exist_sims_dict, total_required_sims_dict = create_exist_and_total_required_sims_dict(default_config_dict, cmp_config_dict_list)
    
    remaining_need_sims_dict = {}
    for data_path in total_required_sims_dict.keys():
        remaining_needed_sims = total_required_sims_dict[data_path] - exist_sims_dict[data_path]
        if remaining_needed_sims > 0:
            remaining_need_sims_dict[data_path] = remaining_needed_sims
    
    return remaining_need_sims_dict