from glob import glob
from utils.common.create_path_utils import create_data_path

    
# 'param_env_alg_dict' で指定された実験設定の 'データパス' と '既存データ数' を取得
def get_exist_sims(param_env_alg_dict, is_animdata):
    param_dict = param_env_alg_dict["param_dict"]
    env_namedtuple = param_env_alg_dict["env_namedtuple"]
    env = env_namedtuple.env
    num_epis = env_namedtuple.epis
    algname = param_env_alg_dict["alg_name"]
    
    data_dir, fname = create_data_path(env, num_epis, param_dict, algname, is_animdata, is_save=False)
    reg_exp_data_path = f"./data/{data_dir}/{fname}"
    pickle_paths_all = glob(reg_exp_data_path)
    exist_sims = len(pickle_paths_all)
    
    return reg_exp_data_path, exist_sims
