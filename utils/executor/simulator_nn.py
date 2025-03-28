from utils.common.agent import Agent, MaintainedHypotheticalReplayAgent
from utils.common.collector import Collector
from utils.common.policy import LSTMQ, LSTMQNet, Softmax, AdditiveMaintainedLSTMBiasedPerSASoftmax, OnlyERBias
from utils.common.replay_buffer import HypotheticalEpisodeReplayBuffer
from utils.common.simulator import simulation

from utils.common.create_path_utils import create_data_path
from utils.common.validate_sims_utils import get_exist_sims
from utils.common.create_setting_from_json_utils import (
    load_default_dict_and_cmp_dict,
    create_intermediate_reprosentation_from_json
)
from utils.common.convert_json_to_dict_utils import (
    create_load_path_dict_from_json
)

"""
    各種実験のセットアップ関数
"""
def setup_lstm_q_learning(env, num_epis, param_dict, algname, is_save_animdata):
    # LSTM-Q に必要な param の展開
    alpha = param_dict["alpha"]
    beta = param_dict["beta"]
    gamma = param_dict["gamma"]
    batch = param_dict["batch"]

    params = {
        "observation_space": env.observation_space,
        "state_space": env.state_space,
        "action_space": env.action_space,
        "alpha": alpha,
        "beta": beta,
        "gamma": gamma,
        "learner": LSTMQ,
        "model": LSTMQNet,
        "batch": batch
    }
    settings = {}
    # policy = EpsironGreedy(env=env, **params)
    policy = Softmax(env=env, **params)
    agent = Agent(policy)

    # PATHの作成
    dataname, fname = create_data_path(env, num_epis, param_dict, algname, is_animdata=is_save_animdata, is_save=True)

    return Collector(dataname, fname, agent, env, params, settings)


def setup_maintained_action_selection_with_er_lstm(env, num_epis, param_dict, algname, is_save_animdata, observation_repr="onehot"):
    # param の展開
    alpha = param_dict["alpha"]
    beta = param_dict["beta"]
    gamma = param_dict["gamma"]
    buffer_size = param_dict["buffer_size"]
    er_ratio = param_dict["er_ratio"]
    er_deactivate_epis = param_dict["er_deactivate_epis"]
    retrieval_size = param_dict["retrieval_size"]
    trajectory_length = param_dict["trajectory_length"]
    bias_definition = param_dict["bias_definition"]
    bias_decay_rate = param_dict["bias_decay_rate"]
    bias_var_power = param_dict["bias_var_power"]
    max_bias_policy_weight = param_dict["max_bias_policy_weight"]
    var_bias_policy_weight = param_dict["var_bias_policy_weight"]
    batch = param_dict["batch"]

    params = {
        "observation_space": env.observation_space,
        "state_space": env.state_space,
        "observation_repr": observation_repr,  # "serial" or "onehot".
        "action_space": env.action_space,
        "alpha": alpha,
        "beta": beta,
        "gamma": gamma,
        "bias_definition": bias_definition,
        "bias_decay_rate": bias_decay_rate,
        "bias_var_power": bias_var_power,
        "er_ratio": er_ratio,
        "er_deactivate_epis": er_deactivate_epis,
        "max_bias_policy_weight": max_bias_policy_weight,
        "var_bias_policy_weight": var_bias_policy_weight,
        "learner": LSTMQ,
        "model": LSTMQNet,
        "batch": batch
    }
    settings = {
        "retrieval_size": retrieval_size,
        "trajectory_length": trajectory_length,
        "buffer_size": buffer_size,
    }
    policy = AdditiveMaintainedLSTMBiasedPerSASoftmax(env=env, **params)
    rb = HypotheticalEpisodeReplayBuffer(buffer_size, retrieval_size, trajectory_length)  # 現エピソードの軌跡については想起しないBuffer
    agent = MaintainedHypotheticalReplayAgent(policy, rb)

    # PATHの作成
    dataname, fname = create_data_path(env, num_epis, param_dict, algname, is_animdata=is_save_animdata, is_save=True)

    return Collector(dataname, fname, agent, env, params, settings)


def setup_only_action_selection_with_er(env, num_epis, param_dict, algname, is_save_animdata, observation_repr="onehot"):
    # param の展開
    alpha = param_dict["alpha"]
    beta = param_dict["beta"]
    gamma = param_dict["gamma"]
    buffer_size = param_dict["buffer_size"]
    er_ratio = param_dict["er_ratio"]
    er_deactivate_epis = param_dict["er_deactivate_epis"]
    retrieval_size = param_dict["retrieval_size"]
    trajectory_length = param_dict["trajectory_length"]
    bias_definition = param_dict["bias_definition"]
    bias_decay_rate = param_dict["bias_decay_rate"]
    bias_var_power = param_dict["bias_var_power"]
    max_bias_policy_weight = param_dict["max_bias_policy_weight"]
    var_bias_policy_weight = param_dict["var_bias_policy_weight"]
    batch = param_dict["batch"]

    params = {
        "observation_space": env.observation_space,
        "state_space": env.state_space,
        "observation_repr": observation_repr,  # "serial" or "onehot".
        "action_space": env.action_space,
        "alpha": alpha,
        "beta": beta,
        "gamma": gamma,
        "bias_definition": bias_definition,
        "bias_decay_rate": bias_decay_rate,
        "bias_var_power": bias_var_power,
        "er_ratio": er_ratio,
        "er_deactivate_epis": er_deactivate_epis,
        "er_ratio": er_ratio,
        "max_bias_policy_weight": max_bias_policy_weight,
        "var_bias_policy_weight": var_bias_policy_weight,
        "learner": LSTMQ,
        "model": LSTMQNet,
        "batch": batch
    }
    settings = {
        "retrieval_size": retrieval_size,
        "trajectory_length": trajectory_length,
        "buffer_size": buffer_size,
    }
    policy = OnlyERBias(env=env, **params)
    rb = HypotheticalEpisodeReplayBuffer(buffer_size, retrieval_size, trajectory_length)
    agent = MaintainedHypotheticalReplayAgent(policy, rb)

    # PATHの作成
    dataname, fname = create_data_path(env, num_epis, param_dict, algname, is_animdata=is_save_animdata, is_save=True)

    return Collector(dataname, fname, agent, env, params, settings)


"""
    各種実験を回すための関数
"""
def lstm_q_learning(env, num_sims, num_epis, collector, is_save_animdata=False):
    if is_save_animdata:
        collector.enable_create_animation()
    simulation(num_sims, num_epis, collector._agent, env, collector)


def maintained_action_selection_with_er_lstm(env, num_sims, num_epis, collector, is_save_animdata=False):
    if is_save_animdata:
        collector.enable_create_animation()
    simulation(num_sims, num_epis, collector._agent, env, collector)


def only_action_selection_with_er(env, num_sims, num_epis, collector, is_save_animdata=False):
    if is_save_animdata:
        collector.enable_create_animation()
    simulation(num_sims, num_epis, collector._agent, env, collector)


# 読み込んだjson設定に従ってsimulationを回す関数
def simulation_with_json(total_required_sims, param_env_alg_dict_list, is_save_animdata):
    # 直積の param設定, env設定, alg_name を一つずつ展開
    for param_env_alg_dict in param_env_alg_dict_list:
        param_dict = param_env_alg_dict["param_dict"]
        env_namedtuple = param_env_alg_dict["env_namedtuple"]
        algname = param_env_alg_dict["alg_name"]

        # env情報の展開
        env = env_namedtuple.env
        epis = env_namedtuple.epis
        
        # 現在の実験設定で既に実施済みの実験数を取得
        _, exist_sims = get_exist_sims(param_env_alg_dict, is_save_animdata)
        remaining_needed_sims = max(total_required_sims - exist_sims, 0)

        ## simulation 実行部分
        if algname == "LSTMQnet":
            lstm_q_collector = setup_lstm_q_learning(env, epis, param_dict, algname, is_save_animdata) # 各実験のセットアップ
            lstm_q_learning(env, remaining_needed_sims, epis, lstm_q_collector, is_save_animdata)  # simulation実行
        elif algname == "M_AS_ER_LSTMQnet":
            er_q_collector = setup_maintained_action_selection_with_er_lstm(env, epis, param_dict, algname, is_save_animdata) # 各実験のセットアップ
            maintained_action_selection_with_er_lstm(env, remaining_needed_sims, epis, er_q_collector, is_save_animdata) # simulation実行
        elif algname == "M_AS_ER_Only":
            er_x_collector = setup_only_action_selection_with_er(env, epis, param_dict, algname, is_save_animdata) # 各実験のセットアップ
            only_action_selection_with_er(env, remaining_needed_sims, epis, er_x_collector, is_save_animdata) # simulation実行            

 
if __name__ == "__main__":

    config_dir = "config/simulation"
    load_path_json_filename = "main.json"

    # load_path_dict から default_config_dict&cmp_config_dic を作成
    load_path_dict = create_load_path_dict_from_json(config_dir=config_dir, json_filename=load_path_json_filename)
    default_config_dict, cmp_config_dict_list = load_default_dict_and_cmp_dict(config_dir=config_dir, load_path_dict=load_path_dict)

    # main.jsonに複数指定した"cmp"対象をfor_each
    for cmp_config_dict in cmp_config_dict_list:
        param_env_alg_dict_list, info = create_intermediate_reprosentation_from_json(default_config_dict, cmp_config_dict)
    
        # 中間形式から各実験を実行
        total_required_sims = cmp_config_dict["total_required_sims"]
        is_save_animdata = cmp_config_dict["is_save_animdata"]
        simulation_with_json(total_required_sims, param_env_alg_dict_list, is_save_animdata)
