import numpy as np
import os
import pickle as pkl
import uuid


# GridEnvironment 用Collector
class Collector(object):
    def __init__(self, taskname, filename, agent, env, params, settings):
        self._epi = 0
        self._agent = agent
        self._env = env
        self._taskname = taskname
        self._filename = filename

        self._params = params
        self._settings = settings

        # sim 単位収集
        self._q_after_sim = None
        self._q_after_sim_append = None

        # epi 単位収集
        self._return = None
        self._return_append = None
        self._steps = None
        self._steps_append = None
        self._terminal_reward_population_means = None
        self._terminal_reward_population_means_append = None

        # step 単位収集
        self._reward = None
        self._reward_append = None
        """
           step毎に生データを保存して、後でfigure_plotterのほうで最大値なり分散なりを計算したほうがいい気がする.
           なのでここではstepデータとして保存する.
           実行時間や処理の煩雑性の観点等から問題がありそうならcollectorで事前にデータ整形してepisodic_dataとして保存する方法もあり.
           分布プロットの際にNoneType/intエラーが発生したため、ここだけnp.nanに変更
           ↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓
        """
        self._biased_policy_distributions = np.nan
        self._biased_policy_distributions_append = np.nan
        self._unbiased_policy_distributions = np.nan
        self._unbiased_policy_distributions_append = np.nan
        """
           ↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑
        """
        # animation 再現用
        self._create_animation = False
        self._hidden = None
        # 環境遷移の再現用
        self._states = None
        self._states_append = None
        self._actions = None
        self._actions_append = None
        # agent 内部状態再現用
        self._q = None
        self._q_append = None
        # ER_A 再現用
        self._bias = None
        self._bias_append = None
        self._focused_trajectory = None
        self._focused_trajectory_append = None
        # Maintain 再現用
        self._bias_table = None
        self._bias_table_append = None

    def _format(self):
        data = {}
        data["params"] = self._params
        data["settings"] = self._settings
        data["reward"] = self._reward
        data["return"] = self._return
        data["steps"] = self._steps
        data["q_after_sim"] = self._q_after_sim
        data["terminal_reward_population_means"] = self._terminal_reward_population_means
        data["biased_policy_distributions"] = self._biased_policy_distributions
        data["unbiased_policy_distributions"] = self._unbiased_policy_distributions
        data["hidden"] = self._hidden
        data["states"] = self._states
        data["actions"] = self._actions
        if self._create_animation:
            data["q_each_step"] = self._q_each_step
            data["bias"] = self._bias
            data["f_tj"] = self._focused_trajectory
            data["bias_table"] = self._bias_table
        return data

    def _env_state(self):
        return self._env.state

    def _env_observation(self):
        return self._env.observation

    def _env_observations(self):
        return self._env.observation_table

    def enable_create_animation(self):
        self._create_animation = True

    def initialize(self):
        self._q_after_sim = []

        self._return = []
        self._steps = []
        self._terminal_reward_population_means = []
        

        self._reward = []
        self._biased_policy_distributions = []
        self._unbiased_policy_distributions = []
        self._states = []
        self._actions = []
        self._q_each_step = []
        self._bias = []
        self._focused_trajectory = []
        self._bias_table = []

        self._return_append = self._return.append
        self._steps_append = self._steps.append
        self._terminal_reward_population_means_append = self._terminal_reward_population_means.append

        self._hidden = []
        self._hidden_append = self._hidden.append

    def reset(self, epi):
        self._epi = epi
        self._reward.append([])
        self._reward_append = self._reward[epi].append
        self._biased_policy_distributions.append([])

        self._biased_policy_distributions_append = self._biased_policy_distributions[epi].append
        self._unbiased_policy_distributions.append([])
        self._unbiased_policy_distributions_append = self._unbiased_policy_distributions[epi].append
        self._states.append([])
        self._states_append = self._states[epi].append
        self._actions.append([])
        self._actions_append = self._actions[epi].append
        self._states_append(self._env_state())
        if self._create_animation:
            self._q_each_step.append([])
            self._q_each_step_append = self._q_each_step[epi].append
            self._bias.append([])
            self._bias_append = self._bias[epi].append
            self._focused_trajectory.append([])
            self._focused_trajectory_append = self._focused_trajectory[epi].append
            self._bias_table.append([])
            self._bias_table_append = self._bias_table[epi].append

            self._q_each_step_append(self._agent._policy.q_table(self._env_observations()))

    def collect_step_data(self):
        self._reward_append(self._env._reward)
        self._biased_policy_distributions_append(self._agent.biased_policy_distribution)
        self._unbiased_policy_distributions_append(self._agent.unbiased_policy_distribution)
        self._states_append(self._env_state())
        self._actions_append(self._agent._last_action)
        if self._create_animation:
            self._q_each_step_append(self._agent._policy.q_table(self._env_observations()))
            if hasattr(self._agent._policy, "focused_trajectory"):
                self._focused_trajectory_append(
                    self._agent._policy.focused_trajectory
                )
                self._bias_append(
                    self._agent._policy.bias
                )
            if hasattr(self._agent._policy, "bias_table"):
                self._bias_table_append(
                    self._agent._policy.bias_table(self._env_observations())
                )

    def collect_episodic_data(self):
        if hasattr(self._env, "hidden"):
            self._hidden_append(self._env.hidden)
        self._return_append(self._env._return)
        self._steps_append(self._env._steps)
        if self._env.kind == "SubOptima":
            self._terminal_reward_population_means_append(self._env._terminal_reward_population_mean)

    def collect_simulation_data(self):
        self._q_after_sim.append(self._agent._policy.q_table(self._env.observation_table))

    def return_nsim_mean(self, n):
        return_list = self._return[-n:]
        if not return_list:
            return 0
        return sum(return_list) / len(return_list)

    def steps_nsim_mean(self, n):
        steps_list = self._steps[-n:]
        if not steps_list:
            return 0
        return sum(steps_list) / len(steps_list)

    def save(self):
        data = self._format()
        folder = f"./data/{self._taskname}/"
        fname = f"{self._filename}_{uuid.uuid4().hex[:6]}.pickle"
        os.makedirs(folder, exist_ok=True)
        with open(f"{folder}{fname}", "wb") as f:
            pkl.dump(data, f)
