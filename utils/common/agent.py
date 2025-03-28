import numpy as np


class Agent(object):
    def __init__(self, policy):
        self._policy = policy
        self._current_observation = None
        self._last_observation = None
        self._last_action = None
        self.biased_policy_distribution = [np.nan]*self._policy._action_space 
        self.unbiased_policy_distribution = [np.nan]*self._policy._action_space

    def initialize(self):
        self._policy.initialize()

    def reset(self, observation):
        self._policy.reset()
        self._current_observation = observation
        self._last_observation = None
        self._last_action = None

    def observe(self, observation, reward, done, **kwargs):
        self._last_observation = self._current_observation

        if self._last_observation is not None:
            self._policy.update(self._last_observation, self._last_action, reward, observation, done)

        self._current_observation = observation

    def act(self):
        action = self._policy.decision_making(self._current_observation)
        self.unbiased_policy_distribution = self._policy.get_unbiased_policy_distribution(self._current_observation)
        self.biased_policy_distribution = self._policy.get_biased_policy_distribution(self._current_observation)
        self._last_action = action
        return action


# ER_A...
# 行動選択時に類似の経験を想起し価値にバイアスをかけ意思決定
class HypotheticalReplayAgent(Agent):
    def __init__(self, policy, replaybuffer):
        super().__init__(policy)
        self._replay_buffer = replaybuffer

    def initialize(self):
        super().initialize()
        self._replay_buffer.initialize()

    def reset(self, observation):
        super().reset(observation)
        self._replay_buffer.reset()

    def observe(self, observation, reward, done, epis):
        super().observe(observation, reward, done)
        self._policy.should_ignore_bias(epis)
        if self._last_observation is not None:
            self._replay_buffer.add(self._last_observation, self._last_action, reward, self._current_observation, done)

    def act(self):
        trajectories = self._replay_buffer.retrieval(self._current_observation)

        self._policy.calc_bias(trajectories)
        action = self._policy.decision_making(self._current_observation)
        self._last_action = action
        return action


# Maintained ER_A
# 行動選択時に類似の経験を想起し価値にバイアスをかけ意思決定
class MaintainedHypotheticalReplayAgent(HypotheticalReplayAgent):
    def act(self):
        trajectories = self._replay_buffer.retrieval(self._current_observation)
        self._policy.update_bias_table(trajectories)
        self._policy.calc_bias(self._current_observation)
        action = self._policy.decision_making(self._current_observation)
        self.unbiased_policy_distribution = self._policy.get_unbiased_policy_distribution(self._current_observation)
        self.biased_policy_distribution = self._policy.get_biased_policy_distribution(self._current_observation)
        self._policy.decay_bias()
        self._last_action = action
        return action
