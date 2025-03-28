import numpy as np
import torch
import torch.nn as nn
import torch.optim as opt
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence, pad_packed_sequence, pack_padded_sequence
from utils.common.replay_buffer import EpisodeReplayBuffer, ReplayBuffer
import sys
import random


class TabularQ(object):
    def __init__(self, **kwargs):
        self._observation_space = kwargs["observation_space"]
        self._action_space = kwargs["action_space"]
        self._alpha = kwargs.get("alpha", 0.1)
        self._gamma = kwargs.get("gamma", 0.9)
        self._q = np.zeros((self._observation_space, self._action_space))

    def reset(self):
        pass

    def initialize(self):
        self._q = np.zeros((self._observation_space, self._action_space))

    def q_table(self, observations):
        return self._q[observations].copy()

    def q(self, observation):
        return self._q[observation].copy()

    def calc_temporal_difference(self, observation, action, reward, next_observation, next_action, done):
        if type(reward) is float:
            reward = [reward]
        target = reward[0] + self._gamma * self._q[next_observation, next_action] * (1 - done)
        for r in reward[1:]:
            target = r + self._gamma * target
        return target - self._q[observation, action]

    def calc_nstep_q(self, observation, action, reward, next_observation, next_action, done):
        if type(reward) is float:
            reward = [reward]
        target = reward[0] + self._gamma * self._q[next_observation, next_action] * (1 - done)
        for r in reward[1:]:
            target = r + self._gamma * target
        return target

    def update(self, observation, action, reward, next_observation, next_action, done):
        delta = self.calc_temporal_difference(observation, action, reward, next_observation, next_action, done)
        self._q[observation, action] += self._alpha * delta


class QNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)
        # self.fc1.register_forward_hook(lambda m, i, o: print("[out]", o))

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


class NeuralNetQ(object):
    def __init__(self, model=QNet, **kwargs):
        self._observation_space = kwargs["observation_space"]
        self._action_space = kwargs["action_space"]
        self._alpha = kwargs.get("alpha", 0.01)
        self._gamma = kwargs.get("gamma", 0.9)
        # self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._device = torch.device("cpu")
        self._model_class = model
        self._hidden_size = kwargs.get("hidden_size", 32)
        self._model = model(self._observation_space, self._hidden_size, self._action_space)
        self._model.to(self._device)
        self._optimizer = opt.RMSprop(self._model.parameters(), lr=self._alpha)
        self._criterion = nn.SmoothL1Loss()
        self._batch_size = kwargs.get("batch", 1)
        self._replay_buffer = ReplayBuffer(10000, self._batch_size)

    def q_table(self, observations):
        q = self.q(observations)
        return q

    def q(self, observation):
        s = torch.tensor(observation, dtype=torch.float32).to(self._device)
        with torch.no_grad():
            return self._model(s.reshape(-1, self._observation_space)).squeeze().to("cpu").detach().numpy().copy()

    def reset(self):
        self._replay_buffer.reset()

    def initialize(self):
        self._model = self._model_class(self._observation_space, self._hidden_size, self._action_space)
        self._model.to(self._device)
        self._optimizer = opt.RMSprop(self._model.parameters(), lr=self._alpha)
        self._replay_buffer.initialize()

    def calc_temporal_difference(self, observation, action, reward, next_observation, next_action, done):
        if type(reward) is float:
            reward = [reward]
        o = torch.tensor(observation, device=self._device, dtype=torch.float32)
        a = torch.tensor([[action]], device=self._device, dtype=torch.long)
        r = torch.tensor(reward, device=self._device, dtype=torch.float32)
        od = torch.tensor(next_observation, device=self._device, dtype=torch.float32)
        ad = torch.tensor([[next_action]], device=self._device, dtype=torch.long)

        with torch.no_grad():
            q = self._model(o.unsqueeze(0))
            q = q.gather(1, a)
            qd = self._model(od.unsqueeze(0))
            target = r[0] + self._gamma * qd.gather(1, ad) * (1 - done)

            for _r in r[1:]:
                target = _r + self._gamma * target
        return (target - q).squeeze().to("cpu").detach().numpy().copy()

    def calc_nstep_q(self, observation, action, reward, next_observation, next_action, done):
        if type(reward) is float:
            reward = [reward]
        o = torch.tensor(observation, device=self._device, dtype=torch.float32)
        a = torch.tensor([[action]], device=self._device, dtype=torch.long)
        r = torch.tensor(reward, device=self._device, dtype=torch.float32)
        od = torch.tensor(next_observation, device=self._device, dtype=torch.float32)
        ad = torch.tensor([[next_action]], device=self._device, dtype=torch.long)

        with torch.no_grad():
            q = self._model(o.unsqueeze(0))
            q = q.gather(1, a)
            qd = self._model(od.unsqueeze(0))
            target = r[0] + self._gamma * qd.gather(1, ad) * (1 - done)

            for _r in r[1:]:
                target = _r + self._gamma * target
        return target.squeeze().to("cpu").detach().numpy().copy()

    def update(self, observation, action, reward, next_observation, next_action, done):
        self._replay_buffer.add(observation, action, reward, next_observation, done)

        o, a, r, od, d = self._replay_buffer.retrieval()
        o = torch.tensor(o, device=self._device, dtype=torch.float32)
        a = torch.tensor(a.reshape(-1, 1), device=self._device, dtype=torch.long)
        od = torch.tensor(od, device=self._device, dtype=torch.float32)
        q = self._model(o)
        q = q.gather(1, a)
        qd = self._model(od).max(1)[0].detach() * (1 - torch.tensor(d, device=self._device, dtype=torch.long))
        target = torch.tensor(r, device=self._device, dtype=torch.float32) + self._gamma * qd

        self._optimizer.zero_grad()
        loss = self._criterion(q.view(-1, 1), target.view(-1, 1))
        loss.backward()
        self._optimizer.step()


class LSTMQNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.hidden_size = hidden_size
        self.lstm = nn.LSTM(input_size, self.hidden_size, 1, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
        # self.lstm.register_forward_hook(lambda m, i, o: print("[out]", o))#[1][1]))

    def forward(self, x, ri, length=None):
        if length is not None:
            x = pack_padded_sequence(x, length, batch_first=True, enforce_sorted=False)
        x, ri = self.lstm(x, ri)
        if length is not None:
            x, _ = pad_packed_sequence(x, batch_first=True)
        x = self.fc(x)
        return x, ri


class LSTMQ(NeuralNetQ):
    def __init__(self, model=LSTMQNet, **kwargs):
        super().__init__(model, **kwargs)
        self._hidden_size = kwargs.get("hidden_size", 8)
        self._recurrent_input = None
        self._pre_recurrent_input = None
        self._replay_buffer = EpisodeReplayBuffer(1000, self._batch_size, -1)

    def q_table(self, observations):
        o = torch.tensor(observations, device=self._device, dtype=torch.float32)
        with torch.no_grad():
            q, _ = self._model(o.unsqueeze(1), None)
            q = q.squeeze().to("cpu").detach().numpy().copy()
        return q

    def q(self, observation):
        self._pre_recurrent_input = self._recurrent_input
        o = torch.tensor(observation, device=self._device, dtype=torch.float32)
        with torch.no_grad():
            q, self._recurrent_input = self._model(o.unsqueeze(0).unsqueeze(0), self._recurrent_input)
            q = q.squeeze().to("cpu").detach().numpy().copy()
        return q

    def reset(self):
        self._recurrent_input = None
        self._pre_recurrent_input = None
        self._replay_buffer.reset()

    def calc_temporal_difference(self, observation, action, reward, next_observation, next_action, done):
        o = torch.tensor(np.flip(observation).copy(), device=self._device, dtype=torch.float32)
        r = torch.tensor(reward, device=self._device)
        od = torch.tensor(next_observation[0], device=self._device, dtype=torch.float32)
        with torch.no_grad():
            q, ri = self._model(o.unsqueeze(0), self._recurrent_input)
            qd = self._model(od.unsqueeze(0).unsqueeze(0), ri)[0].max(2)[0] * (1 - done[0])
            target = r[0] + self._gamma * qd

            for _r in r[1:]:
                target = _r + self._gamma * target

        return (target - q[0, 0, action[-1]]).squeeze().to("cpu").detach().numpy().copy()

    def calc_nstep_q(self, observation, action, reward, next_observation, next_action, done):
        o = torch.tensor(np.flip(observation).copy(), device=self._device, dtype=torch.float32)
        r = torch.tensor(reward, device=self._device)
        od = torch.tensor(next_observation[0], device=self._device, dtype=torch.float32)
        with torch.no_grad():
            q, ri = self._model(o.unsqueeze(0), self._recurrent_input)
            qd = self._model(od.unsqueeze(0).unsqueeze(0), ri)[0].max(2)[0] * (1 - done[0])
            target = r[0] + self._gamma * qd

            for _r in r[1:]:
                target = _r + self._gamma * target

        return target.squeeze().to("cpu").detach().numpy().copy()

    # episode 単位での学習
    # 連続状態を扱う場合、規定長を取得、学習する用な処理にする（R2D2 を参考のこと）

    def update(self, observation, action, reward, next_observation, next_action, done):
        self._replay_buffer.add(observation, action, reward, next_observation, done)
        self._recurrent_input = self._pre_recurrent_input  # 方策でestimate からqを呼ぶ都合上、1つ戻す

        if done:
            # mini batch をreplay buffer から取得（ndarray のlist）
            # [batch_size, episode_length, input_size]
            o, a, r, od, d = self._replay_buffer.retrieval()

            # 各batch の系列長が異なるので、tensor のlist に変換
            # モデルに入力する状態s は、系列長を揃えて行列計算するため、padding する
            obs_lengths = [len(_o) for _o in o]
            o = [torch.tensor(_o, device=self._device, dtype=torch.float32) for _o in o]
            o = pad_sequence(o, batch_first=True, padding_value=-1)
            a = [torch.tensor(_a, device=self._device, dtype=torch.long).reshape(-1, 1) for _a in a]
            r = [torch.tensor(_r, device=self._device, dtype=torch.float32) for _r in r]
            od = torch.tensor(np.array([_od[-1] for _od in od]), device=self._device, dtype=torch.float32)
            d = torch.tensor(np.array([_d[-1] for _d in d]).reshape(-1, 1), device=self._device, dtype=torch.long)

            # Q 値とtarget の計算
            # Q 値の計算 [batch_size, episode_length, input_size] -> [batch_size, episode_length, num_actions]
            # ri は系列の最後の状態 [batch_size, 1, hidden_size]
            qs, ri = self._model(o, None, obs_lengths)

            # target の計算
            # 上で計算したQ 値の[1:episode_length]のmax と系列の最終遷移状態sd のQ 値のmax を結合する

            # sd は系列長1 なので、unsqueeze(1) で[batch_size, input_size] -> [batch_size, 1, input_size] にして入力
            # [batch_size, 1, num_actions] からaxis=2 のmax をとってmax(Q) し、detach で勾配計算しないように切り離す
            # 系列末端が終端状態の場合、0 にする
            lastqd = self._model(od.unsqueeze(1), ri)[0].max(2)[0].detach()
            lastqd *= (1 - d)
            # 各batch で[episode_length-1, num_actions] からaxis=1 のmax をとってdetach
            # qds とlastqd を結合し、[batch_size, episode_length, 1] に
            # target を計算後、cat して[1, batch_size * episode_length] に
            qds = [_qs[1:_ol].max(1)[0].detach() for _qs, _ol in zip(qs, obs_lengths)]
            qds = [torch.cat([_qds, _lastqd]) for _qds, _lastqd in zip(qds, lastqd)]
            targets = torch.cat([_r + self._gamma * _qds for _r, _qds in zip(r, qds)])

            # 各batch で[episode_length, num_actions] からaxis=1 でgather し、a をQ 値のindex にしてQ(a) を取得
            # cat して[batch_size * episode_length, 1] に
            qs = torch.cat([_qs[:_ol].gather(1, _a) for _qs, _ol, _a in zip(qs, obs_lengths, a)])

            self._optimizer.zero_grad()
            loss = self._criterion(qs.view(-1, 1), targets.view(-1, 1))
            # print(loss)
            # torch.set_printoptions(precision=2, edgeitems=1000, linewidth=400)
            # prei = self._model.lstm.weight_ih_l0.clone().detach()
            # preh = self._model.lstm.weight_hh_l0.clone().detach()
            loss.backward()
            # print("ihi_g", self._model.lstm.weight_ih_l0.grad[:self._hidden_size])
            # print("ihf_g", self._model.lstm.weight_ih_l0.grad[self._hidden_size:self._hidden_size*2])
            # print("ihg_g", self._model.lstm.weight_ih_l0.grad[self._hidden_size*2:self._hidden_size*3])
            # print("iho_g", self._model.lstm.weight_ih_l0.grad[self._hidden_size*3:self._hidden_size*4])
            # print("hhi_g", self._model.lstm.weight_hh_l0.grad[:self._hidden_size])
            # print("hhf_g", self._model.lstm.weight_hh_l0.grad[self._hidden_size:self._hidden_size*2])
            # print("hhg_g", self._model.lstm.weight_hh_l0.grad[self._hidden_size*2:self._hidden_size*3])
            # print("hho_g", self._model.lstm.weight_hh_l0.grad[self._hidden_size*3:self._hidden_size*4])
            # print(self._model._fc.weight.grad)
            self._optimizer.step()
            # print("ih", self._model.lstm.weight_ih_l0-prei)
            # print("hh", self._model.lstm.weight_hh_l0-preh)
            # print(self._model._fc.weight)

            # X = self._input(range(0, self._observation_space))

            # W_ii = self._model.lstm.weight_ih_l0[:self._hidden_size]
            # W_if = self._model.lstm.weight_ih_l0[self._hidden_size:self._hidden_size*2]
            # W_ig = self._model.lstm.weight_ih_l0[self._hidden_size*2:self._hidden_size*3]
            # W_io = self._model.lstm.weight_ih_l0[self._hidden_size*3:self._hidden_size*4]

            # W_hi = self._model.lstm.weight_hh_l0[:self._hidden_size]
            # W_hf = self._model.lstm.weight_hh_l0[self._hidden_size:self._hidden_size*2]
            # W_hg = self._model.lstm.weight_hh_l0[self._hidden_size*2:self._hidden_size*3]
            # W_ho = self._model.lstm.weight_hh_l0[self._hidden_size*3:self._hidden_size*4]

            # B_ii = self._model.lstm.bias_ih_l0[:self._hidden_size]
            # B_if = self._model.lstm.bias_ih_l0[self._hidden_size:self._hidden_size*2]
            # B_ig = self._model.lstm.bias_ih_l0[self._hidden_size*2:self._hidden_size*3]
            # B_io = self._model.lstm.bias_ih_l0[self._hidden_size*3:self._hidden_size*4]

            # B_hi = self._model.lstm.bias_hh_l0[:self._hidden_size]
            # B_hf = self._model.lstm.bias_hh_l0[self._hidden_size:self._hidden_size*2]
            # B_hg = self._model.lstm.bias_hh_l0[self._hidden_size*2:self._hidden_size*3]
            # B_ho = self._model.lstm.bias_hh_l0[self._hidden_size*3:self._hidden_size*4]

            # X = self._input([3,4,1])
            # print("3-1-5 inputs")
            # c_t = torch.zeros(1, self._hidden_size)
            # h_t = torch.zeros(1, self._hidden_size)
            # for x in X:
            #     i_t = torch.sigmoid(torch.matmul(W_ii, x.T).T + B_ii + torch.matmul(W_hi, h_t.T).T + B_hi)
            #     f_t = torch.sigmoid(torch.matmul(W_if, x.T).T + B_if + torch.matmul(W_hf, h_t.T).T  + B_hf)
            #     g_t = torch.tanh(torch.matmul(W_ig, x.T).T + B_ig + torch.matmul(W_hg, h_t.T).T  + B_hg)
            #     o_t = torch.sigmoid(torch.matmul(W_io, x.T).T + B_io + torch.matmul(W_ho, h_t.T).T  + B_ho)
            #     c_t = f_t * c_t + i_t * g_t
            #     h_t = o_t * torch.tanh(c_t)
            #     y = self._model.fc(h_t)
            # print(i_t, f_t, g_t, o_t)
            # print(c_t, h_t)
            # print(y)
            # print(F.softmax(y*10))


class Softmax(object):
    def __init__(self, learner=TabularQ, **kwargs):
        self._observation_space = kwargs["observation_space"]
        self._action_space = kwargs["action_space"]
        self._learner = learner(**kwargs)
        self._beta = kwargs.get("beta", 1.0)
        self._behavior = self._softmax
        self._estimate = self._greedy

    @property
    def q_table(self):
        return self._learner.q_table

    def get_biased_policy_distribution(self, observation):
        values = self._learner.q(observation) * self._beta
        values -= values.max()
        exp_values = np.exp(values)

        probs = exp_values / exp_values.sum()
        return probs
    
    def get_unbiased_policy_distribution(self, observation):
        values = self._learner.q(observation) * self._beta
        values -= values.max()
        exp_values = np.exp(values)

        probs = exp_values / exp_values.sum()
        return probs
    
    def _greedy(self, observation):
        values = self._learner.q(observation)
        maxQ = values.max()
        indices = np.where(values == maxQ)[0]
        action = np.random.choice(indices)
        return action

    def _calc_value(self, observation):
        return self._learner.q(observation)

    def _softmax(self, observation):
        values = self._calc_value(observation) * self._beta
        values -= values.max()
        exp_values = np.exp(values)

        probs = exp_values / exp_values.sum()
        action = np.random.choice(self._action_space, p=probs)
        return action

    def initialize(self):
        self._learner.initialize()

    def reset(self):
        self._learner.reset()

    def decision_making(self, observation):
        action = self._behavior(observation)
        return action

    def calc_temporal_difference(self, observation, action, reward, next_observation, done):
        estimate_action = self._estimate(next_observation)
        td = self._learner.calc_temporal_difference(observation, action, reward, next_observation, estimate_action, done)
        return td

    def update(self, observation, action, reward, next_observation, done):
        estimate_action = self._estimate(next_observation)
        self._learner.update(observation, action, reward, next_observation, estimate_action, done)


class EpsilonGreedy(Softmax):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._epsiron = kwargs.get("epsiron", 0.1)
        self._behavior = self._epsiron_greedy

    def _epsiron_greedy(self, observation):
        values = self._learner.q(observation)
        if np.random.random() < self._epsiron:
            indices = self._action_space
        else:
            maxQ = values.max()
            indices = np.where(values == maxQ)[0]
        action = np.random.choice(indices)
        return action


class MaintainedBiasedSoftmax(Softmax):
# bias-table を使ってエピソード内で bias を持続させる

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._env = kwargs.get("env")
        self._bias = np.zeros(self._action_space)  # 現状態におけるバイアス
        self._er_ratio = kwargs.get("er_ratio")
        self._er_deactivate_epis = kwargs.get("er_deactivate_epis")
        self._er_ratio = kwargs.get("er_ratio")
        self._bias_decay_rate = kwargs.get("bias_decay_rate")
        self._bias_definition = kwargs.get("bias_definition", "nstep_q")
        self._observation_repr = kwargs.get("observation_repr")
        self._bias_table = np.zeros([self._observation_space, self._action_space])
        self.focused_trajectory = None
        self.is_ignore_bias_weight = False

    def reset(self):
        super().reset()
        self._bias_table = np.zeros_like(self._bias_table)
        self.is_ignore_bias_weight = True if random.random() <= -1 else False

    @property  # animation作成に必要
    def bias(self):
        return self._bias.copy()

    def should_ignore_bias(self, epis):
        # 現在のepi数が "er_deactivate_epis" 以上であれば想起バイアスを無視する為のフラグをTrueにする
        if epis >= self._er_deactivate_epis:
            self.is_ignore_bias_weight = True
        else:
            self.is_ignore_bias_weight = False


    def get_biased_policy_distribution(self, observation):
        values = self._calc_value(observation) * self._beta
        values -= values.max()
        exp_values = np.exp(values)

        probs = exp_values / exp_values.sum()
        return probs
    
    def get_unbiased_policy_distribution(self, observation):
        values = self._learner.q(observation) * self._beta
        values -= values.max()
        exp_values = np.exp(values)

        probs = exp_values / exp_values.sum()
        return probs
    
    def bias_table(self, observations):
        _observations = [self._env._onehot2serial(observation) for observation in observations]
        return self._bias_table[_observations].copy()

    def _calc_value(self, observation):
        # biasを無視する場合
        if self.is_ignore_bias_weight:
            return self._learner.q(observation)
        # biasを無視しない場合
        else:
            return (
                (1 - self._er_ratio) * self._learner.q(observation)
                + self._er_ratio * self._bias
            )

    def update_bias_table(self, trajectories):
        self.focused_trajectory = [None for _ in range(self._action_space)]
        new_bias_table = np.zeros_like(self._bias_table)
        max_bias = -1 * np.inf
        for o, a, r, no, d in trajectories:
            if self._bias_definition == "nstep_q":
                na = self._estimate(no[0])
                tmp_bias = self._learner.calc_nstep_q(o[-1], a[-1], r, no[0], na, d[0])
            elif self._bias_definition == "sum_rew":
                tmp_bias = r.sum()
            else:
                print('Set the parameter ``bias_definition'' to one of the following; ["nstep_q", "sum_rew"]')

            if tmp_bias > max_bias:
                max_bias = tmp_bias
                self.focused_trajectory = [None for _ in range(self._action_space)]
                self.focused_trajectory[a[-1]] = (o, a, r, no)
                new_bias_table = np.zeros_like(new_bias_table)
                for _o, _a in zip(o, a):
                    if self._observation_repr == "onehot":
                        _o = self._env._onehot2serial(_o)
                    new_bias_table[_o, _a] = max_bias
        self._bias_table = new_bias_table

    def calc_bias(self, observation):
        if self._observation_repr == "onehot":
            observation = self._env._onehot2serial(observation)
        self._bias = self._bias_table[observation]

    def decay_bias(self):
        self._bias_table *= self._bias_decay_rate
        # pprint(self._bias_table)


class AdditiveMaintainedBiasedSoftmax(MaintainedBiasedSoftmax):
# bias_valの分散により全(s,a)に対するbiasの更新率を調整する

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._bias_var_power = kwargs.get("bias_var_power")
        del self.focused_trajectory

    def update_bias_table(self, trajectories):
        tl = len(trajectories)
        if tl == 0:
            return
        new_bias_table = np.zeros_like(self._bias_table)
        bias_vals = np.zeros(tl)
        for ti, (o, a, r, no, d) in enumerate(trajectories):
            if self._bias_definition == "nstep_q":
                na = self._estimate(no[0])
                bias_vals[ti] = self._learner.calc_nstep_q(o[-1], a[-1], r, no[0], na, d[0])
            elif self._bias_definition == "sum_rew":
                bias_vals[ti] = r.sum()
            else:
                print('Set the parameter ``bias_definition'' to one of the following; ["nstep_q", "sum_rew"]')
                sys.exit(1)

        ti_max = np.argmax(bias_vals)
        (o, a, r, no, d) = trajectories[ti_max]

        for _o, _a in zip(o, a):
            if self._observation_repr == "onehot":
                _o = self._env._onehot2serial(_o)
            new_bias_table[_o, _a] = bias_vals[ti_max]
        var_upper_bound = 1  # (rew_upper_bound - rew_lower_bound) ** 2 / 4
        bias_update_rate = 1 - (np.var(bias_vals) / var_upper_bound) ** self._bias_var_power
        self._bias_table += new_bias_table * bias_update_rate


class AdditiveMaintainedBiasedPerSASoftmax(AdditiveMaintainedBiasedSoftmax):
# bias_valの分散により(s,a)ごとにbiasの更新率を調整する

    def update_mean(self, data, old_mean, n):
        return (n * old_mean + data) / (n + 1)

    def update_var(self, data, old_mean, old_var, new_mean, n):
        v = (n * (old_var + old_mean**2) + data**2) / (n + 1) - (new_mean**2)
        if v < 0:
            return 0.0
        if v > 1:
            return 1.0
        return v

    def update_bias_table(self, trajectories):
        tl = len(trajectories)
        if tl == 0:
            return
        sample_mean_bias_table = np.zeros_like(self._bias_table)
        sample_var_bias_table = np.zeros_like(self._bias_table)
        freq_table = np.zeros_like(self._bias_table)

        for (o, a, r, no, d) in trajectories:
            if self._bias_definition == "nstep_q":
                na = self._estimate(no[0])
                bias_val = self._learner.calc_nstep_q(o[-1], a[-1], r, no[0], na, d[0])
            elif self._bias_definition == "sum_rew":
                bias_val = r.sum()
            else:
                print('Set the parameter ``bias_definition'' to one of the following; ["nstep_q", "sum_rew"]')

            for _o, _a in zip(o, a):
                if self._observation_repr == "onehot":
                    _o = self._env._onehot2serial(_o)
                # 想起したtrajectoryについて、(s,a)毎にバイアスの平均と分散の逐次更新を行う
                freq_table[_o, _a] += 1
                old_mean = sample_mean_bias_table[_o, _a]
                old_var = sample_var_bias_table[_o, _a]
                sample_mean_bias_table[_o, _a] = self.update_mean(bias_val, old_mean, freq_table[_o, _a])
                sample_var_bias_table[_o, _a] = self.update_var(bias_val, old_mean, old_var, sample_mean_bias_table[_o, _a], freq_table[_o, _a])

        var_upper_bound = 1  # (rew_upper_bound - rew_lower_bound) ** 2 / 4
        bias_update_rates = 1 - (sample_var_bias_table / var_upper_bound) ** self._bias_var_power
        self._bias_table += sample_mean_bias_table * bias_update_rates


# 各(s,a)におけるbias_valの分散を求め、それによりbiasの重みを調整する
class AdditiveMaintainedLSTMBiasedPerSASoftmax(AdditiveMaintainedBiasedPerSASoftmax):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.max_bias_policy_weight = kwargs.get("max_bias_policy_weight")
        self.var_bias_policy_weight = kwargs.get("var_bias_policy_weight")
    
    def update_bias_table_origin(self, trajectories):
        tl = len(trajectories)
        if tl == 0:
            return
        sample_mean_bias_table = np.zeros_like(self._bias_table)
        sample_var_bias_table = np.ones_like(self._bias_table)
        freq_table = np.zeros_like(self._bias_table)

        for (o, a, r, no, d) in trajectories:
            if self._bias_definition == "nstep_q":
                na = self._estimate(no[0])
                bias_val = self._learner.calc_nstep_q(o, a, r, no, na, d)
            elif self._bias_definition == "sum_rew":
                bias_val = r.sum()
            else:
                print('Set the parameter ``bias_definition'' to one of the following; ["nstep_q", "sum_rew"]')

            for _o, _a in zip(o, a):
                if self._observation_repr == "onehot":
                    _o = self._env._onehot2serial(_o)
                # 想起したtrajectoryについて、(s,a)毎にバイアスの平均と分散の逐次更新を行う
                freq_table[_o, _a] += 1
                old_mean = sample_mean_bias_table[_o, _a]
                old_var = sample_var_bias_table[_o, _a]
                sample_mean_bias_table[_o, _a] = self.update_mean(bias_val, old_mean, freq_table[_o, _a])
                sample_var_bias_table[_o, _a] = self.update_var(bias_val, old_mean, old_var, sample_mean_bias_table[_o, _a], freq_table[_o, _a])

        var_upper_bound = 1  # (rew_upper_bound - rew_lower_bound) ** 2 / 4
        bias_update_rates = 1 - ((sample_var_bias_table / var_upper_bound) ** self._bias_var_power)
        # if (sample_mean_bias_table * bias_update_rates).min() < 0.0:
        #     print(sample_mean_bias_table)
        #     print(bias_update_rates)
        #     print("######## minus update #########")
        self._bias_table += sample_mean_bias_table * bias_update_rates

    # 複数の policy での想起結果を統合して bias_table を更新
    # ToDo: 一回のfor文内で複数のpolicyを計算できるように修正すべきな気がする？(とりあえず後ででいい気はするが)
    def update_bias_table(self, trajectories):
        if self.max_bias_policy_weight > 0:
            max_policy_bias_table = self.update_bias_table_max_bias_policy(trajectories)
        else:
            max_policy_bias_table = np.zeros_like(self._bias_table)
        if self.var_bias_policy_weight > 0:
            var_policy_bias_table = self.update_bias_table_var_bias_policy(trajectories)
        else:
            var_policy_bias_table = np.zeros_like(self._bias_table)

        self._bias_table += self.max_bias_policy_weight*max_policy_bias_table + self.var_bias_policy_weight*var_policy_bias_table

    def update_bias_table_var_bias_policy(self, trajectories):
        sum_bias_table_for_mean = np.zeros_like(self._bias_table, dtype=np.float64)
        sum_bias_table_for_var = np.zeros_like(self._bias_table, dtype=np.float64)
        freq_table = np.zeros_like(self._bias_table, dtype=np.int32)

        if len(trajectories) == 0:
            return sum_bias_table_for_mean

        obs_list = []
        act_list = []
        bias_val_list = []

        for (o, a, r, no, d) in trajectories:
            if self._bias_definition == "nstep_q":
                na = self._estimate(no[0])
                bias_val = self._learner.calc_nstep_q(o, a, r, no, na, d)
            elif self._bias_definition == "sum_rew":
                bias_val = r.sum()
            else:
                print('Set the parameter ``bias_definition'' to one of the following; ["nstep_q", "sum_rew"]')
                continue

            for _o, _a in zip(o, a):
                if self._observation_repr == "onehot":
                    _o = self._env._onehot2serial(_o)
                obs_list.append(_o)
                act_list.append(_a)
                bias_val_list.append(bias_val)

        obs_array = np.array(obs_list)
        act_array = np.array(act_list)
        bias_val_array = np.array(bias_val_list)

        np.add.at(sum_bias_table_for_mean, (obs_array, act_array), bias_val_array)
        np.add.at(sum_bias_table_for_var, (obs_array, act_array), bias_val_array ** 2)
        np.add.at(freq_table, (obs_array, act_array), 1)

        sample_mean_bias_table = np.zeros_like(sum_bias_table_for_mean)
        sample_var_bias_table = np.zeros_like(sum_bias_table_for_var)

        nonzero = freq_table > 0
        sample_mean_bias_table[nonzero] = sum_bias_table_for_mean[nonzero] / freq_table[nonzero]
        sample_var_bias_table[nonzero] = (sum_bias_table_for_var[nonzero] / freq_table[nonzero]) - sample_mean_bias_table[nonzero] ** 2

        ma = 1
        if ma != 0:
            bias_update_rates = 1 - ((sample_var_bias_table / ma) ** self._bias_var_power)
        else:
            bias_update_rates = np.zeros_like(sample_var_bias_table)

        return sample_mean_bias_table * bias_update_rates
    
    # bias値のスケールでランク付けする
    def update_bias_table_max_bias_policy(self, trajectories):
        sum_bias_table_for_mean = np.zeros_like(self._bias_table, dtype=np.float64)
        freq_table = np.zeros_like(self._bias_table, dtype=np.int32)

        if len(trajectories) == 0:
            return sum_bias_table_for_mean

        obs_list = []
        act_list = []
        bias_val_list = []

        for (o, a, r, no, d) in trajectories:
            if self._bias_definition == "nstep_q":
                na = self._estimate(no[0])
                bias_val = self._learner.calc_nstep_q(o, a, r, no, na, d)
            elif self._bias_definition == "sum_rew":
                bias_val = r.sum()
            else:
                print('Set the parameter ``bias_definition'' to one of the following; ["nstep_q", "sum_rew"]')
                continue

            for _o, _a in zip(o, a):
                if self._observation_repr == "onehot":
                    _o = self._env._onehot2serial(_o)
                obs_list.append(_o)
                act_list.append(_a)
                bias_val_list.append(bias_val)

        obs_array = np.array(obs_list)
        act_array = np.array(act_list)
        bias_val_array = np.array(bias_val_list)

        np.add.at(sum_bias_table_for_mean, (obs_array, act_array), bias_val_array)
        np.add.at(freq_table, (obs_array, act_array), 1)

        sample_mean_bias_table = np.zeros_like(sum_bias_table_for_mean)

        nonzero = freq_table > 0
        sample_mean_bias_table[nonzero] = sum_bias_table_for_mean[nonzero] / freq_table[nonzero]

        # ma = sample_mean_bias_table.max()
        ma = 2
        if ma != 0:
            bias_update_rates = (sample_mean_bias_table / ma) ** self._bias_var_power
        else:
            bias_update_rates = np.zeros_like(sample_mean_bias_table)

        return sample_mean_bias_table * bias_update_rates


class OnlyERBias(AdditiveMaintainedLSTMBiasedPerSASoftmax):
    def _calc_value(self, observation):
        return self.bias


if __name__ == "__main__":
    def test(a, ad):
        torch.manual_seed(0)
        n = QNet(5, 2, 5)
        n.to("cuda")
        s = torch.zeros((len(a), 5)).to("cuda")
        for i, _a in enumerate(a):
            s[i][_a] = 1
        y = n(s)
        b = y.detach().clone()

        for i, _a in enumerate(a):
            b[i][_a] = 1
        print((y - b)**2)
        print((y.gather(1, ad) - b.gather(1, ad))**2)

        return n, y, b

    a = [[2], [3]]
    a = [[2]]
    ad = torch.tensor(a, device="cuda", dtype=torch.long)
    criterion = nn.MSELoss(reduction="sum")

    # n, y, b = test(a, ad)
    # l1 = criterion(b, y)

    torch.manual_seed(0)
    n = LSTMQNet(2, 5, 2)
    # s = torch.tensor([[[0, 1], [1, 0]]], dtype=torch.float)
    ri = (torch.zeros(1, 1, 5), torch.zeros(1, 1, 5))
    s = torch.tensor([[[0, 1], [1, 0]]], dtype=torch.float)
    y, _ri = n(s, ri)
    print(y, _ri)

    # n.zero_grad()
    # l1.backward()
    # print(n._fc1.weight.grad)
    # print(n._fc2.weight.grad)
