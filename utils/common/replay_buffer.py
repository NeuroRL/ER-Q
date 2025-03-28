from collections import defaultdict, deque
import numpy as np


# 純正Replay Buffer
class ReplayBuffer(object):
    def __init__(self, memory_capacity, batch_size):
        self._memory_capacity = memory_capacity
        self._memory = []
        self._memory_append = self._memory.append
        self._index = 0
        self._batch_size = batch_size

    def initialize(self):
        self._memory = []
        self._memory_append = self._memory.append
        self._index = 0

    def reset(self):
        pass

    def add(self, state, action, reward, next_state, done):
        transition = (state, action, reward, next_state, done)

        if self._memory_capacity > len(self._memory):
            self._memory_append(transition)
        else:
            self._memory[self._index] = transition
        self._index = (self._index + 1) % self._memory_capacity

    def _encode(self, indices):
        s, a, r, sd, d = self._memory[0]
        states = np.empty((len(indices),) + s.shape, dtype=np.float32)
        actions = np.empty((len(indices),), dtype=np.int32)
        rewards = np.empty((len(indices),), dtype=np.float32)
        next_states = np.empty((len(indices),) + sd.shape, dtype=np.float32)
        dones = np.empty((len(indices),), dtype=np.bool_)

        for i, index in enumerate(indices):
            s, a, r, sd, d = self._memory[index]
            states[i] = s
            actions[i] = a
            rewards[i] = r
            next_states[i] = sd
            dones[i] = d
        return states, actions, rewards, next_states, dones

    def retrieval(self):
        indices = np.random.randint(0, len(self._memory), (self._batch_size, ))

        return self._encode(indices)


class EpisodeReplayBuffer(ReplayBuffer):
    def __init__(self, memory_capacity, batch_size, forward_length):
        super().__init__(memory_capacity, batch_size)
        self._temp_memory = []
        self._forward_length = forward_length

    def initialize(self):
        super().initialize()
        self._temp_memory = []

    def add(self, state, action, reward, next_state, done):
        transition = (state, action, reward, next_state, done)

        self._temp_memory.append(transition)
        if done:
            if self._memory_capacity > len(self._memory):
                self._memory_append(self._temp_memory)
            else:
                self._memory[self._index] = self._temp_memory
            self._index = (self._index + 1) % self._memory_capacity
            self._temp_memory = []

    def _encode(self, trajectories):
        states, actions, rewards, next_states, dones = [], [], [], [], []
        for trajectory in trajectories:
            _ss = []
            _as = []
            _rs = []
            _nss = []
            _ds = []
            for s, a, r, sd, d in trajectory:
                _ss.append(s)
                _as.append(a)
                _rs.append(r)
                _nss.append(sd)
                _ds.append(d)
            states.append(np.array(_ss))
            actions.append(np.array(_as))
            rewards.append(np.array(_rs))
            next_states.append(np.array(_nss))
            dones.append(np.array(_ds))

        return states, actions, rewards, next_states, dones

    def retrieval(self):
        indices = np.random.randint(0, len(self._memory), (self._batch_size, ))

        sl = slice(self._forward_length) if self._forward_length != -1 else slice(None)
        trajectories = [
            self._memory[idx][sl] for idx in indices
        ]

        return self._encode(trajectories)


# ER_A
# 状態s で検索して特定の長さの軌跡を想起するためのReplay Buffer
class HypotheticalReplayBuffer(ReplayBuffer):
    def __init__(self, memory_capacity, batch_size, reverse_length):
        super().__init__(memory_capacity, batch_size)
        self._hypothetical_memory = defaultdict(deque)
        self._reverse_length = reverse_length

    def initialize(self):
        super().initialize()
        self._hypothetical_memory = defaultdict(deque)

    def add(self, state, action, reward, next_state, done):
        # buffer容量の上限に達しているか判定
        # 達している場合、最古のtransition をretrieval_memory から削除
        if self._memory_capacity == len(self._memory):
            s = self._memory[self._index][0]  # sが先頭に格納されているため0を指定
            _s = s.tobytes()
            if _s in self._hypothetical_memory:
                hypothetical_target = self._hypothetical_memory[_s]
                hypothetical_target.popleft()
                if len(hypothetical_target) == 0:
                    del self._hypothetical_memory[_s]
        self._hypothetical_memory[state.tobytes()].append(self._index)
        super().add(state, action, reward, next_state, done)

    # 状態s に関連する軌跡を想起する
    def retrieval(self, state):
        retrieval_target = self._hypothetical_memory[state.tobytes()]
        trajectories = []
        trajectories_append = trajectories.append
        if len(retrieval_target) > 0:
            indices = np.random.randint(0, len(retrieval_target), (self._batch_size, ))
            for index in indices:
                target_index = retrieval_target[index]
                retrieval_indices = []
                retrieval_indices_append = retrieval_indices.append
                memory_length = len(self._memory)
                idx = target_index
                while (idx < memory_length) and (idx != (self._reverse_length + target_index) % self._memory_capacity):
                    retrieval_indices_append(idx)
                    if (self._memory[idx][-1]) or (idx == self._index - 1):
                        break
                    idx = (idx + 1) % self._memory_capacity
                trajectories_append(self._encode(list(reversed(retrieval_indices))))
        return trajectories


# 現エピソードの軌跡については想起しないER_A用のBuffer
class HypotheticalEpisodeReplayBuffer(HypotheticalReplayBuffer):
    def __init__(self, memory_capacity, batch_size, reverse_length):
        super().__init__(memory_capacity, batch_size, reverse_length)
        self._memory_current_episode = []

    def initialize(self):
        super().initialize()
        self._memory_current_episode = []

    def add(self, state, action, reward, next_state, done):
        transition = (state, action, reward, next_state, done)
        self._memory_current_episode.append(transition)
        if done:
            # 現エピソードの内容をまとめて追加
            for (_state, _action, _reward, _next_state, _done) in self._memory_current_episode:
                super().add(_state, _action, _reward, _next_state, _done)
            self._memory_current_episode = []  # 現エピソードのtemporaryな記憶をinit
