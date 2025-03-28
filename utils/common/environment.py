import numpy as np
import matplotlib.pyplot as plt
from matplotlib import patches
import sys


class Environment(object):
    def __init__(self):
        self.name = "Environment"
        self.tag = ""

        self._return = 0
        self._steps = 0
        self._episodes = -1
        self._reward = 0
        self._done = False
        self._limit = -1

    @property
    def limit(self):
        return self._limit

    @property
    def state_space(self):
        return self._state_space

    @property
    def observation_space(self):
        return self._observation_space

    @property
    def action_space(self):
        return self._action_space

    @property
    def done(self):
        return self._done

    @property
    def state(self):
        return self._state()

    @property
    def observation(self):
        return self._observation()

    @property
    def observation_table(self):
        return self._observation_table

    @property
    def envsize(self):
        return self._width, self._height

    def _state(self):
        return None

    def _update(self, action):
        return 0.0

    def initialize(self):
        self._episodes = -1

    def reset(self):
        self._return = 0

        self._steps = 0
        self._done = False
        self._episodes += 1

        return self.observation  # 初期状態における観測

    def step(self, action):
        self._steps += 1
        reward = self._update(action)
        self._return += reward

        return self.observation, reward, self.done, None

    def render(self):
        pass


class GridEnvironment(Environment):
    X = 1
    Y = 0

    MOVABLE = 0
    AGENT = 1
    OBSTACLE = 2
    TERMINAL = 3

    def __init__(self, width, height, limit=50, state_type="serial", tag="", is_render_agt=False):
        self._limit = limit
        self.name = f"GridEnvironment_lm{self._limit}"
        self.kind = "GridEnvironment"
        self.tag = tag

        self._state_type = state_type
        self._width = width
        self._height = height

        # environment 定義用
        self._environment = None
        self._term = None
        self._init_agent_position = None
        self._agent_position = None
        self._actions = None
        self._state = None
        self._observation_table = None
        self._restore = None
        self._observation = None

        # render 用
        self._render_width = 640
        self._render_height = 480
        self.xlim = [-1.0, self._width]
        self.ylim = [self._height, -1.0]
        self._fig = None
        self._ax = None
        self._bg = None
        self._agt = None
        self._color = ["#ffffff", "#ffffff", "#000000", "#77ff77"]
        self._is_render_agt = is_render_agt

        self._init_environment()
        self._create_observation_table()

        # 変数
        self._return = 0
        self._steps = 0
        self._episodes = -1
        self._reward = 0
        self._done = False

    @property
    def agent_position(self):
        return self._agent_position[GridEnvironment.X], self._agent_position[GridEnvironment.Y]

    @property
    def available_positions(self):
        x, y = np.meshgrid(np.arange(self._width), np.arange(self._height))
        return x, y

    @property
    def available_actions(self):
        ax = np.array([0.0, 0.0, -1.0, 1.0])
        ay = np.array([-1.0, 1.0, 0.0, 0.0])
        return ax, ay

    def _init_environment(self):
        self._environment_design()

        self._action_space = len(self._actions)

        if self._state_type == "serial":
            self._state_space = self._width * self._height
            self._state = self._serial
            self._restore = self._restore_serial
        elif self._state_type == "onehot":
            self._state_space = self._width * self._height
            self._state = self._onehot()
            self._restore = self._restore_onehot
        elif self._state_type == "grid":
            self._state_space = (self._height, self._width, 1)
            self._state = self._grid
            self._restore = self._restore_grid
        elif self._state_type == "image":
            self._state_space = (self._render_height, self._render_width, 3)
            self._state = self._image
            # self._restore = self._restore_grid
        self._observation = self._state  # NOTE: For normal tasks, observation == state.
        self._observation_space = self._state_space

    def initialize(self):
        self._episodes = -1
        self._init_environment()

    def _environment_design(self):
        self._environment = np.full((self._height, self._width), GridEnvironment.MOVABLE)
        self._term = (int(self._height - 1), int(self._width - 1))
        self._init_agent_position = np.array([0, 0])
        self._actions = [
            np.array([-1, 0]),  # UP
            np.array([1, 0]),  # DOWN
            np.array([0, -1]),  # LEFT
            np.array([0, 1])   # RIGHT
        ]
        self._environment[self._term] = GridEnvironment.TERMINAL

    def _create_observation_table(self):  # 環境全体のQ値を計算する用
        self._agent_position = np.array([0, 0])
        s = []
        x, y = self.available_positions
        for _x, _y in zip(x.reshape(-1), y.reshape(-1)):
            self._agent_position[GridEnvironment.X] = _x
            self._agent_position[GridEnvironment.Y] = _y
            s.append(self.observation)
        self._observation_table = np.array(s)

    def _serial(self):
        return np.array(self._agent_position[GridEnvironment.X] + self._agent_position[GridEnvironment.Y] * self._width)

    def _restore_serial(self, state):
        x = state % self._width
        y = state // self._width
        self._agent_position[GridEnvironment.X] = x
        self._agent_position[GridEnvironment.Y] = y

    def _onehot(self):

        def _encoder():
            eye = np.eye(self._state_space)
            return eye[self._serial()]
        return _encoder

    def _onehot2serial(self, onehot):
        return onehot.argmax()

    def _restore_onehot(self, state):
        idx = np.arange(self._state_space)
        serial = state @ idx
        self._restore_serial(serial)

    def _grid(self):
        env = self._environment.copy()
        env[self._agent_position[GridEnvironment.Y], self._agent_position[GridEnvironment.X]] = GridEnvironment.AGENT
        return env

    def _restore_grid(self, state):
        pos = np.where(state == GridEnvironment.AGENT)
        self._agent_position[GridEnvironment.X] = pos[GridEnvironment.X]
        self._agent_position[GridEnvironment.Y] = pos[GridEnvironment.Y]

    def _image(self):
        return self.render(mode="rgb_array")

    def _update(self, action):
        move = self._actions[action]
        self._check_movable(move)
        reward = self._check_event()
        return reward

    def _check_movable(self, move):
        self._agent_position += move
        if (self._agent_position[GridEnvironment.X] < 0) or (self._agent_position[GridEnvironment.X] >= self._width) \
                or (self._agent_position[GridEnvironment.Y] < 0) or (self._agent_position[GridEnvironment.Y] >= self._height) \
                or (self.on_target(GridEnvironment.OBSTACLE)):
            self._agent_position -= move

    def _check_event(self):
        reward = 0.0
        if self.on_target(GridEnvironment.TERMINAL):
            self._done = True
            reward = 1.0
        self._reward = reward
        if self._steps >= self._limit:
            self._done = True
        return reward

    def _init_renderer(self):  # TODO: figure 再生成をどうにかする
        # plt.ion()
        self._fig, self._ax = plt.subplots(figsize=(self._render_width / 100, self._render_height / 100))
        self._renderer_setting()
        self._create_background()
        self._fig.canvas.draw()
        self._bg = self._fig.canvas.copy_from_bbox(self._ax.bbox)
        x, y = self.renderer_pos(*self.agent_position)
        if self._is_render_agt:
            self._agt = self._ax.scatter(x, y, 80, c='b', zorder=10)

    def _create_background(self):
        st_x, st_y = self.available_positions
        r_x, r_y = self.renderer_pos(st_x, st_y)
        for x, y, rx, ry in zip(st_x.reshape(-1), st_y.reshape(-1), r_x.reshape(-1), r_y.reshape(-1)):
            if self._environment[y, x] == GridEnvironment.OBSTACLE:
                continue
            c = self._color[int(self._environment[y, x])]
            rct = patches.Rectangle((rx - 0.5, ry - 0.5), width=1, height=1, facecolor=c, edgecolor="black", zorder=1)
            self._ax.add_patch(rct)

    def _renderer_setting(self):
        self._ax.set_aspect("equal")
        self._ax.axis("off")
        self._ax.set_xlim(self.xlim)
        self._ax.set_ylim(self.ylim)

    def renderer_pos(self, x, y):
        return x, y

    def reset(self):
        super().reset()
        self._create_observation_table()
        self._agent_position = self._init_agent_position.copy()
        return self.observation

    def on_target(self, target):
        return self._environment[self._agent_position[GridEnvironment.Y], self._agent_position[GridEnvironment.X]] == target

    def restore_state(self, state):
        self._restore(state)
        self._check_event()

    def render(self, mode="human"):
        if self._fig is None:
            self._init_renderer()
        if mode == "human":
            self._fig.show()
        self._fig.canvas.restore_region(self._bg)
        x, y = self.renderer_pos(*self.agent_position)
        if self._is_render_agt:
            self._agt.set_offsets([x, y])
            self._ax.draw_artist(self._agt)
        if mode == "create_animation":
            return self._fig, self._ax, self._agt
        self._fig.canvas.blit(self._ax.bbox)
        self._fig.canvas.flush_events()
        if mode == "human":
            plt.pause(0.1)
        elif mode == "rgb_array":
            return np.fromstring(self._fig.canvas.renderer.tostring_rgb(), dtype=np.uint8).reshape((self._render_height, self._render_width, 3))


class SubOptima(GridEnvironment):
    TERMINAL2 = 5
    TERMINAL3 = 6
    TERMINAL4 = 7
    TERMINAL5 = 8
    TERMINAL6 = 9
    TERMINAL7 = 10
    TERMINAL8 = 11

    def __init__(self, width, height, sd, reward_upper, limit=50, state_type="serial", is_render_agt=False):
        self._reward_upper = reward_upper
        super().__init__(width, height, limit, state_type, tag="", is_render_agt=is_render_agt)
        self.name = f"SubOptima_lm{limit}_W{self._width}_H{self._height}_sd{sd}_rew-upper{reward_upper}"
        self.kind = f"SubOptima"
        self._color += ["#dcdcdc", "#ffaaaa", '#f4a6a6', '#a6d1f5', '#a1db9a', '#cf99d4', '#ffbb80', '#ffff99', '#d7b795', '#fccfe3']
        self._sd = [sd] * 8
        self._terminal_reward_population_mean = None

    def _environment_design(self):
        self._environment = np.full((self._height, self._width), GridEnvironment.MOVABLE)
        self._environment[:, :] = 0
        # self._term = [(0, 2), (0, 6), (2, 0), (2, 8), (6, 0), (6, 8), (8, 2), (8, 6)]
        self._term = [
            (0, 1+(self._width-7)//2),
            (0, 5+(self._width-7)//2),
            (1+(self._height-7)//2, 0),
            (1+(self._height-7)//2, self._width-1),
            (5+(self._height-7)//2, 0),
            (5+(self._height-7)//2, self._width-1),
            (self._height-1, 1+(self._width-7)//2),
            (self._height-1, 5+(self._width-7)//2)
        ]
        print(f"{self._term=}")
        self._init_agent_position = np.array([self._height//2, self._width//2])
        self._actions = [
            np.array([-1, 0]),  # UP
            np.array([1, 0]),  # DOWN
            np.array([0, -1]),  # LEFT
            np.array([0, 1])   # RIGHT
        ]
        if self._reward_upper == 8:
            self._mean = [8.0, 1.0, 3.0, 5.0, 6.0, 4.0, 2.0, 7.0]
        elif self._reward_upper == 1:
            self._mean = [1.0, 0.1, 0.3, 0.7, 0.8, 0.4, 0.2, 0.9]
        elif self._reward_upper == 2:
            self._mean = [2.0, 0.25, 0.75, 1.25, 1.5, 1.0, 0.5, 1.75]
        else:
            print("現在考慮されていない報酬設定です")
            exit()
        self._environment[self._term[0]] = SubOptima.TERMINAL
        self._environment[self._term[1]] = SubOptima.TERMINAL2
        self._environment[self._term[2]] = SubOptima.TERMINAL3
        self._environment[self._term[3]] = SubOptima.TERMINAL4
        self._environment[self._term[4]] = SubOptima.TERMINAL5
        self._environment[self._term[5]] = SubOptima.TERMINAL6
        self._environment[self._term[6]] = SubOptima.TERMINAL7
        self._environment[self._term[7]] = SubOptima.TERMINAL8

        

    def _check_event(self):
        reward = 0.0
        if self.on_target(SubOptima.TERMINAL):
            self._done = True
            reward = np.random.normal(loc=self._mean[0], scale=self._sd[0])
            self._terminal_reward_population_mean = self._mean[0]
        elif self.on_target(SubOptima.TERMINAL2):
            self._done = True
            reward = np.random.normal(loc=self._mean[1], scale=self._sd[1])
            self._terminal_reward_population_mean = self._mean[1]
        elif self.on_target(SubOptima.TERMINAL3):
            self._done = True
            reward = np.random.normal(loc=self._mean[2], scale=self._sd[2])
            self._terminal_reward_population_mean = self._mean[2]
        elif self.on_target(SubOptima.TERMINAL4):
            self._done = True
            reward = np.random.normal(loc=self._mean[3], scale=self._sd[3])
            self._terminal_reward_population_mean = self._mean[3]
        elif self.on_target(SubOptima.TERMINAL5):
            self._done = True
            reward = np.random.normal(loc=self._mean[4], scale=self._sd[4])
            self._terminal_reward_population_mean = self._mean[4]
        elif self.on_target(SubOptima.TERMINAL6):
            self._done = True
            reward = np.random.normal(loc=self._mean[5], scale=self._sd[5])
            self._terminal_reward_population_mean = self._mean[5]
        elif self.on_target(SubOptima.TERMINAL7):
            self._done = True
            reward = np.random.normal(loc=self._mean[6], scale=self._sd[6])
            self._terminal_reward_population_mean = self._mean[6]
        elif self.on_target(SubOptima.TERMINAL8):
            self._done = True
            reward = np.random.normal(loc=self._mean[7], scale=self._sd[7])
            self._terminal_reward_population_mean = self._mean[7]
        self._reward = reward
        if self._steps >= self._limit:
            self._done = True
            self._terminal_reward_population_mean = -1 # step上限でタスクが終了した場合は終端報酬平均は-1とする。報酬設計を負にしない前提なので後々修正の必要アリかも？

        return reward


class HiddenExplore(GridEnvironment):
    HIDDEN = 4

    def __init__(self, width, height, limit=50, swap=-1, state_type="serial", is_render_agt=False):
        self._mean = None
        self._sd = None
        self._hidden1 = None
        self._hidden2 = None
        self._flag = False
        self._swap_episode = int(swap)

        super().__init__(width, height, limit, state_type, tag="", is_render_agt=is_render_agt)

        self.name = f"HiddenExplore_lm{self._limit}"
        self.kind = "HiddenExplore"
        if swap > 0:
            self.name += f"_sw{self._swap_episode}"
            self.kind += f"_sw{self._swap_episode}"

        self._color += ["#c0c0c0"]

    def _environment_design(self):
        self._environment = np.full((self._height, self._width), GridEnvironment.MOVABLE)
        self._term = (int(self._height / 2), int(self._width - 1))
        self._init_agent_position = np.array([int((self._height) / 2), 0])
        self._hidden1 = (0, int(self._width / 2))
        self._hidden2 = (int(self._height - 1), int(self._width / 2))
        self._actions = [
            np.array([-1, 0]),  # UP
            np.array([1, 0]),  # DOWN
            np.array([0, -1]),  # LEFT
            np.array([0, 1])   # RIGHT
        ]
        self._mean = 2.0
        self._sd = .0
        self._environment[self._term] = GridEnvironment.TERMINAL
        self._environment[self._hidden1] = HiddenExplore.HIDDEN

    def _check_event(self):
        reward = 0.0
        if self.on_target(GridEnvironment.TERMINAL):
            if self._flag:
                self._done = True
                reward = np.random.normal(loc=self._mean, scale=self._sd)
        elif self.on_target(HiddenExplore.HIDDEN):
            self._flag = True
        self._reward = reward
        if self._steps >= self._limit:
            self._done = True
        return reward

    def _swap(self):
        temp = np.copy(self._environment[self._hidden1])
        self._environment[self._hidden1] = np.copy(self._environment[self._hidden2])
        self._environment[self._hidden2] = np.copy(temp)

    def reset(self):
        observation = super().reset()
        self._flag = False
        if (self._episodes == (self._swap_episode - 1)):
            self._swap()
        if self._fig is None:
            self._init_renderer()
        return observation


class HiddenExploreHex(HiddenExplore):
    RT3 = np.sqrt(3)

    def __init__(self, width, height, limit=50, swap=-1, state_type="serial", is_render_agt=False):
        super().__init__(width, height, limit, swap, state_type, is_render_agt=is_render_agt)
        self.name = f"HiddenExploreHex_lm{self._limit}_W{self._width}_H{self._height}"
        self.kind = "HiddenExploreHex"
        if swap > 0:
            self.name += f"_sw{self._swap_episode}"
            self.kind += f"_sw{self._swap_episode}"

        self.xlim = [-1.5, self._width]
        self.ylim = [self._height * HiddenExploreHex.RT3 / 2, -HiddenExploreHex.RT3 / 2]

    @property
    def available_actions(self):
        ax = np.array([-np.sin(np.pi / 6), np.sin(np.pi / 6), -np.sin(np.pi / 6), np.sin(np.pi / 6), -1.0, 1.0])
        ay = np.array([-np.cos(np.pi / 6), -np.cos(np.pi / 6), np.cos(np.pi / 6), np.cos(np.pi / 6), 0, 0])
        return ax, ay

    def _environment_design(self):
        super()._environment_design()
        self._actions = np.array([
            [[-1, 0], [-1, -1]],  # UPLEFT
            [[-1, 1], [-1, 0]],  # UPRIGHT
            [[1, 0], [1, -1]],  # DOWNLEFT
            [[1, 1], [1, 0]],  # DOWNRIGHT
            [[0, -1], [0, -1]],  # LEFT
            [[0, 1], [0, 1]]   # RIGHT
        ])

    def _update(self, action):
        action = (action, self._agent_position[GridEnvironment.Y] % 2)
        return super()._update(action)

    def _create_background(self):
        st_x, st_y = self.available_positions
        r_x, r_y = self.renderer_pos(st_x, st_y)
        for x, y, rx, ry in zip(st_x.reshape(-1), st_y.reshape(-1), r_x.reshape(-1), r_y.reshape(-1)):
            if self._environment[y, x] == GridEnvironment.OBSTACLE:
                continue
            c = self._color[int(self._environment[y, x])]
            rp = patches.RegularPolygon((rx, ry), 6, radius=1 / HiddenExploreHex.RT3, facecolor=c, edgecolor="black", zorder=1)
            self._ax.add_patch(rp)

    def renderer_pos(self, x, y):
        hx = x - 0.5 * (y % 2)
        hy = y * HiddenExploreHex.RT3 / 2
        return hx, hy


class HallwayMaze(GridEnvironment):
    def __init__(self, width, height, limit=50, state_type="serial"):
        super().__init__(width, height, limit, state_type)
        if self._width < 5 or self._height < 5:
            try:
                raise ValueError("This task must have a width >= 5 and height >= 5")
            except ValueError as e:
                print(e)
        self.name = f"HallwayMaze_lm{self._limit}_W{self._width}"
        self.kind = f"HallwayMaze"

    def _init_environment(self):
        super()._init_environment()
        self._environment_design()
        self._observation_space = 2**self._action_space
        self._observation = self._surroundings_onehot

    def _restore_onehot(self, state):
        idx = np.arange(self._state_space)
        serial = state @ idx
        self._restore_serial(serial)

    def _surroundings2serial(self, surroundings):
        vec = np.array([2**i for i in range(self._action_space)])
        serial = surroundings @ vec
        return int(serial)

    def _surroundings2onehot(self, surroundings):
        serial = self._surroundings2serial(surroundings)
        return np.eye(self._observation_space)[serial]

    def _surroundings_onehot(self):
        return self._surroundings2onehot(self._surroundings())

    def _surroundings(self):
        # FIXME: スケーラブルに記述するのが大変なのでとりあえずaction_space=4でのみ動作する
        if self.action_space != 4:
            print("action_space must be 4 for current implementation.")
            sys.exit(1)
        surroundings = np.zeros(self._action_space)
        env = self._environment.copy()
        if self._agent_position is not None:
            if self._agent_position[GridEnvironment.Y] == 0:
                surroundings[0] = 1  # 現在位置が0行目なら上に壁があるFlagを立てる
            elif env[self._agent_position[GridEnvironment.Y] - 1, self._agent_position[GridEnvironment.X]] == GridEnvironment.OBSTACLE:
                surroundings[0] = 1  # あるいは、上のマスが壁のときFlagを立てる
            if self._agent_position[GridEnvironment.Y] == self._height - 1:
                surroundings[1] = 1  # 現在位置が最終行なら下に……(以下同様)
            elif env[self._agent_position[GridEnvironment.Y] + 1, self._agent_position[GridEnvironment.X]] == GridEnvironment.OBSTACLE:
                surroundings[1] = 1
            if self._agent_position[GridEnvironment.X] == 0:
                surroundings[2] = 1
            elif env[self._agent_position[GridEnvironment.Y], self._agent_position[GridEnvironment.X] - 1] == GridEnvironment.OBSTACLE:
                surroundings[2] = 1
            if self._agent_position[GridEnvironment.X] == self._width - 1:
                surroundings[3] = 1
            elif env[self._agent_position[GridEnvironment.Y], self._agent_position[GridEnvironment.X] + 1] == GridEnvironment.OBSTACLE:
                surroundings[3] = 1
        return surroundings

    def step(self, action):
        self._steps += 1
        reward = self._update(action)
        self._return += reward
        # print(f"ser: {np.argmax(self.state)}, (x,y): ({self._agent_position[GridEnvironment.X]}, {self._agent_position[GridEnvironment.Y]}), [上,下,左,右]: {self.observation}")
        return self.observation, reward, self.done, None

    def _environment_design(self):
        self._environment = np.full((self._height, self._width), GridEnvironment.OBSTACLE)
        self._environment[0, :] = 0
        self._environment[-1, :] = 0
        self._environment[:, 0] = 0
        self._environment[:, -1] = 0
        self._environment[:, self._width // 2] = 0
        self._term = (self._height // 2, self._width // 2)
        self._actions = [
            np.array([-1, 0]),  # UP
            np.array([1, 0]),  # DOWN
            np.array([0, -1]),  # LEFT
            np.array([0, 1])   # RIGHT
        ]
        self._mean = 2.0
        self._sd = 0.0
        self._environment[self._term] = GridEnvironment.TERMINAL

    def _check_event(self):
        reward = 0.0
        if self.on_target(GridEnvironment.TERMINAL):
            self._done = True
            reward = np.random.normal(loc=self._mean, scale=self._sd)
        self._reward = reward
        if self._steps >= self._limit:
            self._done = True
        return reward

    def reset(self, hidden=None):  # hiddenは、ランダム性のある環境を再現するための変数
        self._hidden = np.random.choice(4) if hidden is None else hidden
        init_agent_positions = [
            np.array([0, 0]), np.array([0, self._width - 1]),
            np.array([self._height - 1, 0]), np.array([self._height - 1, self._width - 1])]
        self._init_agent_position = init_agent_positions[self._hidden]
        observation = super().reset()
        if self._fig is not None:
            self._init_renderer()
        return observation


class Tmaze(GridEnvironment):
    def __init__(self, width, height=3, limit=50, state_type="serial", is_render_agt=False):
        super().__init__(width, height, limit, state_type, tag="", is_render_agt=is_render_agt)
        self.name = f"Tmaze_lm{self._limit}_W{self._width}"
        self.kind = f"Tmaze"

    def _environment_design(self):
        self._environment = np.full((self._height, self._width), GridEnvironment.OBSTACLE)
        self._environment[1, :] = 0
        self._environment[:, -1] = 0
        self._term = (0, int(self._width - 1))
        self._init_agent_position = np.array([1, 0])
        self._actions = [
            np.array([-1, 0]),  # UP
            np.array([1, 0]),  # DOWN
            np.array([0, -1]),  # LEFT
            np.array([0, 1])   # RIGHT
        ]
        self._mean = 2.0
        self._sd = 0.0
        self._environment[self._term] = GridEnvironment.TERMINAL

    def _check_event(self):
        reward = 0.0
        if self.on_target(GridEnvironment.TERMINAL):
            self._done = True
            reward = np.random.normal(loc=self._mean, scale=self._sd)
        self._reward = reward
        if self._steps >= self._limit:
            self._done = True
        return reward


class TmazeHidden(Tmaze):
    HIDDEN = 4
    TERMINAL2 = 5

    def __init__(self, width, height=3, limit=50, state_type="serial", is_render_agt=False):
        self._hidden = 0
        super().__init__(width, height, limit, state_type, is_render_agt=is_render_agt)
        self.name = f"TmazeHidden_lm{self._limit}_W{self._width}"
        self.kind = f"TmazeHidden"
        self._color += ["#c0c0c0", "#ff4444"]

    @property
    def hidden(self):
        return self._hidden

    def _init_environment(self):
        super()._init_environment()

        if self._state_type == "serial":
            self._state_space *= 2
            self._state = self._serial
        elif self._state_type == "onehot":
            self._state_space = self._state_space + 1  # 隠れ状態を示す1bitを加える
            self._state = self._onehot()
        self._observation = self._state  # In this task, observation == state.
        self._observation_space = self._state_space

    def _environment_design(self):
        super()._environment_design()
        self._environment[self._term] = GridEnvironment.MOVABLE
        self._term = [(0, int(self._width - 1)), (2, int(self._width - 1))]
        self._environment[1, 1] = TmazeHidden.HIDDEN
        self._mean = [2.0, 0.0]
        self._sd = [0.0, 0.0]

    def _serial(self):
        state = super()._serial()
        if self.on_target(TmazeHidden.HIDDEN):
            state += (self._width * self._height) * self._hidden
        return state

    def _restore_serial(self, state):
        size = (self._width * self._height)
        super()._restore_serial(state % size)
        self._hidden = int(state // size)

    # NOTE: 「通常の状態のonehot + 隠れ状態を示す1bit」 なので、実際にはonehotではない
    def _onehot(self):
        eye = np.eye(self._width * self._height)

        def _encoder():
            if self.on_target(TmazeHidden.HIDDEN) and self._hidden == 1:
                return np.concatenate(
                    [eye[super(TmazeHidden, self)._serial()], [1]]
                )
            else:
                return np.concatenate(
                    [eye[super(TmazeHidden, self)._serial()], [0]]
                )
        return _encoder

    def _onehot2serial(self, onehot):
        return onehot[:-1].argmax()

    def _restore_onehot(self, state):
        size = self._width * self._height
        indices = np.arange(size)
        serial = state[:size] @ indices  # NOTE: @は行列積
        super()._restore_serial(serial)

        if np.sum(state[-1]) != 0:
            self._hidden = 1

    def _grid(self):
        env = super()._grid()
        if (not self.on_target(TmazeHidden.HIDDEN)):
            if (not self.on_target(TmazeHidden.TERMINAL)):
                env[self._term[1 - self._hidden]] = GridEnvironment.MOVABLE
            if (not self.on_target(TmazeHidden.TERMINAL2)):
                env[self._term[self._hidden]] = GridEnvironment.MOVABLE
        return env

    def _restore_grid(self, state):
        super()._restore_grid(state)
        if (self.on_target(TmazeHidden.HIDDEN)):
            if state[self._term[0]] == TmazeHidden.TERMINAL2:
                self._hidden = 0
            else:
                self._hidden = 1
            self._environment[self._term[1 - self._hidden]] = TmazeHidden.TERMINAL
            self._environment[self._term[self._hidden]] = TmazeHidden.TERMINAL2

    def _check_event(self):
        reward = 0.0
        if self.on_target(TmazeHidden.TERMINAL):
            self._done = True
            reward = np.random.normal(loc=self._mean[0], scale=self._sd[0])
        elif self.on_target(TmazeHidden.TERMINAL2):
            self._done = True
            reward = np.random.normal(loc=self._mean[1], scale=self._sd[1])
        self._reward = reward
        if self._steps >= self._limit:
            self._done = True

        return reward

    def reset(self, hidden=None):
        self._hidden = np.random.choice(2) if hidden is None else hidden
        self._environment[self._term[1 - self._hidden]] = TmazeHidden.TERMINAL
        self._environment[self._term[self._hidden]] = TmazeHidden.TERMINAL2
        observation = super().reset()
        if self._fig is None:
            self._init_renderer()
        return observation


class ExtendedTmazeHidden(TmazeHidden):
    # 隠れ状態を踏まなくとも終端状態に到達すればエピソードが終了

    def __init__(self, width, height, limit=50, swap=-1, state_type="serial", is_render_agt=False):
        super().__init__(width, height, limit, state_type, is_render_agt=is_render_agt)
        self._swap_episode = int(swap)
        self.name = f"ExtendedTmazeHidden_lm{self._limit}_W{self._width}_H{self._height}"
        self.kind = "ExtendedTmazeHidden"
        if swap > 0:
            self.name += f"_sw{self._swap_episode}"
            self.kind += f"_sw{self._swap_episode}"

    def _environment_design(self):
        super()._environment_design()
        self._environment = np.full((self._height, self._width), GridEnvironment.MOVABLE)
        self._term = [(0, int(self._width - 1)), (int(self._height - 1), int(self._width - 1))]
        self._init_agent_position = np.array([int((self._height - 1) / 2), 0])
        self._environment[:, int(self._width - 2)] = TmazeHidden.OBSTACLE  # 一列に障害物を設定
        self._environment[int((self._height - 1) / 2), int(self._width - 2)] = TmazeHidden.MOVABLE  # 一列の障害物の内の１マスを開ける
        self._hidden1 = (0, int((self._width - 2) / 2))  # 右2列のT字路を除いた中心に隠れ状態を設置
        self._hidden2 = (int(self._height - 1), int((self._width - 2) / 2))
        self._environment[self._hidden1] = TmazeHidden.HIDDEN
        self._mean = [2.0, 0.0]
        self._sd = [0.0, 0.0]

    def _swap(self):
        temp = np.copy(self._environment[self._hidden1])
        self._environment[self._hidden1] = np.copy(self._environment[self._hidden2])
        self._environment[self._hidden2] = np.copy(temp)

    def reset(self, hidden=None):
        observation = super().reset(hidden)
        if (self._episodes == (self._swap_episode - 1)):
            self._swap()
        return observation


class FlaggedExtendedTmazeHidden(ExtendedTmazeHidden):
    # 隠れ状態を踏まなければ終端状態に到達してもエピソードが終了しない
    def __init__(self, width, height, limit=50, swap=-1, state_type="serial", is_render_agt=False):
        super().__init__(width, height, limit, swap, state_type, is_render_agt=is_render_agt)
        self.name = f"FlaggedExtendedTmazeHidden_lm{self._limit}_W{self._width}_H{self._height}"
        self.kind = "FlaggedExtendedTmazeHidden"
        if swap > 0:
            self.name += f"_sw{self._swap_episode}"
            self.kind += f"_sw{self._swap_episode}"
        self._flag = False

    def _check_event(self):
        reward = 0.0
        if self.on_target(HiddenExplore.HIDDEN):
            self._flag = True
        elif self.on_target(TmazeHidden.TERMINAL):
            if self._flag:
                self._done = True
                reward = np.random.normal(loc=self._mean[0], scale=self._sd[0])
        elif self.on_target(TmazeHidden.TERMINAL2):
            if self._flag:
                self._done = True
                reward = np.random.normal(loc=self._mean[1], scale=self._sd[1])
        self._reward = reward
        if self._steps >= self._limit:
            self._done = True
        return reward

    def reset(self, hidden=None):
        self._flag = False
        observation = super().reset(hidden)
        return observation


if __name__ == "__main__":
    def env_test(env):
        print(env.name)
        print(env.state_space)
        # print(env.state_table)
        s = env.reset()
        env.render()
        print(env.state, env.agent_position)
        env.step(3)
        env.render()
        print(env.state, env.agent_position)
        env.step(0)
        env.render()
        print(env.state, env.agent_position)
        env.step(3)
        env.render()
        print(env.state, env.agent_position)
        s2, _, _, _ = env.step(3)
        env.render()
        print(env.state.shape, env.agent_position)
        env.restore_state(s)
        env.render()
        print(env.state, env.agent_position)
        env.restore_state(s2)
        env.render()
        print(env.state, env.agent_position)

    # env_test(GridEnvironment(3, 3, state_type="serial"))
    # env_test(GridEnvironment(3, 3, state_type="onehot"))
    # env_test(GridEnvironment(3, 3, state_type="grid"))
    # env_test(GridEnvironment(3, 3, state_type="image"))

    # env_test(HiddenExplore(3, 5, state_type="serial"))
    # env_test(HiddenExplore(3, 5, state_type="onehot"))
    # env_test(HiddenExplore(3, 5, state_type="grid"))
    # env_test(HiddenExplore(3, 5, state_type="image"))

    # env_test(HiddenExploreHex(3, 5, state_type="serial"))
    # env_test(HiddenExploreHex(3, 5, state_type="onehot"))
    # env_test(HiddenExploreHex(3, 5, state_type="grid"))
    # env_test(HiddenExploreHex(3, 5, state_type="image"))

    # env_test(Tmaze(3, state_type="serial"))
    # env_test(Tmaze(3, state_type="onehot"))
    # env_test(Tmaze(3, state_type="grid"))
    # env_test(Tmaze(3, state_type="image"))

    # env_test(TmazeHidden(3, state_type="serial"))
    # env_test(TmazeHidden(3, state_type="onehot"))
    # env_test(TmazeHidden(3, state_type="grid"))
    # env_test(TmazeHidden(3, state_type="image"))

    env_test(SubOptima(9, 9, sd=0, reward_upper=1, state_type="serial"))
    # env_test(SubOptima(9, 9, sd=0, reward_upper=1, state_type="onehot"))
    # env_test(SubOptima(9, 9, sd=0, reward_upper=1, state_type="grid"))
    # env_test(SubOptima(9, 9, sd=0, reward_upper=1, state_type="image"))
