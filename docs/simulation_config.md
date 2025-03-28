#  このドキュメントに記載されていること
このドキュメントには、「**シミュレーションを実施するための設定に関する説明や注意事項**」が記載されています。

# 概要
## ディレクトリ&ファイル構成
```bash
.
└──  config
    ├── ...
    ├── simulation
    │   ├── main.json # 実施するシミュレーションを統括するJSON
    │   ├── default
    │   │   └── sim_config_default.json # デフォルト設定を記述するJSON
    │   ├── sim-01-bw-cmp
    │   │   └── sim_config.json # 実施したいシミュレーション設定を記述するJSON
    │   ├── sim-02-tl_cmp
    │   │   └── ... # 以下同様の為、省略する
    │   ├── ... # 実施したい実験設定ごとにJSONが定義されている
    ├── ...
```

## 実験実行方法
**パターン１：シミュレーション完了時にSlackに通知を送る実行**（← 基本的にはコレで実行する）
```bash
# main.sh内で "python -m utils.executor.simulator_nn" と ""./slack.sh" が実行される
./main.sh
```

**パターン２：シミュレーションのみの実行**
```bash
python -m utils.executor.simulator_nn
```

## シミュレーション実行のために修正する必要のあるJSON
### その１：実施する全てのシミュレーション設定を統括するJSON（.../config/simulation/main.json）
後述の「各種ディレクトリ & ファイル」説明内の「[.../simulation/main.json](#simulationmainjson)」を参照


### その２：実施する各種シミュレーションを定義するためのJSON（.../config/simulation/****/sim_config.json）
後述の「各種ディレクトリ & ファイル」説明内の「[.../config/simulation/****/sim_config.json](#configsimulationsim_configjson)」を参照


# 詳細
## 各種ディレクトリ & ファイルの説明
### .../simulation/main.json
**【一言まとめ】** <br>
実施する全てのシミュレーション設定を統括するJSON。`sim_config.json` をリスト形式で複数指定することで、複数シミュレーションを実行可能。

↓↓ 設定例 ↓↓
```json
{
  "load_path": {
    // "default" に指定したJSONをベースに設定が作成される （詳細は後述の説明を参照）
    "default": "default/sim_config_default.json", 
    // "cmp" に指定した "sim_config.json" の相対PATHの設定が上から順に実行される。
    "cmp": [
      "sim-common-alpha-cmp/sim_config.json",
      "sim-common-batch-cmp/sim_config.json",
      "sim-common-beta-cmp/sim_config.json",
      "sim-common-gamma-cmp/sim_config.json",
      "sim-env-size-cmp/sim_config.json",
      "sim-steps_limit-hex-cmp/sim_config.json",
      "sim-steps_limit-suboptima-cmp/sim_config.json",
      "sim-steps_limit-tmaze-cmp/sim_config.json",
      "sim-01-bw-cmp/sim_config.json",
      "sim-02-tl_cmp/sim_config.json",
      "sim-03-rs-cmp/sim_config.json",
      "sim-04-bdr-cmp/sim_config.json",
      "sim-05-bvp-cmp/sim_config.json",
      "sim-07-bs-cmp/sim_config.json",
      "sim-08-bpw-cmp/sim_config.json",
      "sim-09-visual/sim_config.json"
    ]
    }
}

```

### .../simulation/default/sim_config_default.json
**【一言まとめ】** <br>
シミュレーションのデフォルト設定を定義するためのJSON。全てのシミュレーションに影響を与える可能性があるので、基本的に書き換えない方が良い。（"全てのシミュレーションの設定を変えたい"等、書き換えた方が良い場合もある）

↓↓ 設定例 ↓↓
```json
{
  "total_required_sims": 100, // 最終的に必要なsim数を指定。不足している数が計算され、必要最低限な実行になるように調整される
  "is_save_animdata": false, // アニメーションなどのvisual系画像を実行する時のみ "true" にする
  "params": { // アルゴリズムパラメータの指定
    "alpha": [0.01], // 学習率
    "beta": [10], // softmaxの逆温度 
    "gamma": [0.9] , // Q学習の減衰率
    "lamb": [0.9] , // 適格度（現在不使用）
    "buffer_size": [7000] , // 想起バッファのサイズ
    "er_ratio": [0.1], // Q値と想起バイアス値を考慮する割合を指定（0に近いほどQ値だけに頼り、1に近いほどERだけに頼る）
    "er_deactivate_epis": [20000], // ERを非活性にするエピソードを指定
    "retrieval_size": [30] , // 想起バイアス計算時に取得する軌跡数
    "trajectory_length": [10] , // 想起バイアス計算時に取得する軌跡の長さ
    "bias_difinition": ["sum_rew"] , // 想起バイアスとして採用する値の定義（単純報酬和以外を指定する予定無し）
    "bias_weight": [1.0] , // 想起バイアスの係数（1.0で固定して無視する予定）
    "bias_decay_rate": [0.95] , // 想起バイアス更新時の減衰率
    "bias_var_power": [4.0] , // 想起バイアス更新重みを累乗する際の指数
    "replay_beta_max": [1] , // 想起バイアス値の最大値を定義（現在不使用）
    "replay_beta_min": [0] , // 想起バイアス値の最小値を定義（現在不使用）
    "max_bias_policy_weight": [0, 1], // Meanバイアス方策（Varianceと足して1になる設定のみ採用される）
    "var_bias_policy_weight": [0, 1], // Varianceバイアス方策（Meanと足して1になる設定のみ採用される）
    "batch": [10] // Q値学習時のバッチサイズを指定
  },
  "envs": { // タスクパラメータの指定
    "suboptima": {
      "available": [true], // タスクを使用する場合にTrue、使用しない場合にはFalse
      "epis": [1000], // エピソード数を指定
      "width": [9], // 環境サイズの幅を指定
      "height": [9], // 環境サイズの高さを指定
      "sd": [0], // 報酬源の分散を指定
      "rew_upper": [2], // 報酬源の平均を指定
      "steps_limit": [100], // ステップ数の上限を指定
      "is_render_agt": [false] // エージェントを描画する場合にTrue、しない場合にはFalse（simulationでは使用しないので無視してOK）
    },
    "tmaze_hidden": {
      "available": [true],
      "epis": [5000],
      "width": [7],
      "height": [5],
      "swap_timing": [0.5], // 隠れ状態の位置をスワップするエピソードを指定
      "steps_limit": [100],
      "is_render_agt": [false]
    },
    "hex_hidden": {
      "available": [true],
      "epis": [5000],
      "width": [11],
      "height": [11],
      "swap_timing": [0.5],
      "steps_limit": [100],
      "is_render_agt": [false]
    }
  },
  "alg_names": ["ER-Q", "LSTM-Q", "ER-X"] // モデル名を指定（errで全てのアルゴリズムを表現できるようになったため、現在は不使用）
}

```

### .../config/simulation/****/sim_config.json
**【一言まとめ】** <br>
実施したい各種シミュレーションの設定を定義するためのJSONファイル。`null`にした場合はデフォルトの設定値が採用され、任意の値を指定した場合は、指定した値が採用される。値はリスト形式で複数指定できる。複数指定した際は、直積された設定がすべて適用される（重複や不適切な設定は自動で除外される、はず...）

【注意点】 <br>
- 一部、複数指定できない設定があります
  - default.jsonで、リストで指定されていない設定（total_required_simsなど）はnullに指定することはできない
  - そのため、各種シミュレーション定義JSONで逐一指定する必要があるので注意が必要

↓↓ "trajectory_length" のパラメータを振る際の設定例 ↓↓
```json
{
  "total_required_sims": 10, // nullを指定できない
  "is_save_animdata": false, // nullを指定できない
  "cmp_params": {
    "alpha": null,
    "beta": null,
    "gamma": null,
    "lamb": null,
    "buffer_size": null,
    "er_ratio": null,
    "er_deactivate_epis": null,
    "retrieval_size": null,
    "trajectory_length": [1, 2, 3, 5, 10, 20, 30, 50, 100], // 他のパラメータは全てデフォルトで、軌跡の長さのみ変化させたシミュレーションが実行される
    "bias_difinition": null,
    "bias_weight": null,
    "bias_decay_rate": null,
    "bias_var_power": null,
    "replay_beta_max": null,
    "replay_beta_min": null,
    "max_bias_policy_weight": null,
    "var_bias_policy_weight": null,
    "batch": null
  },
  "cmp_envs": {
    "suboptima": {
      "available": null,
      "epis": null,
      "width": null,
      "height": null,
      "sd": null,
      "rew_upper": null,
      "steps_limit": null,
      "is_render_agt": null
    },
    "tmaze_hidden": {
      "available": null,
      "epis": null,
      "width": null,
      "height": null,
      "swap_timing": null,
      "steps_limit": null,
      "is_render_agt": null
    },
    "hex_hidden": {
      "available": null,
      "epis": null,
      "width": null,
      "height": null,
      "swap_timing": null,
      "steps_limit": null,
      "is_render_agt": null
    }
  },
  "cmp_alg_names": ["ER-Q", "LSTM-Q"]
}

```
