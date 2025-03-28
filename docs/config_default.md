# 実験設定
基本的に`env_config_default.json`と`param_config_default.json`は変更しない

## 目次
- [env_config_default.json](#env_config_defaultjson)
  - **（変更不可）** デフォルト環境設定
- [param_config_default.json](#param_config_defaultjson)
  - **（変更不可）** デフォルトパラメータ設定
- [sim_config_default.json](#sim_config_defaultjson)
  - シミュレーション数設定
- [sim-01-bw_cmp](#sim-01-bw_cmp)
  - [env_config.json](#env_configjson)
    - 実験用の環境設定
  - [param_config.json](#param_configjson)
    - 実験用のパラメータ設定

## 階層
```
.
└── config
    └── simulation
        ├── sim-01-bw_cmp
        │   ├── env_config.json
        │   └── param_config.json
        ├── env_config_default.json
        ├── param_config_default.json
        └── sim_config_default.json
```

## env_config_default.json
現在宣言されているenvは`suboptima`, `tmaze_hidden`, `hex_hidden`の3種類ある

### 共通パラメータ
```json
"available": [true]     // この環境を実験で使用するか
"epis": [1000]          // エピソード数
"width": [9],           // 環境の幅
"height": [9],          // 環境の高さ
```

### suboptima
```json
"sd": [0],              // 報酬に与える正規分布の標準偏差
"rew_upper": [1]        // 報酬の上限値（1 or 8）
```

### tmaze_hidden and hex_hidden
```json
"swap_timing": [2]      // エピソード数の1/nで隠れ状態をスワップ
```

```json
{
  "envs": {
    "suboptima": {
      "available": [true],
      "epis": [1000],
      "width": [9],
      "height": [9],
      "sd": [0],
      "rew_upper": [1]

    },
    "tmaze_hidden": {
      "available": [true],
      "epis": [5000],
      "width": [7],
      "height": [5],
      "swap_timing": [2]
    },
    "hex_hidden": {
      "available": [true],
      "epis": [1000],
      "width": [11],
      "height": [11],
      "swap_timing": [2]
    }
  }
}
```

## param_config_default.json
```json
{
  "params": {
    "alpha": [0.01],                    // 学習率
    "beta": [10],                       // softmaxの探索率
    "gamma": [0.9],                     // 割引率
    "buffer_size": [5000],              // 価値関数用バッファ、想起用バッファのサイズ
    "er_ratio": [0.1],                  // ERとQの重み付け割合
    "retrieval_size": [20],             // 想起する軌跡の本数
    "trajectory_length": [20],          // 想起された軌跡の長さ
    "bias_definition": ["sum_rew"],     // バイアス値の定義
    "bias_decay_rate": [0.95],          // バイアス減衰率
    "bias_var_power": [1],              // 分散を先鋭化するためのパラメータ
    "max_bias_policy_weight": [0],      // bias値のスケールでランク付け
    "var_bias_policy_weight": [1],      // bias値のスケールでランク付け
    "batch": [10]                       // 価値関数用バッファのバッチサイズ
  }
}
```

## sim_config_default.json
```json
{
  "sims": {
    "sims": 100     // シミュレーション回数
  }
}
```


## sim-01-bw_cmp
仮の実験設定として定義。デフォルトパラメータ以外で実験を回す場合、複数パラメータを同時に回す場合に使用。

### env_config.json
回したいパラメータのみ記入、それ以外は`null`にする \
複数のパラメータを回す場合、例えば`"epis": [1000, 2000], "width": [9, 10],`であれば、`(epis, width) = (1000, 9), (1000, 10), (2000, 9), (2000, 10)`の4つの実験設定を生成する
```json
{
  "cmp_envs": {
    "suboptima": {
      "available": null,
      "epis": [1000, 2000],
      "width": null,
      "height": null,
      "sd": null,
      "rew_upper": null
    },
    "tmaze_hidden": {
      "available": null,
      "epis": [5000, 10000],
      "width": null,
      "height": null,
      "swap_timing": null
    },
    "hex_hidden": {
      "available": null,
      "epis": [5000, 10000],
      "width": null,
      "height": null,
      "swap_timing": null
    }
  }
}
```

### param_config.json
回したいパラメータのみ記入、それ以外は`null`にする \
複数のパラメータを回す場合、例えば`"alpha": [0, 1], "beta": [2, 3],`であれば、`(alpha, beta) = (0, 2), (0, 3), (1, 2), (1, 3)`の4つの実験設定を生成する
```json
{
  "cmp_params": {
    "alpha": null,
    "beta": null,
    "gamma": null,
    "buffer_size": null,
    "retrieval_size": null,
    "trajectory_length": null,
    "bias_definition": null,
    "bias_decay_rate": null,
    "bias_var_power": null,
    "max_bias_policy_weight": null,
    "var_bias_policy_weight": null,
    "batch": null
  }
}
```
