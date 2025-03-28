#  このドキュメントに記載されていること
このドキュメントには、「**画像プロットを実行するための設定に関する説明や注意事項**」が記載されています。

# 概要
## ディレクトリ構成
```bash
.
└──  config
    ├── ...
    ├── plot
    │   ├── main.json # 実施するplotを統括するJSON
    │   ├── baseline
    │   │   └── plot_config.json # 各種plotの設定を指定するJSON
    │   ├── default
    │   │   └── plot_config.json # plot用のデフォルト設定を定義するJSON
    │   ├── envsize
    │   │   └── ... # 以下同様の為省略
    │   ├── ... # 作成したいfigごとにディレクトリが分かれている
    ├── ...
```
## 実験実行方法
```bash
python -m utils.executor.plot_figures
```

## 実装済みのplotter（figure_type）の種類
### アルゴリズムパラメータ比較関連
| plotter<br>(figure_type) | 画像の概要 | 画像の縦軸・横軸 | 補足 
| ---- | ---- | ---- | ---- |
| cmp_err plot<br>(cmp_err_figures) | errが各種モデルの性能に与える影響を大雑把に確認する為の画像 | 横軸 : errの設定値<br>縦軸 : 収益 or ステップ数 | ※実装の都合上、param plotに統合できなかった。作成したい画像の形式はparam plotと同じ。 |
| model plot<br>(model_figures) | 各種パラメータがモデルの性能に与える影響を、エピソード経過も含めて確認する為の画像。各折線が各設定値のデータに対応。 | 横軸 : エピソード数<br>縦軸 : 収益 or ステップ数 | なし |
| param plot<br>(param_figures) | 各種パラメータが各種モデルの性能に与える影響を大雑把に確認する為の画像。各折れ線がER-QやLSTM等の各モデルのデータに対応。 | 横軸 : 各種パラメータの設定値<br>縦軸 : 収益 or ステップ数の、エピソード全体を通じた代表値 | なし |



### タスクパラメータ比較関連
| plotter<br>(figure_type) | 画像の概要 | 画像の縦軸・横軸 | 補足 
| ---- | ---- | ---- |---- |
| envsize plot<br>(envsize_figures) | 環境サイズが各種モデルの性能に与える影響を大雑把に確認する為の画像。各折れ線がER-QやLSTM等の各モデルのデータに対応。 | 横軸 : 環境サイズの設定値<br>縦軸 : 収益 or ステップ数の、エピソード全体を通じた代表値 | なし |
| envsize_model plot<br>(envsize_model_figures) | 環境サイズがモデルの性能に与える影響を、エピソード経過も含めて確認する為の画像。各折れ線が、 環境サイズの設定値のデータに対応。 | 横軸 : エピソード数<br>縦軸 : 収益 or ステップ数 |なし |
| steps_limit plot<br>(steps_limit_figures) | ステップ数上限が各種モデルの性能に与える影響を大雑把に確認する為の画像。各折れ線がER-QやLSTM等の各モデルのデータに対応。 | 横軸 : ステップ数上限の設定値<br>縦軸 : 収益 or ステップ数の、エピソード全体を通じた代表値 | なし |
| steps_limit_model plot<br>(steps_limit_model_figures) | ステップ数上限がモデルの性能に与える影響を、エピソード経過も含めて確認する為の画像。各折れ線が、 ステップ数上限の設定値のデータに対応。 | 横軸 : エピソード数<br>縦軸 : 収益 or ステップ数 | なし |


### シミュレーション実行の可視化関連
| plotter<br>(figure_type) | 画像の概要 | 画像の縦軸・横軸 | 補足 
| ---- | ---- | ---- |---- |
| visual plot<br>(visual_figures) | 1sim数しか指定できないケースのsim_eachと同義 | ※ グリッド画像の為、縦軸横軸は無い | ※現在使用していない可能性が高い。使用していないのであれば削除する。 |
| visual_task plot<br>(visual_task_figures) | グリッドやエージェント等、タスクを説明するために必要な最低限のコンポーネントが含まれる。 | ※ グリッド画像の為、縦軸横軸は無い | ※使用できない可能性有 |
| visual_sim_all plot<br>(visual_sim_all_figures) | 指定したシミュレーション分の累積移動経路を視認するための画像 | ※ グリッド画像の為、縦軸横軸は無い | なし |
| visual_anim_sim_any plot<br>(visual_anime_sim_any_figures) | 指定したエピソードのタスク実行の様子のアニメーション。想起バイアスも可視化される。 | ※ グリッド画像の為、縦軸横軸は無い | エピソードの指定は辞書から指定可能 |
| visual_sim_each plot<br>(visual_sim_each_figures) | 指定したステップ、エピソードのタスク実行の様子の画像。想起バイアスも可視化される。 | ※ グリッド画像の為、縦軸横軸は無い | ステップ、エピソードの指定は辞書から指定可能 |


### その他
| plotter<br>(figure_type) | 画像の概要 | 画像の縦軸・横軸 | 補足 
| ---- | ---- | ---- |---- |
| baseline plot<br>(baseline_figures) | デフォルトパラメータでの実行結果をエピソード経過も含めて確認する為の画像 | 横軸 : エピソード数<br>縦軸 : 収益 or ステップ数 | なし |
| interval plot<br>(interval_figures) | 報酬を得てから次の報酬を得るまでのエピソード間隔を度数分布表で表現した画像 | 横軸 : 報酬獲得インターバル（エピソード数）<br>縦軸 : 度数 | ※現状モデルごとに棒グラフの横幅が異なるので注意 |
| terminal_return plot<br>(terminal_return_figures) | 最終的に獲得された収益のを度数分布表で表現した画像 | 横軸 : 最終獲得収益<br>縦軸 : 度数 | ※基本的にSubOptimaでのみ使用する(他の環境では0 or 2しか取り得ないため) |






## plot実行のために修正する必要のあるJSON
### その１：実施する全てのplot設定を統括するJSON（.../config/plot/main.json）
後述の「各種ディレクトリ & ファイル」説明内の「[.../plot/main.json](#plotmainjson)」を参照


### その２：実施する各種plotを定義するためのJSON（.../config/plot/****/plot_config.json）
後述の「各種ディレクトリ & ファイル」説明内の「[.../config/plot/****/plot_config.json](#configplotplot_configjson)」を参照


# 詳細
## 各種ディレクトリ & ファイルの説明
### .../plot/main.json
**【一言まとめ】** <br>
実施する全てのplot設定を統括するJSON。`plot_config.json` をリスト形式で複数指定することで、複数シミュレーションを実行可能。

↓↓ 設定例 ↓↓
```json
{
  "load_path": {
    "default": "default/plot_config.json", // パラメータのデフォルト設定を定義したファイルのPATHを指定。
    "cmp": [ // 作成したい画像の設定ファイルのPATHを指定
      "cmp_err/fig-err/plot_config.json",
      "envsize_model/fig-hex/plot_config.json",
      "envsize/fig-hex/plot_config.json",
      "model/fig-alpha/plot_config.json",
      "model/fig-ede/hex/plot_config.json",
      "param/fig-alpha/plot_config.json",
      "steps_limit_model/fig-hex/plot_config.json",
      "steps_limit/fig-hex/plot_config.json",
      "baseline/fig-baseline/plot_config.json",
      "visual/fig-sim_each/plot_config.json",
      "visual/fig-sim_any/plot_config.json",
      "visual/fig-sim_all/plot_config.json"
    ]
  }
}

```

### .../plot/default/plot_config.json
**【一言まとめ】** <br>
plotのデフォルト設定を定義するためのJSON。全てのplotに影響を与える可能性があるので、基本的に書き換えない方が良い。（"全てのplotの設定を変えたい"等、書き換えた方が良い場合もある）

↓↓ 設定例 ↓↓

```json
{
  "figures_type": "types of figures", // 作成したいfigの形式を指定
  "total_required_sims": 10, // plotしたいsim数を指定
  "params": { // アルゴリズムパラメータの指定
    "alpha": [0.01], // 学習率
    "beta": [10], // softmaxの逆温度 
    "gamma": [0.9] , // Q学習の減衰率
    "buffer_size": [7000] , // 想起バッファのサイズ
    "er_ratio": [0.1], // Q値と想起バイアス値を考慮する割合を指定（0に近いほどQ値だけに頼り、1に近いほどERだけに頼る）
    "er_deactivate_epis": [20000], // ERを非活性にするエピソードを指定
    "retrieval_size": [30] , // 想起バイアス計算時に取得する軌跡数
    "trajectory_length": [10] , // 想起バイアス計算時に取得する軌跡の長さ
    "bias_difinition": ["sum_rew"] , // 想起バイアスとして採用する値の定義（単純報酬和以外を指定する予定無し）
    "bias_decay_rate": [0.95] , // 想起バイアス更新時の減衰率
    "bias_var_power": [4.0] , // 想起バイアス更新重みを累乗する際の指数
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
      "is_render_agt": [false] // エージェントを描画する場合にTrue、しない場合にはFalse
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

### .../config/plot/****/plot_config.json
**【一言まとめ】** <br>
実施したい各種シミュレーションの設定を定義するためのJSONファイル。`null`にした場合はデフォルトの設定値が採用され、任意の値を指定した場合は、指定した値が採用される。値はリスト形式で複数指定できる。複数指定した際は、直積された設定がすべて適用される（重複や不適切な設定は自動で除外される、はず...）

【注意点】 <br>
- 一部、複数指定できない設定があります
  - default.jsonで、リストで指定されていない設定（total_required_simsなど）はnullに指定することはできない
  - そのため、各種plot定義JSONで逐一指定する必要があるので注意が必要

↓↓ "trajectory_length" のパラメータを振ったfigをplotする際の設定例 ↓↓
```json
{
  "figures_type": "param_figures", // nullを指定できない
  "total_required_sims": 10, // nullを指定できない
  "cmp_params": {
    "alpha": null,
    "beta": null,
    "gamma": null,
    "buffer_size": null,
    "er_ratio": null,
    "er_deactivate_epis": null,
    "retrieval_size": null,
    "trajectory_length": [1, 2, 3, 5, 10, 20, 30, 50, 100], // 他のパラメータは全てデフォルトで、軌跡の長さのみ変化させたシミュレーションが実行される
    "bias_difinition": null,
    "bias_decay_rate": null,
    "bias_var_power": null,
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
