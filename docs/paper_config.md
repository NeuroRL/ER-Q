#  このドキュメントに記載されていること
このドキュメントには、「**論文用の画像プロットを実行するための設定に関する説明や注意事項**」が記載されています。

# 概要
## ディレクトリ構成
```bash
.
└──  config
    ├── ...
    ├── paper
    │   ├── main.json # 実施する論文用の画像作成を統括するJSON
    │   ├── paper_fig_02-outline_of_hex.json # 各種の論文用画像作成の設定を指定するJSON
    │   ├── paper_fig_03-outline_of_tmaze.json # 各種の論文用画像作成の設定を指定するJSON
    │   ├── paper_fig_04-outline_of_suboptima.json # 各種の論文用画像作成の設定を指定するJSON
    │   ├── ... # 作成したい論文用画像によってファイルが分けられている
    ├── ...
```
## 実験実行方法
```bash
python -m utils.executor.paper_figures
```

## 任意の論文用画像を作成するために修正する必要のあるJSON
### その１：実施する全ての論文用の画像作成の設定を統括するJSON（.../config/paper/main.json）
後述の「各種ディレクトリ & ファイル」説明内の「[.../paper/main.json]()」を参照


### その２：各種作成する論文用画像を定義するためのJSON（.../config/paper/paper_fig_xx_yy.json）
後述の「各種ディレクトリ & ファイル」説明内の「[.../config/paper/paper_fig_xx_yy.json]()」を参照


### その３（必要に応じて）：作成する論文用画像で**新たに**必要になる画像を定義するためのJSON（.../config/paper/****/paper_config.json）
後述の「各種ディレクトリ & ファイル」説明内の「[.../config/paper/****/paper_config.json]()」を参照


# 詳細
## 各種ディレクトリ & ファイルの説明
### .../paper/main.json
**【一言まとめ】** <br>
実施する全ての論文用の画像作成の設定を統括するJSON。`paper_fig_xx-yy.json` をリスト形式で複数指定することで、複数の論文用画像を作成することが可能。

↓↓ 設定例 ↓↓
```json
{
    "paper_fig_json_path_list": [
        "paper_fig_02-outline_of_hex.json",
        "paper_fig_03-outline_of_tmaze.json",
        "paper_fig_04-outline_of_suboptima.json",
        "paper_fig_05-parameter_dependency_return_of_hex.json",
        "paper_fig_06-parameter_dependency_return_of_tmaze.json",
        "paper_fig_07-parameter_dependency_return_of_suboptima.json",
        "paper_fig_08-parameter_dependency_steps_of_hex.json",
        "paper_fig_09-parameter_dependency_steps_of_tmaze.json",
        "paper_fig_10-parameter_dependency_steps_of_suboptima.json",
        "paper_fig_11-envsize_dependency_of_hex.json",
        "paper_fig_12-envsize_dependency_of_tmaze.json",
        "paper_fig_13-envsize_dependency_of_suboptima.json",
        "paper_fig_14-steps_limit_dependency_of_hex.json",
        "paper_fig_15-steps_limit_dependency_of_tmaze.json",
        "paper_fig_16-steps_limit_dependency_of_suboptima.json"
    ]
}

```

### .../paper/paper_fig_xx_yy.json
**【一言まとめ】** <br>
作成したい論文用画像の設定を定義するためのJSONファイル。  
`img_position_dict` では、key に座標（画像の配置位置）、value に保存されている画像のパスを指定する。  

【注意点】 <br>
指定された画像が存在しない場合、該当する画像を生成する。  
そのため、`img_position_dict` で画像を指定する際には、**該当する画像を作成するための `paper_config.json` を `cmp` に設定する必要がある**。


↓↓ 設定例 ↓↓

```json
{
  "load_path": {
    "default": "default/paper_config.json",  // パラメータのデフォルト設定を定義するファイル
    "cmp": [  // 下記(img_position_dict)で指定した画像が1つでも存在しない場合、この中にある`paper_config.json`で画像生成
      "baseline/fig-baseline/paper_config.json",
      "visual/fig-sim_each/paper_config.json",
      "visual/fig-sim_all/paper_config.json"
    ]
  },
  "img_position_dict": {  // key に座標（画像の配置位置）、value に保存されている画像のパスを指定
    "(0, 0)": "paper_img/visual/HiddenExploreHex_lm100_W11_H11_sw10000/epi5000/bs10000_tl20_rs20_alpha0.01_beta10_gamma0.9_bdr0.95_bvp5.0_mabpw0_varbpw1_batch10/visual_bias_table/sim_id0_epi1000_0perstep.png",
    "(0, 1)": "paper_img/visual/HiddenExploreHex_lm100_W11_H11_sw10000/epi5000/bs10000_tl20_rs20_alpha0.01_beta10_gamma0.9_bdr0.95_bvp5.0_mabpw1_varbpw0_batch10/visual_bias_table/sim_id0_epi1000_0perstep.png",
    "(0, 2)": "paper_img/blank/blank.png",  // 自動で生成
    "(1, 0)": "paper_img/visual/HiddenExploreHex_lm100_W11_H11_sw10000/epi5000/bs10000_tl20_rs20_alpha0.01_beta10_gamma0.9_bdr0.95_bvp5.0_mabpw0_varbpw1_batch10/visual_sim_all.png",
    "(1, 1)": "paper_img/visual/HiddenExploreHex_lm100_W11_H11_sw10000/epi5000/bs10000_tl20_rs20_alpha0.01_beta10_gamma0.9_bdr0.95_bvp5.0_mabpw1_varbpw0_batch10/visual_sim_all.png",
    "(1, 2)": "paper_img/visual/HiddenExploreHex_lm100_W11_H11_sw10000/epi5000/alpha0.01_beta10_gamma0.9_batch10/visual_sim_all.png",
    "(2, 0)": "paper_img/baseline/HiddenExploreHex_lm100_W11_H11_sw10000/epi5000/baseline_plot_return_mean.png",
    "(2, 1)": "paper_img/baseline/HiddenExploreHex_lm100_W11_H11_sw10000/epi5000/baseline_plot_steps_mean.png",
    "(2, 2)": "paper_img/blank/blank.png"
  }
}

```

### .../config/paper/****/paper_config.json
**【一言まとめ】** <br>
`paper_fig_xx-yy.json` を実行した際に、指定された画像が存在しない場合、`load_path` の `cmp` にある `paper_config.json` を利用して不足している画像を作成するファイル。
`null`にした場合はデフォルトの設定値が採用され、任意の値を指定した場合は、指定した値が採用される。値はリスト形式で複数指定できる。複数指定した際は、直積された設定がすべて適用される（重複や不適切な設定は自動で除外される、はず...）

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
    "bias_definition": null,
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
```
