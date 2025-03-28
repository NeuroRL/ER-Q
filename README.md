# Overvaluation

## リポジトリ構成
```bash
.
├── config                  # 設定jsonを管理
│   ├── paper               # 複数の画像を1枚の画像にまとめるためのjsonを管理
│   ├── plot                # plotをするためのjsonを管理
│   └── simulation          # シミュレーションを実行するためのjsonを管理
├── csv                     # pickleデータのキャッシュとして利用するためのcsvを管理
├── data                    # シミュレーションデータを管理
├── docs                    # ドキュメントを管理
├── img                     # 画像を管理
├── paper_csv               # 論文用画像作成時のデータキャッシュとして利用する為のcsvを管理
├── paper_img               # 論文用画像を管理
├── plotter                 # plotで使用するpythonファイルを管理
├── utils                   # プログラム中で使用する関数などを管理
│   ├── common              # 共通で使用するコンポーネントを管理
│   ├── executor            # 実行する際のエンドポイントを管理
│   └── plotter             # plotで使用するコンポーネントを管理
├── README.md               # リポジトリの概要を記載（当READNE）
├── main.sh                 # シミュレーションの実行&シミュレーション終了時のSlackへの通知を行う為のシェルスク。
├── requirements-py13.txt   # 本リポジトリを実行する際に必要な依存関係を記載したテキストファイル。
└── slack.sh                # Slackへの通知を行う為のシェルスク。
```

## シミュレーション設定
docs配下の[シミュレーション実施方法](./docs/simulation_config.md)に従い、"main.json"と"sim_config.json"の2つのJSONファイルを修正

## 実行方法
```bash
# Working DIR: ./.../Overvaluation

./main.sh
```

## 補足 - シミュレーションを途中で中断した際のした際の復旧方法
【結論】<br>
[#実行方法](#実行方法) に従って、シミュレーションを実行し直すだけでOK！<br>
【理由】<br>
シミュレーション実行の際に、収集済みデータとシミュレーション設定JSONを比較するようになっている。
比較した結果、データ数が不足している設定のシミュレーションのみが実行されるようになっている為


## 各種ドキュメントの内容と保管場所
| 概要 | 場所 | 詳細 |
| ---- | ---- | ---- |
| 環境構築方法 | [./docs/create_env.md](./docs/create_env.md) | python3.12.4のインストール & venv仮想環境の作成方法の説明。 |
| シミュレーション実施方法 | [./docs/simulation_config.md](./docs/simulation_config.md) | なし |
| 論文用画像作成方法 | [./docs/plot_config.md](./docs/plot_config.md) | なし |
| シミュレーション結果画像作成方法 | [./docs/paper_plot_config.md](./docs/paper_plot_config.md) | なし |

