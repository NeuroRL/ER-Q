# パラメータ名を短縮表記に変換する為の辞書
CONVERT_PARAMNAME_DICT = {
    "alpha": "alpha",
    "beta": "beta",
    "gamma": "gamma",
    "buffer_size": "bs",
    "er_ratio": "err",
    "er_deactivate_epis": "ede",
    "retrieval_size": "rs",
    "trajectory_length": "tl",
    "bias_decay_rate": "bdr",
    "bias_var_power": "bvp",
    "max_bias_policy_weight": "SRBP",
    "var_bias_policy_weight": "VRBP",
    "batch": "batch"
}

# パラメータ名からplot名に変換する為の辞書
CONVERT_PLOTNAME_DICT = {
    "alpha": "alpha",
    "beta": "beta",
    "gamma": "gamma",
    "buffer_size": "buffer size",
    "er_ratio": "er ratio",
    "er_deactivate_epis": "er deactivate epis",
    "retrieval_size": "retrieval size",
    "trajectory_length": "trajectory length",
    "bias_decay_rate": "bias decay rate",
    "bias_var_power": "bias var power",
    "max_bias_policy_weight": "scale bias policy weight",
    "var_bias_policy_weight": "variance bias policy weight",
    "batch": "batch"
}

# 環境ごとの理論的な最小ゴール到達ステップ数
THEORETICAL_MIN_GOAS_STEPS_DICT = {
    "FlaggedExtendedTmazeHidden": {
        "W5_H3": 7,
        "W7_H5": 12,
        "W9_H7": 17,
        "W11_H9": 22,
    },
    "SubOptima": {
        "W7_H7": 5,
        "W9_H9": 6,
        "W11_H11": 7,
        "W13_H13": 8,
        "W15_H15": 9,
        "W17_H17": 10,
        "W19_H19": 11,
        "W29_H29": 16,
        "W39_H39": 21,
    },
    "HiddenExploreHex": {
        "W9_H9": 12,
        "W11_H11": 15,
        "W13_H13": 18,
        "W15_H15": 21,
    }
}

# 環境名の変換辞書
ENVNAME_CONVERT_DICT = {
    "FlaggedExtendedTmazeHidden": "Cued T-maze",
    "SubOptima": "Distributed Foraging",
    "HiddenExploreHex": "Key-Tile Maze"
}

# プロット名からアルゴリズム名への変換辞書
PLOTNAME_TO_ALGNAME_DICT = {
    "LSTM-Q": "LSTMQnet",
    "ER-Q": "M_AS_ER_LSTMQnet",
    "ER-X": "M_AS_ER_Only"
}

# アルゴリズム名からプロット名への変換辞書
ALGNAME_TO_PLOTNAME_DICT = {
    "LSTMQnet": "LSTM-Q",
    "M_AS_ER_LSTMQnet": "ER-Q",
    "M_AS_ER_Only": "ER-X"
}

# visual系のplotで使用するエピソード辞書
VISUAL_EPISODE_INDEX_DICT = {
    "FlaggedExtendedTmazeHidden": [1, 100, 500, 1000, 3000, 5000],
    "SubOptima": [1, 100, 500, 1000],
    "HiddenExploreHex": [1, 100, 500, 1000, 3000, 5000]
}

# visual系のplotで使用する進行割合（パーセント）辞書
# 0%: 開始時, 100%: 終了時
VISUAL_PERCENT_STEP_DICT = {
    "FlaggedExtendedTmazeHidden": [0, 50, 60, 100],
    "SubOptima": [0, 50, 60, 100],
    "HiddenExploreHex": [0, 50, 60, 100]
}

# visual系のplotで使用するエピソードのレンジ辞書
VISUAL_EPISODE_INDEX_RANGE_DICT = {
    "FlaggedExtendedTmazeHidden": [(1, 3), (97, 100), (497, 500), (997, 1000), (2997, 3000), (4997, 5000)],
    "SubOptima": [(1, 3), (97, 100), (497, 500), (997, 1000)],
    "HiddenExploreHex": [(1, 3), (97, 100), (497, 500), (997, 1000), (2997, 3000), (4997, 5000)]
}

# metricsをplot用の表記に変換するための辞書
CONVERT_METRICS_DICT = {
    "return": "Return",
    "steps": "Steps",
    "return_mean": "Return (mean)",
    "steps_mean": "Steps (mean)"
}

# 保存する画像の拡張子
SAVE_IMAGE_EXT = {
    "ext": ["png", "eps", "pdf"]
}
