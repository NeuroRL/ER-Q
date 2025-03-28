import pandas as pd
from glob import glob
import pickle as pkl
import os
from tqdm.contrib import tenumerate
from utils.data_creater_utils import (
    valid_convolve,
)
from utils.common.custom_exception_utils import InvalidJSONConfigException
from utils.common.dict_mappings import VISUAL_EPISODE_INDEX_DICT, VISUAL_EPISODE_INDEX_RANGE_DICT
from utils.plotter.env_info_utils import extract_env_type_from_envkind

"""
    df作成の最小単位関数群 (実際にデータをロードしてpd.concatする関数群)
    ToDo: Atom, Molecules的な感じでディレクトリ分割してもいいかも?
"""
# とある実験設定のsim毎のdfを作成する関数 (行:epi, 列:param)
def create_metrics_data_df_sim_each(pickle_path, save_csv_dir):
    df_data_dict = {}
    save_csv_path = os.path.join(save_csv_dir, os.path.basename(pickle_path).replace('.pickle', '.csv'))

    if not os.path.exists(save_csv_path):
        # データのロード
        with open(pickle_path, "rb") as f:
            load_data_dict = pkl.load(f)
        # データをdfに変換
        # ToDo: 後で他のデータ (entropy等) も取得する。最初は必要最低限のデータ。
        df_data_dict["return"] = load_data_dict["return"]
        df_data_dict["steps"] = load_data_dict["steps"]
        df_data_dict["return_mean"] = valid_convolve(load_data_dict["return"])
        df_data_dict["steps_mean"] = valid_convolve(load_data_dict["steps"])
        df_data_dict["episodes"] = range(len(load_data_dict["steps"]))

        df = pd.DataFrame(df_data_dict)
        df.to_csv(save_csv_path)
    else:
        df = pd.read_csv(save_csv_path)

    return df

# visual_plot用のsim毎のdfを作成する関数
def create_visual_data_df_sim_each(pickle_path, save_csv_dir, env_kind):
    df_data_dict = {}

    # NOTE:
    #   ndarrayなどの複雑なデータをcsvに保存すると文字列として保存されてしまう
    #   複雑な構造体をcsvで効率的に扱うことは難しそう。 
    #   そのため、一旦visualデータはcsvファイルをキャッシュとして参照しないようにする

    # データのロード
    with open(pickle_path, "rb") as f:
        load_data_dict = pkl.load(f)
    # データをdfに変換
    df_data_dict["states"] = load_data_dict["states"]
    df_data_dict["actions"] = load_data_dict["actions"]
    # データに "Hidden" 存在が存在しない（HexやSuboptima）場合、"長さ epi のNone配列" で初期化する
    if load_data_dict["hidden"]:
        df_data_dict["hidden"] = load_data_dict["hidden"]
    else:
        df_data_dict["hidden"] = [None]*len(df_data_dict["states"])

    bias_key_list = ["bias", "bias_table"]
    if all([bias_key in load_data_dict for bias_key in bias_key_list]):
        env_kind = extract_env_type_from_envkind(env_kind)
        bias_table_epi_indices = VISUAL_EPISODE_INDEX_DICT[env_kind]
        
        epis = len(load_data_dict["bias_table"])
        save_bias_table_list = [None] * epis

        for index in bias_table_epi_indices:
            save_bias_table_list[index-1] = load_data_dict["bias_table"][index-1]
        
        df_data_dict["bias"] = load_data_dict["bias"]
        df_data_dict["bias_table"] = save_bias_table_list

    df = pd.DataFrame(df_data_dict)

    return df


# visual_anime_sim_any_plot用のdfを作成する関数
def create_visual_data_df_sim_any(pickle_path, save_csv_dir, env_kind):
    df_data_dict = {}

    # データのロード
    with open(pickle_path, "rb") as f:
        load_data_dict = pkl.load(f)
    # データをdfに変換
    df_data_dict["states"] = load_data_dict["states"]
    df_data_dict["actions"] = load_data_dict["actions"]
    # データに "Hidden" 存在が存在しない（HexやSuboptima）場合、"長さ epi のNone配列" で初期化する
    if load_data_dict["hidden"]:
        df_data_dict["hidden"] = load_data_dict["hidden"]
    else:
        df_data_dict["hidden"] = [None]*len(df_data_dict["states"])

    bias_key_list = ["bias", "bias_table"]
    if all([bias_key in load_data_dict for bias_key in bias_key_list]):
        bias_table_epi_indices_range = VISUAL_EPISODE_INDEX_RANGE_DICT[env_kind]
        
        epis = len(load_data_dict["bias_table"])
        save_bias_table_list = [None] * epis

        bias_table_epi_indices = []
        for start_epi, end_epi in bias_table_epi_indices_range:
            bias_table_epi_indices.extend(range(start_epi, end_epi + 1))

        for index in bias_table_epi_indices:
            save_bias_table_list[index-1] = load_data_dict["bias_table"][index-1]
        
        df_data_dict["bias"] = load_data_dict["bias"]
        df_data_dict["bias_table"] = save_bias_table_list

    df = pd.DataFrame(df_data_dict)
    
    return df

# visual_task_plot用のdfを作成する関数
def create_visual_task_data_df_sim_each(pickle_path, save_csv_dir):
    df_data_dict = {}

    # NOTE:
    #   ndarrayなどの複雑なデータをcsvに保存すると文字列として保存されてしまう
    #   複雑な構造体をcsvで効率的に扱うことは難しそう。 
    #   そのため、一旦visualデータはcsvファイルをキャッシュとして参照しないようにする

    # データのロード
    with open(pickle_path, "rb") as f:
        load_data_dict = pkl.load(f)
    # データをdfに変換
    df_data_dict["states"] = load_data_dict["states"]
    df_data_dict["actions"] = load_data_dict["actions"]

    df = pd.DataFrame(df_data_dict)
    df_last = df.iloc[-1]

    return df_last

# 1つの実験に対して、各simで平均を取ったdfを返す関数 (pickleがない場合はNoneを返す)
def create_metrics_data_df_sim_all(target_data_path, save_figures_dir, save_csv_dir, sims):
    df = pd.DataFrame()

    # 各simulationのpickleをdfに変換する
    # ToDo: コード再利用性が低そうであれば "create_param_df_sim_mean" と "create_param_df_sim_each" を同階層で呼び出すようにする
    pickle_paths_all = glob(target_data_path)
    
    # 実際の実験データ数が "sims" より少ない場合はjson設定値がおかしいので Exception を発生させる    
    if len(pickle_paths_all) < sims:
        raise InvalidJSONConfigException("'sims'の設定値に満たないデータ数の実験設定がります。")
    
    pickle_paths = pickle_paths_all[:sims] # TODO: random sampleとかでもいいかも？後でいい感じに調整したい

    # img保存用、csv保存用ディレクトリを作成
    os.makedirs(save_figures_dir, exist_ok=True)
    os.makedirs(save_csv_dir, exist_ok=True)

    for each_sim_id, target_pickle_path in enumerate(pickle_paths):
        additional_df = create_metrics_data_df_sim_each(target_pickle_path, save_csv_dir)
        additional_df["each_sim_id"] = each_sim_id
        df = pd.concat([df, additional_df])
    
    return df

# 1つの実験に対して、各simで平均を取ったdfを返す関数 (pickleがない場合はNoneを返す)
def create_metrics_data_df_sim_mean(target_data_path, save_figures_dir, save_csv_dir, sims):
    df = create_metrics_data_df_sim_all(target_data_path, save_figures_dir, save_csv_dir, sims)
    # dfをgroupbyしてsim平均 (epiでgroupbyした平均) を計算
    df_sim_mean = df.groupby("episodes").mean(numeric_only=True)
    
    return df_sim_mean


# visual_sim_each_plotとvisual_sim_all_plotを作成するためのdfを返す関数 (pickleがない場合はNoneを返す)
def create_visual_data_df_sim_all(target_data_path, save_figures_dir, save_csv_dir, sims, env_kind):
    df = pd.DataFrame()

    pickle_paths_all = glob(target_data_path)

    # 実際の実験データ数が "sims" より少ない場合はjson設定値がおかしいので Exception を発生させる    
    if len(pickle_paths_all) < sims:
        raise InvalidJSONConfigException("'sims'の設定値に満たないデータ数の実験設定がります。")
    
    pickle_paths = pickle_paths_all[:sims] # TODO: random sampleとかでもいいかも？後でいい感じに調整したい
    
    # img保存用、csv保存用ディレクトリを作成
    os.makedirs(f"{save_figures_dir}/visual_bias_table", exist_ok=True)
    os.makedirs(f"{save_figures_dir}/visual_count_table", exist_ok=True)
    os.makedirs(f"{save_csv_dir}/visual_bias_table", exist_ok=True)
    os.makedirs(f"{save_csv_dir}/visual_count_table", exist_ok=True)
    
    # dfを作成
    for sim_id, target_pickle_path in tenumerate(pickle_paths, desc="create df for visual plot."):
        additional_df = create_visual_data_df_sim_each(target_pickle_path, save_csv_dir, env_kind)
        additional_df["sim_id"] = sim_id
        df = pd.concat([df, additional_df])
    
    return df


# visual_anime_sim_any_plotを作成するためのdfを返す関数 (pickleがない場合はNoneを返す)
def create_visual_anime_data_df_one_sim_only(target_data_path, save_figures_dir, save_csv_dir, sims, env_kind):
    df = pd.DataFrame()

    pickle_paths_all = glob(target_data_path)

    # 実際の実験データ数が "sims" より少ない場合はjson設定値がおかしいので Exception を発生させる    
    if len(pickle_paths_all) < sims:
        raise InvalidJSONConfigException("'sims'の設定値に満たないデータ数の実験設定がります。")

    pickle_paths = pickle_paths_all[:sims] # TODO: random sampleとかでもいいかも？後でいい感じに調整したい

    # img保存用、csv保存用ディレクトリを作成
    os.makedirs(f"{save_figures_dir}/create_animation", exist_ok=True)
    os.makedirs(save_csv_dir, exist_ok=True)
    
    # dfを作成
    for sim_id, target_pickle_path in tenumerate(pickle_paths, desc="create df for visual plot."):
        additional_df = create_visual_data_df_sim_any(target_pickle_path, save_csv_dir, env_kind)
        additional_df["sim_id"] = sim_id
        df = pd.concat([df, additional_df])
    
    return df


# visual_plot作成するためのdf(1sim, random)を返す関数 (pickleがない場合はNoneを返す)
def create_visual_data_df_sim_mean(target_data_path, save_figures_dir, save_csv_dir, sims, env_kind):
    if sims != 1:
        raise InvalidJSONConfigException("'visual plot'ではsimに'1'以外の値を設定できません。")
    
    df = pd.DataFrame()

    pickle_paths_all = glob(target_data_path)

    # 実際の実験データ数が "sims" より少ない場合はjson設定値がおかしいので Exception を発生させる    
    if len(pickle_paths_all) < sims:
        raise InvalidJSONConfigException("'sims'の設定値に満たないデータ数の実験設定がります。")
    
    pickle_path = pickle_paths_all[:sims] # TODO: random sampleとかでもいいかも？後でいい感じに調整したい
    
    # img保存用、csv保存用ディレクトリを作成
    os.makedirs(f"{save_figures_dir}/visual_bias_table", exist_ok=True)
    os.makedirs(f"{save_figures_dir}/visual_count_table", exist_ok=True)
    os.makedirs(f"{save_csv_dir}/visual_bias_table", exist_ok=True)
    os.makedirs(f"{save_csv_dir}/visual_count_table", exist_ok=True)
    
    # dfを作成
    # ToDo: 現状は1simのみのデータを扱うが、見にくい場合は複数simのデータを扱う
    df = create_visual_data_df_sim_each(pickle_path, save_csv_dir, env_kind)
    
    return df


# visual_task_plot作成するためのdf(1sim, random)を返す関数 (pickleがない場合はNoneを返す)
def create_visual_task_data_df_sim_mean(target_data_path, save_figures_dir, save_csv_dir, sims):
    if sims != 1:
        raise InvalidJSONConfigException("'visual plot'ではsimに'1'以外の値を設定できません。")
    
    df = pd.DataFrame()

    pickle_paths_all = glob(target_data_path)

    # 実際の実験データ数が "sims" より少ない場合はjson設定値がおかしいので Exception を発生させる    
    if len(pickle_paths_all) < sims:
        raise InvalidJSONConfigException("'sims'の設定値に満たないデータ数の実験設定がります。")
    
    pickle_path = pickle_paths_all[:sims] # TODO: random sampleとかでもいいかも？後でいい感じに調整したい
    
    # img保存用、csv保存用ディレクトリを作成
    os.makedirs(f"{save_figures_dir}/visual_bias_table", exist_ok=True)
    os.makedirs(f"{save_figures_dir}/visual_count_table", exist_ok=True)
    os.makedirs(f"{save_csv_dir}/visual_bias_table", exist_ok=True)
    os.makedirs(f"{save_csv_dir}/visual_count_table", exist_ok=True)
    
    # dfを作成
    df = create_visual_task_data_df_sim_each(pickle_path, save_csv_dir)
    
    return df


"""
    plot用のdfを作成する関数群
"""
# 縦軸実験データのplot用dfを作成する関数
def create_df_for_metrics_kind_figures(df_setting, plot_setting, sim_info, figure_type):
    print(f"{sim_info=}")
    df = pd.DataFrame()
    sims = sim_info['commom_info']["total_required_sims"]
    for target_data_path, each_sim_info in zip(df_setting["reg_exp_data_path_list"], sim_info["each_sim_info_list"]):
        sim_id = each_sim_info["sim_id"]
        save_figures_dir = plot_setting["save_figures_dir"]
        save_csv_dir = plot_setting["save_csv_dir"]
        # ToDo: plotする際に、df上のどのデータが何なのか一意に指定するためにplot_settingとdf_settingで値を共通理解できるよなものに設定しておく
        df_sim = None
        if figure_type in {"interval_figures", "terminal_return_figures"}:
            df_sim = create_metrics_data_df_sim_all(target_data_path, save_figures_dir, save_csv_dir, sims)
        else:
            df_sim = create_metrics_data_df_sim_mean(target_data_path, save_figures_dir, save_csv_dir, sims)
        # df_simが問題なく作成されていることを確認
        if isinstance(df_sim, pd.core.frame.DataFrame):
            df_sim["sim_id"] = sim_id
            df = pd.concat([df, df_sim])
    return df


# 全simの移動経路画像を作成するためのdfを作成する関数
def create_df_for_visual_kind_figures(df_setting, plot_setting, sim_info, figure_type):
    target_data_path = ""
    if len(df_setting["reg_exp_data_path_list"]):
        target_data_path = df_setting["reg_exp_data_path_list"][0]
    else:
        raise ValueError("data_path_list of dfsetting for visual figure settings shoud have only one element.")
    
    save_figures_dir = ""
    save_csv_dir = ""
    # TODO: 条件文が意図した挙動になっているか確認する
    if len(plot_setting["save_csv_dir"]) and len(plot_setting["save_figures_dir"]):
        save_figures_dir = plot_setting["save_figures_dir"]
        save_csv_dir = plot_setting["save_csv_dir"]
        env_kind = plot_setting["env"].kind
    else:
        raise ValueError("plot_settig_list of plot_setting for visual figure settings shoud have only one element.")
    
    sims = sim_info['commom_info']["total_required_sims"]
    df = None
    if figure_type in {"visual_sim_all_figures", "visual_sim_each_figures"}:
        df = create_visual_data_df_sim_all(target_data_path, save_figures_dir, save_csv_dir, sims, env_kind)
    elif figure_type == "visual_figures":
        df = create_visual_data_df_sim_mean(target_data_path, save_figures_dir, save_csv_dir, sims, env_kind)
    elif figure_type == "visual_anime_sim_any_figures":
        # TODO: ここの引数simsを追加し、動作確認
        df = create_visual_anime_data_df_one_sim_only(target_data_path, save_figures_dir, save_csv_dir, sims, env_kind)
        
    return df
