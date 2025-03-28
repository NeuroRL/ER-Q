from utils.executor.plot_figures import plot_figures
import json
import os
import matplotlib.pyplot as plt
from PIL import Image

from utils.common.convert_json_to_dict_utils import (
    create_load_path_dict_from_json,
    create_img_position_dict_from_json
)
from utils.common.create_setting_from_json_utils import (
    load_default_dict_and_cmp_dict
)
from utils.common.dict_mappings import SAVE_IMAGE_EXT
from utils.plotter.create_blank_fig_utils import create_blank_fig

def create_paper_figure(img_position_dict, save_path):
    # サブプロットのサイズを計算
    max_row = max(eval(key)[0] for key in img_position_dict.keys()) + 1
    max_col = max(eval(key)[1] for key in img_position_dict.keys()) + 1

    # プロットの準備
    fig, axes = plt.subplots(max_row, max_col, figsize=(5 * max_col, 5 * max_row))

    # img_position_dict内の画像をプロット
    for key, path in img_position_dict.items():
        row, col = eval(key)  # サブプロットの座標
        if os.path.exists(path):  # 画像が存在する場合のみ処理
            img = Image.open(path)
            if max_row > 1 and max_col > 1:
                # 両方複数行・複数列の場合
                axes[row][col].imshow(img)
                axes[row][col].axis("off")
            elif max_row == 1:
                # 1行の場合 (axes は1次元配列)
                axes[col].imshow(img)
                axes[col].axis("off")
            elif max_col == 1:
                # 1列の場合 (axes は1次元配列)
                axes[row].imshow(img)
                axes[row].axis("off")
            else:
                # サブプロットが1つだけの場合
                axes.imshow(img)
                axes.axis("off")
        else:
            # TODO: loggerで出力できるようにする
            print(
                    f"画像が存在しません: {path}。\n"
                    "'paper_config'に指定している必要画像 'img_position_dict' と、"
                    "足りない画像を生成するための設定ファイルである'cmp'での指定が一致しているかを確認して下さい。"
                )

    # 余白を調整して保存
    plt.subplots_adjust(hspace=0, wspace=5)
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight")
    plt.close(fig)
    print(f"論文用画像を保存しました: {save_path}")

if __name__ == "__main__":
    config_dir_paper = "config/paper"
    config_dir_plot = "config/plot"
    paper_img_save_dir = "paper_img"

    # config/paper/main.jsonを読み込む。
    with open(f"{config_dir_paper}/main.json") as paper_fig_main_file:
        paper_fig_json_path_list = json.load(paper_fig_main_file)["paper_fig_json_path_list"]

    # paper_fig毎にそれぞれ処理
    for paper_fig_json_path in paper_fig_json_path_list:
        img_position_dict  = create_img_position_dict_from_json(config_dir=config_dir_paper, json_filename=paper_fig_json_path)

        # 画像が存在しない場合は必要画像を作成してからpaper figuresを作成
        is_need_create_image = False
        for subpllot_pos, img_path in img_position_dict.items():
            if not os.path.exists(img_path):
                is_need_create_image = True

        # paper_figを保存するディレクトリの作成
        os.makedirs(paper_img_save_dir, exist_ok=True)

        # 空きスペースに入れる白紙の画像を毎回作成
        create_blank_fig(save_dir=f"{paper_img_save_dir}/blank")

        # TODO: 存在しない画像を作成するplot jsonのみ実行するように修正 (plot figuresの引数をjsonにしてしまうとかで対応できそう)
        if is_need_create_image:
            # paper_fig.jsonからload_path_dictを取得
            load_path_dict = create_load_path_dict_from_json(config_dir=config_dir_paper, json_filename=paper_fig_json_path)
            print(f"{load_path_dict=}")
            # load_path_dict から default_config_dict&cmp_config_dic を作成
            default_config_dict, cmp_config_dict_list = load_default_dict_and_cmp_dict(config_dir=config_dir_plot, load_path_dict=load_path_dict)
            plot_figures(default_config_dict=default_config_dict, cmp_config_dict_list=cmp_config_dict_list, is_paper_fig=True)


        paper_img_save_filename = paper_fig_json_path.split(".")[0]
        for ext in SAVE_IMAGE_EXT["ext"]:
            paper_img_save_path = f"{paper_img_save_dir}/{paper_img_save_filename}.{ext}"
            # paper_figの作成
            create_paper_figure(img_position_dict=img_position_dict, save_path=paper_img_save_path)
