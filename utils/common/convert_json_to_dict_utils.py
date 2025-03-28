import json


# 指定されたjsonを読みに行って、load_path_dictを返す関数
# TODO: LoadPathDictで型アノテーションができるようにする
def create_load_path_dict_from_json(config_dir, json_filename):
    with open(f"{config_dir}/{json_filename}") as file:
        load_path_dict = json.load(file)["load_path"]
        return load_path_dict


# 指定されたjsonを読みに行って、img_position_dictを返す関数
# TODO: LoadPathDictで型アノテーションができるようにする
def create_img_position_dict_from_json(config_dir, json_filename):
    with open(f"{config_dir}/{json_filename}") as file:
        img_position_dict = json.load(file)["img_position_dict"]
        return img_position_dict