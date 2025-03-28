import re


def trimming_trailing_zero(value):
    if isinstance(value, float):
        value = str(value)
        value = re.sub(r'(\.\d*?[1-9])0+$', r'\1', value)  # 末尾の 0 を削除
        value = re.sub(r'\.0+$', '', value)  # .0 だけの場合は削除
    return value
