# 環境構築（windows）

### 1. [こちら](https://www.python.org/downloads/)からpython3.13をダウンロード（2025/02/04現在）

### 2. 下画像より、`Add Python 3.XX to PATH`を**チェックせずに**インストール開始
(pylauncherで複数バージョンのpythonを管理するため)
![python_install](https://github.com/user-attachments/assets/f83a566e-3268-44c4-82eb-a12b15de119d)


### 3. 下記コマンドを実行してpython3.13がインストールされているか確認
```bash
py --list
```

### 4. 下記コマンドを実行し、venv仮想環境を作成・起動
```bash
py -m venv .venv
.venv/Scripts/activate
```

### 5. 下記コマンドを実行し、必要なライブラリをインストール
```bash
pip install -r requirements-py13.txt
```
