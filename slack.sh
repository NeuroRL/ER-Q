#!/bin/sh

set -eu

# .env ファイルが存在するかチェック
if [ ! -f .env ]; then
  echo "Error: .env file not found. Please create a .env file with SLACK_WEBHOOK_URL defined." >&2
  exit 1
fi

# .env ファイルを読み込む
. ./.env

# Webhook URL を環境変数から取得
WEBHOOKURL="${SLACK_WEBHOOK_URL:-""}"

MESSAGE=""
while getopts m: opts; do
    case $opts in
        m) MESSAGE=$OPTARG"\n" ;;
    esac
done

# メッセージ本文の取得（標準入力）
if [ -p /dev/stdin ]; then
    WEBMESSAGE=$(cat - | tr '\n' '\\' | sed 's/\\/\\n/g')
else
    echo "nothing stdin"
    exit 1
fi

# Slackへ送信
curl -X POST -H 'Content-type: application/json' \
    --data "{\"text\": \"${MESSAGE}${WEBMESSAGE}\" }" \
    "$WEBHOOKURL"
