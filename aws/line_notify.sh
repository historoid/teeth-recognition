#!/bin/bash
LINE_ACCESS_TOKEN="VYaH66svomjcq73ilslEwwla1yn6FsHIuJswf1En1je"
function line_notify() {
  MESSAGE=$1
  curl -X POST -H "Authorization: Bearer ${LINE_ACCESS_TOKEN}" -F "message=$MESSAGE" https://notify-api.line.me/api/notify
}
line_notify "テストメッセージです"