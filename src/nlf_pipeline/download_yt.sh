#!/usr/bin/env bash
set -euo pipefail

for id in $*; do
    downloaded_path=${INFERENCE_ROOT}/videos_in/${id}.mp4
    yt-dlp -f 'bestvideo[ext=mp4][height<=1080][vcodec!^=av01]+bestaudio[ext=m4a]' --prefer-ffmpeg -o "$downloaded_path" --no-mtime -- "$id"
done