#!/bin/sh
set -eu

# Capture the HDMI card without resizing or a lossy intermediate. AVFoundation
# reports its native input as limited-range BT.709 NV12; FFV1 stores the same
# Y/Cb/Cr samples losslessly in planar 4:2:0 form.

usage() {
    echo "usage: $0 OUTPUT_DIR LABEL [SECONDS] [DEVICE_INDEX]" >&2
    exit 2
}

[ "$#" -ge 2 ] && [ "$#" -le 4 ] || usage

output_dir=$1
label=$2
seconds=${3:-10}
device_index=${4:-0}
capture="$output_dir/$label.mkv"
preview="$output_dir/$label.png"
frame_hashes="$output_dir/$label.framemd5"
stream_info="$output_dir/$label.ffprobe.json"
signal_stats="$output_dir/$label.signalstats.txt"

mkdir -p "$output_dir"

ffmpeg -hide_banner -loglevel info \
    -f avfoundation \
    -framerate 30 \
    -video_size 1920x1080 \
    -pixel_format nv12 \
    -i "$device_index:none" \
    -t "$seconds" \
    -an \
    -c:v ffv1 \
    -level 3 \
    -pix_fmt yuv420p \
    -color_range tv \
    -colorspace bt709 \
    -color_primaries bt709 \
    -color_trc bt709 \
    -y "$capture"

ffprobe -v error \
    -show_format \
    -show_streams \
    -show_frames \
    -select_streams v:0 \
    -print_format json \
    "$capture" >"$stream_info"

# Exact duplicate hashes reveal a slow/repeated HDMI source underneath the
# nominal 30 fps transport. The signal-stat ledger exposes clipping and range.
ffmpeg -v error -i "$capture" -map 0:v:0 -f framemd5 "$frame_hashes"
ffmpeg -hide_banner -v info -i "$capture" \
    -vf "signalstats,metadata=print:file=$signal_stats" \
    -an -f null - >/dev/null 2>&1

# Use a frame one second in, after the capture path has settled, for quick
# visual verification. The lossless sequence remains the measurement source.
ffmpeg -v error -ss 1 -i "$capture" -frames:v 1 -y "$preview"

echo "capture:      $capture"
echo "preview:      $preview"
echo "frame hashes: $frame_hashes"
echo "stream info:  $stream_info"
echo "signal stats: $signal_stats"
