#!/bin/sh
# Build the static city once, then record from a frozen response field.
set -eu
cd "$(dirname "$0")/../../.."
profile=${1:-final}
case "$profile" in
 final) width=1280; height=720; fw=641; fh=361; normals=49; fps=60; seconds=18; stem=city_rain_frozen_transport_720p60 ;;
 preview) width=960; height=540; fw=241; fh=137; normals=9; fps=30; seconds=12; stem=rain_preview ;;
 *) echo 'Usage: run_m4.sh [final|preview]' >&2; exit 2 ;;
esac
experiments/photonic_transport_field/city_rain/build_m4.sh
compute_host=$(/Users/ultimussecundai/.local/bin/m4host)
remote_dir=/tmp/city_rain_$profile
output=experiments/photonic_transport_field/city_rain/output/$profile
mkdir -p "$output"
ssh "$compute_host" "set -eu; mkdir -p '$remote_dir'; /tmp/city_rain_native test > '$remote_dir/checks.log' 2>&1; /tmp/city_rain_native bake --field '$remote_dir/response.field' --stats '$remote_dir/precompute.json' --images '$remote_dir/city' --width $width --height $height --field-width $fw --field-height $fh --normals $normals > '$remote_dir/bake.stdout' 2> '$remote_dir/bake.log'"
scp "$compute_host:$remote_dir/precompute.json" "$compute_host:$remote_dir/city_clear.ppm" "$output/"
if [ "$profile" = final ]; then
 ssh "$compute_host" "/tmp/city_rain_native reference --field '$remote_dir/response.field' --stats '$remote_dir/reference.json' --images '$remote_dir/reference_12s' > '$remote_dir/reference.stdout' 2> '$remote_dir/reference.log'"
fi
ssh "$compute_host" "set -o pipefail; /tmp/city_rain_native play --field '$remote_dir/response.field' --stats '$remote_dir/playback.json' --images '$remote_dir/frame' --duration $seconds --fps $fps 2> '$remote_dir/play.log' | /opt/homebrew/bin/ffmpeg -hide_banner -loglevel warning -y -f rawvideo -pixel_format rgb24 -video_size ${width}x${height} -framerate $fps -i pipe:0 -an -vf scale=in_range=pc:out_range=tv:out_color_matrix=bt709,format=yuv420p -c:v h264_videotoolbox -b:v 16000k -color_range tv -colorspace bt709 -color_primaries bt709 -color_trc bt709 -movflags +faststart '$remote_dir/$stem.mp4' 2> '$remote_dir/encode.log'; /opt/homebrew/bin/ffprobe -v error -select_streams v:0 -show_entries stream=width,height,r_frame_rate,avg_frame_rate,nb_frames,codec_name,pix_fmt:format=duration,size -of json '$remote_dir/$stem.mp4' > '$remote_dir/video.json'"
scp "$compute_host:$remote_dir/*.json" "$compute_host:$remote_dir/*.ppm" "$compute_host:$remote_dir/*.log" "$compute_host:$remote_dir/$stem.mp4" "$output/"
# Stream the archive directly home, avoiding a second large file on the Mini.
# build/ is excluded by m4build, so this cache is not mirrored back on builds.
cache_dir=experiments/photonic_transport_field/city_rain/build
mkdir -p "$cache_dir"
cache=$cache_dir/response.field.gz
if [ "$profile" = preview ]; then cache=$cache_dir/response_preview.field.gz; fi
ssh "$compute_host" "/usr/bin/gzip -1 -c '$remote_dir/response.field'" > "$cache.tmp"
/usr/bin/gzip -t "$cache.tmp"
mv "$cache.tmp" "$cache"
/usr/bin/shasum -a 256 "$cache" > "$cache.sha256"
if [ "${KEEP_CITY_RAIN_FIELD:-0}" != 1 ]; then
 ssh "$compute_host" "rm '$remote_dir/response.field'"
fi
printf 'Video and measurements: %s/%s.mp4\n' "$output" "$stem"
