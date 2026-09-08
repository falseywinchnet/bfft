#!/bin/sh
set -eu
cd "$(dirname "$0")/../../.."
/Users/ultimussecundai/.local/bin/m4build -- sh -c '
 set -eu
 root=experiments/photonic_transport_field
 cc -O3 -DNDEBUG -std=c11 -pthread -mcpu=native -c standalone_conv_resize_demo/native/conv_native.c -o /tmp/night_screen_conv.o
 c++ -O3 -DNDEBUG -std=c++20 -mcpu=native "$root/city_rain/night_screen.cpp" "$root/native/retained_transport.cpp" /tmp/night_screen_conv.o -pthread -o /tmp/night_screen_native
 /tmp/night_screen_native test
 /tmp/night_screen_native bake /tmp/night_screen
'
compute_host=$(/Users/ultimussecundai/.local/bin/m4host)
output=experiments/photonic_transport_field/city_rain/output/night_screen
mkdir -p "$output"
scp "$compute_host:/tmp/night_screen/bake.json" "$compute_host:/tmp/night_screen/night_clear.ppm" "$output/"
ssh "$compute_host" 'set -o pipefail; /tmp/night_screen_native play /tmp/night_screen | /opt/homebrew/bin/ffmpeg -hide_banner -loglevel warning -y -f rawvideo -pixel_format rgb24 -video_size 1280x720 -framerate 60 -i pipe:0 -an -vf scale=in_range=pc:out_range=tv:out_color_matrix=bt709,format=yuv420p -c:v h264_videotoolbox -b:v 16000k -color_range tv -colorspace bt709 -color_primaries bt709 -color_trc bt709 -movflags +faststart /tmp/night_screen/night_city_screen_rain_720p60.mp4'
scp "$compute_host:/tmp/night_screen/*.json" "$compute_host:/tmp/night_screen/*.ppm" "$compute_host:/tmp/night_screen/*.mp4" "$compute_host:/tmp/night_screen/illumination.screen" "$output/"
ssh "$compute_host" '/opt/homebrew/bin/ffprobe -v error -select_streams v:0 -show_entries stream=width,height,r_frame_rate,avg_frame_rate,nb_frames,codec_name,pix_fmt:format=duration,size -of json /tmp/night_screen/night_city_screen_rain_720p60.mp4 > /tmp/night_screen/video.json; /opt/homebrew/bin/ffmpeg -v error -i /tmp/night_screen/night_city_screen_rain_720p60.mp4 -f null -'
scp "$compute_host:/tmp/night_screen/video.json" "$output/"
.venv-jpeg/bin/python experiments/photonic_transport_field/city_rain/summarize_night_screen.py
