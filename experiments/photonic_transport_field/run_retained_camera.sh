#!/bin/sh
# Paired camera representations in the actual retained renderer, on the M4 CPU.
set -eu
cd "$(dirname "$0")/../.."
/Users/ultimussecundai/.local/bin/m4build -- sh -c '
  set -eu
  native=experiments/photonic_transport_field/native
  cc -O3 -DNDEBUG -std=c11 -pthread -mcpu=native -c \
    standalone_conv_resize_demo/native/conv_native.c -o /tmp/retained_camera_conv.o
  c++ -O3 -DNDEBUG -std=c++20 -Wall -Wextra -Wpedantic -mcpu=native \
    "$native/regime_scene_native.cpp" "$native/retained_transport.cpp" \
    /tmp/retained_camera_conv.o -pthread -o /tmp/retained_camera_native
  /tmp/retained_camera_native --self-test
  /tmp/retained_camera_native --update-self-test
  /tmp/retained_camera_native --camera-self-test
  /tmp/retained_camera_native --benchmark-camera /tmp/retained_camera_benchmark.json --width 800 --height 600
  /tmp/retained_camera_native --benchmark-updates /tmp/retained_camera_updates.json --width 800 --height 600
  for scene in aperture-canyon mirror-relay occlusion-garden; do
    /tmp/retained_camera_native --scene "$scene" --acceleration bvh --width 800 --height 600 \
      --camera-gather eager --camera-crossings dense --out "/tmp/camera_${scene}_eager.ppm" > "/tmp/camera_${scene}_eager.json"
    /tmp/retained_camera_native --scene "$scene" --acceleration bvh --width 800 --height 600 \
      --out "/tmp/camera_${scene}_demand.ppm" > "/tmp/camera_${scene}_demand.json"
    cmp "/tmp/camera_${scene}_eager.ppm" "/tmp/camera_${scene}_demand.ppm"
    echo "camera image equivalence: $scene ok"
  done
  cc -O1 -g -std=c11 -fsanitize=address,undefined -fno-omit-frame-pointer -pthread -c \
    standalone_conv_resize_demo/native/conv_native.c -o /tmp/retained_camera_conv_sanitized.o
  c++ -O1 -g -std=c++20 -fsanitize=address,undefined -fno-omit-frame-pointer \
    "$native/regime_scene_native.cpp" "$native/retained_transport.cpp" \
    /tmp/retained_camera_conv_sanitized.o -pthread -o /tmp/retained_camera_sanitized
  /tmp/retained_camera_sanitized --camera-self-test
  /tmp/retained_camera_sanitized --update-self-test
  cc -O1 -g -std=c11 -fsanitize=thread -pthread -c \
    standalone_conv_resize_demo/native/conv_native.c -o /tmp/retained_camera_conv_tsan.o
  c++ -O1 -g -std=c++20 -fsanitize=thread \
    "$native/regime_scene_native.cpp" "$native/retained_transport.cpp" \
    /tmp/retained_camera_conv_tsan.o -pthread -o /tmp/retained_camera_tsan
  /tmp/retained_camera_tsan --camera-self-test
'
mkdir -p experiments/photonic_transport_field/camera_gather_m4
scp "$(/Users/ultimussecundai/.local/bin/m4host):/tmp/retained_camera_benchmark.json" \
  "$(/Users/ultimussecundai/.local/bin/m4host):/tmp/retained_camera_updates.json" \
  "$(/Users/ultimussecundai/.local/bin/m4host):/tmp/camera_*_eager.json" \
  "$(/Users/ultimussecundai/.local/bin/m4host):/tmp/camera_*_demand.json" \
  experiments/photonic_transport_field/camera_gather_m4/
