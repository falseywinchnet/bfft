#!/bin/sh
set -eu

root=$(CDPATH= cd -- "$(dirname -- "$0")/../.." && pwd)
toolchain="$root/.venv-jpeg"
fusion="$root/third_party/jpeg_fusion"
jpegli="$fusion/jpegli"
mozjpeg="$fusion/mozjpeg"
jpegli_commit=031a0077f5799a6041004267fc12b956c1f52a20
mozjpeg_commit=08265790774cd0714832c9e675522acbe5581437

if [ ! -d "$jpegli/.git" ]; then
  git clone https://github.com/google/jpegli.git "$jpegli"
  git -C "$jpegli" checkout --detach "$jpegli_commit"
  git -C "$jpegli" submodule update --init --recursive
fi
if [ "$(git -C "$jpegli" rev-parse HEAD)" != "$jpegli_commit" ]; then
  echo "jpegli checkout does not match SOURCE_LOCK.json" >&2
  exit 1
fi
if git -C "$jpegli" apply --reverse --check \
    "$fusion/jpegli-ownership-fusion.patch" 2>/dev/null; then
  : # Patch is already present in this working checkout.
else
  git -C "$jpegli" apply --check "$fusion/jpegli-ownership-fusion.patch"
  git -C "$jpegli" apply "$fusion/jpegli-ownership-fusion.patch"
fi

if [ ! -d "$mozjpeg/.git" ]; then
  git clone https://github.com/mozilla/mozjpeg.git "$mozjpeg"
  git -C "$mozjpeg" checkout --detach "$mozjpeg_commit"
fi
if [ "$(git -C "$mozjpeg" rev-parse HEAD)" != "$mozjpeg_commit" ]; then
  echo "mozjpeg checkout does not match SOURCE_LOCK.json" >&2
  exit 1
fi

if [ ! -x "$toolchain/bin/cmake" ] || [ ! -x "$toolchain/bin/ninja" ]; then
  echo "Missing repository-local .venv-jpeg CMake/Ninja toolchain" >&2
  exit 1
fi

PATH="$toolchain/bin:/usr/bin:/bin:/usr/sbin:/sbin"
export PATH

cmake -S "$jpegli" \
  -B "$root/third_party/jpeg_fusion/build/jpegli" -G Ninja \
  -DCMAKE_BUILD_TYPE=Release -DBUILD_TESTING=OFF
cmake --build "$root/third_party/jpeg_fusion/build/jpegli" \
  --target cjpegli djpegli -j 4

cmake -S "$mozjpeg" \
  -B "$root/third_party/jpeg_fusion/build/mozjpeg" -G Ninja \
  -DCMAKE_BUILD_TYPE=Release -DENABLE_SHARED=OFF -DWITH_JPEG8=ON
cmake --build "$root/third_party/jpeg_fusion/build/mozjpeg" \
  --target cjpeg jpegtran djpeg -j 4
