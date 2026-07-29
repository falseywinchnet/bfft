#!/bin/zsh
set -u

script_dir=${0:A:h}
binary=/tmp/xa5_libusb_ptp
user_domain="gui/$(id -u)"

restore_camera_agents() {
    launchctl enable "$user_domain/com.apple.icdd" 2>/dev/null || true
    launchctl enable "$user_domain/com.apple.ptpcamerad" 2>/dev/null || true
    launchctl kickstart "$user_domain/com.apple.icdd" 2>/dev/null || true
    launchctl kickstart "$user_domain/com.apple.ptpcamerad" 2>/dev/null || true
}

trap restore_camera_agents EXIT INT TERM HUP

clang -std=c11 -Wall -Wextra -Werror \
    $(pkg-config --cflags libusb-1.0) \
    "$script_dir/xa5_libusb_ptp.c" \
    $(pkg-config --libs libusb-1.0) \
    -o "$binary" || exit $?

launchctl disable "$user_domain/com.apple.icdd"
launchctl disable "$user_domain/com.apple.ptpcamerad"
killall -KILL icdd ptpcamerad 2>/dev/null || true

"$binary" "$@"
