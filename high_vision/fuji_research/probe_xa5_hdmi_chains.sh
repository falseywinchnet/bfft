#!/bin/zsh
set -u
setopt ERR_EXIT PIPE_FAIL

script_dir=${0:A:h}
binary=/tmp/xa5_libusb_ptp
user_domain="gui/$(id -u)"
capture_device="USB3.0 HD VIDEO"
capture_pixel_format="nv12"
capture_size="1280x960"
capture_rate="30"
capture_frames=16
work_dir=$(mktemp -d /tmp/xa5-hdmi-probe.XXXXXX)
holder_pid=""

restore_camera_agents() {
    launchctl enable "$user_domain/com.apple.icdd" 2>/dev/null || true
    launchctl enable "$user_domain/com.apple.ptpcamerad" 2>/dev/null || true
    launchctl kickstart "$user_domain/com.apple.icdd" 2>/dev/null || true
    launchctl kickstart "$user_domain/com.apple.ptpcamerad" 2>/dev/null || true
}

stop_holder() {
    if [[ -n "$holder_pid" ]] && kill -0 "$holder_pid" 2>/dev/null; then
        kill -INT "$holder_pid" 2>/dev/null || true
        wait "$holder_pid" 2>/dev/null || true
    fi
    holder_pid=""
}

cleanup() {
    stop_holder
    restore_camera_agents
}

trap cleanup EXIT INT TERM HUP

if (( $# == 0 )); then
    chains=(direct_movie)
else
    chains=("$@")
fi

for chain in "${chains[@]}"; do
    if [[ "$chain" != "idle" &&
          "$chain" != "open" &&
          "$chain" != "plain_open" &&
          "$chain" != "plain_direct_movie" &&
          "$chain" != "plain_cancel_direct_movie" &&
          "$chain" != "plain_open_movie" &&
          "$chain" != "direct_movie" &&
          "$chain" != "direct_movie_reset" &&
          "$chain" != "execute_direct_movie" &&
          "$chain" != "same_session_direct_movie" &&
          "$chain" != "same_session_execute_direct_movie" &&
          "$chain" != "same_session_open_movie" &&
          "$chain" != "open_movie" ]]; then
        print -u2 "unknown chain: $chain"
        print -u2 \
            "valid chains: idle open plain_open plain_direct_movie"
        print -u2 "              plain_open_movie"
        print -u2 "              plain_cancel_direct_movie"
        print -u2 "              direct_movie direct_movie_reset"
        print -u2 "              execute_direct_movie open_movie"
        print -u2 "              same_session_direct_movie"
        print -u2 "              same_session_execute_direct_movie"
        print -u2 "              same_session_open_movie"
        exit 2
    fi
done

clang -std=c11 -Wall -Wextra -Werror \
    $(pkg-config --cflags libusb-1.0) \
    "$script_dir/xa5_libusb_ptp.c" \
    $(pkg-config --libs libusb-1.0) \
    -o "$binary"

device_report=$(
    ffmpeg -hide_banner -f avfoundation -list_devices true -i '' 2>&1 || true
)
if [[ "$device_report" != *"$capture_device"* ]]; then
    print -u2 "AVFoundation device not found: $capture_device"
    exit 1
fi

launchctl disable "$user_domain/com.apple.icdd" 2>/dev/null || true
launchctl disable "$user_domain/com.apple.ptpcamerad" 2>/dev/null || true
killall -KILL icdd ptpcamerad 2>/dev/null || true

# Recovered from the nine-entry table at firmware address 0xc05eee78.
# XA5_DEBUG_KEYS permits a focused run, for example:
# XA5_DEBUG_KEYS="0 9" ./probe_xa5_hdmi_chains.sh same_session_direct_movie
if [[ -n "${XA5_DEBUG_KEYS:-}" ]]; then
    debug_keys=(${=XA5_DEBUG_KEYS})
else
    debug_keys=(0 1 3 4 5 6 9 A FF)
fi

print "logs: $work_dir"
print \
    "key,chain,set_result,command_result,frames,effective_fps,signal_present,"\
"yavg,ymin,ymax,ydif_avg,ydif_max"

for key in "${debug_keys[@]}"; do
    for chain in "${chains[@]}"; do
        # These chains never consume CamDebugUsbMode. Run them once even when
        # the default nine-key sweep is active; repeating movie start can
        # itself latch the camera into DeviceBusy after the first success.
        if [[ "$key" != "$debug_keys[1]" &&
              ( "$chain" == "idle" || "$chain" == plain_* ) ]]; then
            continue
        fi

        stop_holder
        killall -KILL icdd ptpcamerad 2>/dev/null || true

        if [[ "$chain" == "idle" || "$chain" == plain_* ]]; then
            set_status=0
            set_result=skipped
        elif [[ "$chain" == same_session_* ]]; then
            set_status=0
            set_result=inline
        else
            set_log="$work_dir/key_${key}_${chain}_set.log"
            if [[ "$chain" == "execute_direct_movie" ]]; then
                initial_command=(
                    --sxc-execute CamDebugUsbMode "$key"
                    --allow-sxc-execute
                )
            else
                initial_command=(--sxc-set CamDebugUsbMode "$key")
            fi
            if "$binary" "${initial_command[@]}" >"$set_log" 2>&1; then
                set_status=0
            else
                set_status=$?
            fi
            if (( set_status != 0 )); then
                print \
                    "$key,$chain,transport_$set_status,none,0,0,no,"\
"0,0,0,0,0"
                continue
            fi
            if [[ "$chain" == "execute_direct_movie" ]]; then
                set_result=queued
                sleep 1
            else
                set_result=$(
                    rg -o '<result>[0-9]+' /tmp/xa5-DRSPONSE.SXC |
                        tail -1 | tr -cd '0-9'
                )
            fi
        fi

        holder_log="$work_dir/key_${key}_${chain}_holder.log"
        case "$chain" in
            idle)
                ;;
            open)
                "$binary" --skip-device-reset --hold \
                    >"$holder_log" 2>&1 &
                holder_pid=$!
                ;;
            plain_open)
                "$binary" --hold \
                    >"$holder_log" 2>&1 &
                holder_pid=$!
                ;;
            plain_direct_movie)
                "$binary" --direct-movie --movie --hold \
                    >"$holder_log" 2>&1 &
                holder_pid=$!
                ;;
            plain_cancel_direct_movie)
                "$binary" --cancel-capture --direct-movie --movie --hold \
                    >"$holder_log" 2>&1 &
                holder_pid=$!
                ;;
            plain_open_movie)
                "$binary" --movie --hold \
                    >"$holder_log" 2>&1 &
                holder_pid=$!
                ;;
            direct_movie)
                "$binary" --skip-device-reset --direct-movie --movie --hold \
                    >"$holder_log" 2>&1 &
                holder_pid=$!
                ;;
            direct_movie_reset)
                "$binary" --direct-movie --movie --hold \
                    >"$holder_log" 2>&1 &
                holder_pid=$!
                ;;
            execute_direct_movie)
                "$binary" --direct-movie --movie --hold \
                    >"$holder_log" 2>&1 &
                holder_pid=$!
                ;;
            same_session_direct_movie)
                "$binary" --sxc-set CamDebugUsbMode "$key" \
                    --continue-after-sxc --direct-movie --movie --hold \
                    >"$holder_log" 2>&1 &
                holder_pid=$!
                ;;
            same_session_execute_direct_movie)
                "$binary" --sxc-execute CamDebugUsbMode "$key" \
                    --allow-sxc-execute --continue-after-sxc \
                    --direct-movie --movie --hold \
                    >"$holder_log" 2>&1 &
                holder_pid=$!
                ;;
            same_session_open_movie)
                "$binary" --sxc-set CamDebugUsbMode "$key" \
                    --continue-after-sxc --movie --hold \
                    >"$holder_log" 2>&1 &
                holder_pid=$!
                ;;
            open_movie)
                "$binary" --skip-device-reset --movie --hold \
                    >"$holder_log" 2>&1 &
                holder_pid=$!
                ;;
        esac

        if [[ -n "$holder_pid" ]]; then
            for _ in {1..60}; do
                if rg -q \
                    'sent operation 0x9020|claim interface|OpenSession failed' \
                    "$holder_log" 2>/dev/null; then
                    break
                fi
                kill -0 "$holder_pid" 2>/dev/null || break
                sleep 0.1
            done
        fi

        capture_log="$work_dir/key_${key}_${chain}_capture.log"
        ffmpeg -hide_banner -loglevel info \
            -f avfoundation -pixel_format "$capture_pixel_format" \
            -framerate "$capture_rate" -video_size "$capture_size" \
            -i "$capture_device:none" \
            -frames:v "$capture_frames" \
            -vf 'signalstats,metadata=print:file=-' \
            -an -f null - >"$capture_log" 2>&1 || true

        stop_holder

        command_result=none
        if [[ -f "$holder_log" ]]; then
            if rg -q 'claim interface .*LIBUSB_ERROR' "$holder_log"; then
                command_result=claim_failed
            else
                command_result=$(
                    awk '
                    /sent operation 0x101c|sent operation 0x9020/ {
                        awaiting = 1
                        next
                    }
                    awaiting && /received type 3 code/ {
                        print $5
                        exit
                    }
                    ' "$holder_log"
                )
                [[ -n "$command_result" ]] || command_result=none
            fi
        fi

        metrics=$(
            awk -F= '
                /lavfi.signalstats.YAVG=/ {
                    frames++
                    yavg += $2
                }
                /lavfi.signalstats.YMIN=/ {
                    if (!have_min || $2 < ymin) ymin = $2
                    have_min = 1
                }
                /lavfi.signalstats.YMAX=/ {
                    if (!have_max || $2 > ymax) ymax = $2
                    have_max = 1
                }
                /lavfi.signalstats.YDIF=/ {
                    ydif += $2
                    if ($2 > ydif_max) ydif_max = $2
                }
                /^frame:/ {
                    timestamp = $0
                    sub(/^.*pts_time:/, "", timestamp)
                    timestamp += 0
                    if (!pts_frames) first_pts = timestamp
                    last_pts = timestamp
                    pts_frames++
                }
                END {
                    if (!frames) {
                        print "0,0,no,0,0,0,0,0"
                    } else {
                        fps = 0
                        if (pts_frames > 1 && last_pts > first_pts) {
                            fps = (pts_frames - 1) / (last_pts - first_pts)
                        }
                        signal = ((ymax - ymin) >= 4 || ydif_max >= 0.5) \
                            ? "yes" : "no"
                        printf "%d,%.6f,%s,%.6f,%.6f,%.6f,%.6f,%.6f\n",
                            frames, fps, signal, yavg / frames, ymin, ymax,
                            ydif / frames, ydif_max
                    }
                }
            ' "$capture_log"
        )
        print \
            "$key,$chain,${set_result:-none},$command_result,$metrics"
    done
done
