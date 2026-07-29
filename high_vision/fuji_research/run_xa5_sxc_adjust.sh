#!/bin/zsh
set -u
setopt EXTENDED_GLOB

script_dir=${0:A:h}
binary=/tmp/xa5_libusb_ptp
user_domain="gui/$(id -u)"
log_dir=$(mktemp -d /tmp/xa5-sxc-adjust.XXXXXX)
restore_diag=0
hold_open=0
probe_preview=""
produce_preview=""

arguments=("$@")
if (( ${#arguments} >= 2 )) &&
   [[ "${arguments[-2]}" == "--produce-preview" ]]; then
    produce_preview="${arguments[-1]}"
    arguments[-2,-1]=()
fi
if (( ${#arguments} >= 2 )) &&
   [[ "${arguments[-2]}" == "--probe-preview" ]]; then
    probe_preview="${arguments[-1]}"
    arguments[-2,-1]=()
fi
if (( ${#arguments} > 0 )) && [[ "${arguments[-1]}" == "--hold-open" ]]; then
    hold_open=1
    arguments[-1]=()
fi
set -- "${arguments[@]}"

restore_camera_agents() {
    launchctl enable "$user_domain/com.apple.icdd" 2>/dev/null || true
    launchctl enable "$user_domain/com.apple.ptpcamerad" 2>/dev/null || true
    launchctl kickstart "$user_domain/com.apple.icdd" 2>/dev/null || true
    launchctl kickstart "$user_domain/com.apple.ptpcamerad" 2>/dev/null || true
}

cleanup() {
    if (( restore_diag )); then
        "$binary" --skip-device-reset \
            --sxc-set CamDiagLong 00000000 \
            >"$log_dir/99_restore_diaglong.log" 2>&1 || true
    fi
    restore_camera_agents
}

trap cleanup EXIT INT TERM HUP

if (( $# < 3 || ($# - 1) % 2 != 0 )) || [[ "$1" != "--allow" ]]; then
    print -u2 \
        "usage: $0 --allow OPCODE ARGUMENT [OPCODE ARGUMENT ...]"
    print -u2 \
        "       [--hold-open | --probe-preview OUTPUT |"
    print -u2 \
        "        --produce-preview OUTPUT]"
    print -u2 "       values may be decimal or 0x-prefixed"
    exit 2
fi
terminal_modes=$(( hold_open + (${#probe_preview} > 0) +
                   (${#produce_preview} > 0) ))
if (( terminal_modes > 1 )); then
    print -u2 \
        "--hold-open, --probe-preview, and --produce-preview are mutually exclusive"
    exit 2
fi

shift
command_values=("$@")
for value in "${command_values[@]}"; do
    if [[ "$value" != <-> && "$value" != 0[xX][0-9a-fA-F]## ]]; then
        print -u2 "opcode and argument must be unsigned integers"
        exit 2
    fi
done

for (( index = 1; index <= ${#command_values}; index += 2 )); do
    opcode=$(( command_values[index] ))
    argument=$(( command_values[index + 1] ))
    if (( opcode < 0 || opcode > 1023 || argument < 0 ||
          argument > 0xffffffff )); then
        print -u2 "opcode must be 0..1023 and argument must fit in 32 bits"
        exit 2
    fi
done

clang -std=c11 -Wall -Wextra -Werror \
    $(pkg-config --cflags libusb-1.0) \
    "$script_dir/xa5_libusb_ptp.c" \
    $(pkg-config --libs libusb-1.0) \
    -o "$binary" || exit $?

launchctl disable "$user_domain/com.apple.icdd"
launchctl disable "$user_domain/com.apple.ptpcamerad"
killall -KILL icdd ptpcamerad 2>/dev/null || true

run_step() {
    local name=$1
    shift
    print "SXC adjustment: $name"
    "$binary" "$@" >"$log_dir/$name.log" 2>&1
    local result=$?
    if (( result != 0 )); then
        print -u2 "step $name failed; log: $log_dir/$name.log"
        tail -40 "$log_dir/$name.log" >&2
        exit "$result"
    fi
}

restore_diag=1
step=1
first_session=1
for (( index = 1; index <= ${#command_values}; index += 2 )); do
    opcode=$(( command_values[index] ))
    argument=$(( command_values[index + 1] ))
    opcode_hex=$(printf "%08X" "$opcode")
    argument_hex=$(printf "%08X" "$argument")
    command_number=$(( (index + 1) / 2 ))

    session_options=()
    if (( ! first_session )); then
        session_options=(--skip-device-reset)
    fi
    run_step "$(printf '%02d_cmd%02d_set_argument' "$step" "$command_number")" \
        "${session_options[@]}" \
        --sxc-set CamDiagLong "$argument_hex"
    first_session=0
    (( step++ ))

    run_step "$(printf '%02d_cmd%02d_stage_argument' "$step" "$command_number")" \
        --skip-device-reset \
        --sxc-execute CamDiagOp 14 --allow-sxc-execute
    (( step++ ))
    run_step "$(printf '%02d_cmd%02d_set_opcode' "$step" "$command_number")" \
        --skip-device-reset \
        --sxc-set CamDiagLong "$opcode_hex"
    (( step++ ))
    run_step "$(printf '%02d_cmd%02d_dispatch' "$step" "$command_number")" \
        --skip-device-reset \
        --sxc-execute CamDiagOp 16 --allow-sxc-execute
    (( step++ ))
    print "SXC adjustment dispatched: opcode=$opcode argument=$argument"
done

run_step "$(printf '%02d_restore_diaglong' "$step")" \
    --skip-device-reset \
    --sxc-set CamDiagLong 00000000
restore_diag=0

if (( hold_open )); then
    print "SXC adjustment: entering hidden open-capture state"
    print "Unplug Fuji USB only after the client reports that it is holding."
    "$binary" --skip-device-reset --hold
elif [[ -n "$probe_preview" ]]; then
    print "SXC adjustment: probing hidden capture-preview producer"
    "$binary" --skip-device-reset \
        --preview "$probe_preview" --delay-ms 1000
elif [[ -n "$produce_preview" ]]; then
    print "SXC adjustment: invoking CamTakePreviewOp and probing in-session"
    "$binary" --skip-device-reset \
        --sxc-execute CamTakePreviewOp 00 --allow-sxc-execute \
        --preview "$produce_preview" --delay-ms 1500
fi

print "logs: $log_dir"
