# Fujifilm X-A5 firmware research

This directory contains firmware-analysis and explicitly invoked live PTP
research tooling for the ODM `LENGTH=` / `@DFI` firmware family used by the
Fujifilm X-A5 (`XZ01`) and its X-T100 (`XZ01A`) sibling.

The analyzer never communicates with a camera and cannot repack or flash an
image:

```sh
python3 len_dfi_inspect.py /path/to/FWUP0016.DAT
python3 len_dfi_inspect.py --json /path/to/FWUP0016.DAT
python3 len_dfi_inspect.py --extract-dir out /path/to/FWUP0016.DAT
```

Current evidence:

- `FWUP0016.DAT` is a concatenation of multiple `LENGTH=` records.
- Records may contain an `@DFI` image or raw tables/configuration payloads.
- Observed DFI headers are 0x200 bytes. Some prospective payloads are packed;
  others, including the image at file offset `0x77e600`, are followed by a
  high-confidence ARM32 exception-vector table.
- Packed payloads use a 4 KiB-window LZSS stream. The analyzer includes a
  read-only decompressor and bounds output using the size from the DFI header.
- This is a different firmware and service family from Fujifilm's normal
  X-Processor `04cb:ff80` jig path.

Until the button masks and SD filename checks are recovered from executable
code, do not infer that generic Fujifilm service chords or scripts apply.

The recovered X-A5 service-card evidence and safety boundary are documented in
[`XA5_SERVICE_CARD.md`](XA5_SERVICE_CARD.md). Direct ARM-state call references
can be reproduced with:

```sh
python3 arm_bl_xrefs.py ../out/fuji_xa5_203/00_00000100_anon_c0000000.bin \
  0xc01cc9e4 0xc01cbd2c 0xc02d59e8
```

On macOS, the live capability probe is hard-coded to Fujifilm `04cb:02d5` and
sends only standard PTP `GetDeviceInfo` (`0x1001`):

```sh
clang -fobjc-arc -framework Foundation -framework ImageCaptureCore \
  xa5_ptp_deviceinfo.m -o /tmp/xa5_ptp_deviceinfo
/tmp/xa5_ptp_deviceinfo
```

By default it does not send a property getter, property setter, capture
command, vendor operation, or adjustment event. The explicit
`--describe-set-usb-mode` option additionally sends the read-only standard
`GetDevicePropDesc` operation for property `0xd15d`:

```sh
/tmp/xa5_ptp_deviceinfo --describe-set-usb-mode
```

Neither mode sends `SetDevicePropValue`.

## Hidden live PTP surface

Static recovery of the X-A5 2.03 PTP dispatchers and live probing found two
unadvertised standard operations and a larger unadvertised Fuji operation
table:

- `0x101c InitiateOpenCapture` and `0x1018 TerminateOpenCapture` are active.
  Termination must carry the transaction id of the matching initiate command.
- `0x9020..0x9027`, `0x902b..0x902e`, and `0x9030` reach real vendor
  handlers even though they are absent from `GetDeviceInfo`.
- `0x101c 0 0` returns `OK` and visibly moves the rear display out of the
  ordinary white USB-mode screen.
- `0x9022 GetCapturePreview` returns `OK`, but both ImageCaptureCore and a
  direct USB bulk client receive an exactly 12-byte PTP data container: a
  header with zero payload. This proves ImageCaptureCore was not discarding a
  hidden JPEG in the tested state.
- The device has one configuration and one `06/01/01` PTP interface: bulk OUT
  `0x01`, bulk IN `0x82`, and interrupt IN `0x83`. It has no UVC interface,
  isochronous video endpoint, or alternate interface.

The flexible ImageCaptureCore probe sends arbitrary operation containers and
supports ordered sequences, response-transaction references, delays, outgoing
data, and a held session:

```sh
clang -fobjc-arc -framework Foundation -framework ImageCaptureCore \
  xa5_ptp_vendor_probe.m -o /tmp/xa5_ptp_vendor_probe

# Enter and hold the hidden open-capture state.
/tmp/xa5_ptp_vendor_probe 0x101c 0 0 --hold-open

# Enter, dwell, inspect the capture-preview data phase, and terminate using
# the initiate response transaction id.
/tmp/xa5_ptp_vendor_probe \
  0x101c 0 0 --delay-ms 2000 \
  --next 0x9022 \
  --next 0x1018 @0
```

The direct bulk transport bypasses `ptpcamerad`, issues the Still Image class
reset when recovering a stale session, performs open capture, optionally tries
the hidden movie state, and records the literal `0x9022` payload:

```sh
clang -std=c11 -Wall -Wextra -Werror \
  $(pkg-config --cflags libusb-1.0) xa5_libusb_ptp.c \
  $(pkg-config --libs libusb-1.0) -o /tmp/xa5_libusb_ptp

/tmp/xa5_libusb_ptp --describe
/tmp/xa5_libusb_ptp --preview /tmp/xa5-preview.bin
/tmp/xa5_libusb_ptp --movie --preview /tmp/xa5-preview.bin --hold
/tmp/xa5_libusb_ptp --direct-movie --preview /tmp/xa5-preview.bin --hold
/tmp/xa5_libusb_ptp --cancel-capture --direct-movie --movie --hold
/tmp/xa5_libusb_ptp --terminate-movie-id 2
/tmp/xa5_libusb_ptp --get-video-out
/tmp/xa5_libusb_ptp --set-video-out 1
/tmp/xa5_libusb_ptp --set-usb-mode 5
/tmp/xa5_libusb_ptp --set-force-mode 1
```

The routing-property setters are deliberately limited to Fuji's known
`VideoOutOnOff` (`0xd168`), `SetUSBMode` (`0xd15d`), and `ForceMode`
(`0xd230`) properties. On this X-A5 retail endpoint, all three currently
return `0x200a`, including after selecting debug mode 6.

On macOS, `ptpcamerad` owns the PTP interface exclusively. The raw client can
claim it only while that agent is stopped; the caller is responsible for
restoring the agent afterward. `run_xa5_libusb_ptp.sh` performs that handoff
and uses signal/exit traps to restore both Apple camera agents:

```sh
./run_xa5_libusb_ptp.sh \
  --direct-movie --preview /tmp/xa5-preview.bin --hold
```

Descriptor-only mode does not claim the interface and can be run directly.

`probe_xa5_hdmi_chains.sh` is the command-to-frame oracle for the attached
MacroSilicon HDMI decoder. It opens the exact AVFoundation device name rather
than a reorderable numeric index, requests NV12 at 1280x960, waits until the
PTP command has actually been sent, and scores 16 frames for luma range,
temporal difference, signal presence, and measured frame cadence:

```sh
XA5_DEBUG_KEYS="0 9" \
  ./probe_xa5_hdmi_chains.sh same_session_direct_movie

XA5_DEBUG_KEYS=none \
  ./probe_xa5_hdmi_chains.sh plain_cancel_direct_movie
```

The decoder's proven no-input frame is spatially and temporally flat
`Y=7, U=128, V=128`. In that fallback state AVFoundation actually timestamps
frames at about 8 fps even when FFmpeg requests 30 fps, so the harness reports
cadence from PTS deltas instead of trusting the requested rate.

The same client can enumerate the camera's SXC factory script objects and
make read-only registry requests:

```sh
./run_xa5_libusb_ptp.sh --sxc-list
./run_xa5_libusb_ptp.sh --sxc-discover
./run_xa5_libusb_ptp.sh --sxc-get CamVersion1
```

SXC `Set` and `Execute` are state-changing factory operations. In particular,
`CamDiagOp 0` starts calibration selector zero; it is not a benign query or a
way to submit adjustment opcode zero. The client refuses every SXC Execute
unless the caller also supplies the explicit `--allow-sxc-execute`
interlock. See `XA5_SERVICE_CARD.md` for the recovered dispatch path and the
recorded host-watchdog incident. Execute responses are cross-session on this
camera: the client closes after queueing the operation, and its next SXC
invocation saves the deferred `DRSPONSE.SXC` before sending a new request.
For a deliberately atomic state experiment, `--continue-after-sxc` keeps the
same PTP session and immediately runs the requested capture operation.

The `idle` and `plain_*` harness chains do not consume
`CamDebugUsbMode`, so they run exactly once even when the default nine-key
sweep is enabled. This matters because a successful `0x9020` start can leave
later starts returning `0x2019`; repeating a key-independent control would
destroy the clean-boot observation.

`plain_open` is the no-SXC `0x101c` control. The older `open` chain retains
its original meaning: set `CamDebugUsbMode` for each requested key, then enter
open capture.

The CSV `command_result` column reports the response to the chain's actual
`0x101c` or `0x9020` operation. An interface ownership failure is emitted as
`claim_failed`, rather than leaving a misleading empty movie-only field.

`run_xa5_sxc_adjust.sh` performs the recovered adjustment transaction,
restores `CamDiagLong`, and keeps Apple's camera agents disabled for the whole
sequence. Multiple opcode/argument pairs form one atomic chain: only the first
PTP session issues a class reset, preserving volatile camera RAM between
commands.

```sh
# Arm both raw-analysis markers and test the hidden PTP preview producer.
./run_xa5_sxc_adjust.sh --allow 328 1 399 1 \
  --probe-preview /tmp/xa5-raw-preview.bin

# Invoke the factory preview producer and query 0x9022 before closing the
# same PTP session.
./run_xa5_sxc_adjust.sh --allow 328 1 399 1 \
  --produce-preview /tmp/xa5-produced-preview.bin

# Arm both markers, enter hidden open capture, and wait for a physical USB
# disconnect without rebooting the camera.
./run_xa5_sxc_adjust.sh --allow 328 1 399 1 --hold-open
```

The X-A5 and HDMI capture dongle are not simultaneous observation channels.
Fujifilm's manual explicitly says USB cannot be used while HDMI is connected
and prescribes camera off, HDMI connected, then camera on. A live experiment
nevertheless established a no-reboot bridge: after the atomic raw-marker
chain and successful `0x101c InitiateOpenCapture`, hot-unplugging Fuji USB
made ordinary HDMI resume. The resulting 1280x960 Rec.709 NV12 stream retained
normal color/gamma and Fuji overlays; it was not a Bayer or high-bit-depth
transport.

The armed `0x9022` probe remained an exactly zero-byte data phase. The SXC
registry's explicit `CamTakePreviewOp` was then invoked immediately before
open capture and `0x9022` in the same PTP session; that also returned zero
bytes. An all-format object inventory can be reproduced with:

```sh
./run_xa5_libusb_ptp.sh --skip-device-reset --list-objects
```

After the factory preview operation, that inventory contained only the empty
`DCIM` association—no JPEG, RAF, vendor object, or hidden image object.
Adjustment opcodes 601 and 857 alias the volatile
`SIGPRO_IMAGE_DUMP_MODE` byte. Combining opcode 601 with `RAWMODE`,
`RAW_DATA_MODE`, and `CamTakePreviewOp` still produced neither a `0x9022`
payload nor a new PTP object.

The firmware does contain a second PTP data-plane scaffold named
`SXC_VIRT.BIN`. `c02039c0` builds its virtual ObjectInfo and `c0203a60`
handles its object transfer, but the retail path currently supplies a zero
size/source. This is a plausible narrow firmware-patch target: bind the
virtual object to the last completed SRAW pool buffer, then retrieve it with
ordinary bulk PTP or `GetPartialObject`. The host cannot read arbitrary camera
RAM merely by knowing an address; a camera-side operation must perform or
authorize that mapping.

The negative retail result does not rule out the camera's debug USB state.
`CamDebugUsbMode` is absent from the SXC Get schema, so the earlier
`FFFFFFFF` result from attempting to read it was a host-side
missing/unsupported-value sentinel. It is present in the Set and Execute
schemas, and its recovered setter searches a nine-entry key-to-mode table
before posting two USB reconfiguration events. The accepted keys are `0`,
`1`, `3`, `4`, `5`, `6`, `9`, `A`, and `FF`: key `0` selects mode 0,
key `A` selects mode 6, and the remaining keys select mode 1 submodes 0-6.
Debug mode may still authorize additional operations or back `SXC_VIRT.BIN`
while retaining the same `04cb:02d5` PTP descriptor.

At `0x3f5000` bytes per SRAW pool surface, 30 fps would require about
124.5 MB/s before protocol overhead and cannot fit through the camera's USB
2.0 link. A carefully implemented bulk reader could still plausibly deliver
several frames per second. It must select a completed—not actively written—
triple buffer and carry a sequence number or generation check to reject torn
frames.

The MacroSilicon `534d:2109` dongle exposes no UVC vendor extension unit, so
the host cannot query HDMI input lock or timing through a vendor UVC control.
`inspect_uvc_descriptors.c` reproduces that descriptor audit.

Fuji's network remote protocol uses PTP as a command channel but delivers live
JPEGs over a separate TCP socket on port 55742 after open capture (with events
on 55741). The X-A5 USB personality exposes no network interface, so a custom
USB PTP driver alone cannot create that JPEG channel. The current live leads
are therefore:

1. open capture may enable a usable HDMI acquisition state;
2. the recovered `CamDebugUsbMode` transition may expose or authorize an
   additional PTP data plane;
3. a deeper remote-mode prerequisite may make `0x9020` start the producer;
4. an unlocated factory buffer-copy or publication operation may expose a
   different transport.

The `--inspect-usb-mode` option sends the read-only standard
`GetDevicePropDesc` and `GetDevicePropValue` operations for libfuji's active
transport property `0xd16e`:

```sh
/tmp/xa5_ptp_deviceinfo --inspect-usb-mode
```

It does not query or change `SetUSBMode` (`0xd15d`).

The minimal mode-3 adjustment-card firmware candidate is built only from the
known X-A5 2.03 stock package. The builder refuses an unexpected source hash,
never overwrites its input or an existing output, repairs the outer byte sum,
and verifies that decompression changes exactly one executable byte:

```sh
python3 build_xa5_mode3_service_firmware.py \
  /path/to/stock/FWUP0016.DAT \
  /new/output/path/FWUP0016.DAT
```

This creates an offline research artifact. It does not access the camera,
prepare an SD card, or initiate a firmware update.
