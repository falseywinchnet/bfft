# OBS on macOS: `420v` capture

## Result

The installed OBS 32.1.1 capture adapter already supports native Apple
`420v`. OBS calls the memory layout **NV12** and keeps the video/full-range
distinction separately:

| AVFoundation FourCC | OBS label | OBS frame format | Range |
|---|---|---|---|
| `420v` | `NV12 (420v)` | `VIDEO_FORMAT_NV12` | partial/video |
| `420f` | `NV12 (420f)` | `VIDEO_FORMAT_NV12` | full |
| `yuvs` | `YUY2 (yuvs)` | `VIDEO_FORMAT_YUY2` | partial/video |

Consequently, "YUY2 versus NV12" in the OBS property control is the choice
between the packed 4:2:2 layout and the bi-planar 4:2:0 layout. Selecting the
camera's NV12 mode selects its `420v` AVFoundation format; `420v` is not a
third layout missing from the menu.

This is confirmed at all three relevant layers on this development Mac:

1. A live AVFoundation probe reports the MacroSilicon `USB3.0 HD VIDEO`
   device active at `1920x1080 420v`, with native `420v` modes from 640x480
   through 1920x1080.
2. OBS logs identify the modes as `NV12 (420v)`.
3. The saved OBS source's `supported_format` value ends in decimal
   `875704438`, which is hexadecimal `0x34323076`, or ASCII `420v`.

The High Vision OBS filter is already correct for this representation. It
accepts `VIDEO_FORMAT_NV12`, reads the native Y plane without resizing, and
uses `obs_source_frame.full_range` plus `color_range_min/max` to normalize
video-range luma. No new pixel-format case is required in that filter.

## Exact OBS implementation

The installed build is OBS 32.1.1. Its matching upstream source:

- names `kCVPixelFormatType_420YpCbCr8BiPlanarVideoRange` as
  `NV12 (420v)`;
- maps both `420v` and `420f` to `VIDEO_FORMAT_NV12`; and
- maps `VIDEO_FORMAT_NV12` back to `420v` unless the requested range is full.

See
[`OBSAVCapture.m` at the installed version](https://github.com/obsproject/obs-studio/blob/7272af1375b38bc3cf4e0f98a5d999e8b76e9309/plugins/mac-avcapture/OBSAVCapture.m#L790-L1029)
and
[`video-io.h`](https://github.com/obsproject/obs-studio/blob/7272af1375b38bc3cf4e0f98a5d999e8b76e9309/libobs/media-io/video-io.h#L32-L42).

## Recheck after an OBS or camera change

Run:

```sh
python3 osx_obs_features/420v_capture/audit_obs_420v.py
```

The audit is read-only. It reports the installed OBS version and plugin,
decodes the selected formats in every scene collection, and prints recent
capture-format evidence from the OBS logs. A unit test for the scene decoder
is included:

```sh
python3 -m unittest \
  osx_obs_features/420v_capture/test_audit_obs_420v.py
```

For direct hardware enumeration, the existing probe remains the stronger
device-level check:

```sh
xcrun clang++ -std=c++17 -fobjc-arc \
  high_vision/apple_capture/avcapture_probe.mm \
  -o /tmp/high_vision_avcapture_probe \
  -framework AVFoundation -framework CoreMedia \
  -framework CoreVideo -framework Foundation

/tmp/high_vision_avcapture_probe
```

## When a change really would be needed

A separate capture source is only warranted if future work requires the
original `CVPixelBuffer` and Apple attachments themselves rather than the
equivalent OBS frame representation. An OBS CPU filter receives
`VIDEO_FORMAT_NV12` plus range/color metadata, not the original CoreVideo
FourCC. For pixel processing, those fields fully describe `420v`; for auditing
capture provenance or CoreVideo-only attachments, they do not.

The OBS Virtual Camera is a different boundary. On this Mac its native
AVFoundation format is BGRA. AVFoundation advertises `420v` as a convertible
output for it, but that conversion is not native camera evidence.
