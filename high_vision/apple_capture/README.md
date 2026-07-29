# Apple camera capability probe

`avcapture_probe.mm` is the go/no-go test for routing a macOS camera directly
into High Vision. It distinguishes:

- capture formats emitted natively by the device;
- CoreVideo pixel formats that `AVCaptureVideoDataOutput` can convert those
  native frames into; and
- exposure controls the device actually publishes.

That distinction is important. An output option such as BGRA, full-range
`420f`, or packed `2vuy` is not additional sensor evidence when the device's
native capture format is limited-range `420v`; it is a conversion of that
already processed 8-bit YUV frame.

Build and run:

```sh
xcrun clang++ -std=c++17 -fobjc-arc \
  high_vision/apple_capture/avcapture_probe.mm \
  -o /tmp/high_vision_avcapture_probe \
  -framework AVFoundation -framework CoreMedia \
  -framework CoreVideo -framework Foundation

/tmp/high_vision_avcapture_probe
```

`avcapture_record.mm` is the matching raw-frame recorder for UVC devices whose
AVFoundation negotiation is stricter than FFmpeg's. The MacroSilicon HDMI
adapter accepts literal advertised frame-duration fractions; FFmpeg otherwise
falls back to slow `yuvs`. The recorder selects native video-range `420v`,
writes exact NV12 planes, and records source presentation timestamps:

```sh
xcrun clang++ -std=c++17 -fobjc-arc \
  high_vision/apple_capture/avcapture_record.mm \
  -o /tmp/high_vision_avcapture_record \
  -framework AVFoundation -framework CoreMedia \
  -framework CoreVideo -framework Foundation

/tmp/high_vision_avcapture_record /tmp/fuji 1920 1080 30 4
```

## FaceTime HD Camera result

On the development Mac, the built-in camera exposes four native modes:

| Resolution | Rate | Native subtype |
|---|---:|---|
| 1920x1080 | 30 fps | `420v` |
| 1280x720 | 30 fps | `420v` |
| 1024x768 | 30 fps | `420v` |
| 640x480 | 30 fps | `420v` |

It exposes no AVFoundation exposure mode or exposure point. macOS also marks
AVFoundation's Bayer RAW photo-format API unavailable; it is not a hidden
video path.

Consequently, a standalone FaceTime viewer could force 1920x1080, preserve
native timestamps and planes, and avoid accidental resizing before High
Vision. It would not supply RAW/Bayer samples, additional luma levels, or
manual shutter/ISO control. The current 640x480 OBS input is therefore a
configuration loss, not evidence that OBS is hiding a richer FaceTime stream.

## External-camera decision rule

Run the same probe after connecting the Fuji:

- A higher-resolution `2vuy`/`yuvs` native mode is useful: it preserves luma
  per pixel and avoids 4:2:0 chroma subsampling, though it remains ISP output.
- A native 10-bit or Bayer subtype would justify a dedicated capture bridge.
- Native `420v` alone means a standalone viewer only buys format selection and
  timing. Prefer configuring OBS at the native resolution unless measurement
  shows that OBS adds another conversion or resize.
- Output formats listed only under `video_output_pixel_formats` do not count as
  better native evidence.
