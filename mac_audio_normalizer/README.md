# TV Normalizer for this Mac

A native macOS menu bar host for the **actual GVST GMax 3.1 VST2 plug-in**.
It routes system playback through the already-installed VB-Cable device and
plays the processed stereo signal through the output selected when enabled.
No recording, upload, replacement DSP model, or additional audio driver is used.

## Everyday use

Open `~/Applications/TV Normalizer.app`. Click **GMax ON / GMax OFF** in the
menu bar:

- **Enable / Disable normalization** changes between GMax playback and direct
  output. Disable and Quit restore the speaker device selected before enabling.
- **Volume boost** selects 0–24 dB in 3 dB steps. The default is +12 dB and the
  selection is remembered. GMax's ceiling is −0.3 dB; release is 0.5 seconds.
- **Launch at login** is optional and initially off. It creates/removes the
  user's `Library/LaunchAgents/local.personal.TVNormalizer.plist`.
- **Quit and restore audio** exits and restores direct playback.

The app starts with normalization enabled. To use another output, disable
normalization, select that output in macOS Sound, then enable it again. An
external output switch stops normalization rather than overriding that choice.
Sleep stops routing; wake restarts it when it had been active. A sample-rate
change or stalled audio callbacks returns playback to the direct output.

macOS audio input permission is needed for **VB-Cable**, even though the app
does not select the built-in microphone. The original input device is not changed.

## Installation and rebuild

The installed application is:
`/Users/ultimussecundai/Applications/TV Normalizer.app`.
The GMax plug-in is installed separately in:
`~/Library/Audio/Plug-Ins/VST/GMax.vst`.
Its license is beside it as `GMax-LICENSE.txt`.

Get the Mac/Silicon download from the official vendor:
https://gvst.uk/Downloads/GMax/Select

GMax is proprietary freeware and is **not included in this repository**.
Its license limits redistribution; download the plug-in from GVST rather than
packaging it into a distributed copy of this app.

Build locally with `mac_audio_normalizer/build.sh` (Xcode command-line tools).
This is a small platform integration compile, not a heavy compute job.
The build uses Cocoa, Core Audio, Audio Unit, and AVFoundation; no package
manager or additional library is required. Quit the running app before replacing
its executable. The built app is locally ad-hoc signed.

## Audio and recovery design

Separate Core Audio HAL units capture VB-Cable and drive the original speaker
device. A bounded single-producer/single-consumer stereo ring provides 32 ms of
buffering. Slow fractional resampling tracks input/output clock drift. GMax adds
48 frames of lookahead at 48 kHz (1 ms), plus the audio devices' own latency.
The real-time callbacks allocate no memory and perform no disk or network I/O.

VST parameter changes happen on the processing callback. GMax supplies the gain
and limiter processing. A final finite-value / sample-ceiling guard protects the
output against invalid plug-in samples.

Routing is saved before changing the default output. A separate watchdog restores
the previous output if the app exits abnormally, provided VB-Cable is still the
default. The next app start also checks the saved recovery record. Ordinary
shutdown restores output before stopping the audio units. The watchdog does not
change an output the user has independently selected.

A one-second diagnostic snapshot is written to:
`~/Library/Application Support/TV Normalizer/status.json`.
It contains counters, peak amplitudes, current settings, and menu visibility;
it never contains audio samples. `recovery.json` holds the temporary device IDs.

## Validation (2026-09-20)

Tested on this Apple Silicon Mac, macOS 26.6.1, with VB-Cable and MacBook Neo
Speakers at 48 kHz:

- Real GMax quiet-signal gain: **11.9989 dB** for the +12 dB setting.
- Overload test: peak **0.966051**, matching the **−0.3 dB** sample ceiling.
- Stereo equality and post-release silence tests passed.
- Live playback captured and rendered nonzero audio with **zero underruns and
  overruns** during the observed run. User confirmed the improved loudness.
- Forced process termination restored **MacBook Neo Speakers** as the default
  output through the separate watchdog.
- Route enable/disable integration test processed 24,064 captured / 22,528
  rendered frames and restored the original output.
- Final installed app reported enabled normalization, a visible menu bar item,
  +12 dB gain, and active speaker playback after restart.

Re-run offline DSP checks:

```sh
"$HOME/Applications/TV Normalizer.app/Contents/MacOS/TVNormalizer" --self-test
```

With the normalizer **quit**, the short live routing check is:

```sh
"$HOME/Applications/TV Normalizer.app/Contents/MacOS/TVNormalizer" --route-test
```

`--devices` lists the device IDs and current output. `--restore` restores the
saved routing record; quit the app first. These are development/recovery
commands, not necessary for everyday use.
