# Fujifilm X-A5 2.03 service-card findings

This records only behavior observed directly in the decompressed official X-A5
2.03 executable. Service procedures for other Fujifilm processor families are
not assumed to apply to the X-A5.

## Identity

- Firmware container: `FWUP0016.DAT`
- SHA-256:
  `0d35e4104b98513154c6fe0f827ef771f7531712386a468612ea652f792baeff`
- Camera/product identifier: `C713A`
- Main executable load address: `0xc0000000`
- Main executable size: `0x831000`

## Adjustment-card filesystem

Before enabling the adjustment engine, `0xc01cdc04` constructs the packed FAT
8.3 key `C713AADJ` from the embedded strings `C713A` and `ADJ`, then compares
the first eight bytes against up to 128 records in the camera's active
directory cache at `0x5fd59800`. Therefore the card requires an empty
activation marker:

- `C713A.ADJ`

Root-only placement on a valid FAT32 card did not activate the handler. The
builder currently emits the marker both at the root and inside `ADJ`, beside
the model payload directory, until the cache's directory source is fully
resolved.

The surrounding directory code reads classic 32-byte FAT directory records
and tests the attribute byte at offset 11. An exFAT card does not encode its
directory this way and did not activate in two camera boots even with the
marker present. The current research card is a single full-capacity MBR/FAT32
filesystem. The camera successfully wrote `DSCF0273.JPG` to it, proving that
the filesystem is writable and that card-format failure is not the present
service-mode blocker.

The executable constructs `ADJ/C713A` and contains these packed FAT 8.3 names:

- `INPUT   DAT` -> `INPUT.DAT`
- `CARDVER DAT` -> `CARDVER.DAT`
- `DATE    DAT` -> `DATE.DAT`
- `FINISH  MRK` -> `FINISH.MRK`
- `RESPONSEDAT` -> `RESPONSE.DAT`
- `RESPONS2TXT` -> `RESPONS2.TXT`
- `SCRIPT  DAT` -> `SCRIPT.DAT`

The service-card handler is at `0xc01cdfb4`. It reads the packed 8.3 files into
a shared work buffer. The initializer copies exactly 12 bytes from
`INPUT.DAT` into its result structure and later compares a 13-byte field
against the camera identity at `c0d6ea20 + 0x148`; the thirteenth byte is
structural padding or a terminator, not evidence for a 13-character input
identity. The parser recognizes process/step headers with the fixed form
`#Pddd-Sddd`, with three decimal digits per field.

## Proven read-only adjustment backup operation

The adjustment command handler begins at `0xc02d59e8`.

Event `1100` (`0x44c`) prints `DUMP ADJUST DATA`, then calls the output routine
at `0xc02d87d0` with:

- source address `0x5ffa0000`
- length `0x40000` (256 KiB)
- packed FAT name `CAM256KBADJ`, meaning `CAM256KB.ADJ`

This is an outward copy of the camera's adjustment block.

## Dangerous inverse operation

Event `1101` (`0x44d`) prints `WRITE ADJUST DATA`, reads the adjustment image,
and follows a flash/programming path. It must not be invoked during discovery.
Event `1102` is a cleanup/exit-like branch.

Any generated experiment must be constrained to event 1100 and reject 1101.

## Boot path

The adjustment engine entry is `0xc01cc9e4`. It is called from normal boot at
`0xc001e024`, after two gates:

1. the byte returned by `0xc01cbb1c` equals one; and
2. the signed byte at `0xc0e5d1b8` is nonzero.

The first byte is populated from `0xc01cdc04` during card initialization when
the active directory cache contains `C713A.ADJ`, but that scan is itself
disabled unless `c0d6ea20 + 0x155` is one of `2`, `4`, `16`, or `17`. Invalid
or empty persisted values are normalized to retail value `3` at
`0xc01b0610`. This explains why rearranging correct files and trying boot
chords produced no response: a normal retail boot never reaches the marker
comparison.

The firmware's parameter registry names this byte
`CamSelfCalibrationMode`. Internal set command `255` dispatches to
`0xc01c12e0`, which accepts `1..4`, `16..17`, `256`, and `257`. Values
`1..4` and `16..17` are written to the mode byte and followed by
`0xc01b031c`; changed configuration is then scheduled for persistent storage
through `0xc01df5a0`. Consequently command 255 is **not** a harmless
in-memory unlock and must not be emitted speculatively.

The corresponding read-only getter is `0xc01c1890`. It returns the mode byte
(or the related 16-bit extended value) without storing configuration. A safe
next dynamic step is therefore to recover the external transport and issue
the getter before considering any mode transition.

There is also a dedicated self-calibration transition, distinct from the
generic registry setter. Adjustment-script command **109** reaches
`0xc01c10cc`, which
calls `0xc01cbb74` and then `0xc01b031c`. The first function:

- writes `1` to `CamSelfCalibrationMode`;
- clears bit `0x4` in the related word at `c0d6ea20 + 0x2e4`.

The second function compares the complete live adjustment configuration
against its saved copy and calls `0xc01df5a0` when they differ. Command 109 is
therefore a persistent service transition, not a safe probe.

The script parser at `0xc01d1578` accepts records beginning with byte `0x1b`
and a family byte in the inclusive range `0x43..0x45`. It reconstructs the
inner command number from that family byte and packet byte 5 before invoking
the command table at `0xc01c6158`. Its sole caller is inside the adjustment
engine's card-processing loop; nearby literals include `RESPONSE.DAT`,
`RESPONS2.TXT`, and `RETRY_COUNT`. This is therefore the adjustment-card
record envelope, not evidence that ordinary PTP accepts command 109.

Other factory-facing registry entries recovered nearby include:

- `CamAdjustStatus`, id `0x83`
- `CamFactoryCode`, id `0x8c`
- `CamUsbInitMode`, id `0xa5`
- `CamDebugUsbMode`, id `0x1d8`

These ids are registry ids, not necessarily valid ids in every factory
packet namespace. In particular, the type-`E` packet path accepts queued
operations `496..732`; blindly translating registry id `0x1d8` by adding
496 produces 968, which that path rejects. Do not treat 968 as a recovered
debug-USB event.

## X-A5 USB transport

The executable contains the X-A5 USB PTP descriptor builder at
`0xc01c4004` with:

- vendor id `0x04cb` (Fujifilm)
- product id `0x02d5`
- product string `USB PTP Camera`
- model string `X-A5`
- firmware string `2.03`

`CamUsbInitMode` reads persistent word `c0d6ee74 + 0x1d0`. It returns one
when the word equals `0x554f4646`. This value is the compiler's
little-endian representation of the four-character literal `UOFF`, strongly
indicating a USB-off sentinel. Its set handler replaces the sentinel with
`-1` and invokes the persistent configuration path. This proves that a
separate USB-initialization option exists, but it does **not** make the
`04cb:02d5` endpoint a factory endpoint. That USB id is the ordinary X-A5 PTP
identity in libgphoto2's camera database.

The correct next target remains the read-only side of the service transport:
recover the button/boot condition or temporary debug path whose endpoint
differs observably from ordinary `04cb:02d5`, then query `CamUsbInitMode` and
`CamSelfCalibrationMode` before changing either value. Command 109 only
demonstrates that an already-running adjustment session can persistently
select self-calibration mode; it does not bootstrap that session from retail
PTP.

### Comparison with `tiredboffin/fffw`

The public `fffw` tooling clarifies that there are two distinct USB gates on
the conventional Fujifilm service family:

1. The camera must already have entered the service route and enumerated as
   `04cb:ff80`. Its documented entry procedure is to connect USB, then power
   on while holding **Shutter + Up**.
2. Once that endpoint is open, `fffw` can temporarily change configuration
   byte `0xf7` to enable the debug operations used for RAM access and synthetic
   key events. The implementation reads the previous byte and restores it
   after the operation.

Consequently the `0xf7` toggle and synthetic `KEY_USB`/`KEY_JIGBOOT` events
cannot bootstrap a camera from ordinary retail USB: all of them are carried
inside a service session that must already exist.

The X-A5 is not known to be an `ff80` firmware target. The **Shutter + Up** procedure is
therefore evidence about another family's architecture, not a recovered X-A5
chord and must not be presented as one. X-A5 enumeration should be inspected
for a change away from its ordinary `04cb:02d5` id while its actual boot
predicates are recovered from this firmware.

The X-A5's `CamUsbInitMode` is the more plausible analogue of the first gate.
`CamDebugUsbMode` is now proven to drive a live USB-state transition and is
therefore a plausible analogue of the second, although no equivalence to the
other family's byte `0xf7` has been established. Registry id `0x1d8` is not
itself the value to send, and the mode must not be treated as a Boolean.

### Recovered ordinary X-A5 PTP boot dispatcher

A live X-A5 has now enumerated in its ordinary PTP mode as:

- product string `USB PTP Camera`
- vendor/product `04cb:02d5`
- USB 2.0 high speed

The static call chain that creates this endpoint is:

```text
c0008204 -> c0008bcc -> c01fd1b4 -> c01fd0e8 -> c01fd03c
          -> c032cf54 -> c01c4340 -> c01c4004
```

At `c0008bcc`, boot calls `c0185298`, tests return bit `0x4`, and only then
calls `c01fd1b4` with transport mode `4`. `c0185298`:

1. reads SoC register `0x2d000314` through `c01f46a0(0x314)`;
2. isolates register bit 1;
3. returns zero immediately when that bit is set;
4. otherwise queries a global connection-mode value through `c00dc538` and
   returns either `1` or `4`.

This layer is a hardware/boot-state dispatcher, not a direct comparison of
named camera buttons. Register `0x2d000314` belongs to a memory-mapped input
bank used throughout the adjacent switch/hardware abstraction; bit 1 is the
hard gate for this ordinary transport branch. A physical service chord must
be decoded elsewhere or alter an earlier latched state. No X-A5 service chord
has yet been recovered.

The ordinary PTP descriptor initializer at `c01c4004` is unconditional once called:
it installs Fujifilm vendor id `0x04cb`, fixed product id `0x02d5`, and the
normal PTP strings. Other callers invoke the same higher transport dispatcher
with modes `13` and `17`, confirming that `c01fd1b4` is a shared mode
initializer rather than the physical-input decoder itself.

### Read-only live PTP capability probe

`xa5_ptp_deviceinfo.m` uses macOS ImageCaptureCore to send only the standard,
read-only PTP `GetDeviceInfo` operation (`0x1001`) to a device whose identity
is exactly `04cb:02d5`. A live X-A5 in ordinary USB/card-reader mode returned
success (`0x2001`) and advertised these 25 operations:

```text
1001 1002 1003 1004 1005 1006 1007 1008
1009 100a 100b 100c 100d 100f 1014 1015
1016 101b 9801 9802 9803 9805 900c 900d 901d
```

It advertised only these eight device properties:

```text
5001 5005 5007 500a 5010 d407 d406 d303
```

In particular, the cross-model libgphoto2 name `SetUSBMode` at property
`0xd15d` is **not advertised by this X-A5 endpoint**. A separate read-only
`GetDevicePropDesc(0xd15d)` probe returned `0x200a`
(`DevicePropNotSupported`) with no data phase, so it is not merely an omitted
descriptor for a readable property in the current mode. No property setter
was sent. The DeviceInfo record reports vendor-extension id `6`, the string
`fujifilm.co.jp: 1.0`, and device version `2.03`.

`libfuji` distinguishes that setting property from active transport property
`USBMode` at `0xd16e` (values `5` tether, `6` raw conversion, and `8`
webcam). A live read-only X-A5 probe found that `0xd16e` is also unadvertised.
Both `GetDevicePropDesc(0xd16e)` and `GetDevicePropValue(0xd16e)` returned
`0x200a` (`DevicePropNotSupported`) with zero data bytes. Thus the ordinary
X-A5 card-reader endpoint does not expose even a read-only active USB-mode
property through the cross-model interface. No property setter was sent.

The ordinary vendor-operation dispatcher begins at `0xc020dc7c`. Its compiled
table includes operations beyond those advertised in the current mode, but
the three live operations resolve as follows:

- `0x900c` -> `0xc020d3a0`
- `0x900d` -> `0xc020d4c8`
- `0x901d` -> `0xc020d7e0`

None of these handlers reaches either recovered writer of
`CamSelfCalibrationMode`. The generic writer at `0xc01c131c` remains reachable
through internal command 255, and the dedicated writer at `0xc01cbb74`
remains reachable through adjustment-card command 109. No third direct writer
exists in the executable.

The result is a useful negative: **there is no presently recovered,
advertised ordinary-PTP command that enters calibration mode**. It remains
possible that a hidden operation is accepted without being advertised, but
sending unclassified vendor setters is not justified. The next bootstrap
target is the earlier boot/control route that changes transport mode or USB
identity; once that route is entered, its read-only capability surface can be
queried again.

### Boot-time USB-listener audit

The main executable's startup sequence does not contain a pre-PTP loop that
receives a USB command and selects calibration mode.

The relevant order in the startup function is:

```text
c0008204  call c0008bcc       USB connection branch
c0008208  continue ordinary subsystem initialization
...
c0008250  enter the normal boot-event loop
```

`c0008bcc` first calls `c0185298`. That function reads hardware register
`0x2d000314`, tests bit 1, and queries the already-latched connection state.
It does not initialize an endpoint or inspect host data. When the returned
mask contains bit `0x4`, `c0008bcc` calls:

```text
c01fd1b4(4) -> c01fd0e8 -> c01fd03c -> c032cf54
             -> c01c4340 -> c01c4004
```

This constructs and installs the ordinary `04cb:02d5` PTP descriptors and
starts the normal transport. `c01fd1b4` contains initialization and error
handling but no receive-and-compare loop before it returns.

Startup next calls `c00e3c30`. Its conditional 50 ms delay initially looked
like a possible host-command window. The gate at `c00e43b0`, however, only
tests a byte in an existing global structure. The remainder starts the
ordinary transport task and propagates state; it does not read a setup packet,
bulk packet, command signature, or mode value from USB.

After that point the endpoint can receive commands concurrently with the
remaining camera boot, but those commands enter the same ordinary PTP
dispatcher audited above. There is no separate boot-only operation table and
no path from that dispatcher to either `CamSelfCalibrationMode` writer.

This result applies to the decompressed X-A5 2.03 main executable. It cannot
exclude a command recognized by immutable SoC boot ROM or by an earlier
loader that is not present in this firmware image. Such a route would have to
run before this executable reaches `c0008bcc`, and would likely be observable
as a different USB identity or descriptor set before `04cb:02d5` appears.

### Normal-versus-debug USB decision audit

There is no recovered `normal USB -> debug USB` *descriptor-identity*
decision in the X-A5 2.03 main executable. There is, however, a recovered
live debug-USB mode setter that can reconfigure the USB state while retaining
the possibility of the same descriptor identity.

The only meaningful USB device-descriptor constructor in the image is
`0xc01c4004`. It installs vendor id `0x04cb` and obtains its product id from
`0xc00e4480`; that function unconditionally returns `0x02d5`. A search of the
complete decompressed executable found no second `0x04cb` descriptor
constructor and no meaningful `0xff80` USB product-id constant. The apparent
`0xff80` immediates elsewhere belong to image arithmetic and floating-point
special-value construction.

Transport mode `17` must not be confused with
`CamSelfCalibrationMode == 17`. In `0xc01fd1b4`, transport mode `17` takes a
special teardown/state-transition route and bypasses the descriptor
initialization path entirely. It therefore does not supply an alternate USB
identity that can simply be selected in place of transport mode `4`.

The extracted `NROG` record is a RIFF/WAVE resource, not a secondary USB
loader, and the extracted `TEST` records inspected so far are resources such
as fonts. Thus the update package has not exposed an earlier executable
containing a second USB personality.

`CamDebugUsbMode` is present in the SXC Set and Execute schemas, but not in the
SXC Get schema. The earlier live `Get` result of `FFFFFFFF` was therefore the
host parser's missing/unsupported-value sentinel, not a camera mode value.
That probe did not test debug USB.

The Set schema maps `CamDebugUsbMode` to setter case `0x41`, whose call chain
reaches `0xc01e12dc`. This function searches a runtime table of exactly nine
eight-byte entries. Each entry contains:

- a 32-bit external key;
- a primary mode byte at offset `+4`; and
- a signed submode byte at offset `+5`.

After selecting an entry, the setter stores the primary mode, handles the
mode-1 submode mapping, calls the USB/state routine at `0xc00d76b8`, and posts
events `0x1c000003` and `0x1c000002` through `0xc00c8e28`. This is a real USB
personality/state switch, not dead registry metadata.

The nine-entry source table is at `0xc05eee78`. Its accepted external keys and
selected `(primary mode, submode)` pairs are:

- `0` -> `(0, 0)`;
- `10` (`0x0a`) -> `(6, 0)`;
- `1` -> `(1, 0)`;
- `255` (`0xff`) -> `(1, 1)`;
- `9` -> `(1, 2)`;
- `3` -> `(1, 3)`;
- `4` -> `(1, 4)`;
- `5` -> `(1, 5)`;
- `6` -> `(1, 6)`.

All nine values are accepted by the live SXC Set path and return a complete
`DRSPONSE.SXC`. The response field `<result>11</result>` is not an
invalid-key error: a nonexistent registry name produces no response object,
while other real fields return different result values.

No consumer has yet been tied to a second USB descriptor or product id. The
most plausible current interpretation is that debug mode gates additional
operations behind the same `04cb:02d5` PTP transport. In particular, it may
populate or authorize the otherwise zero-length `SXC_VIRT.BIN` data plane,
enable a memory/buffer operation table, or expose a factory handle-based image
path. A pre-main-image loader remains possible, but it is no longer required
to explain `CamDebugUsbMode`.

The cross-family `fffw` configuration byte at offset `0xf7` was also checked
as a possible structural anchor. X-A5 does share at least one suggestive
configuration offset with that family—its proven self-calibration byte is at
offset `0x155`—but the X-A5 main executable contains no load from
`c0d6ea20 + 0xf7`. The few instructions using immediate `0xf7` address stack,
UI, and unrelated runtime structures. There is therefore no static evidence
that forcing the conventional family's byte `0xf7` would enable X-A5 debug
USB.

Each key was tested with descriptor enumeration before and after the
transition. The camera retained `04cb:02d5`, its single Still Image interface,
and the same bulk and interrupt endpoints; no alternate setting or
isochronous endpoint appeared. Atomic same-session chains then issued hidden
movie start immediately after every key, through both SXC Set and Execute.
Every chain returned the same `0x2019` movie result and the HDMI decoder
remained at its exact flat no-signal frame. The later clean-boot control showed
that this sweep began with the persistent movie predicate already set, so it
does not discriminate among debug keys as movie-start prerequisites. A
separate full nine-key sweep using the independently repeatable `0x101c`
open-capture path did discriminate routing: all nine open operations returned
`0x2001`, and every HDMI sample remained the identical fallback frame.

### Passive boot-identity observations

A passive IOKit notification trace, filtered only by Fujifilm vendor ID
`04cb`, observed the following:

- an ordinary USB boot enumerates only `04cb:02d5`, named `USB PTP Camera`;
- entering the DISP/BACK firmware-upgrade screen removes `04cb:02d5`;
- while the firmware-upgrade header displays `NO MEMORY CARD`, no replacement
  Fujifilm USB identity enumerates;
- the additional tested button-chord boots produced only brief or stable
  appearances of the same ordinary `04cb:02d5` identity.

The tracer sent no control, bulk, PTP, or vendor packets. These observations
substantially constrain the immutable-loader caveat above: the observed
firmware-update branch does not expose a host-facing USB bootloader.

### Firmware-upgrade branch

The update path beginning near `0xc001e300` is a removable-media loader, not a
second USB dispatcher. Its update-specific region calls neither
`0xc01fd1b4` nor `0xc01c4004`, the ordinary USB/PTP initializers. It instead
opens and parses `FWUP0016DAT` through `0xc01cb644` and `0xc01cb384`, then
displays the embedded `BATTERY LOW`, `FIRM UP ERR`, or `FIRM UP` status.
Failure to find usable media stops earlier in this same state machine, which
matches the observed firmware-upgrade header plus `NO MEMORY CARD`.

The stock package's outer integrity field is presently reproducible as a
plain 32-bit sum of payload bytes. For example, the first record declares
`SUM=939407882`, exactly equal to the byte sum of its 7,721,472-byte payload.
The other records use the same scheme when their tag-prefixed 256-byte headers
(`IPL`, `PTBL`, `SUB1`, and `ND1`) are accounted for. This proves that the
outer `SUM` layer is not cryptographic authentication. It does not yet prove
that every embedded `@DFI` validation layer is unsigned or safely repackable.

### Minimal custom-firmware candidate

The narrowest recovered patch does not overwrite the persistent
`CamSelfCalibrationMode` value. The adjustment-card scan at `0xc01cdc04`
locally accepts modes `2`, `16`, `4`, and `17`, then exits before scanning in
ordinary retail mode `3`. Replacing only its final `17` comparison with `3`
would make retail mode eligible while leaving the shared mode predicates and
the stored configuration untouched:

```text
c01cdc18  cmpne r0, #16
c01cdc1c  cmpne r0, #4
c01cdc20  cmpne r0, #17   ->   cmpne r0, #3
c01cdc24  bne   exit
```

This intentionally trades away mode 17 only at this one call site. In the
stock LZSS stream, the immediate byte for `c01cdc20` is a literal at package
offset `0x15beaa`. Full-stream provenance analysis finds exactly one
decompressed consumer of that literal. Therefore changing package byte
`0x15beaa` from `0x11` to `0x03` changes exactly one byte in the 8,589,312-byte
main executable; it does not require recompression or move any later record.
The containing outer byte sum would decrease by 14, from `939407882` to
`939407868`.

This is a patch candidate, not yet a flash instruction. Before placing it on
an SD card, the remaining work is to reproduce all loader checks over the
embedded `@DFI` image, generate the candidate from the known stock SHA-256,
decompress it again, and assert that the intended instruction is the sole
executable difference.

The guarded offline builder now performs those package-level checks. From
stock SHA-256
`0d35e4104b98513154c6fe0f827ef771f7531712386a468612ea652f792baeff`,
it generated a 22,030,584-byte candidate with SHA-256
`0f2f3e874cfba729383b0602aaad8f63ea520b5757bb4732b683093d225c76d8`.
An independent extraction confirmed the patched instruction bytes
`03 00 50 13` at executable offset `0x1cdc20`. At the package level, `cmp`
reports only the two ASCII digits needed to change the declared sum and the
one compressed literal at `0x15beaa`.

### First patched-camera observation

The X-A5 accepted and installed the candidate, then remained bootable. With
the research SD card inserted, entering the camera's USB mode caused the
ordinary `04cb:02d5` endpoint to disappear and no alternate Fujifilm USB
identity enumerated. After powering off, removing the SD card, and booting the
same installed firmware normally, `04cb:02d5` returned immediately.

A read-only `GetDeviceInfo` request then completed successfully and reported
model `X-A5`, device version `2.03`, the same 25 operations, seven events, and
eight properties as the stock ordinary PTP surface. This A/B result isolates
the transport change to the inserted research card: retail mode 3 now reaches
the adjustment-card route, while ordinary cardless PTP remains intact. It
also confirms that this service route is card-driven rather than an alternate
USB identity.

## Runtime raw-mode path

The firmware also contains a more direct sensor-data path. The adjustment
dispatcher at `0xc01c961c` maps operation **328** to `0xc02d3680`. That
handler accepts only argument zero or one:

- zero writes `0` to `c0d6ee74 + 0x9d8`;
- one writes the runtime marker `0x1234` to the same word;
- other values do not change the word.

It then reports `RAWMODE OFF` or `RAWMODE ON`. There is no persistent-save
call in this handler. Adjacent firmware strings and configuration reporting
also expose `RAW_MODE=`, `RAW_SIM=`, `RAW_DATA_MODE`, `ChangeSensorMode=`,
raw width/height/bit-size fields, and the raw-output pipeline stages.

Operation 328 is therefore a substantially better target than forcing the
self-calibration boot byte: it is an explicit, apparently volatile raw-mode
toggle. Its enclosing adjustment transport and the destination of the raw
frames still need to be recovered before any packet is emitted.

The most coherent interpretation is that persisted self-calibration mode `1`
selects a calibration session intended to cooperate with factory tooling,
while operation 328 enables raw acquisition within that session. Calibration software commonly needs
unprocessed sensor samples to measure and program defect, shading, and color
tables. The firmware supports that architecture, but the names alone do not
prove that mode `1` automatically exposes a continuous Bayer stream over USB.

The mapping was recovered from the compressed dispatcher table at
`c05ee116`: key `0x4e`, whose command range begins at 250, resolves to
operation `250 + 0x4e = 328` and branches to `0xc01c9fac`, which calls the
raw-mode handler.

Queued factory operation 502, reached from type-`E` command 6, enters the
larger adjustment command session at `0xc01c7bcc`.

### Live SXC control plane and `CamDiagOp` correction

The retail PTP endpoint exposes a second, script-object control plane. Sending
`HDISCVRY.SXC` makes the camera publish `DDISCVRY.SXC`; sending
`HREQUEST.SXC` with Sanyo SXC XML publishes `DRSPONSE.SXC` through a PTP
`RequestObjectTransfer` event. Read-only `Get` requests have returned live
camera values, so this is a real factory registry rather than dead firmware
data.

Static tracing after the first live `Execute CamDiagOp` request corrects the
earlier interpretation of operation 502:

- the SXC executor queues operation 502 and one byte of caller input;
- the operation-502 worker stores `input + 1` at `0xc0d74a98`, reads it back,
  and subtracts one;
- the recovered input is passed to `0xc01c7bcc` as a calibration-routine
  selector;
- selector zero calls `0xc01c7904`, which reconfigures imaging hardware,
  includes a fixed 500 ms delay, and participates in the adjustment session.

Consequently, `CamDiagOp 0` is not a neutral probe and `CamDiagOp 328` is not
a route to adjustment opcode 328 (the SXC argument is only one byte anyway).
The raw-mode operation belongs to a nested adjustment command transport that
still has to be recovered.

The next static trace recovered most of that nesting:

- `CamDiagOp` selector **22** maps to `0xc01c80f4`, which calls the
  0--899 adjustment dispatcher at `0xc01c961c`;
- `CamDiagLong` is registry ID 32, and its setter at `0xc01c0c48` writes
  directly to the command word at `0xc0d91240`;
- the dispatcher extracts the low ten bits of that word before selecting an
  adjustment opcode;
- the per-command numeric argument is separately loaded from the adjustment
  global at `0xc0eefce8`.

A reset-camera read-only audit returned `CamDiagLong = 0`. It also confirmed
that self-calibration mode, readiness, adjustment status, and inspection
status were unchanged from before the selector-zero request. Thus selector
zero did not persistently change those controls.

The SXC `Set` grammar was then recovered and verified live. Scalar values use
a nested `<value>` element; direct element text is the `Execute` grammar.
With the nested form, setting `CamDiagLong` to its current value round-trips
exactly. The earlier `0x0fffffff` readback was the parser's missing-value
sentinel, not an authorization failure.

The apparent missing argument channel is also now resolved. Calibration
selector **20** copies the current `CamDiagLong` command word at `0xc0d91240`
into the adjustment argument slot at `0xc0eefce8`. Selector **22** then loads
that staged value and calls the 0--899 adjustment dispatcher, whose opcode is
still taken from the current `CamDiagLong`. The transaction shape is
therefore:

1. set `CamDiagLong` to the numeric argument;
2. execute `CamDiagOp 0x14` (decimal selector 20) to stage it;
3. set `CamDiagLong` to the adjustment opcode;
4. execute `CamDiagOp 0x16` (decimal selector 22) to dispatch it;
5. restore `CamDiagLong` to zero after the experiment.

A live zero-value staging probe was accepted by PTP. Its
`DRSPONSE.SXC` is not readable in the submitting PTP session even after a
multi-second delay, but is immediately readable after a clean close/reopen.
The deferred response reports SXC result `05` with an empty `CamDiagOp`
element. The client therefore closes cleanly after queuing `Execute`; the
next SXC invocation harvests that response before sending its own request.

Selector 22 has now been exercised once with the least invasive dispatcher
case: staged argument zero and adjustment opcode 399. Static disassembly shows
that pair only clears the runtime `RAW_DATA_MODE` flag and mode fields. The
SXC upload and PTP close both completed normally; this selector did not
publish a nonempty deferred response object. `CamDiagLong` was restored to
zero, `CamAdjustStatus` remained `9E033721`, the ordinary USB PTP personality
remained present, and the Apple camera agents restarted normally. This proves
the host-to-dispatcher route without yet enabling raw mode.

Host memory also remained stable through these probes: swap-outs did not
increase, swap use fell slightly, compressor occupancy stayed essentially
flat, and no probe process remained resident.

Some `CamDiagInfo*` readbacks contain unique device identifiers and should not
be retained in ordinary probe logs.

The Mac running the first live selector-zero request subsequently suffered a
system watchdog panic. The panic report says `watchdogd` missed check-ins for
90 seconds while the VM compressor was at its compressed-page limit with 41
swapfiles. Its kernel backtrace contains Apple watchdog and interrupt
controller code, not USB, PTP, libusb, or the test client. The temporal
association is therefore recorded but causality is unproved. SXC Execute is
now protected by a separate `--allow-sxc-execute` command-line interlock;
read-only SXC discovery and `Get` remain available without it.

A second runtime control is even more explicit. Adjustment operation **399**
(compressed-table key `0x95`) calls `0xc02d65e8` and reports
`RAW_DATA_MODE`. Arguments zero, one, and two select:

- `(flag, mode) = (0, 0)`
- `(flag, mode) = (0x1234, 0x456)`
- `(flag, mode) = (0x1234, 0)`

respectively. These are written to runtime configuration fields
`c0d6ee74 + 0x1854` and `c0d6ee74 + 0x6a`; this handler also contains no
persistent-save call.

The direct consumer of operation 399's marker is now identified. Adjustment
opcode **21**, reached through the direct 0--29 dispatcher table, calls
`0xc02d3888`. Argument zero selects the primary ten-sample sensor acquisition
at `0xc02d030c`; arguments two and three select separate twelve-step
AGC-related sweeps. The primary path calls `0xc02cff2c`, whose checks at
`0xc02d000c` and `0xc02d02b4` consume operation 399's flag and mode. It
accumulates and copies 68-byte acquisition/evidence structures and reports
AGC and white-balance results. No persistent calibration-save call or PTP
object producer is present in this path.

`run_xa5_sxc_adjust.sh` accepts multiple opcode/argument pairs so gates and
their consumers run under one Apple-agent handoff with no intervening PTP
class reset. The live chain `(399, 1) -> (21, 0)` completed, restored
`CamDiagLong`, left ordinary `04cb:02d5` PTP enumerated, created no additional
PTP object, and did not change `CamAdjustStatus`. Host swap-outs remained flat
during the command and both Apple camera agents restarted. This proves
`RAW_DATA_MODE` changes the internal sensor-analysis acquisition, but does not
itself select a raw PTP or HDMI transport.

The only direct consumer so far found for operation 328's `+0x9d8` marker
gates a defect-acquisition/raw-analysis routine at `0xc0340094`. This means
`RAWMODE` is proven to alter the adjustment acquisition pipeline, but is not
yet proof of a continuous Bayer stream. The remaining transport problem is to
find a producer that exports one of these internal buffers.

### USB-to-HDMI transition

Fujifilm's X-A5 owner's manual explicitly states that USB cannot be used while
HDMI is connected. Its documented sequence is camera off, connect HDMI,
select the receiver input, then turn the camera on. This explains the invalid
green/no-lock frames observed while retail PTP was active.

A no-reboot transition is nevertheless possible. The atomic chain
`(328, 1) -> (399, 1)` armed both volatile raw-analysis markers; the direct
client then issued hidden standard operation `0x101c InitiateOpenCapture`
without a PTP reset and held transaction 2. Physically removing Fuji USB made
the camera resume valid HDMI while remaining powered. OBS received a stable
1280x960, 30 fps, Rec.709 NV12 stream.

That output still contained normal color/gamma and Fuji display overlays. It
was therefore demosaiced and composited before the HDMI encoder, not Bayer or
full sensor bit depth. This establishes the mode bridge but not raw HDMI.
Hidden `0x9022 GetCapturePreview` was then issued inside the same armed chain.
The camera returned a valid type-2 data container and `0x2001`, but the
payload remained exactly zero bytes. The two raw-analysis markers are
therefore insufficient to enable the retail PTP preview producer.

The SXC registry contains the explicit factory operation
`CamTakePreviewOp` (ID `0x1f7`) adjacent to `CamTakePicOp`. A coordinated
client invoked `CamTakePreviewOp`, entered open capture, and requested
`0x9022` before closing the same PTP session. The payload was still zero.
An all-format object inventory after the operation contained only the empty
`DCIM` association; no JPEG, RAF, vendor-format, or hidden image object had
been published.

The adjustment dispatcher also exposes `SIGPRO_IMAGE_DUMP_MODE`.
Conservative dispatcher emulation maps opcodes 601 and 857 to the same case
at `c01ca2f4`; argument one only sets volatile byte `c0d6ed39`, and the
handler has no persistent-save call. A live chain armed opcodes 328, 399, and
601 before the in-session factory preview operation. It again produced
neither a `0x9022` payload nor a new PTP object. Thus the named RAW, RAW_DATA,
and SIGPRO dump flags do not themselves construct an exporter in the retail
USB personality.

The MacroSilicon `534d:2109` decoder's UVC VideoControl descriptor contains an
input terminal, processing unit, and output terminal but no vendor extension
unit. It exposes no host-readable HDMI lock or input-timing control. The
hub's separate Silicon Motion USB display function is an output, not an HDMI
capture input.

### Raw-output geometry, not a frame-buffer pointer

The signal-processor graph dump at `0xc03a7e40` labels the four-register block
`0x301128c0..0x301128cc` as `SIG0 RAWOUT`. Its helper `0xc03aa7ac` only reads
and prints MMIO words; these addresses are registers, not image memory.

The signal-register descriptor builder at `0xc03e86fc` independently maps
those four registers into the SIG0 shadow beginning at `0x302104d4`. The
writer at `0xc03a43d0` establishes the layout:

- register zero contains two 14-bit fields in its low and high halfwords;
- register one contains another two 14-bit fields;
- register two contains two 13-bit fields;
- register three receives a one-bit enable flag.

Callers construct these values from image extents and crop calculations, so
`RAWOUT` is a geometry/control stage in the processing graph. It provides a
way to recover the raw plane's live extent, but it is not the endpoint that
holds its pixels. `decode_rawout_regs.py` reverses the proven bit layout while
deliberately leaving the six geometry fields unnamed until their exact
coordinate semantics are established.

### SRAW capture task

`SRAW0000.XXX` is not presently evidence of a file named `SRAW0000.XXX`.
The structure at `0xc05aef30` is passed to the RTOS object creator at
`0xc0002160`: it contains the entry point `0xc00ba2bc`, priority `0x12`, and
stack size `0x1000`. It is therefore the name/configuration of an internal
SRAW task.

That task is started by the normal still-capture sequence at `0xc01384f4`
and stopped at `0xc01388d4`. Its main acquisition path begins at
`0xc00ba4b8` and participates in the same buffer/acquisition calls used by
the ordinary shooting pipeline.

The path contains a concrete image-memory pool. `c0069538` selects these
three bases:

- `0x00f35400`
- `0x0132a400`
- `0x0171f400`

Adjacent bases are exactly `0x3f5000` bytes apart. The pool initializer at
`0xc006958c` registers `0x3f5000` as the normal allocation size. This is
`1920 * 1080 * 2 + 0x800`, and the nearby runtime getters at `0xc003f8a4`
and `0xc003f8f0` explicitly return width 1920 and height 1080 in the default
mode. The layout is therefore strongly consistent with a triple-buffered
1080-class, two-byte-per-sample image surface plus a small guard/header
allowance; the arithmetic alone does not prove its pixel format.

`c008b258` supplies the source passed by SRAW to the generic image operation
at `c0077a34`: it returns `0x05538c00` in one capture state and otherwise
selects the third pool buffer. `c0077a34` builds a 65-word image descriptor
and submits it to the processing engine at `c0406e70`, with the selected
source, a destination inside the triple-buffer pool, dimensions, offsets,
and format controls.

This is the strongest pixel-endpoint lead so far. The remaining proof is to
decode the format controls supplied to `c0077a34` and establish whether the
conditional `0x05538c00` source is Bayer, corrected raw, or an already
processed two-byte surface. No runtime packet is needed for that analysis.

The retail PTP implementation also contains a virtual binary-object scaffold.
`c02039c0` constructs ObjectInfo for `SXC_VIRT.BIN`, and `c0203a60` is its
GetObject-side transfer handler. In the present retail path the inline
size/source getters resolve to zero, matching the observed empty object
behavior. It is therefore not an existing arbitrary-memory reader, but it is
a focused patch surface for exposing a completed SRAW pool buffer without
adding a new USB interface or endpoint.

Such a binding would be a polled pseudo-stream rather than an isochronous
stream. Each `0x3f5000`-byte surface is about 4.15 MB, so 30 fps would require
about 124.5 MB/s before PTP overhead—well beyond USB 2.0. Several frames per
second may be practical if the camera-side handler selects the previous
completed triple buffer and validates its generation before and after the
bulk transfer.

The second boot gate is set by the normal boot event flow after its startup
queue drains; event flag `0x40` is an RTOS queue notification, not evidence of
a physical button bit.

## Ordinary-USB hidden capture dispatcher

The retail `04cb:02d5` PTP personality contains callable operations beyond the
25 codes returned by `GetDeviceInfo`. The standard dispatcher at `c0200944`
routes the unadvertised pair:

- `0x101c InitiateOpenCapture` to `c0201798`;
- `0x1018 TerminateOpenCapture` to `c0201810`.

Live invocation of `0x101c` with storage and format parameters both zero
returns `0x2001` and visibly changes the camera from its white USB-mode screen
to an acquisition-like display. The handler stores the initiating PTP
transaction id and `0x1018` requires that exact id; a mismatched id returns
`0x201d`.

The actual start wrapper at `c0207730` checks the runtime mode word at
`c0d96ff8`. Only value `13` invokes the registered acquisition callback;
other values return success without calling it. The observed display change
therefore supplies runtime evidence that the ordinary USB state reached the
active branch during the tested session.

The adjacent unadvertised Fuji dispatcher contains:

- `0x9020` initiate movie capture;
- `0x9021` terminate movie capture;
- `0x9022` get capture preview;
- `0x9023..0x9025` zoom controls;
- `0x9026/0x9027` focus-point or S1 lock/unlock controls;
- `0x902b` remote device information;
- `0x902c..0x902e` shutter, aperture, and exposure-compensation controls;
- `0x9030` cancel initiate capture.

These are real handlers, not merely symbols. Several return success in the
retail USB personality, while `0x902b` reaches its handler and returns a state
error rather than `OperationNotSupported`. Hidden properties used by newer
Fuji remote sessions remain unavailable through the retail property
dispatcher.

`0x9022` reaches `c020da08`, which calls the registered preview getter
`c01c5bac -> c00e4274 -> c019b4e8`. A direct libusb client claimed the sole
PTP interface, issued the Still Image class reset, opened a fresh PTP session,
entered open capture, and requested `0x9022`. The camera itself returned:

1. a type-2 PTP data container of exactly 12 bytes, with zero payload;
2. a normal `0x2001` response container.

ImageCaptureCore had returned the same zero-length data phase. This proves the
host framework did not discard a JPEG in the tested state: the camera-side
preview producer was empty.

Static inspection explains the separation. `c019b4e8` reads a dedicated
preview-cache descriptor around `c0d8ed94` and copies from the selected cache
region. It does not directly select the SRAW source at `0x05538c00` or the
three acquisition-pool bases. The missing operation is therefore the producer
that fills or publishes that cache, not a custom host PTP decoder.

The raw descriptor has one `06/01/01` interface with bulk OUT `0x01`, bulk IN
`0x82`, and interrupt IN `0x83`. There is no alternate setting, UVC interface,
or isochronous video pipe. Fuji's separate network-remote transport delivers
live JPEG blobs on TCP port 55742 after open capture and events on 55741; the
USB personality exposes no network interface. A custom PTP driver is therefore
useful for hidden control and exact transport diagnosis, but cannot by itself
turn the present USB descriptors into a live-video endpoint.

The second vendor dispatcher begins at `c020dc7c`. Its reverse lookup table at
`c05f28a8` maps `0x9020` to handler `c020d930`, which calls the registered
movie-start callback at `c00f3408`. The exact `0x2019` path is now recovered:

1. `c019e758(2, 1)` reports that the ordinary acquisition object is not
   already running;
2. `c00f3408` checks the two movie-state words at `c0eec5e4` and
   `c0eec604`;
3. either word being nonzero returns internal status `-2`;
4. translator `c020d110` maps `-2` to PTP `0x2019 DeviceBusy`.

The separate local movie-start routine at `c00f3ad4` sets the first word,
whereas the PTP movie-start callback sets the second. `0x9021` only invokes
its termination callback when the saved PTP initiating transaction at
`c0ef23c8` is nonzero and matches the supplied id. A cleanup attempt with id
two returned `0x2001` but did not change the subsequent `0x2019`, consistent
with no saved PTP transaction to terminate. Hidden `0x9030` likewise returned
`0x2001` but did not clear the movie-state predicate.

The analytical HDMI harness confirmed all of these paths against the exact
AVFoundation device name. The dongle's no-signal fallback is precisely
`Y=7, U=128, V=128`, with zero luma range and zero frame difference. All
plain, open-capture, cancel-then-movie, cross-session debug, and atomic
same-session debug chains remained on that frame. In no-signal state the
receiver actually timestamps NV12 1280x960 frames at about 8 fps despite a
30-fps FFmpeg request; the harness therefore computes effective cadence from
PTS rather than reporting the requested value.

A clean-boot run resolved that experiment. Plain `0x9020` returned `0x2001`
for transaction two and remained held while the HDMI decoder captured 16
frames. All frames were still the exact no-signal fallback (`Y=7, U=128,
V=128`, zero range and difference, about 8.0046 fps). The matching `0x9021`
termination returned `0x2001`, but subsequent starts remained at `0x2019`.
Thus movie start itself sets the persistent predicate, yet successful USB
movie state does not route the producer to HDMI.

Plain `0x101c` open capture independently returned `0x2001` and produced the
same fallback. The full nine-key `CamDebugUsbMode -> 0x101c` sweep then
returned `0x2001` for open capture in every recovered mode/submode, again with
bit-identical fallback frames.

Three cross-model routing properties were also tested through their standard
PTP setters. `VideoOutOnOff` (`0xd168`) rejected descriptor, value, and u16
setter operations with `0x200a`; `SetUSBMode` (`0xd15d`) and `ForceMode`
(`0xd230`) rejected their u16 setters with `0x200a`. Repeating those probes
after selecting the distinct `CamDebugUsbMode` key `A` (mode 6) did not unlock
them. The retail/debug-mode command surface therefore has no observed route
that permits simultaneous PTP ownership and HDMI output.

## Safety boundary

The builder is deliberately constrained to dispatcher pair `(326, 1100)` and
raises on all other pairs, including adjustment write event 1101. The first
camera-side test should be limited to generating `CAM256KB.ADJ`; do not accept
any firmware-update or adjustment-write prompt.
