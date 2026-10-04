#!/bin/zsh
set -euo pipefail
cd "${0:A:h}"
app="${1:-$HOME/Applications/TV Normalizer.app}"
mkdir -p "$app/Contents/MacOS"
clang++ -std=c++17 -O2 -fobjc-arc main.mm -framework Cocoa -framework CoreAudio -framework AudioUnit -framework AVFoundation -o "$app/Contents/MacOS/TVNormalizer"
cat > "$app/Contents/Info.plist" <<'PLIST'
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0"><dict>
<key>CFBundleIdentifier</key><string>local.personal.TVNormalizer</string>
<key>CFBundleName</key><string>TV Normalizer</string>
<key>CFBundleDisplayName</key><string>TV Normalizer</string>
<key>CFBundleExecutable</key><string>TVNormalizer</string>
<key>CFBundlePackageType</key><string>APPL</string>
<key>CFBundleShortVersionString</key><string>1.0</string>
<key>CFBundleVersion</key><string>1</string>
<key>LSUIElement</key><true/>
<key>NSMicrophoneUsageDescription</key><string>TV Normalizer reads system audio from the VB-Cable virtual audio device, boosts it with GMax, and plays it through your speakers. It does not record audio.</string>
<key>NSHighResolutionCapable</key><true/>
</dict></plist>
PLIST
codesign --force --sign - --identifier local.personal.TVNormalizer "$app"
printf 'Built %s\n' "$app"
