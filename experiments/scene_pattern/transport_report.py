"""Assemble the private fusion reviewer from retained numerical receipts."""
import argparse
import hashlib
import json
import shutil
from pathlib import Path


def build(fusion, attachment, out):
    packet = json.loads((fusion / 'fusion.json').read_text())
    graph = json.loads((attachment / 'dense.json').read_text())
    if packet['records'] != graph['records']:
        raise ValueError('Fusion and attachment records differ')
    photo_contributors = sorted({p['capture'] for v in packet['views']
                               for p in v['participation']
                               if p['pixels_above_quarter_weight'] > 0
                               and not packet['records'][p['capture']]['panoramic']})
    packet['attachment'] = dict(
        anchored_captures=graph['anchored_captures'],
        unresolved_captures=graph['unresolved_captures'],
        component_sizes=[len(c) for c in graph['capture_components']],
        photo_contributors=photo_contributors,
        validated_directed_sites=graph['cycles']['validated_directed_sites'],
        rounds=graph['attachment_rounds'],
        receipt_sha256=hashlib.sha256((attachment / 'dense.json').read_bytes()).hexdigest())
    out.mkdir(parents=True, exist_ok=True)
    for v in packet['views']:
        for suffix in ('before.png', 'after.png', 'reference.png', 'support.png', 'overlay.jpg'):
            name = f"{v['anchor']:03d}-{suffix}"
            shutil.copyfile(fusion / name, out / name)
    (out / 'fusion.json').write_text(json.dumps(packet, indent=2))
    shutil.copyfile(Path(__file__).with_name('transport-viewer.html'), out / 'index.html')
    return packet


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--fusion', type=Path, required=True)
    p.add_argument('--attachment', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    args = p.parse_args()
    result = build(args.fusion, args.attachment, args.out)
    print(json.dumps(dict(views=len(result['views']), attachment=result['attachment'])))


if __name__ == '__main__':
    main()
