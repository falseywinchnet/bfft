"""Restore checksum-addressed evidence without overwriting a divergent file.

Only the public recovery manifest is read. The private scene snapshot is a
separate local archive and is deliberately outside this restoration path.
"""
import argparse
import hashlib
import json
from pathlib import Path
import shutil


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--evidence-root', type=Path, default=Path(
        '/Users/ultimussecundai/Documents/CodexRecovery/bfft-20261003/evidence'))
    parser.add_argument('--destination', type=Path, required=True,
                        help='Separate directory under which original relative paths are reconstructed')
    parser.add_argument('--only', default='', help='Optional source-relative path prefix')
    parser.add_argument('--verify-only', action='store_true')
    args = parser.parse_args()
    rows = json.loads(Path(__file__).with_name('manifest.json').read_text())
    count = 0
    for row in rows:
        if 'external-evidence' not in row['disposition'] or not row['source_path'].startswith(args.only):
            continue
        relative = Path(row['source_path'])
        if relative.is_absolute() or '..' in relative.parts:
            raise ValueError('Unsafe relative destination in manifest')
        sha = row['sha256']
        source = args.evidence_root / sha[:2] / sha
        if source.stat().st_size != row['bytes'] or digest(source) != sha:
            raise ValueError('Evidence checksum mismatch: ' + str(relative))
        target = args.destination / relative
        if not args.verify_only:
            if target.exists():
                if digest(target) != sha:
                    raise FileExistsError('Refusing to replace divergent evidence: ' + str(target))
            else:
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source, target)
                if digest(target) != sha:
                    raise ValueError('Restored checksum mismatch: ' + str(target))
        count += 1
    print(json.dumps({'verified_records': count, 'restored': not args.verify_only,
                      'destination': str(args.destination)}, indent=2))


if __name__ == '__main__':
    main()
