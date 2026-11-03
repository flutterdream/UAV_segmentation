"""Read-only verification after copying this complete folder. Python + Pillow."""
from pathlib import Path, PurePosixPath
import hashlib, io, json, sys, zipfile
from PIL import Image

CHECKS = Path(__file__).resolve().parent
ROOT = CHECKS.parents[1]

def sha256(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()

def main():
    manifest = json.loads((CHECKS/'MANIFEST_SHA256.json').read_text(encoding='utf-8'))
    expected = set()
    for i, record in enumerate(manifest['files'], 1):
        rel = PurePosixPath(record['path'])
        if rel.is_absolute() or '..' in rel.parts:
            raise RuntimeError('Unsafe manifest path')
        path = ROOT.joinpath(*rel.parts)
        if not path.is_file() or path.is_symlink() or not path.resolve().is_relative_to(ROOT):
            raise RuntimeError(f'Missing/unsafe file: {rel}')
        if path.stat().st_size != record['bytes'] or sha256(path) != record['sha256']:
            raise RuntimeError(f'Integrity mismatch: {rel}')
        expected.add(str(rel))
        if i % 1000 == 0:
            print(f'SHA-256 {i}/{len(manifest["files"])}', flush=True)
    expected.add((CHECKS/'MANIFEST_SHA256.json').relative_to(ROOT).as_posix())
    extras = sorted(str(p.relative_to(ROOT).as_posix()) for p in ROOT.rglob('*')
                    if p.is_file() and p.relative_to(ROOT).as_posix() not in expected)
    if extras:
        print(f'NOTE: {len(extras)} extra files are not part of the original archive: {extras[:10]}')
    names_list = json.loads((ROOT/'-算法代码'/'configs'/'TEST2_FILENAMES.json').read_text(encoding='utf-8'))
    if len(names_list)!=1300 or len(set(names_list))!=1300 or any('/' in n or '\\' in n or not n.endswith('.png') for n in names_list):
        raise RuntimeError('Invalid original test2 filename manifest')
    names = set(names_list)
    result = ROOT/'-任务结果文档'/'S2_stage47_i600_c1_alpha075_eight_probability.zip'
    with zipfile.ZipFile(result) as z:
        if len(z.infolist()) != 1300 or len(set(z.namelist())) != 1300 or set(z.namelist()) != names:
            raise RuntimeError('ZIP must contain exactly 1300 original-name root PNG files')
        if z.testzip() is not None:
            raise RuntimeError('ZIP CRC failure')
        for name in z.namelist():
            with Image.open(io.BytesIO(z.read(name))) as im:
                if im.mode != 'L' or im.size != (1024,1024) or im.format != 'PNG':
                    raise RuntimeError(f'PNG mode/size mismatch: {name}')
                lo, hi = im.getextrema()
                if lo < 1 or hi > 8:
                    raise RuntimeError(f'Class IDs out of range: {name}')
    print(json.dumps({'status':'passed', 'files_verified':len(manifest['files']),
                      'prediction_count':1300,
                      'checks':'file SHA-256 and prediction ZIP format/CRC'},
                     ensure_ascii=False, indent=2))

if __name__ == '__main__':
    main()
