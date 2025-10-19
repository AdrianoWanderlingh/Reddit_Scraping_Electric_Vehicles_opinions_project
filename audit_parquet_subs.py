from pathlib import Path
import os
from evrepo.utils import read_subreddit_map

def norm(s: str) -> str:
    s = (s or '').strip().lower()
    if s.startswith('r/'): s = s[2:]
    if s.startswith('/r/'): s = s[3:]
    return s

parquet_root = r'C:\Users\awand\Documents\evrepo\data\parquet'   # adjust if needed
year, month = 2025, 1

yaml_path = r'config/subreddits.yaml'                             # same file you checked
yaml_map = read_subreddit_map(yaml_path)
whitelist = {norm(k) for k in yaml_map.keys()}

base = Path(parquet_root) / f'year={year:04d}' / f'month={month:02d}'
on_disk = set()
if base.exists():
    for p in base.iterdir():
        if p.is_dir() and p.name.startswith('subreddit='):
            sub = p.name.split('=',1)[1]
            on_disk.add(norm(sub))

unexpected = sorted(on_disk - whitelist)
missing    = sorted(whitelist - on_disk)

print('Checked:', base)
print('Whitelist subs:', len(whitelist))
print('On-disk subs  :', len(on_disk))
print('Unexpected (not in YAML):', len(unexpected))
print('  ', unexpected[:50])
print('Missing (in YAML but not on disk):', len(missing))
print('  ', missing[:50])
