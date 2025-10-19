from pathlib import Path
from collections import Counter
import os

# project imports
from evrepo.utils import read_subreddit_map
from evrepo.normalize import normalize
from evrepo.filters import CandidateFilter
from tools.PushshiftDumps.personal.utils import read_obj_zst  # Watchful1

def norm(s):
    s = (s or '').strip().lower()
    if s.startswith('r/'): s = s[2:]
    if s.startswith('/r/'): s = s[3:]
    return s

# config paths (use the same CWD you use to run the CLI)
ideology_map = r'config/subreddits.yaml'
keywords     = r'config/keywords.yaml'
neg_filters  = r'config/neg_filters.yaml'

# input file to sample (comments Jan 2025)
zst = r'C:\Users\awand\Downloads\reddit_pushshift_dump_2025\comments\RC_2025-01.zst'

ideology_by_sub = read_subreddit_map(ideology_map)
yaml_whitelist  = {norm(s) for s in ideology_by_sub.keys()}

# build EV candidate filter
from evrepo.utils import load_yaml
kw = load_yaml(keywords)
nf = load_yaml(neg_filters)
cand = CandidateFilter.from_config(kw, nf)

seen = kept = skipped_nonwhite = skipped_nontext = skipped_kw = 0
offenders = Counter()
examples  = []

for i, obj in enumerate(read_obj_zst(zst)):
    if i >= 200000: break  # small sample
    seen += 1
    rec = normalize(obj, ideology_by_sub)

    sub_raw = rec.get('subreddit') or obj.get('subreddit') or ''
    sub = norm(sub_raw)

    if sub not in yaml_whitelist:
        skipped_nonwhite += 1
        if len(offenders) < 40:
            offenders[sub] += 1
        continue

    text = rec.get('text') or ''
    if not text:
        skipped_nontext += 1
        continue

    if not cand.is_candidate(text):
        skipped_kw += 1
        continue

    kept += 1
    if len(examples) < 8:
        examples.append((sub, rec.get('is_submission'), text[:120].replace('\n',' ')))

print("Sampled:", seen)
print("Kept (white+EV):", kept)
print("Skipped non-whitelisted:", skipped_nonwhite)
print("Skipped empty text:", skipped_nontext)
print("Skipped by keywords:", skipped_kw)
print("Top unexpected subs:", offenders.most_common(10))
print("Examples:", examples)
