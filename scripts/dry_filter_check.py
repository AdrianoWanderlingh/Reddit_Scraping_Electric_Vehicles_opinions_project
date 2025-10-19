from __future__ import annotations
import sys
from pathlib import Path
from collections import Counter

# ---- match your ingester's path setup ----
ROOT = Path(__file__).resolve().parents[1]  # repo root (this file will live under scripts/)
SRC_ROOT = ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

PUSHSHIFT_PERSONAL = ROOT / "tools" / "PushshiftDumps" / "personal"
if str(PUSHSHIFT_PERSONAL) not in sys.path:
    sys.path.insert(0, str(PUSHSHIFT_PERSONAL))

# Watchful1 helper (expects personal/ on sys.path so it can import zst_blocks)
from utils import read_obj_zst  # type: ignore

# ---- project imports ----
from evrepo.utils import read_subreddit_map, load_yaml
from evrepo.normalize import normalize
from evrepo.filters import CandidateFilter

def norm(s: str) -> str:
    s = (s or "").strip().lower()
    if s.startswith("r/"): s = s[2:]
    if s.startswith("/r/"): s = s[3:]
    return s

# ---- config (absolute) ----
ideology_map = str((ROOT / "config" / "subreddits.yaml").resolve())
keywords     = str((ROOT / "config" / "keywords.yaml").resolve())
neg_filters  = str((ROOT / "config" / "neg_filters.yaml").resolve())

# monthly COMMENTS file to sample
zst = r"C:\Users\awand\Downloads\reddit_pushshift_dump_2025\comments\RC_2025-01.zst"

# ---- build whitelist & filters ----
ideology_by_sub = read_subreddit_map(ideology_map)
yaml_whitelist  = {norm(s) for s in ideology_by_sub.keys()}

kw_cfg = load_yaml(keywords)
nf_cfg = load_yaml(neg_filters)
cand = CandidateFilter.from_config(kw_cfg, nf_cfg)

LIMIT = 200_000  # sample size

seen = kept = 0
skipped_nonwhite = skipped_nontext = skipped_kw = 0
changed_sub = 0
offenders = Counter()
changed_examples = []

for i, obj in enumerate(read_obj_zst(zst)):
    if i >= LIMIT: break
    seen += 1

    rec = normalize(obj, ideology_by_sub)

    raw_sub   = norm(obj.get("subreddit") or "")
    norm_subr = norm(rec.get("subreddit") or raw_sub)

    if norm_subr != raw_sub:
        changed_sub += 1
        if len(changed_examples) < 8:
            changed_examples.append((raw_sub, norm_subr))

    if not norm_subr or norm_subr not in yaml_whitelist:
        skipped_nonwhite += 1
        if len(offenders) < 200:
            offenders[norm_subr or "(missing)"] += 1
        continue

    text = rec.get("text") or ""
    if not text:
        skipped_nontext += 1
        continue

    if not cand.is_candidate(text):
        skipped_kw += 1
        continue

    kept += 1

print("Config paths:")
print("  ideology_map:", ideology_map)
print("  keywords    :", keywords)
print("  neg_filters :", neg_filters)
print()
print(f"Sampled lines: {seen:,}")
print(f"Kept (whitelisted + keyword match): {kept:,}")
print(f"Skipped non-whitelisted: {skipped_nonwhite:,}")
print(f"Skipped empty/none text: {skipped_nontext:,}")
print(f"Skipped by EV keywords : {skipped_kw:,}")
print(f"Subreddit changed by normalize(): {changed_sub:,}")
print("Examples of raw->normalized subreddit changes (up to 8):")
for raw, new in changed_examples:
    print(f"  {raw} -> {new}")
print()
print("Top unexpected subs (not in YAML) in sample:")
for sub, n in offenders.most_common(15):
    print(f"  {sub:30s} {n}")
