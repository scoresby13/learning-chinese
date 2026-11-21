#!/usr/bin/env python3
"""
Safe, measured sync fixes for committed transcripts (public/data/*_track*.json).

What it changes, and why (see scripts/check_sync.py for the measurements):
  1. English timings move EN_OFFSET seconds later. Across the course, Whisper's English
     word timestamps lead the voice by a near-constant ~0.36 s (IQR 0.34–0.38 s,
     200 of 201 tracks within 0.25 s of the median). Tracks whose own English lag is
     far from that are left untouched and reported, because they need real alignment.
  2. Segments that start after the audio has ended are removed (Whisper hallucinations
     such as "Awesome." 30 s past the end), and anything running past the end is clamped.
  3. Segments stay in order: a segment never starts before the previous one ends.

Chinese timings are NOT shifted: they don't share the English offset (they were spread
evenly and partly pulled into preceding silence), so they need forced alignment instead.

Usage:
  python3 scripts/fix_sync.py --all            # rewrite every track in place
  python3 scripts/fix_sync.py --all --dry-run  # report only
"""
import argparse
import json
import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import check_sync  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
EN_OFFSET = 0.36          # seconds; keep in sync with scripts/process_cd.py
OUTLIER_TOLERANCE = 0.25  # skip the English shift if this track's own lag is further from EN_OFFSET
MIN_WORD = 0.05


def audio_duration(path: Path) -> float:
    return float(subprocess.check_output(
        ['ffprobe', '-v', 'error', '-show_entries', 'format=duration', '-of', 'csv=p=0', str(path)]).decode())


def _r(x):
    return round(x, 2)



def fit_words(seg):
    """Keep word timings inside their segment without overlaps.
    If the words no longer fit (the segment was trimmed), map them proportionally
    into the segment, which preserves their order and relative spacing."""
    words = seg.get('words') or []
    if not words:
        return
    ws, we = words[0]['start'], max(w['end'] for w in words)
    if ws < seg['start'] or we > seg['end']:
        span = max(we - ws, 1e-6)
        scale = (seg['end'] - seg['start']) / span
        for w in words:
            w['start'] = round(seg['start'] + (w['start'] - ws) * scale, 2)
            w['end'] = round(seg['start'] + (w['end'] - ws) * scale, 2)
    prev = seg['start']
    for w in words:  # tidy rounding: monotonic, non-empty, inside the segment
        w['start'] = round(min(max(w['start'], prev), seg['end']), 2)
        w['end'] = round(min(max(w['end'], w['start'] + 0.01), seg['end']), 2)
        prev = w['end']


def fix_track(json_path: str, dry_run: bool = False) -> dict:
    p = Path(json_path)
    d = json.loads(p.read_text(encoding='utf-8'))
    dur = audio_duration(ROOT / 'public' / d['audioUrl'])
    report = {'track': p.name, 'shifted_en': False, 'dropped': 0, 'clamped': 0, 'reordered': 0, 'note': ''}

    # Measure this track's English lag before deciding to shift
    speech = check_sync.speech_mask(ROOT / 'public' / d['audioUrl'])
    en_mask = check_sync.transcript_mask(d['segments'], len(speech), 'en')
    lag_en = check_sync.best_lag(speech, en_mask) if en_mask.any() else EN_OFFSET
    shift = abs(lag_en - EN_OFFSET) <= OUTLIER_TOLERANCE
    report['lag_en'] = round(lag_en, 2)
    if not shift:
        report['note'] = f'English lag {lag_en:+.2f}s is unusual — not shifted, needs alignment'

    segs = []
    for s in d['segments']:
        if s['start'] >= dur - 0.05:
            report['dropped'] += 1          # starts after the audio has ended: hallucinated
            continue
        if shift and s.get('language') != 'zh':
            s['start'] = _r(s['start'] + EN_OFFSET)
            s['end'] = _r(s['end'] + EN_OFFSET)
            for w in s.get('words', []):
                w['start'] = _r(w['start'] + EN_OFFSET)
                w['end'] = _r(w['end'] + EN_OFFSET)
            report['shifted_en'] = True
        segs.append(s)

    prev_end = 0.0
    for s in segs:
        if s['start'] < prev_end:
            report['reordered'] += 1
            s['start'] = _r(prev_end)
        if s['end'] > dur:
            report['clamped'] += 1
            s['end'] = _r(dur)
        s['end'] = _r(max(s['end'], s['start'] + 0.1))
        fit_words(s)
        prev_end = s['end']

    # Renumber ids after dropping
    for i, s in enumerate(segs):
        s['id'] = f'seg-{i + 1:03d}'
    d['segments'] = segs

    if not dry_run:
        p.write_text(json.dumps(d, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    return report


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('files', nargs='*')
    ap.add_argument('--all', action='store_true')
    ap.add_argument('--dry-run', action='store_true')
    args = ap.parse_args()
    files = sorted(str(p) for p in (ROOT / 'public' / 'data').glob('*_track*.json')) if args.all else args.files
    if not files:
        ap.error('give transcript files or --all')
    with ProcessPoolExecutor() as ex:
        reports = list(ex.map(fix_track, files, [args.dry_run] * len(files)))
    shifted = sum(r['shifted_en'] for r in reports)
    print(f"{len(reports)} tracks: English shifted +{EN_OFFSET}s in {shifted}; "
          f"{sum(r['dropped'] for r in reports)} hallucinated tail segments removed; "
          f"{sum(r['clamped'] for r in reports)} clamped to the audio end; "
          f"{sum(r['reordered'] for r in reports)} starts moved to keep segments in order")
    for r in reports:
        if r['note'] or r['dropped']:
            print(f"  {r['track']}: {r['note'] or ''}{' ' if r['note'] and r['dropped'] else ''}{'removed ' + str(r['dropped']) + ' tail segment(s)' if r['dropped'] else ''}")


if __name__ == '__main__':
    main()
