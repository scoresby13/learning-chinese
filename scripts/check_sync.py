#!/usr/bin/env python3
"""
Audio ⇄ transcript sync checker for Learning Chinese.

Measures how well each transcript's timestamps match where speech actually is in
its MP3 — no speech recognition needed, just the audio's loudness over time:

  1. Decode the track with ffmpeg (16 kHz mono) and compute loudness per 20 ms frame.
  2. Mark frames as speech/silence with a per-track adaptive threshold, then close
     short gaps (speech has brief dips between syllables) and drop tiny blips.
  3. Compare against the transcript:
       • words_in_silence  – share of timed words that sit mostly (≥70 %) in silence
       • lag               – the time shift that best lines up transcript speech with
                             audio speech (a constant lag = the whole track is early/late)
       • drift             – how much that best lag changes between the start and end
                             of the track (a growing lag = timings slide out of sync)
       • onset_error_med   – median gap between a line's start and the nearest real
                             speech onset
       • overrun           – how far the last line ends after the audio ends

Usage:
  python3 scripts/check_sync.py public/data/foundation_cd01_track01.json [...]
  python3 scripts/check_sync.py --all            # every track, sorted worst first
  python3 scripts/check_sync.py --all --json out.json

Exit code is 1 if any checked track is rated "bad", so it can gate CI.
"""
import argparse
import json
import os
import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
FRAME = 0.02            # seconds per analysis frame
SR = 16000


def speech_mask(audio_path: Path) -> np.ndarray:
    pcm = subprocess.run(
        ['ffmpeg', '-v', 'error', '-i', str(audio_path), '-ac', '1', '-ar', str(SR), '-f', 's16le', '-'],
        check=True, capture_output=True).stdout
    x = np.frombuffer(pcm, dtype=np.int16).astype(np.float32) / 32768.0
    n = int(len(x) / (SR * FRAME))
    frames = x[: n * int(SR * FRAME)].reshape(n, -1)
    db = 10 * np.log10(np.mean(frames ** 2, axis=1) + 1e-10)
    # Adaptive threshold between the noise floor and typical speech level
    floor, loud = np.percentile(db, 10), np.percentile(db, 90)
    speech = db > floor + 0.35 * (loud - floor)
    # Close gaps < 200 ms (between syllables), then remove blips < 80 ms
    speech = _close(speech, int(0.2 / FRAME))
    speech = ~_close(~speech, int(0.08 / FRAME))
    return speech


def _close(mask: np.ndarray, max_gap: int) -> np.ndarray:
    """Fill runs of False shorter than max_gap that sit between True runs."""
    m = mask.copy()
    idx = np.flatnonzero(np.diff(np.concatenate(([1], m.astype(np.int8), [1]))))
    # idx pairs mark starts/ends of False runs
    for a, b in zip(idx[0::2], idx[1::2]):
        if 0 < a and b < len(m) and (b - a) < max_gap:
            m[a:b] = True
    return m


def transcript_mask(segments, n_frames: int, lang=None) -> np.ndarray:
    m = np.zeros(n_frames, dtype=bool)
    for s in segments:
        if lang and (s.get('language') == 'zh') != (lang == 'zh'):
            continue
        spans = [(w['start'], w['end']) for w in s.get('words', [])] or [(s['start'], s['end'])]
        for a, b in spans:
            m[max(0, int(a / FRAME)): min(n_frames, int(b / FRAME) + 1)] = True
    return m


def best_lag(audio: np.ndarray, text: np.ndarray, max_lag_s=3.0) -> float:
    """Shift (s) to apply to the transcript that maximises overlap with audio speech. + = transcript is early."""
    L = int(max_lag_s / FRAME)
    a = audio.astype(np.float32) - audio.mean()
    t = text.astype(np.float32) - text.mean()
    best, best_k = -1e18, 0
    for k in range(-L, L + 1):
        if k >= 0:
            v = float(np.dot(a[k:], t[: len(t) - k]))
        else:
            v = float(np.dot(a[: len(a) + k], t[-k:]))
        if v > best:
            best, best_k = v, k
    return best_k * FRAME


def onsets(mask: np.ndarray) -> np.ndarray:
    return np.flatnonzero(np.diff(mask.astype(np.int8)) == 1) * FRAME + FRAME


def check_track(json_path: str) -> dict:
    d = json.load(open(json_path, encoding='utf-8'))
    audio_path = ROOT / 'public' / d['audioUrl']
    speech = speech_mask(audio_path)
    n = len(speech)
    dur = n * FRAME
    segs = d['segments']
    text = transcript_mask(segs, n)

    # Words that sit mostly in silence (only words long enough to judge), per language
    judged = {'en': 0, 'zh': 0}
    silent = {'en': 0, 'zh': 0}
    for s in segs:
        lang = 'zh' if s.get('language') == 'zh' else 'en'
        for w in s.get('words', []):
            a, b = int(w['start'] / FRAME), min(int(w['end'] / FRAME), n)
            if b - a < 5:
                continue
            judged[lang] += 1
            if speech[a:b].mean() < 0.3:
                silent[lang] += 1
    words_silent_by = {k: silent[k] / judged[k] if judged[k] else 0.0 for k in judged}
    lag_by = {k: best_lag(speech, transcript_mask(segs, n, k)) if judged[k] else 0.0 for k in judged}

    lag = best_lag(speech, text)
    # Drift: compare best lag in the first and last thirds of the track
    third = n // 3
    lag_start = best_lag(speech[:third], text[:third]) if third > 250 else lag
    lag_end = best_lag(speech[-third:], text[-third:]) if third > 250 else lag

    ons = onsets(speech)
    errs = []
    for s in segs:
        if len(ons):
            j = np.searchsorted(ons, s['start'])
            cands = [ons[k] for k in (j - 1, j) if 0 <= k < len(ons)]
            errs.append(min((c - s['start'] for c in cands), key=abs))
    onset_med = float(np.median(np.abs(errs))) if errs else 0.0
    last_end = max((s['end'] for s in segs), default=0.0)

    overlap = float((speech & text).sum())
    recall = overlap / max(1, speech.sum())       # how much real speech has text on it
    precision = overlap / max(1, text.sum())      # how much highlighted time is real speech

    words_silent = sum(silent.values()) / max(1, sum(judged.values()))
    bad_reasons = []
    if abs(lag) >= 0.5:
        bad_reasons.append(f'whole track shifted {lag:+.2f}s')
    if abs(lag_end - lag_start) >= 0.75:
        bad_reasons.append(f'drifts {lag_start:+.2f}s→{lag_end:+.2f}s')
    if words_silent > 0.25:
        bad_reasons.append(f'{words_silent:.0%} of words in silence')
    if last_end - dur > 0.5:
        bad_reasons.append(f'runs {last_end - dur:.1f}s past the audio')
    warn = words_silent > 0.12 or onset_med > 0.35 or precision < 0.7

    return {
        'track': os.path.basename(json_path),
        'duration': round(dur, 1),
        'segments': len(segs),
        'words_judged': sum(judged.values()),
        'words_in_silence_en': round(words_silent_by['en'], 3),
        'words_in_silence_zh': round(words_silent_by['zh'], 3),
        'lag_en': round(lag_by['en'], 2),
        'lag_zh': round(lag_by['zh'], 2),
        'words_in_silence': round(words_silent, 3),
        'precision': round(precision, 3),
        'recall': round(recall, 3),
        'lag': round(lag, 2),
        'lag_start': round(lag_start, 2),
        'lag_end': round(lag_end, 2),
        'onset_error_med': round(onset_med, 2),
        'overrun': round(max(0.0, last_end - dur), 1),
        'rating': 'bad' if bad_reasons else ('warn' if warn else 'ok'),
        'problems': bad_reasons,
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('files', nargs='*')
    ap.add_argument('--all', action='store_true', help='check every track in public/data')
    ap.add_argument('--json', help='write the full report to this file')
    ap.add_argument('--jobs', type=int, default=os.cpu_count() or 2)
    args = ap.parse_args()

    files = args.files or []
    if args.all:
        files = sorted(str(p) for p in (ROOT / 'public' / 'data').glob('*_track*.json'))
    if not files:
        ap.error('give transcript files or --all')

    with ProcessPoolExecutor(max_workers=args.jobs) as ex:
        results = list(ex.map(check_track, files))

    order = {'bad': 0, 'warn': 1, 'ok': 2}
    results.sort(key=lambda r: (order[r['rating']], -r['words_in_silence']))
    print(f"{'track':34} {'rating':6} {'lag':>6} {'drift':>13} {'lag en/zh':>12} {'silent en/zh':>13} {'prec':>5}  problems")
    for r in results:
        drift = f"{r['lag_start']:+.2f}→{r['lag_end']:+.2f}"
        lags = f"{r['lag_en']:+.2f}/{r['lag_zh']:+.2f}"
        sil = f"{r['words_in_silence_en']:.0%}/{r['words_in_silence_zh']:.0%}"
        print(f"{r['track']:34} {r['rating']:6} {r['lag']:+6.2f} {drift:>13} {lags:>12} {sil:>13} {r['precision']:5.2f}  {'; '.join(r['problems'])}")
    counts = {k: sum(r['rating'] == k for r in results) for k in order}
    print(f"\n{len(results)} tracks: {counts['ok']} ok, {counts['warn']} warn, {counts['bad']} bad")

    if args.json:
        Path(args.json).write_text(json.dumps(results, indent=2))
    sys.exit(1 if counts['bad'] else 0)


if __name__ == '__main__':
    main()
