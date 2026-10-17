#!/usr/bin/env python3
"""Single-sample CNVscope visualization report.

Reads a CNVscope VCF (mode=0 or mode=1) plus the binary probe dump produced
by `--dump_probes`, and emits an HTML report with a genome ideogram,
per-chromosome panels (log2-ratio scatter, segment overlay, mapping-quality
track from FORMAT/MQS, optional rolling-variance and MQ0 mask tracks, segment
ribbon) and a sidebar of visualization controls.

Probe tracks are written as a bin-size pyramid cut into JSONP tiles that the
viewer loads on demand, so the browser only ever holds the visible window.
OUTPUT ending in .html gives one self-contained file with the tiles inlined;
any other OUTPUT is a directory holding index.html + viewer.js + data/.

Usage:
    python3 scripts/cnvscope_viz.py sample.vcf.gz sample.probes.bin report_dir
    python3 scripts/cnvscope_viz.py --serve report_dir

read_probes_binary() describes the probe dump format (CNVP v1 and v2).

The VCF may also be a combine_cnv.py output (CNVscope + indel2cnv-converted
SV calls, INFO SOURCE=CNV|SV_CNV): SV-derived records get their own ribbon
lane, a dotted segment overlay, and CNDIFF/repeat-array hover annotations.

Segments the CNV model rejected (FILTER=MLrejected; CNVModelApply --emit_conf 0
writes all of them) are not calls: they are drawn as thin grey lines and stay out of
the ribbon, the ideogram and the call counts. A sidebar menu puts the lines at the
model's CN (the default) or at the measured depth (rejected_y()).
"""

__version__ = '1.0.0'

import argparse
import base64
import gzip
import json
import logging
import math
import os
import shutil
import struct
import sys
import tempfile
import time
from collections import defaultdict

import numpy as np

logger = logging.getLogger(__name__)

DUMP_MAGIC = b'CNVP'
DUMP_VERSION = 2
# v1 stored log2r/log2r0 directly (r<=0 written as NaN, losing hom-dels);
# v2 stores linear r/r0 so r=0 round-trips and log2r is computed here.
LOG2R_ZERO_FLOOR = -8.0  # r=0 hom-del -> finite floor so it plots/classifies as a deletion
PROBE_RECORD_SIZE = 20
BLOCK_RECORD_SIZE = 12
EOF_MARKER = 0xCAFEBABE

CN_COLORS = {
    0: '#881337',  # hom_del — rose-900
    1: '#4f46e5',  # loss — indigo-600 (matches UI accent)
    2: '#94a3b8',  # neutral — slate-400
    3: '#e11d48',  # gain — rose-600
    4: '#be123c',  # amp — rose-700
}
GAIN_COLOR = '#e11d48'     # rose-600
LOSS_COLOR = '#4f46e5'     # indigo-600 (matches UI accent)
HOM_DEL_COLOR = '#881337'  # rose-900
AMP_COLOR = '#be123c'      # rose-700
NEUTRAL_COLOR = '#cbd5e1'  # slate-300
SV_COLOR = '#0d9488'       # teal-600 (SV-derived segment overlay)
REJECTED_COLOR = '#94a3b8' # slate-400 (model-rejected segment overlay)
MAPQ_COLOR = '#64748b'     # slate-500 (mapping-quality track)
KIND_COLORS = {'loss': LOSS_COLOR, 'hom_del': HOM_DEL_COLOR, 'gain': GAIN_COLOR, 'amp': AMP_COLOR}
POSTERIOR_SCALE = [
    [0.0, 'rgba(248, 250, 252, 0)'],   # transparent — blends with card
    [0.2, 'rgba(199, 210, 254, 0.6)'], # indigo-200
    [0.5, 'rgba(129, 140, 248, 0.85)'],# indigo-400
    [1.0, 'rgba(67, 56, 202, 0.95)'],  # indigo-700
]

SUPPORT_EMAIL = 'support@sentieon.com'
LOGO_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'sentieon_logo.png')


def load_logo_data_uri():
    """Base64 data URI of the bundled Sentieon logo, or '' (text title) when the file is missing."""
    try:
        with open(LOGO_PATH, 'rb') as fp:
            return 'data:image/png;base64,' + base64.b64encode(fp.read()).decode('ascii')
    except OSError as e:
        vlog(f'logo not found ({e}); using a text title')
        return ''


VERBOSE = False

def log(msg):
    print(f'[cnvscope_viz] {msg}', file=sys.stderr, flush=True)

def vlog(msg):
    if VERBOSE:
        print(f'[cnvscope_viz] {msg}', file=sys.stderr, flush=True)


def _r_to_log2r(r):
    """Linear ratio -> log2r. r>0: log2(r); r==0 (hom-del): finite floor; NaN: NaN."""
    out = np.full(r.shape, np.nan, dtype=np.float64 if r.dtype == np.float64 else np.float32)
    pos = np.isfinite(r) & (r > 0)
    out[pos] = np.log2(r[pos])
    out[np.isfinite(r) & (r <= 0)] = LOG2R_ZERO_FLOOR
    return out


def read_probes_binary(path):
    """Parse a CNVscope --dump_probes file (CNVP v1 or v2), little-endian and packed.

    Header: b'CNVP', u16 version, u16 flags, u8 sex (0 unknown, 1 M, 2 F), u8 sample_cn_neutral,
    u16 reserved, u32 n_chroms, u16 n + n bytes of UTF-8 sample name.
    Per chromosome: u16 n + n bytes of UTF-8 name, u32 length, u8 cn_neutral, u8 n_blocks,
    u16 reserved, u32 n_probes, then n_blocks block records and n_probes probe records.
    The file ends with u32 0xCAFEBABE (a mismatch is only logged with -v).
    - block, 12 bytes: u32 start, u32 end, u8 cn_neutral, 3 bytes padding; a ploidy block,
      e.g. male chrX outside the PARs with cn_neutral 1.
    - probe, 20 bytes: u32 start (0-based), u32 end, f32 r, f32 r0, u8 mq0, u8 flags, 2 bytes
      padding. r is the normalized depth ratio (1.0 = the sample's neutral copy number), r0 the
      ratio from well-mapped (MAPQ > 0) reads; mq0 == 0 marks a dark probe; flags are not used.
    v2 stores r and r0 as linear ratios, so r = 0 (hom-del) round-trips and log2r is computed
    here. v1 stores log2 ratios with r <= 0 written as NaN, so a hom-del probe reads as missing.

    Returns (meta, chroms) where:
    - meta: {sample, sex, sample_cn_neutral}
    - chroms: list of dicts with keys
        name, length, cn_neutral, blocks (list of {start,end,cn_neutral}),
        pos (np.uint32[n]), end (np.uint32[n]),
        log2r, log2r0, r, r0 (np.float32[n]),
        mq0 (np.uint8[n]), flags (np.uint8[n])
    """
    with open(path, 'rb') as fp:
        buf = fp.read()
    off = 0
    def take(n):
        nonlocal off
        v = buf[off:off+n]
        off += n
        return v
    if take(4) != DUMP_MAGIC:
        raise ValueError(f'{path}: not a CNVP file (bad magic)')
    version, hflags = struct.unpack('<HH', take(4))
    if version not in (1, 2):
        raise ValueError(f'{path}: unsupported version {version}')
    sex, sample_cn_neutral, _hres = struct.unpack('<BBH', take(4))
    n_chroms, = struct.unpack('<I', take(4))
    sname_len, = struct.unpack('<H', take(2))
    sample = take(sname_len).decode('utf-8', errors='replace')

    chroms = []
    for _ in range(n_chroms):
        nlen, = struct.unpack('<H', take(2))
        name = take(nlen).decode('utf-8', errors='replace')
        length, ccn, n_blocks, _cres = struct.unpack('<IBBH', take(8))
        n_probes, = struct.unpack('<I', take(4))
        blocks = []
        for _ in range(n_blocks):
            pstart, pend, bcn, _pad, _pad2 = struct.unpack('<IIBBH', take(12))
            blocks.append({'start': pstart, 'end': pend, 'cn_neutral': bcn})
        rec_bytes = n_probes * PROBE_RECORD_SIZE
        recs = np.frombuffer(buf[off:off+rec_bytes], dtype=np.dtype([
            ('start', '<u4'), ('end', '<u4'),
            ('v', '<f4'), ('v0', '<f4'),
            ('mq0', 'u1'), ('flags', 'u1'), ('pad', '<u2'),
        ]))
        off += rec_bytes
        if version >= 2:
            # v2 stores linear r/r0; compute log2r here (r=0 -> finite floor).
            r = recs['v'].copy()
            r0 = recs['v0'].copy()
            log2r = _r_to_log2r(r)
            log2r0 = _r_to_log2r(r0)
        else:
            # v1 stored log2r directly; r=0 was lost to NaN. Derive r for display.
            log2r = recs['v'].copy()
            log2r0 = recs['v0'].copy()
            r = np.where(np.isfinite(log2r), np.exp2(log2r), np.nan).astype(np.float32)
            r0 = np.where(np.isfinite(log2r0), np.exp2(log2r0), np.nan).astype(np.float32)
        chroms.append({
            'name': name,
            'length': int(length),
            'cn_neutral': int(ccn),
            'blocks': blocks,
            'pos': recs['start'].copy(),
            'end': recs['end'].copy(),
            'log2r': log2r,
            'log2r0': log2r0,
            'r': r,
            'r0': r0,
            'mq0': recs['mq0'].copy(),
            'flags': recs['flags'].copy(),
        })

    eof, = struct.unpack('<I', take(4))
    if eof != EOF_MARKER:
        vlog(f'warning: EOF marker mismatch (got {eof:#x}, expected {EOF_MARKER:#x})')

    sex_str = {0: 'unknown', 1: 'M', 2: 'F'}.get(sex, 'unknown')
    meta = {
        'sample': sample,
        'sex': sex_str,
        'sample_cn_neutral': sample_cn_neutral,
    }
    return meta, chroms


def _int(x):
    try:
        return int(x)
    except (TypeError, ValueError):
        return None


def _num(x):
    """A finite float from a VCF value or its text ('0.00', '-0.000000'); None when missing, '.' or nan/-nan."""
    try:
        f = float(x)
    except (TypeError, ValueError):
        return None
    return f if math.isfinite(f) else None


def _first_num(v):
    """The first number of a VCF value: a scalar, vcflib's list, or raw text ('59.0', '59.0,1'), read as _num()."""
    if isinstance(v, str):
        v = v.split(',')[0]
    elif isinstance(v, (list, tuple)):
        v = v[0] if v else None
    return _num(v)


def dps_log2r(dps):
    """Segment log2r from FORMAT/DPS[0], the trimmed mean normalized depth (probe r scale); None if absent."""
    if isinstance(dps, str):
        dps = dps.split(',')
    if isinstance(dps, (list, tuple)):
        dps = dps[0] if dps else None
    try:
        m = float(dps)
    except (TypeError, ValueError):
        return None
    if m > 0 and math.isfinite(m):
        return math.log2(m)
    return LOG2R_ZERO_FLOOR if m == 0 else None


def parse_floats(v):
    """A VCF list of numbers, such as INFO/PROBS (the model's probabilities for CN 0..4) or FORMAT/GAMMA, from
    vcflib's list or the raw text of an undeclared field: floats, None for a '.' or nan/-nan entry; None if absent."""
    if v is None:
        return None
    if isinstance(v, str):
        v = v.split(',')
    elif not isinstance(v, (list, tuple)):
        v = [v]
    return [_num(x) for x in v] or None


def read_vcf_reference_info(path):
    """Pull reference/assembly metadata from VCF contig and meta lines.

    Returns dict with keys: assembly, n_contigs, total_length, cmdline (or '').
    """
    import vcflib
    vcf = vcflib.VCF(path, 'r')
    assemblies = []
    total = 0
    for name, c in vcf.contigs.items():
        a = c.get('assembly')
        if a and a not in assemblies:
            assemblies.append(a)
        try:
            total += int(c.get('length', 0))
        except (TypeError, ValueError):
            pass
    if not assemblies:
        # combine_cnv.py drops the contig assembly attribute; fall back to ##reference
        for h in vcf.headers:
            if h.startswith('##reference='):
                assemblies.append(os.path.basename(h.split('=', 1)[1].strip()))
                break
    cmdline = ''
    cmd_hdrs = [h for h in vcf.headers if 'SentieonCommandLine' in h and 'CommandLine=' in h]
    # Combined VCFs carry several command lines; show the CNVscope one
    cmd_hdrs.sort(key=lambda h: 'SentieonCommandLine.CNVscope=' not in h)
    if cmd_hdrs:
        try:
            cmdline = cmd_hdrs[0].split('CommandLine=', 1)[1].split('">', 1)[0].lstrip('"')
        except IndexError:
            pass
    n_contigs = len(vcf.contigs)
    vcf.close()
    return {
        'assembly': '/'.join(assemblies) if assemblies else 'unknown',
        'n_contigs': n_contigs,
        'total_length': total,
        'cmdline': cmdline,
    }


# HMM output kept on a segment only for --internal reports
HMM_FIELDS = ('hmmcn', 'gamma', 'lr', 'cdelta', 'entropy', 'ecost', 'maxconf', 'minconf', 'rawcov')


def read_segments_vcf(path, chrom_names, hmm=True):
    """Parse CNVscope VCF into per-chrom segment lists.

    Handles these VCF shapes:
      - mode=1 raw: INFO.HMMCN + FORMAT.GAMMA/LR/...
      - mode=0 raw: just FORMAT.DPS[0] = normalized segment mean, no CN
      - CNVModelApply output: INFO.CN (final call) + INFO.ML_PROB; HMMCN kept in mode=1
      - combine_cnv.py output: the above plus SV-derived records (SOURCE=SV_CNV,
        CN = neutral + CNDIFF, REFSTART/REFSTOP repeat array) and FILTER=SVdup
      - CNVModelApply --emit_conf 0 output (and its merge): every segment, the rejected
        ones with FILTER=MLrejected, CN = CN_NEUTRAL and INFO PROBS

    Each segment gets a 'log2r' = log2(FORMAT/DPS[0]) (never FORMAT/LR, kept as 'lr'),
    an 'mqs' = FORMAT/MQS (the mean MAPQ), a 'cn' (INFO/CN, else HMMCN except on a rejected
    record; None if neither) and 'rejected' (a CNVscope record with FILTER=MLrejected).
    hmm=False (customer reports) sets the HMM_FIELDS to None and takes CN from INFO/CN only.
    Returns (segments_by_chrom, mode).
    """
    import vcflib
    vcf = vcflib.VCF(path, 'r')
    segs = defaultdict(list)
    chrom_set = set(chrom_names)
    has_gamma = False
    has_cn = False
    for chrom in vcf.contigs:
        if chrom not in chrom_set:
            continue
        for v in vcf.range(chrom):
            info = v.info
            end = info.get('END', v.pos + 1)
            svlen = info.get('SVLEN', end - v.pos)
            samp = v.samples[0] if v.samples else {}
            cn = info.get('CN', info.get('HMMCN', None))
            hmmcn = _int(info.get('HMMCN', None))
            source = info.get('SOURCE', None)
            cn_neutral_info = info.get('CN_NEUTRAL', None)
            gamma = samp.get('GAMMA', None)
            if gamma is not None: has_gamma = True
            if cn is not None: has_cn = True
            ml_prob = info.get('ML_PROB', None)
            filt = (';'.join(v.filter) if isinstance(v.filter, (list, tuple)) and v.filter
                    else (v.filter or 'PASS'))
            sv = source is not None and source != 'CNV'
            rejected = not sv and 'MLrejected' in filt.split(';')
            if rejected:
                cn = info.get('CN', None)   # the model's CN only, never HMMCN; rejected_y() falls back to the neutral CN
            seg = {
                'start': int(v.pos),
                'end': int(end),
                'qual': float(v.qual) if v.qual is not None else float('nan'),
                # SV-derived CN (neutral + CNDIFF) can go below 0; clamp like cnv_eval
                'cn': max(int(cn), 0) if cn is not None else None,
                'hmmcn': hmmcn,
                'cn_neutral': int(cn_neutral_info) if cn_neutral_info is not None else None,
                'sv': sv,
                'rejected': rejected,
                'svtype': info.get('SVTYPE', None),
                'cndiff': _int(info.get('CNDIFF', None)),
                'period': _int(info.get('PERIOD', None)),
                'refstart': _int(info.get('REFSTART', None)),
                'refstop': _int(info.get('REFSTOP', None)),
                'dedup': info.get('DEDUP', None),
                'cnvcov': _num(info.get('CNVCOV')),
                'rawcov': _num(info.get('RAWCOV')),
                'gt': samp.get('GT', None),
                'svlen': int(svlen) if svlen is not None else 0,
                'log2r': dps_log2r(samp.get('DPS')),
                'mqs': _first_num(samp.get('MQS')),   # the segment's mean MAPQ
                # CNVModelApply's number text ('0.00', '-nan') and vcflib's rewrite of it read alike
                'ml_prob': _num(ml_prob),
                'probs': parse_floats(info.get('PROBS', None)),
                'gamma': parse_floats(gamma) if gamma else None,
                'lr': _num(samp.get('LR')),
                'cdelta': _num(samp.get('CDELTA')),
                'entropy': _num(samp.get('ENTROPY')),
                'ecost': _num(samp.get('ECOST')),
                'maxconf': _num(samp.get('MAXCONF')),
                'minconf': _num(samp.get('MINCONF')),
                'filter': filt,
            }
            if not hmm:
                seg.update(dict.fromkeys(HMM_FIELDS))
                if info.get('CN', None) is None:
                    seg['cn'] = None   # not the HMM's CN either
            segs[chrom].append(seg)
    vcf.close()
    mode = 1 if has_gamma else 0
    return dict(segs), mode


def chrom_length(c):
    """Plotted length of a chrom: the contig length, never shorter than its last probe."""
    return max(int(c['length'] or 0), int(c['pos'].max()) + 1 if c['pos'].size else 1)


def bin_probes(pos, r, mq0, bin_size):
    """Per-bin median linear ratio, dark fraction and probe count, fully vectorized.

    Bins are keyed by probe start (pos // bin_size); bins with no probe are absent.
    Median is over probes with finite r (r=0 hom-dels count), NaN when none are.
    Returns (bin_index, median_r, dark_frac, n_probes).
    """
    n = pos.size
    if n == 0:
        return (np.zeros(0, dtype=np.int64), np.zeros(0), np.zeros(0),
                np.zeros(0, dtype=np.int64))
    bidx = pos.astype(np.int64) // int(bin_size)
    order = np.lexsort((r, bidx))  # NaN sorts last within a bin
    b = bidx[order]
    rs = r[order].astype(np.float64)
    dark = (mq0[order] == 0).astype(np.int64)
    starts = np.concatenate(([0], np.flatnonzero(np.diff(b)) + 1))
    counts = np.diff(np.concatenate((starts, [n])))
    nvalid = np.add.reduceat(np.isfinite(rs).astype(np.int64), starts)
    med = np.full(starts.size, np.nan)
    ok = nvalid > 0
    lo = (starts + (nvalid - 1) // 2)[ok]
    hi = (starts + nvalid // 2)[ok]
    med[ok] = 0.5 * (rs[lo] + rs[hi])
    return b[starts], med, np.add.reduceat(dark, starts) / counts, counts


def smooth_median(arr, window):
    """Centered rolling nan-median over a level's bin sequence, applied to linear r."""
    if window <= 1 or arr.size < 3:
        return arr
    from numpy.lib.stride_tricks import sliding_window_view
    pad = window // 2
    padded = np.concatenate([np.full(pad, np.nan), arr.astype(np.float64),
                             np.full(pad, np.nan)])
    with np.errstate(invalid='ignore'):
        sm = np.nanmedian(sliding_window_view(padded, window), axis=1)
    return sm[:arr.size]


def rolling_std(x, window):
    """Centered rolling population SD over finite values; NaN where fewer than 3 valid."""
    n = len(x)
    if n == 0 or window <= 1:
        return np.full(n, np.nan)
    v = np.asarray(x, dtype=np.float64)
    ok = np.isfinite(v)
    # centering on the global mean keeps the cumulative sums well conditioned
    d = np.where(ok, v - (v[ok].mean() if ok.any() else 0.0), 0.0)
    c0 = np.concatenate(([0.0], np.cumsum(ok.astype(np.float64))))
    c1 = np.concatenate(([0.0], np.cumsum(d)))
    c2 = np.concatenate(([0.0], np.cumsum(d * d)))
    h = window // 2
    i = np.arange(n)
    lo = np.maximum(0, i - h)
    hi = np.minimum(n, i + h + 1)
    cnt = c0[hi] - c0[lo]
    with np.errstate(invalid='ignore', divide='ignore'):
        var = (c2[hi] - c2[lo]) / cnt - ((c1[hi] - c1[lo]) / cnt) ** 2
    out = np.sqrt(np.maximum(var, 0.0))
    out[cnt < 3] = np.nan
    return out


def build_levels(bin_sizes, probe_width, max_len, points_per_view):
    """Sorted bin-size pyramid, extended x5 until the longest chrom fits the point budget."""
    bins = sorted({int(b) for b in bin_sizes if int(b) > 0})
    if probe_width and int(probe_width) > 0:
        bins = sorted(set(bins) | {int(probe_width)})
    if not bins:
        bins = [10000]
    while max_len / bins[-1] > points_per_view and len(bins) < 32:
        bins.append(bins[-1] * 5)
    return bins


def pick_level(bins, span, points_per_view):
    """Finest level whose visible span fits the point budget (else the coarsest)."""
    for k, b in enumerate(bins):
        if span / b <= points_per_view:
            return k
    return len(bins) - 1


def tile_range(x0, x1, bin_size, tile_points, length, n_tiles):
    """Inclusive tile index range covering [x0, x1] at one level."""
    ts = tile_points * bin_size
    hi = max(length - 1, 0)
    t0 = int(min(max(x0, 0), hi) // ts)
    t1 = int(min(max(x1, 0), hi) // ts)
    if t1 < t0:
        t0, t1 = t1, t0
    return max(0, min(t0, n_tiles - 1)), max(0, min(t1, n_tiles - 1))


def n_tiles_for(length, bin_size, tile_points):
    """Tiles needed to cover a chrom at one level."""
    n_bins = -(-int(length) // int(bin_size))
    return max(1, -(-n_bins // int(tile_points)))


def _jsonable(a):
    """numpy array -> JSON list with NaN replaced by None."""
    lst = a.tolist()
    if a.dtype.kind == 'f' and np.isnan(a).any():
        lst = [None if v != v else v for v in lst]
    return lst


def level_arrays(chrom, bin_size, smooth_window, want_sd):
    """Per-bin (centers, median r, dark fraction, rolling SD) for one chrom at one level."""
    bidx, med, dark, _ = bin_probes(chrom['pos'], chrom['r'], chrom['mq0'], bin_size)
    med = smooth_median(med, smooth_window)
    centers = ((bidx.astype(np.float64) + 0.5) * bin_size).astype(np.int64)
    sd = None
    if want_sd:
        w = max(3, smooth_window if smooth_window > 2 else 5)
        sd = rolling_std(_r_to_log2r(med), w)
    return bidx, centers, med, dark, sd


def level_tiles(chrom, bin_size, tile_points, smooth_window, want_sd):
    """One chrom at one level: (per-tile point counts, tile generator, per-bin median r).

    The generator yields (tile_index, payload_json); counts[t] == 0 tiles are skipped.
    """
    n_tiles = n_tiles_for(chrom_length(chrom), bin_size, tile_points)
    bidx, centers, med, dark, sd = level_arrays(chrom, bin_size, smooth_window, want_sd)
    edges = np.searchsorted(bidx, np.arange(n_tiles + 1) * tile_points)
    counts = np.diff(edges).astype(np.int64)

    def gen():
        for t in range(n_tiles):
            s, e = int(edges[t]), int(edges[t + 1])
            if e <= s:
                continue
            payload = {
                'x': np.diff(centers[s:e], prepend=0).tolist(),
                'r': _jsonable(np.round(med[s:e], 4)),
                'd': _jsonable(np.round(dark[s:e], 3)),
            }
            if sd is not None:
                payload['sd'] = _jsonable(np.round(sd[s:e], 3))
            yield t, json.dumps(payload, separators=(',', ':'))
    return counts, gen(), med


def chrom_qc(chrom):
    log2r = chrom['log2r']
    valid = log2r[np.isfinite(log2r)]
    if valid.size == 0:
        return {'median': float('nan'), 'mad': float('nan'), 'n_dark': int((chrom['mq0'] == 0).sum()), 'n_probes': int(chrom['log2r'].size)}
    med = float(np.median(valid))
    mad = float(np.median(np.abs(valid - med)))
    return {
        'median': med,
        'mad': mad,
        'n_dark': int((chrom['mq0'] == 0).sum()),
        'n_probes': int(chrom['log2r'].size),
    }


def mapd(chroms):
    """Median |log2r[i+1] - log2r[i]| over neighbouring finite probes in genomic order, pooled over chroms."""
    diffs = []
    for c in chroms:
        v = c['log2r']
        if v.size > 1 and np.any(c['pos'][1:] < c['pos'][:-1]):
            v = v[np.argsort(c['pos'], kind='stable')]
        v = v[np.isfinite(v)]
        if v.size > 1:
            diffs.append(np.abs(np.diff(v)))
    return float(np.median(np.concatenate(diffs))) if diffs else float('nan')


LOG2R_LOSS_THRESH = -0.25
LOG2R_GAIN_THRESH = 0.20
LOG2R_HOM_THRESH = -1.5
LOG2R_AMP_THRESH = 0.9


def seg_cn_neutral(seg, chrom_cn_neutral):
    """Effective neutral CN for a segment: per-segment CN_NEUTRAL if set
    (sex chrom blocks), else the chromosome-level default."""
    v = seg.get('cn_neutral')
    return v if v is not None else chrom_cn_neutral


HOMDEL_FLOOR = -10.0  # log2 sentinel: 2^-10 ≈ 0.001 (≈ 0 in linear mode)


def cn_log2(cn, sample_cn_neutral):
    """log2(cn / sample_cn_neutral), the level the dots take for that CN; CN 0 at HOMDEL_FLOOR."""
    return HOMDEL_FLOOR if cn <= 0 else math.log2(cn / max(sample_cn_neutral, 1))


def segment_y(seg, sample_cn_neutral):
    """y-coordinate (log2 ratio) for the segment overlay line.

    Probe-level DP in the dump is sample-wide normalized: the value the
    plotted dots take for a CN=k region is log2(k / sample_cn_neutral),
    regardless of which ploidy block the region falls into. So segment
    overlays must use the same convention to align with the dots.

    Hom-del (CN=0) returns HOMDEL_FLOOR so it lands at r≈0 in linear mode
    (and below the chart in log2 mode, where the ribbon still marks it)."""
    if seg.get('cn') is not None:
        return cn_log2(seg['cn'], sample_cn_neutral)
    if seg.get('log2r') is not None:
        return seg['log2r']
    return None


# Levels of the grey rejected-segment lines, in menu order; 'cn' is the default
REJECTED_LEVELS = ('cn', 'depth', 'hmm')


def rejected_y(seg, level, sample_cn_neutral, chrom_cn_neutral):
    """y (log2 ratio) of a model-rejected segment at one level, on the probes' sample-wide scale; None: no line.
    'cn' (the default, both modes): the model's CN, as segment_y() draws a call; the model gives a rejected record
    the neutral CN, and a record without CN takes seg_cn_neutral(). 'depth': the measured depth, log2(FORMAT/DPS[0]).
    'hmm': CNVscope's HMM state, log2(HMMCN / sample_cn_neutral), when the VCF carries HMMCN. CN or HMMCN 0: HOMDEL_FLOOR."""
    if level == 'depth':
        return seg.get('log2r')
    if level == 'hmm':
        cn = seg.get('hmmcn')
    else:
        cn = seg['cn'] if seg.get('cn') is not None else seg_cn_neutral(seg, chrom_cn_neutral)
    return None if cn is None else cn_log2(cn, sample_cn_neutral)


def p_neutral(seg, chrom_cn_neutral):
    """The model's probability of the neutral CN: PROBS[CN_NEUTRAL] when that index exists, else None."""
    probs = seg.get('probs')
    k = seg_cn_neutral(seg, chrom_cn_neutral)
    if probs and k is not None and 0 <= k < len(probs):
        return probs[k]
    return None


def segment_kind(seg, chrom_cn_neutral):
    """Categorise segment as one of {hom_del, loss, gain, amp} or None
    (neutral / not callable). Uses CN call when present, log2r threshold otherwise."""
    if seg.get('cn') is not None:
        cn0 = seg_cn_neutral(seg, chrom_cn_neutral)
        cn = seg['cn']
        if cn == cn0: return None
        if cn <= 0: return 'hom_del'
        if cn >= cn0 + 2: return 'amp'
        return 'gain' if cn > cn0 else 'loss'
    log2r = seg.get('log2r')
    if log2r is None or math.isnan(log2r): return None
    if log2r < LOG2R_HOM_THRESH: return 'hom_del'
    if log2r < LOG2R_LOSS_THRESH: return 'loss'
    if log2r > LOG2R_AMP_THRESH: return 'amp'
    if log2r > LOG2R_GAIN_THRESH: return 'gain'
    return None


def local_cn_neutral(chrom, pos):
    """Ploidy at pos: cn_neutral of the containing ploidy block (blocks are probe-index ranges)."""
    ppos, pend = chrom['pos'], chrom['end']
    for b in chrom.get('blocks', []):
        if b['end'] > b['start'] and ppos[b['start']] <= pos < pend[b['end'] - 1]:
            return b['cn_neutral']
    return chrom['cn_neutral']


def rederive_sv_cn(segments, chroms):
    """Older combine_cnv.py output has no CN_NEUTRAL on SV-derived records (CN = 2 + CNDIFF);
    redo those against the dump's ploidy blocks. Records carrying CN_NEUTRAL are trusted."""
    by_name = {c['name']: c for c in chroms}
    for name, ss in segments.items():
        c = by_name.get(name)
        if c is None: continue
        for s in ss:
            if s.get('sv') and s.get('cndiff') is not None and s.get('cn_neutral') is None:
                s['cn_neutral'] = local_cn_neutral(c, s['start'])
                s['cn'] = max(0, s['cn_neutral'] + s['cndiff'])


def call_location(chrom_name, s):
    """chrom:POS-END as in the VCF; segment starts are kept 0-based, like the probes."""
    return f"{chrom_name}:{s['start'] + 1:,}-{s['end']:,}"


def _measured(s):
    """Card line of a segment's measured values, log2r (DPS[0]) and MAPQ (MQS); '' when it has neither."""
    parts = []
    if s.get('log2r') is not None:
        parts.append(f"log2r={s['log2r']:.3f}")
    if s.get('mqs') is not None:
        parts.append(f"MAPQ={round(s['mqs'], 1):g}")
    return ' '.join(parts) + '<br>' if parts else ''


def segment_hover(chrom_name, s, chrom_cn_neutral, has_sv=False):
    hov = call_location(chrom_name, s) + '<br>'
    if has_sv:
        hov += 'SV-derived call<br>' if s.get('sv') else 'CNVscope call<br>'
    if s['cn'] is not None:
        hov += f"CN={s['cn']} (neutral {seg_cn_neutral(s, chrom_cn_neutral)})"
        if s.get('hmmcn') is not None and s['hmmcn'] != s['cn']:
            hov += f" HMMCN={s['hmmcn']}"
        hov += '<br>'
    if s.get('sv'):
        if s.get('cndiff') is not None:
            hov += f"{s.get('svtype') or 'SV'} CNDIFF={s['cndiff']:+d}"
            if s.get('gt'): hov += f" GT={s['gt']}"
            hov += '<br>'
        if s.get('refstart') is not None and s.get('refstop') is not None:
            arr_len = s['refstop'] - s['refstart']
            hov += f"repeat array {s['refstart']:,}-{s['refstop']:,} ({arr_len:,} bp"
            if s.get('period'):
                hov += f", {arr_len / s['period']:.1f} x {s['period']:,} bp unit"
            hov += ')<br>'
        if s.get('cnvcov') is not None:
            hov += f"depth support: PASS calls {float(s['cnvcov']):.2f}"
            if s.get('rawcov') is not None:
                hov += f", raw HMM {float(s['rawcov']):.2f}"
            hov += ' of array<br>'
    hov += _measured(s)
    hov += (f"QUAL={s['qual']:.1f} FILTER={s['filter']}<br>"
            f"len={s['end']-s['start']:,} bp")
    if s.get('dedup'): hov += f"<br>DEDUP={s['dedup']}"
    if s.get('ml_prob') is not None: hov += f"<br>ML_PROB={s['ml_prob']:.3f}"
    return hov + _hmm_features(s)


def _hmm_features(s):
    """LR / CDELTA / ENTROPY / ECOST hover lines, as on the call cards."""
    hov = ''
    if s.get('lr') is not None: hov += f"<br>LR={s['lr']:.3f}"
    if s.get('cdelta') is not None: hov += f" CDELTA={s['cdelta']:.3f}"
    if s.get('entropy') is not None: hov += f"<br>ENTROPY={s['entropy']:.3f}"
    if s.get('ecost') is not None: hov += f" ECOST={s['ecost']:.3f}"
    return hov


def rejected_hover(chrom_name, s, chrom_cn_neutral):
    """Hover card of a model-rejected segment: the model's CN and P(neutral), and CNVscope's own values."""
    hov = call_location(chrom_name, s) + '<br>rejected by the model<br>'
    cn_part = []
    if s.get('cn') is not None:
        cn_part.append(f"CN={s['cn']} (neutral {seg_cn_neutral(s, chrom_cn_neutral)})")
    if s.get('hmmcn') is not None:
        cn_part.append(f"HMMCN={s['hmmcn']}")
    if cn_part:
        hov += ' '.join(cn_part) + '<br>'
    pn = p_neutral(s, chrom_cn_neutral)
    prob_part = []
    if pn is not None:
        prob_part.append(f"P(neutral)={pn:.2f}")
    if s.get('ml_prob') is not None:
        prob_part.append(f"ML_PROB={s['ml_prob']:.2f}")
    if prob_part:
        hov += ' '.join(prob_part) + '<br>'
    hov += _measured(s)
    qual = f"QUAL={s['qual']:.1f} " if not math.isnan(s['qual']) else ''
    hov += f"{qual}len={s['end']-s['start']:,} bp"
    return hov + _hmm_features(s)


def compute_event_summary(segments, sample_cn_neutral, chrom_cn_neutral):
    counts = defaultdict(int)
    total_bp = defaultdict(int)
    for chrom, ss in segments.items():
        cn0 = chrom_cn_neutral.get(chrom, sample_cn_neutral)
        for s in ss:
            kind = segment_kind(s, cn0)
            if kind is None:
                continue
            counts[kind] += 1
            total_bp[kind] += s['end'] - s['start']
    return dict(counts), dict(total_bp)


def strip_chr(name):
    """Chrom name without a chr prefix, so chr1 and 1 compare equal."""
    return name[3:] if name[:3].lower() == 'chr' else name


def is_main_chrom(name):
    """Autosome or sex chromosome (1..N, X, Y), with or without the chr prefix."""
    n = strip_chr(name)
    return n.isdigit() or n.upper() in ('X', 'Y')


def select_chroms(chroms, min_probes, wanted, all_chroms=False):
    """Chroms shown in the report, each with at least min_probes probes: the --chroms list (chr prefix optional),
    every contig with --all-chroms, else the autosomes and sex chromosomes (every contig if the dump names none)."""
    shown = [c for c in chroms if c['pos'].size >= min_probes]
    if wanted:
        names = {strip_chr(n.strip()) for n in wanted.split(',')}
        return [c for c in shown if strip_chr(c['name']) in names]
    if all_chroms:
        return shown
    return [c for c in shown if is_main_chrom(c['name'])] or shown


def record_counts(segs):
    """Records by source and FILTER: CNVscope PASS / other FILTER / rejected by the model, SV-derived PASS / other."""
    n = {'cnv_pass': 0, 'cnv_filtered': 0, 'rejected': 0, 'sv_pass': 0, 'sv_filtered': 0}
    for s in segs:
        if s.get('sv'):
            n['sv_pass' if s['filter'] == 'PASS' else 'sv_filtered'] += 1
        elif s.get('rejected'):
            n['rejected'] += 1
        else:
            n['cnv_pass' if s['filter'] == 'PASS' else 'cnv_filtered'] += 1
    return n


def _pass_calls(k):
    return f"{k:,} PASS call" + ('' if k == 1 else 's')


def segment_lines(n, has_sv):
    """Segments tile and chrom chip wording for record_counts() n: one line per source."""
    cnv = [_pass_calls(n['cnv_pass'])]
    if n['cnv_filtered']: cnv.append(f"{n['cnv_filtered']:,} filtered")
    if n['rejected']: cnv.append(f"{n['rejected']:,} rejected")
    if not has_sv:
        return [' &middot; '.join(cnv)]
    sv = [_pass_calls(n['sv_pass'])]
    if n['sv_filtered']: sv.append(f"{n['sv_filtered']:,} filtered")
    return ['CNVscope: ' + ' &middot; '.join(cnv), 'SV-derived: ' + ' &middot; '.join(sv)]


def call_summary(shown_chroms, segments, sample_cn_neutral):
    """Header counts over the shown chroms only. Segments counts every record; events and calls per Mb
    count PASS calls (CNVscope or SV-derived) that change copy number."""
    shown = {c['name']: segments.get(c['name'], []) for c in shown_chroms}
    passing = {name: [s for s in ss if s['filter'] == 'PASS'] for name, ss in shown.items()}
    counts, _ = compute_event_summary(passing, sample_cn_neutral,
                                      {c['name']: c['cn_neutral'] for c in shown_chroms})
    n_events = sum(counts.values())
    span_mb = sum(chrom_length(c) for c in shown_chroms) / 1e6
    return {
        'n_segments': sum(len(ss) for ss in shown.values()),
        'n_sv': sum(1 for ss in shown.values() for s in ss if s.get('sv')),
        'records': record_counts(s for ss in shown.values() for s in ss),
        'n_events': n_events,
        'n_gain': counts.get('gain', 0) + counts.get('amp', 0),
        'n_loss': counts.get('loss', 0) + counts.get('hom_del', 0),
        'calls_per_mb': n_events / span_mb if span_mb else 0,
    }


def fig_genome_ideogram(chroms, segments, sample_cn_neutral, max_height_log2r=2.0):
    """One-row ideogram across all chroms. Segments drawn as colored bars, height encoding
    |log2(cn/cn_neutral)|; bar traces carry meta=chrom and customdata=segment index for the viewer."""
    import plotly.graph_objects as go
    import plotly.subplots as sp

    n = len(chroms)
    fig = sp.make_subplots(
        rows=1, cols=n, shared_yaxes=True, horizontal_spacing=0.002,
        subplot_titles=[c['name'] for c in chroms],
    )
    for col, chrom in enumerate(chroms, 1):
        L = chrom_length(chrom)
        cn0 = chrom['cn_neutral']
        fig.add_trace(go.Scatter(
            x=[0, L], y=[0, 0], mode='lines',
            line=dict(color='#dddddd', width=1),
            hoverinfo='skip', showlegend=False,
        ), row=1, col=col)
        bars = {(d, sv): dict(x=[], y=[], w=[], tip=[], idx=[])
                for sv in (True, False) for d in ('gain', 'loss')}
        chrom_baseline = math.log2(max(cn0, 1) / max(sample_cn_neutral, 1))
        for i, s in enumerate(segments.get(chrom['name'], [])):
            kind = segment_kind(s, cn0)
            if kind is None: continue
            y = segment_y(s, sample_cn_neutral)
            if y is None: continue
            # bar height relative to this chrom's expected baseline
            h = max(min(y - chrom_baseline, max_height_log2r), -max_height_log2r)
            x = (s['start'] + s['end']) / 2
            w = max(s['end'] - s['start'], L * 0.001)
            tip = call_location(chrom['name'], s) + '<br>'
            if s.get('sv'):
                tip += 'SV-derived<br>'
            if s.get('cn') is not None:
                seg_cn0 = seg_cn_neutral(s, cn0)
                tip += f"CN={s['cn']} (neutral {seg_cn0})<br>"
            tip += f"len={s['end']-s['start']:,} bp"
            b = bars[('gain' if kind in ('gain', 'amp') else 'loss', bool(s.get('sv')))]
            b['x'].append(x); b['y'].append(h); b['w'].append(w); b['tip'].append(tip); b['idx'].append(i)
        # SV-derived bars are drawn first (lighter) so CNVscope calls sit on top
        for (d, sv), b in bars.items():
            if not b['x']: continue
            fig.add_trace(go.Bar(
                x=b['x'], y=b['y'], width=b['w'], customdata=b['idx'], meta=chrom['name'],
                marker=dict(color=GAIN_COLOR if d == 'gain' else LOSS_COLOR,
                            line=dict(width=0), opacity=0.45 if sv else 0.8),
                hovertext=b['tip'], hoverinfo='text', showlegend=False,
            ), row=1, col=col)
        fig.update_xaxes(showticklabels=False, showgrid=False, zeroline=False,
                         range=[0, L], row=1, col=col)
    fig.update_yaxes(range=[-max_height_log2r * 1.05, max_height_log2r * 1.05],
                     showgrid=False, zeroline=True, zerolinecolor='#cbd5e1',
                     tickfont=dict(size=11, color='#64748b'),
                     row=1, col=1)
    for col in range(2, n + 1):
        fig.update_yaxes(showticklabels=False, row=1, col=col)
    for ann in fig.layout.annotations:
        ann.font = dict(size=11, color='#475569', weight=500)
        ann.textangle = -45 if n > 12 else 0
        ann.yshift = -8  # pull chrom labels down closer to the bars
    fig.update_layout(
        height=170, margin=dict(l=40, r=10, t=22, b=10),
        plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
        bargap=0,
        hoverlabel=dict(
            bgcolor='#1e293b', bordercolor='#1e293b',
            font=dict(color='#f8fafc', size=13,
                      family='-apple-system, BlinkMacSystemFont, Inter, system-ui, sans-serif'),
            align='left', namelength=-1,
        ),
    )
    return fig


def panel_yranges(med_r, neutral_log2r, y_max):
    """Fixed y ranges for a chrom, from the coarsest level, so zooming never rescales."""
    finite = _r_to_log2r(med_r)
    finite = finite[np.isfinite(finite)]
    if y_max:
        half = y_max
    elif finite.size:
        # 99th-percentile deviation from the baseline: one dark probe must not blow up the range
        half = max(2.0, float(np.percentile(np.abs(finite - neutral_log2r), 99))) * 1.1
    else:
        half = 2.0
    lin = med_r[np.isfinite(med_r)]
    upper = max(2.0, float(np.percentile(lin, 99)) * 1.15) if lin.size else 2.0
    return {'yrange_log2': [neutral_log2r - half, neutral_log2r + half],
            'yrange_linear': [-upper * 0.03, upper],
            'neutral_log2r': neutral_log2r}


def _obj(v):
    """v as a numpy object array: plotly.py validates it in one copy, a list item by item (2 s for 93k records)."""
    return np.array(v, dtype=object)


def _short_num(v):
    """v rounded to 1 decimal, an int when whole, so the panel JSON writes 59 for 59.0."""
    v = round(v, 1)
    return int(v) if v == int(v) else v


def fig_chrom_panel(chrom, segments_for_chrom, sample_cn_neutral, yranges,
                    show_posterior, show_variance, show_mq0, mode, has_sv=False, rejected=(),
                    show_mapq=False):
    """Per-chrom panel split into one Plotly figure per track. Returns a list
    of {'track': str, 'fig': go.Figure, 'height': int} dicts. Track divs are
    independent so the HTML can hide any of them with `display: none` and
    actually reclaim layout space. X-axis sync between divs is handled
    client-side via plotly_relayout events.

    segments_for_chrom are the calls (customdata indexes them, as CNV_CONFIG.segments
    does); rejected are the model-rejected segments, drawn on the depth track only.
    show_mapq adds the mapping-quality pane below the depth track: FORMAT/MQS of every
    CNVscope record, calls and rejected segments alike.

    Probe-fed traces (scatter/variance/mq0) are emitted empty; the viewer fills
    them from tiles at the resolution the visible span asks for.
    """
    import plotly.graph_objects as go

    cn0 = chrom['cn_neutral']
    L = chrom_length(chrom)
    common_x = dict(
        range=[0, L], tickformat=',d',
        showgrid=False, zeroline=False,
        linecolor='rgba(15,23,42,0.10)', tickcolor='rgba(15,23,42,0.20)',
        tickfont=dict(size=12, color='#64748b'),
        showticklabels=False,  # JS turns on for the bottom-most visible pane
    )
    common_y_style = dict(
        linecolor='rgba(15,23,42,0.10)', tickcolor='rgba(15,23,42,0.20)',
        tickfont=dict(size=12, color='#64748b'),
        gridcolor='rgba(15,23,42,0.04)',
    )
    common_layout = dict(
        margin=dict(l=60, r=20, t=10, b=22),
        plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
        showlegend=False,
        font=dict(family='-apple-system, BlinkMacSystemFont, Inter, system-ui, sans-serif',
                  color='#0f172a'),
        hoverlabel=dict(
            bgcolor='#1e293b', bordercolor='#1e293b',
            font=dict(color='#f8fafc', size=13,
                      family='-apple-system, BlinkMacSystemFont, Inter, system-ui, sans-serif'),
            align='left', namelength=-1,
        ),
    )

    # the HMM posterior covers every CNVscope segment, rejected or not, in genome order
    hmm_segments = sorted(list(segments_for_chrom) + list(rejected), key=lambda s: s['start'])
    has_posterior = show_posterior and mode == 1 and any(
        s.get('gamma') for s in hmm_segments)
    panes = []

    # ── log2 ratio scatter + segment overlay ────────────────────────
    scatter_fig = go.Figure()
    scatter_fig.add_trace(go.Scattergl(
        x=[], y=[], mode='markers',
        marker=dict(size=3, color=[],
                    colorscale=[[0, '#475569'], [1, '#f59e0b']],
                    cmin=0, cmax=1, opacity=0.6, showscale=False),
        name='probes', meta='probes',
        hovertemplate='pos=%{x:,.0f}<br>log2r=%{y:.2f}<extra></extra>',
    ))
    # Sample-wide normalization: neutral baseline for this chrom is
    # log2(chrom_cn_neutral / sample_cn_neutral) — non-zero on haploid chroms.
    chrom_neutral_log2r = yranges['neutral_log2r']
    scatter_fig.add_hline(y=chrom_neutral_log2r,
                          line=dict(color='#cbd5e1', width=1, dash='dot'))
    # Rejected segments sit under the calls; no customdata, so the call filters never touch them.
    # The traces' y is the cn level; layout.meta.rejected_levels holds one y per segment for each other level
    levels = [k for k in REJECTED_LEVELS
              if k != 'hmm' or any(s.get('hmmcn') is not None for s in rejected)]
    rej_x, mid_x, rej_hover = [], [], []
    seg_y = {k: [] for k in levels}
    for s in rejected:
        rej_x += [s['start'], s['end'], None]
        mid_x.append((s['start'] + s['end']) // 2)
        rej_hover.append(rejected_hover(chrom['name'], s, cn0))
        for k in levels:
            y = rejected_y(s, k, sample_cn_neutral, cn0)
            seg_y[k].append(None if y is None else round(y, 4))
    scatter_meta = yranges
    if rej_x:
        scatter_fig.add_trace(go.Scatter(
            x=_obj(rej_x), y=_obj([v for y in seg_y['cn'] for v in (y, y, None)]), mode='lines',
            line=dict(color=REJECTED_COLOR, width=1.5), name='rejected segments', meta='rejected',
            hoverinfo='skip',
        ))
        # invisible midpoint markers carry the hover, as in the ribbon. SVG, not WebGL: Plotly scores an
        # SVG marker by its radius, so within 6 px the card beats the probe dots around the same level
        scatter_fig.add_trace(go.Scatter(
            x=_obj(mid_x), y=_obj(seg_y['cn']), mode='markers', marker=dict(size=12, opacity=0),
            hovertext=_obj(rej_hover), hoverinfo='text', name='rejected segments', meta='rejected',
        ))
        scatter_meta = dict(yranges, rejected_levels={k: _obj(seg_y[k]) for k in levels if k != 'cn'})
    # customdata = index into this chrom's CNV_CONFIG.segments, so the viewer can filter calls
    seg_x, seg_y, seg_i = [], [], []
    sv_x, sv_y, sv_i = [], [], []
    for i, s in enumerate(segments_for_chrom):
        y = segment_y(s, sample_cn_neutral)
        if y is None: continue
        xs, ys, idx = (sv_x, sv_y, sv_i) if s.get('sv') else (seg_x, seg_y, seg_i)
        xs += [s['start'], s['end'], None]
        ys += [y, y, None]
        idx += [i, i, None]
    if seg_x:
        scatter_fig.add_trace(go.Scatter(
            x=seg_x, y=seg_y, customdata=seg_i, mode='lines',
            line=dict(color='#0f172a', width=2.5),
            name='segments', meta='segments', hoverinfo='skip',
        ))
    if sv_x:
        scatter_fig.add_trace(go.Scatter(
            x=sv_x, y=sv_y, customdata=sv_i, mode='lines',
            line=dict(color=SV_COLOR, width=2, dash='dot'),
            name='SV-derived segments', meta='segments', hoverinfo='skip',
        ))
    scatter_fig.update_xaxes(**common_x)
    scatter_fig.update_yaxes(range=yranges['yrange_log2'],
                             title='log2 ratio',
                             title_font=dict(size=12, color='#475569', weight=600),
                             **common_y_style)
    scatter_fig.update_layout(height=320, meta=scatter_meta, **common_layout)
    panes.append({'track': 'scatter', 'fig': scatter_fig, 'height': 320,
                  'label': 'log2 ratio'})

    # ── mapping quality (FORMAT/MQS) ────────────────────────────────
    if show_mapq:
        # the track describes the region: every CNVscope record with MQS, no call filter
        recs = sorted((s for s in list(segments_for_chrom) + list(rejected)
                       if not s.get('sv') and s.get('mqs') is not None), key=lambda s: s['start'])
        # 'hv' step line: a point at each record's start, a closing point where a run of adjoining records ends
        step_x, step_y, mid_x, mid_y, pos_end = [], [], [], [], []
        end = None
        for s in recs:
            v = _short_num(s['mqs'])
            if end is not None and s['start'] != end:
                step_x += [end, None]
                step_y += [step_y[-1], None]
            step_x.append(s['start'])
            step_y.append(v)
            end = s['end']
            mid_x.append((s['start'] + s['end']) // 2)
            mid_y.append(v)
            pos_end.append([s['start'] + 1, s['end']])
        if end is not None:
            step_x.append(end)
            step_y.append(step_y[-1])
        mapq_fig = go.Figure()
        if recs:
            mapq_fig.add_trace(go.Scatter(
                x=_obj(step_x), y=_obj(step_y), mode='lines', line=dict(color=MAPQ_COLOR, width=1.5, shape='hv'),
                name='MAPQ', meta='mapq', hoverinfo='skip',
            ))
            # invisible midpoint markers carry the hover; customdata [POS, END] keeps the cards small
            mapq_fig.add_trace(go.Scatter(
                x=_obj(mid_x), y=_obj(mid_y), customdata=_obj(pos_end), mode='markers',
                marker=dict(size=10, opacity=0), name='MAPQ', meta='mapq',
                hovertemplate=(chrom['name'] + ':%{customdata[0]:,d}-%{customdata[1]:,d}'
                               '<br>MAPQ=%{y:.1~f}<extra></extra>'),
            ))
        top = max([60] + mid_y)
        mapq_fig.update_xaxes(**common_x)
        mapq_fig.update_yaxes(range=[-0.05 * top, 1.1 * top], tickmode='array',
                              tickvals=list(range(0, int(top) + 1, 20)), title='MAPQ',
                              title_font=dict(size=12, color='#475569', weight=600),
                              **common_y_style)
        mapq_fig.update_layout(height=100, **common_layout)
        panes.append({'track': 'mapq', 'fig': mapq_fig, 'height': 100,
                      'label': 'mapping quality'})

    # ── HMM posterior heatmap ───────────────────────────────────────
    if has_posterior:
        states = list(range(5))
        # Pass cell EDGES (N+1) instead of centers so heatmap cells align
        # with the actual segment start/end positions. We also break the
        # heatmap at gaps where consecutive segments aren't contiguous,
        # by inserting a NaN column in z.
        x_edges = []
        states_z = [[] for _ in states]
        prev_end = None
        for s in hmm_segments:
            g = s.get('gamma')
            if not g: continue
            if prev_end is None:
                x_edges.append(s['start'])
            elif s['start'] > prev_end:
                # Gap between segments → insert one empty cell spanning the gap
                x_edges.append(s['start'])
                for k in states:
                    states_z[k].append(None)
            # contiguous: prev segment's end edge IS this segment's start edge
            x_edges.append(s['end'])
            for k in states:
                states_z[k].append(g[k] if k < len(g) else 0)
            prev_end = s['end']
        post_fig = go.Figure()
        if x_edges:
            # Plotly: when len(x) == len(z[0]) + 1, x is interpreted as edges.
            post_fig.add_trace(go.Heatmap(
                x=x_edges, y=states, z=states_z,
                colorscale=POSTERIOR_SCALE, zmin=0, zmax=1,
                showscale=False, meta='posterior',
                zsmooth=False, xgap=0, ygap=0,
                hovertemplate='pos=%{x:,.0f}<br>%{y}<br>P=%{z:.3f}<extra></extra>',
            ))
        post_fig.update_xaxes(**common_x)
        post_fig.update_yaxes(tickmode='array', tickvals=states,
                              ticktext=[f'CN={k}' for k in states],
                              title='HMM posterior',
                              title_font=dict(size=12, color='#475569', weight=600),
                              showgrid=False, **common_y_style)
        post_fig.update_layout(height=140, **common_layout)
        panes.append({'track': 'posterior', 'fig': post_fig, 'height': 140,
                      'label': 'HMM posterior'})

    # ── rolling SD ──────────────────────────────────────────────────
    if show_variance:
        var_fig = go.Figure()
        var_fig.add_trace(go.Scattergl(
            x=[], y=[], mode='lines',
            line=dict(color='#818cf8', width=1.25),  # indigo-400, softened
            opacity=0.75,
            name='SD', meta='variance',
            hovertemplate='pos=%{x:,.0f}<br>SD=%{y:.3f}<extra></extra>',
        ))
        var_fig.update_xaxes(**common_x)
        var_fig.update_yaxes(title='rolling SD',
                             title_font=dict(size=12, color='#475569', weight=600),
                             **common_y_style)
        var_fig.update_layout(height=110, **common_layout)
        panes.append({'track': 'variance', 'fig': var_fig, 'height': 110,
                      'label': 'rolling SD'})

    # ── dark fraction ───────────────────────────────────────────────
    if show_mq0:
        mq0_fig = go.Figure()
        # one WebGL area: a bar per bin meant tens of thousands of SVG nodes and stalled every zoom
        mq0_fig.add_trace(go.Scattergl(
            x=[], y=[], mode='lines', fill='tozeroy',
            line=dict(color='#fbbf24', width=1), fillcolor='#fbbf24',
            name='dark fraction', meta='mq0',
            hovertemplate='pos=%{x:,.0f}<br>dark=%{y:.2f}<extra></extra>',
        ))
        mq0_fig.update_xaxes(**common_x)
        mq0_fig.update_yaxes(range=[0, 1], title='dark frac',
                             title_font=dict(size=12, color='#475569', weight=600),
                             **common_y_style)
        mq0_fig.update_layout(height=100, **common_layout)
        panes.append({'track': 'mq0', 'fig': mq0_fig, 'height': 100,
                      'label': 'dark fraction'})

    # ── segment ribbon ──────────────────────────────────────────────
    rib_fig = go.Figure()
    # Two lanes (CNVscope above, SV-derived below) when the VCF mixes sources
    lanes = {False: 0.68, True: 0.32} if has_sv else {False: 0.5, True: 0.5}
    lane_bands = [(0.56, 0.80), (0.20, 0.44)] if has_sv else [(0.35, 0.65)]
    for y0, y1 in lane_bands:
        rib_fig.add_shape(type='rect', x0=0, x1=L, y0=y0, y1=y1,
                          fillcolor='rgba(148,163,184,0.18)',
                          line=dict(width=0), layer='below')
    # Bar trace per (kind, source, filtered); non-PASS segments (e.g. SVdup) are dimmed
    groups = {(kind, sv, failing): dict(base=[], size=[], mid=[], idx=[], hover=[])
              for failing in (False, True) for sv in (False, True) for kind in KIND_COLORS}
    for i, s in enumerate(segments_for_chrom):
        kind = segment_kind(s, cn0)
        if kind is None: continue
        sv = bool(s.get('sv'))
        g = groups[(kind, sv, s['filter'] != 'PASS')]
        g['base'].append(s['start'])
        g['size'].append(s['end'] - s['start'])
        g['mid'].append((s['start'] + s['end']) // 2)
        g['idx'].append(i)
        g['hover'].append(segment_hover(chrom['name'], s, cn0, has_sv))
    bar_width = 0.14 if has_sv else 0.2  # lane axis is [0, 1]: the old 12 / 14 px ribbon lines
    for (kind, sv, failing), g in groups.items():
        if not g['idx']: continue
        name = kind + (' (SV)' if sv else '') + (' filtered' if failing else '')
        lane = [lanes[sv]] * len(g['idx'])
        meta = 'ribbon_sv' if sv else 'ribbon'
        rib_fig.add_trace(go.Bar(
            orientation='h', base=g['base'], x=g['size'], y=lane, width=bar_width,
            customdata=g['idx'], marker=dict(color=KIND_COLORS[kind], line=dict(width=0)),
            opacity=0.3 if failing else 0.8,
            hovertext=g['hover'], hoverinfo='text', name=name, meta=meta,
        ))
        # bar hover needs the cursor inside the bar: invisible midpoint markers keep sub-pixel calls hoverable
        rib_fig.add_trace(go.Scatter(
            x=g['mid'], y=lane, mode='markers', marker=dict(size=8, opacity=0),
            customdata=g['idx'], hovertext=g['hover'], hoverinfo='text', name=name, meta=meta,
        ))
    rib_fig.update_xaxes(**common_x)
    if has_sv:
        rib_fig.update_yaxes(range=[0, 1], tickmode='array', tickvals=[0.68, 0.32],
                             ticktext=['CNVscope', 'SV'],
                             tickfont=dict(size=10, color='#64748b'),
                             showgrid=False, zeroline=False, title='segments',
                             title_font=dict(size=11))
    else:
        rib_fig.update_yaxes(range=[0, 1], showticklabels=False, showgrid=False,
                             zeroline=False, title='segments',
                             title_font=dict(size=11))
    rib_h = 120 if has_sv else 100
    rib_fig.update_layout(height=rib_h, barmode='overlay', **common_layout)
    panes.append({'track': 'ribbon', 'fig': rib_fig, 'height': rib_h,
                  'label': 'segment ribbon'})

    return panes


HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>CNVscope report &middot; {{ sample }}</title>
<style>
  :root {
    --bg:           #eef2f7;
    --bg-card:      #ffffff;
    --bg-subtle:    #f8fafc;
    --text-1:       #0f172a;
    --text-2:       #475569;
    --text-3:       #94a3b8;
    --border:       rgba(15, 23, 42, 0.06);
    --border-strong:rgba(15, 23, 42, 0.10);
    --shadow-sm:    0 1px 2px rgba(15, 23, 42, 0.04);
    --shadow:       0 1px 2px rgba(15, 23, 42, 0.04),
                    0 2px 6px rgba(15, 23, 42, 0.05);
    --shadow-md:    0 4px 16px rgba(15, 23, 42, 0.08), 0 2px 4px rgba(15, 23, 42, 0.04);
    --accent:       #4f46e5;
    --accent-hover: #4338ca;
    --accent-soft:  #eef2ff;
    --accent-text:  #3730a3;
    --gain:         #e11d48;
    --loss:         #4f46e5;
    --hom-del:      #881337;
    --amp:          #be123c;
    --radius:       12px;
    --radius-sm:    8px;
    --radius-pill:  999px;
    --topbar-h:     64px;
  }
  * { box-sizing: border-box; }
  html { -webkit-font-smoothing: antialiased; -moz-osx-font-smoothing: grayscale; }
  body {
    margin: 0; padding: 0;
    font-family: -apple-system, BlinkMacSystemFont, "Inter", "Segoe UI",
      Roboto, "Helvetica Neue", Arial, sans-serif;
    font-feature-settings: 'ss01', 'cv11';
    background: var(--bg); color: var(--text-1); font-size: 14px;
    line-height: 1.5;
  }
  .topbar {
    display: flex; align-items: center; gap: 14px;
    background: rgba(255, 255, 255, 0.78);
    -webkit-backdrop-filter: saturate(180%) blur(14px);
            backdrop-filter: saturate(180%) blur(14px);
    border-bottom: 1px solid var(--border);
    box-shadow: 0 1px 0 rgba(15,23,42,0.02);
    padding: 12px 24px;
    position: sticky; top: 0; z-index: 100;
    height: 64px;
  }
  .topbar img { height: 36px; width: auto; }
  .topbar .title { font-size: 18px; font-weight: 600; color: var(--text-1); letter-spacing: -0.015em; }
  {%- if internal %}
  .topbar .internal-tag {
    display: inline-flex; align-items: center;
    margin-left: 8px; padding: 3px 11px;
    background: #fef3c7; color: #92400e;
    font-size: 12px; font-weight: 700; letter-spacing: 0.04em; text-transform: uppercase;
    border-radius: var(--radius-pill);
  }
  {%- endif %}
  .topbar .sample {
    display: inline-flex; align-items: center;
    margin-left: 6px;
    padding: 3px 11px;
    background: var(--accent-soft); color: var(--accent-text);
    font-size: 13px; font-weight: 600;
    border-radius: var(--radius-pill);
    letter-spacing: -0.005em;
  }
  .layout {
    display: grid; grid-template-columns: 260px 1fr;
    min-height: calc(100vh - var(--topbar-h));
  }
  .sidebar {
    background: var(--bg-card);
    border-right: 1px solid var(--border);
    padding: 20px 18px;
    position: sticky; top: var(--topbar-h);
    height: calc(100vh - var(--topbar-h));
    overflow-y: auto;
  }
  .sidebar h2 {
    font-size: 11px; font-weight: 600; color: var(--text-3);
    text-transform: uppercase; letter-spacing: 0.05em;
    margin: 0 0 14px 0;
  }
  .sidebar .group { margin-bottom: 18px; }
  .sidebar .group-label {
    font-size: 11px; font-weight: 600; color: var(--text-3);
    text-transform: uppercase; letter-spacing: 0.05em;
    margin: 22px 0 12px 0;
    padding-top: 16px;
    border-top: 1px solid var(--border);
  }
  .sidebar label {
    display: block; font-size: 12px; color: var(--text-2);
    margin-bottom: 6px; font-weight: 500;
  }
  .sidebar input[type=text],
  .sidebar input[type=number],
  .sidebar select {
    width: 100%; padding: 9px 11px;
    border: 1px solid var(--border); border-radius: var(--radius-sm);
    background: var(--bg-card); color: var(--text-1);
    font-size: 14px; font-family: inherit;
    transition: border-color 0.15s, box-shadow 0.15s;
  }
  .sidebar input:focus, .sidebar select:focus {
    outline: none; border-color: var(--accent);
    box-shadow: 0 0 0 3px var(--accent-soft);
  }
  .sidebar input[type=checkbox] {
    margin-right: 8px; accent-color: var(--accent);
    cursor: pointer; width: 14px; height: 14px;
  }
  .sidebar .row { display: flex; align-items: center; margin-bottom: 7px; }
  .sidebar .row.sub-row { padding-left: 22px; }
  .sidebar .row label {
    margin-bottom: 0; cursor: pointer; color: var(--text-1);
    font-weight: 400; font-size: 14px;
  }
  .sidebar .level-row { padding-left: 22px; margin: 2px 0 9px; }
  .sidebar .level-row label { margin-bottom: 4px; }
  .sidebar .level-row select { padding: 5px 8px; font-size: 13px; }
  .sidebar button {
    width: 100%; padding: 9px 14px;
    background: var(--accent); color: #fff;
    border: 1px solid transparent;
    border-radius: var(--radius-sm);
    cursor: pointer; font-size: 14px; font-weight: 500;
    font-family: inherit; line-height: 1.5;
    transition: background 0.15s, transform 0.05s;
  }
  .sidebar button:hover { background: var(--accent-hover); }
  .sidebar button:active { transform: translateY(1px); }
  .sidebar button.secondary {
    background: transparent; color: var(--text-2);
    border: 1px solid var(--border);
  }
  .sidebar button.secondary:hover { background: var(--bg); color: var(--text-1); }
  .sidebar .btn-row {
    display: grid; grid-template-columns: 2fr 1fr; gap: 6px;
  }
  .sidebar .jump-row {
    display: grid; grid-template-columns: 1fr auto; gap: 6px;
  }
  .sidebar .jump-row button { padding: 9px 16px; }
  .content { padding: 24px 28px; max-width: 1600px; }
  header {
    background: var(--bg-card);
    box-shadow: var(--shadow);
    padding: 20px 24px;
    border-radius: var(--radius);
    margin-bottom: 16px;
    display: grid;
    grid-template-columns: repeat(5, 1fr);
    gap: 0;
  }
  header .field {
    font-size: 13px;
    padding: 10px 20px;
    border-right: 1px solid var(--border);
  }
  header .field:nth-child(5n) { border-right: 0; }
  header .field:first-child { padding-left: 0; }
  header .field:nth-child(5n+1) { padding-left: 0; }
  header .field .label {
    color: var(--text-3); text-transform: uppercase;
    font-size: 11px; letter-spacing: 0.05em; font-weight: 600;
    margin-bottom: 7px;
  }
  header .field .value {
    font-size: 20px; font-weight: 600; color: var(--text-1);
    letter-spacing: -0.015em; line-height: 1.2;
  }
  header .field .sub {
    font-size: 12px; color: var(--text-3); margin-top: 1px;
    font-weight: 400; line-height: 1.3;
  }
  .panel {
    background: var(--bg-card);
    box-shadow: var(--shadow);
    border-radius: var(--radius);
    margin-bottom: 14px; padding: 14px 16px;
    scroll-margin-top: 72px;
  }
  .panel.chrom-panel { padding: 12px 16px 8px 16px; }
  .panel.hidden { display: none; }
  .chrom-title {
    display: flex; align-items: baseline; flex-wrap: wrap; gap: 12px;
    font-weight: 600; color: var(--text-1);
    font-size: 22px; letter-spacing: -0.02em;
    padding: 4px 4px 14px 4px;
  }
  .chrom-title .chip {
    display: inline-flex; align-items: center;
    padding: 3px 10px;
    background: var(--bg);
    color: var(--text-2);
    font-size: 12px; font-weight: 500;
    letter-spacing: 0; line-height: 1.4;
    border-radius: var(--radius-pill);
  }
  .chrom-title .chip strong {
    color: var(--text-1); font-weight: 600; margin-left: 4px;
  }
  .track-pane { padding: 0; margin: 0; }
  .track-pane + .track-pane:not([data-hidden]) { margin-top: 8px; }
  .track-pane[data-hidden] { display: none; }
  .chrom-jumpbar {
    display: flex; flex-wrap: wrap; gap: 6px;
    margin-bottom: 14px;
    padding: 4px 2px;
  }
  .chrom-jumpbar a {
    padding: 5px 14px;
    text-decoration: none; color: var(--text-2);
    background: var(--bg-card);
    border-radius: var(--radius-pill);
    font-size: 13px; font-weight: 500;
    box-shadow: var(--shadow-sm);
    transition: all 0.15s;
  }
  .chrom-jumpbar a:hover {
    background: var(--accent-soft); color: var(--accent-text);
  }
  .chrom-jumpbar a.active {
    background: var(--accent); color: #fff;
    box-shadow: 0 1px 3px rgba(79, 70, 229, 0.35);
  }
  details.legend, details.cmdline {
    background: var(--bg-card);
    box-shadow: var(--shadow);
    border-radius: var(--radius);
    margin-bottom: 14px;
    padding: 14px 18px;
  }
  details.legend summary, details.cmdline summary {
    cursor: pointer; font-weight: 600; color: var(--text-1);
    user-select: none; font-size: 14px;
    list-style: none;
  }
  details.legend summary::-webkit-details-marker,
  details.cmdline summary::-webkit-details-marker { display: none; }
  details.legend summary::before, details.cmdline summary::before {
    content: '▸'; display: inline-block; margin-right: 8px;
    color: var(--text-3); transition: transform 0.15s;
    font-size: 11px;
  }
  details[open].legend summary::before, details[open].cmdline summary::before {
    transform: rotate(90deg);
  }
  details.legend dl {
    margin: 14px 0 0 0; padding: 0;
    display: grid; grid-template-columns: minmax(160px, 200px) 1fr;
    column-gap: 18px; row-gap: 12px;
  }
  details.legend dt {
    font-weight: 600; color: var(--text-1); font-size: 13px;
    padding-left: 10px;
    border-left: 2px solid var(--accent);
    line-height: 1.6;
  }
  details.legend dd {
    margin: 0; color: var(--text-2);
    font-size: 13px; line-height: 1.6;
  }
  details.legend .swatch {
    display: inline-block; width: 11px; height: 11px;
    border-radius: 3px; vertical-align: middle;
    margin: 0 4px 1px 0; box-shadow: var(--shadow-sm);
  }
  details.cmdline pre {
    white-space: pre-wrap; word-break: break-all;
    color: var(--text-2); margin: 12px 0 0 0;
    padding: 14px 16px; background: var(--bg);
    border-radius: var(--radius-sm); font-size: 12.5px;
    font-family: ui-monospace, "SF Mono", Menlo, Monaco, Consolas, monospace;
    line-height: 1.6;
  }
  .footer {
    color: var(--text-2); font-size: 13px;
    padding: 32px 0 24px 0; text-align: center;
    margin-top: 36px;
    background: linear-gradient(to bottom,
                                rgba(15,23,42,0.0) 0%,
                                rgba(15,23,42,0.02) 100%);
    border-top: 1px solid var(--border);
  }
  .footer a {
    color: var(--accent); text-decoration: none; font-weight: 500;
  }
  .footer a:hover { text-decoration: underline; }
  .footer .muted { color: var(--text-3); display: block; margin-top: 4px; }
</style>
{{ plotly_js }}
</head>
<body>
<div class="topbar">
  {% if logo_uri %}<img src="{{ logo_uri }}" alt="Sentieon">{% endif %}
  <div class="title">{% if not logo_uri %}Sentieon {% endif %}CNVscope report <span class="sample">{{ sample }}</span>{% if internal %}<span class="internal-tag">Internal</span>{% endif %}</div>
</div>
<div class="layout">
<aside class="sidebar">
  <h2>Navigation</h2>

  <div class="group">
    <label>Chromosome</label>
    <select id="chrom-select">
      {% for c in chroms %}<option value="{{ c.name }}">{{ c.name }}</option>{% endfor %}
    </select>
  </div>

  <div class="group">
    <label>Jump to region (chr:start-end)</label>
    <div class="jump-row">
      <input type="text" id="jump-input" placeholder="chr7:50000000-60000000">
      <button id="jump-btn">Go</button>
    </div>
  </div>

  <div class="group-label">Y axis</div>
  <div class="group">
    <label>Scale</label>
    <select id="y-mode">
      <option value="linear" selected>linear ratio (r)</option>
      <option value="log2">log<sub>2</sub> ratio</option>
    </select>
  </div>

  <div class="group">
    <label>Half-range (&plusmn;)</label>
    <input type="number" id="y-range" value="3" min="0.5" max="10" step="0.1">
  </div>

  <div class="group-label">Filters</div>
  <div class="group">
    <label>Min segment size (bp)</label>
    <input type="number" id="min-size" value="0" min="0" step="100">
  </div>

  <div class="group">
    <label>Min QUAL</label>
    <input type="number" id="min-qual" value="0" min="0" step="1">
  </div>

  <div class="group">
    <label>FILTER status</label>
    <select id="filter-mode">
      <option value="all">show all</option>
      <option value="pass" selected>PASS only</option>
      <option value="failing">failing only</option>
    </select>
  </div>

  <div class="group">
    <label>Event type</label>
    <select id="event-type">
      <option value="all">all events</option>
      <option value="gain">gains only</option>
      <option value="loss">losses only</option>
    </select>
  </div>
  {% if has_sv %}
  <div class="group">
    <label>Source</label>
    <select id="source-mode">
      <option value="all" selected>all sources</option>
      <option value="cnv">CNVscope only</option>
      <option value="sv">SV-derived only</option>
    </select>
  </div>
  {% endif %}

  <div class="group-label">Tracks</div>
  <div class="group">
    <div class="row"><label><input type="checkbox" class="track-toggle" data-track="scatter" checked>log<sub>2</sub> ratio + segments</label></div>
    {% if has_rejected %}
    <div class="row sub-row"><label><input type="checkbox" id="show-rejected" checked>rejected segments</label></div>
    <div class="level-row"><label for="rejected-level">grey lines at</label><select id="rejected-level">
      <option value="cn" selected>model CN</option>
      <option value="depth">measured depth</option>{% if rejected_has_hmm %}
      <option value="hmm">HMM CN</option>{% endif %}
    </select></div>
    {% endif %}
    {% if show_mapq %}
    <div class="row"><label><input type="checkbox" class="track-toggle" data-track="mapq" checked>mapping quality</label></div>
    {% endif %}
    {% if show_posterior_legend %}
    <div class="row"><label><input type="checkbox" class="track-toggle" data-track="posterior" checked>HMM posterior</label></div>
    {% endif %}
    {% if show_variance_legend %}
    <div class="row"><label><input type="checkbox" class="track-toggle" data-track="variance" checked>rolling SD</label></div>
    {% endif %}
    {% if show_mq0_legend %}
    <div class="row"><label><input type="checkbox" class="track-toggle" data-track="mq0" checked>dark fraction</label></div>
    {% endif %}
    <div class="row"><label><input type="checkbox" class="track-toggle" data-track="ribbon" checked>segment ribbon</label></div>
  </div>

  <div class="group btn-row">
    <button id="apply-btn">Apply</button>
    <button id="reset-btn" class="secondary">Reset</button>
  </div>

  <div class="group-label">Generated</div>
  <div class="group" style="margin-top:0;">
    <div style="font-size:11px; color:var(--text-3);">{{ generated }}</div>
  </div>
</aside>

<main class="content">

<header>
  <div class="field"><div class="label">Sex</div><div class="value">{{ sex }}</div></div>
  <div class="field"><div class="label">Ploidy</div><div class="value">{{ sample_cn_neutral }}n</div></div>
  <div class="field"><div class="label">Mode</div><div class="value">mode={{ mode }}</div></div>
  <div class="field">
    <div class="label">Reference</div>
    <div class="value">{{ assembly }}</div>
    <div class="sub">{{ n_contigs }} contigs &middot; {{ ref_length_gb }} Gb</div>
  </div>
  <div class="field">
    <div class="label">Probes</div>
    <div class="value">{{ n_probes }}</div>
    <div class="sub">{{ n_dark }} dark ({{ dark_pct }}%)</div>
  </div>
  <div class="field"><div class="label">Segments</div><div class="value">{{ n_segments }}</div>{% for line in segment_lines %}<div class="sub">{{ line }}</div>{% endfor %}</div>
  <div class="field">
    <div class="label">Events</div>
    <div class="value">{{ n_events }}</div>
    <div class="sub">{{ n_gain }} gain &middot; {{ n_loss }} loss</div>
  </div>
  <div class="field"><div class="label">Calls / Mb</div><div class="value">{{ calls_per_mb }}</div></div>
  <div class="field"><div class="label">Median log2r</div><div class="value">{{ median_log2r }}</div></div>
  <div class="field"><div class="label">MAPD</div><div class="value">{{ mapd }}</div></div>
</header>

{% if cmdline %}
<details class="cmdline">
  <summary>Sentieon command line</summary>
  <pre>{{ cmdline }}</pre>
</details>
{% endif %}

<details class="legend">
  <summary>How to read this report</summary>
  <dl>
    <dt>Summary counts</dt>
    <dd>Counts cover the chromosomes in the report. Segments counts every record in the VCF{% if segment_lines %}, split into PASS calls, filtered calls (a FILTER other than PASS{% if has_sv %}, such as SVdup or NoDepth{% endif %}){% if has_rejected %}, segments rejected by the CNV model{% endif %}{% if has_sv %} and SV-derived calls{% endif %}{% endif %}. Events and Calls / Mb count the PASS calls that change copy number.</dd>

    <dt>Genome ideogram</dt>
    <dd>{% if has_cn %}One bar per called event across all chromosomes. Bar height = |log<sub>2</sub>(CN/cn_neutral)|, capped at &plusmn;2.{% else %}One bar per significant segment across all chromosomes. Bar height = |segment log<sub>2</sub> ratio &minus; chromosome baseline|, capped at &plusmn;2.{% endif %} <span class="swatch" style="background:#e11d48;"></span>gain above the line, <span class="swatch" style="background:#4f46e5;"></span>loss below. Click a bar to open its chromosome.</dd>

    <dt>log<sub>2</sub> ratio (scatter)</dt>
    <dd>Per-bin median of log<sub>2</sub>(normalized depth / sample neutral). 0 = neutral; +1 = 2&times; depth ({% if has_cn %}e.g., CN=4 in a diploid{% else %}twice the expected coverage{% endif %}); &minus;1 = half depth{% if has_cn %} (CN=1){% endif %}.{% if has_cn %} The dotted reference line is at log<sub>2</sub>(chrom_cn_neutral / sample_cn_neutral) &mdash; non-zero on haploid chromosomes (e.g. chrX/Y in males).{% endif %} Point shade: <span class="swatch" style="background:#475569;"></span>well-mapped &rarr; <span class="swatch" style="background:#f59e0b;"></span>mostly dark.</dd>

    <dt>Segment lines (black)</dt>
    <dd>{% if has_cn and not has_log2r_only %}The caller's per-segment value at log<sub>2</sub>(CN/sample_cn_neutral).{% elif has_log2r_only and not has_cn %}The caller's per-segment mean log<sub>2</sub> ratio. No copy-number call is attached at this stage &mdash; run CNVModelApply downstream for CN.{% else %}The caller's per-segment value: log<sub>2</sub>(CN/sample_cn_neutral) when a CN call exists, otherwise the segment's mean log<sub>2</sub> ratio.{% endif %}</dd>

    {% if has_rejected %}
    <dt>Rejected segments (grey)</dt>
    <dd>Segments the CNV model rejected (FILTER=MLrejected); CNVModelApply writes all of them with --emit_conf 0. The model gives them the neutral copy number. A thin grey line marks each one, at the level the <em>grey lines at</em> menu under Tracks picks: <em>model CN</em> (the default), log<sub>2</sub>(CN / sample_cn_neutral), which puts a rejected segment on its neutral level{% if rejected_has_hmm %}; <em>measured depth</em>, log<sub>2</sub> of FORMAT/DPS[0]; or <em>HMM CN</em>, CNVscope's HMM state, log<sub>2</sub>(HMMCN / sample_cn_neutral){% else %}, or <em>measured depth</em>, log<sub>2</sub> of FORMAT/DPS[0]{% endif %}. A segment without the chosen value has no line, and zero copies or zero depth sit below the log<sub>2</sub> axis. Hover a line's midpoint for P(neutral), the model's probability of the neutral copy number, and the segment's other values. Rejected segments are not calls: they have no ribbon or overview bar, the Events and Calls / Mb counts leave them out, and the call filters do not apply to them. The <em>rejected segments</em> checkbox under Tracks shows or hides them.</dd>
    {% endif %}

    {% if show_mapq %}
    <dt>Mapping quality</dt>
    <dd>FORMAT/MQS, the segment's mean mapping quality (MAPQ, 0 to 60), drawn as a step line for every CNVscope record in the VCF, whatever its FILTER; with CNVModelApply --emit_conf 0 that is every segment. SV-derived calls have no MQS and no line. Low values mark regions where many reads also align elsewhere, such as segmental duplications, so depth there is less reliable. Hover a segment's midpoint for its location and MAPQ. The call filters and the <em>rejected segments</em> checkbox do not apply to this track.</dd>
    {% endif %}

    {% if show_posterior_legend %}
    <dt>HMM posterior</dt>
    <dd>Per-segment probability of each copy-number state CN=0..4. Darker = more confident. A solid horizontal stripe at one CN row = high-confidence call; vertical smear across rows = ambiguous.</dd>
    {% endif %}

    {% if show_variance_legend %}
    <dt>Rolling SD</dt>
    <dd>Standard deviation of the binned log<sub>2</sub> ratio in a sliding window. High SD = noisy region; calls in spikes are less trustworthy.</dd>
    {% endif %}

    {% if show_mq0_legend %}
    <dt>Dark fraction</dt>
    <dd>Fraction of probes in each bin with no well-mapped reads (mapping-quality dark &mdash; unmappable, or masked by the panel-of-normals). Treat depth signal in dark-heavy regions with caution.</dd>
    {% endif %}

    <dt>Segment ribbon</dt>
    <dd>Bottom strip of each chrom panel. Color = event class: <span class="swatch" style="background:#4f46e5;"></span>loss, <span class="swatch" style="background:#881337;"></span>hom_del, <span class="swatch" style="background:#e11d48;"></span>gain, <span class="swatch" style="background:#be123c;"></span>amp{% if has_cn %} (CN &ge; cn_neutral+2){% else %} (log<sub>2</sub> ratio &gt; +0.9){% endif %}. Dimmed bars are non-PASS records (FILTER in the hover); they appear when FILTER status is &ldquo;show all&rdquo; or &ldquo;failing only&rdquo;. Hover a segment for QUAL, log<sub>2</sub>r{% if show_mapq %}, MAPQ{% endif %}{% if show_posterior_legend %}, LR, CDELTA, ENTROPY, ECOST{% endif %}{% if has_ml_prob %}, ML_PROB{% endif %} and other annotations.</dd>

    {% if has_sv %}
    <dt>SV-derived events</dt>
    <dd>Calls converted from SV-caller INDELs (indel2cnv + combine_cnv, INFO SOURCE=SV_CNV). They occupy the lower <em>SV</em> lane of the ribbon, the lighter ideogram bars, and the dotted <span class="swatch" style="background:#0d9488;"></span> segment lines. CN = local cn_neutral + CNDIFF, where CNDIFF is the change in tandem-repeat unit count reported by the SV caller &mdash; not a read-depth estimate, so the dotted line marks the implied level, and the depth change is spread over the whole repeat array (REFSTART&ndash;REFSTOP in the hover) rather than the record span. The merge sets three FILTER values, all hidden under &ldquo;PASS only&rdquo;: SVdup on CNVscope gains and losses that repeat a converted call (the hover shows DEDUP, the converted interval), NoSV on short CNVscope losses that no converted loss supports (single-end models only), and NoDepth on long converted gains over which CNVscope saw no gain.</dd>
    {% endif %}

    {% if has_cn %}
    <dt>cn_neutral</dt>
    <dd>The expected copy number on a given chromosome &mdash; 2 for autosomes, 1 for chrX/Y in males, 0 for chrY in females. Used as the reference baseline for that chromosome.</dd>
    {% endif %}

    <dt>MAPD</dt>
    <dd>Median Absolute Pairwise Difference: the median absolute difference between the log<sub>2</sub> ratios of neighbouring probes along each chromosome. A noise metric: lower means less noise; about 0.13 for a 30&times; WGS sample with 200 bp probes.</dd>
  </dl>
</details>

<div class="chrom-jumpbar">
  {% for c in chroms %}<a href="#chrom-{{ c.name }}" data-chrom="{{ c.name }}">{{ c.name }}</a>{% endfor %}
</div>

<div class="panel" id="genome-overview">{{ ideogram_div }}</div>

{% for c in chroms %}
<div class="panel chrom-panel" id="chrom-{{ c.name }}" data-chrom="{{ c.name }}" {% if not loop.first %}hidden{% endif %}>
  <div class="chrom-title">
    <span>{{ c.name }}</span>
    <span class="chip">cn_neutral<strong>{{ c.cn_neutral }}</strong></span>
    {% for line in c.segment_lines %}<span class="chip">{{ line }}</span>{% else %}<span class="chip">{{ c.n_segments }} segments</span>{% endfor %}
    <span class="chip" data-role="status">loading…</span>
  </div>
  {% for p in c.panes %}
  <div class="track-pane" data-track="{{ p.track }}">
    <div class="plotly-mount" id="chrom-{{ c.name }}-{{ p.track }}" style="height:{{ p.height }}px;"></div>
  </div>
  {% endfor %}
</div>
{% endfor %}

<div class="footer">
  Questions or issues? Contact <a href="mailto:{{ support_email }}">{{ support_email }}</a>
  <span class="muted">CNVscope viz {{ version }} &middot; generated {{ generated }}</span>
</div>
</main>
</div>

<script>
window.CNV_CONFIG = {{ config_json | safe }};
window.CNV_INLINE = {tiles: {}, panels: {}};
window.cnvTile = function (c, k, t, payload) {
  var key = c + '/' + k + '/' + t;
  if (window.CNV_CONFIG.mode === 'single') window.CNV_INLINE.tiles[key] = payload;
  if (window.__cnvOnTile) window.__cnvOnTile(key, payload);
};
window.cnvPanel = function (c, spec) {
  if (window.CNV_CONFIG.mode === 'single') window.CNV_INLINE.panels[c] = spec;
  if (window.__cnvOnPanel) window.__cnvOnPanel(c, spec);
};
</script>
<!--CNV_DATA-->
{{ viewer_script }}
</body>
</html>
"""


VIEWER_JS = r"""(function () {
  'use strict';

  // Finest level whose visible span fits the point budget (else the coarsest).
  function pickLevel(bins, span, budget) {
    for (var i = 0; i < bins.length; i++) {
      if (span / bins[i] <= budget) return i;
    }
    return bins.length - 1;
  }

  // Inclusive tile index range covering [x0, x1] at one level.
  function tileRange(x0, x1, bin, tilePoints, length, nTiles) {
    var ts = tilePoints * bin;
    var top = Math.max(length - 1, 0);
    var t0 = Math.floor(Math.min(Math.max(x0, 0), top) / ts);
    var t1 = Math.floor(Math.min(Math.max(x1, 0), top) / ts);
    if (t1 < t0) { var s = t0; t0 = t1; t1 = s; }
    return [Math.max(0, Math.min(t0, nTiles - 1)), Math.max(0, Math.min(t1, nTiles - 1))];
  }

  // Delta-decode a tile payload into absolute positions.
  function decodeTile(payload) {
    var dx = (payload && payload.x) || [];
    var x = new Array(dx.length), acc = 0;
    for (var i = 0; i < dx.length; i++) { acc += dx[i]; x[i] = acc; }
    return {x: x, r: (payload && payload.r) || [], d: (payload && payload.d) || [],
            sd: (payload && payload.sd) || null};
  }

  // A call is drawn only if it passes every sidebar filter; kind comes from Python's segment_kind().
  function segmentVisible(s, f) {
    if (s.end - s.start < f.minSize) return false;
    if ((s.qual || 0) < f.minQual) return false;
    if (f.filterMode === 'pass' && s.filter !== 'PASS') return false;
    if (f.filterMode === 'failing' && s.filter === 'PASS') return false;
    if (f.eventType === 'gain' && s.kind !== 'gain' && s.kind !== 'amp') return false;
    if (f.eventType === 'loss' && s.kind !== 'loss' && s.kind !== 'hom_del') return false;
    if (f.sourceMode === 'cnv' && s.sv) return false;
    if (f.sourceMode === 'sv' && !s.sv) return false;
    return true;
  }

  var CALL_KEYS = ['x', 'y', 'base', 'width', 'customdata', 'hovertext'];

  // Points of a call trace whose customdata segment index is kept; a null separator follows its segment.
  function keepPoints(tr, keep) {
    var keys = CALL_KEYS.filter(function (k) { return Array.isArray(tr[k]); });
    var out = {};
    keys.forEach(function (k) { out[k] = []; });
    var on = false;
    (tr.customdata || []).forEach(function (i, j) {
      if (i !== null) on = !!keep[i];
      if (!on) return;
      keys.forEach(function (k) { out[k].push(tr[k][j]); });
    });
    return out;
  }

  // A null break wherever bins are missing, so an area or line never bridges a stretch with no probes.
  function breakGaps(x, y, bin) {
    var ox = [], oy = [];
    for (var i = 0; i < x.length; i++) {
      if (i > 0 && x[i] - x[i - 1] > 1.5 * bin) {
        ox.push(null);
        oy.push(null);
      }
      ox.push(x[i]);
      oy.push(y[i]);
    }
    return {x: ox, y: oy};
  }

  // Report chrom for a typed name, chr prefix optional (chr7 and 7 open the same chrom); null if none.
  function resolveChrom(name, known) {
    if (known[name]) return name;
    var alt = /^chr/i.test(name) ? name.slice(3) : 'chr' + name;
    return known[alt] ? alt : null;
  }

  // Restyle of the rejected-segment traces (meta 'rejected') from a pane's spec: y at the menu's level, the Scale
  // and the checkbox. The traces' own y (log2) is the cn level; layout.meta.rejected_levels holds one y per
  // segment for each other level, spread over each trace's own points. The call filters never apply.
  function rejectedUpdate(pane, level, mode, show) {
    var levels = ((pane && pane.layout && pane.layout.meta) || {}).rejected_levels || {};
    var per = level === 'cn' ? null : (levels[level] || []);   // a level the panel lacks: no line
    var idx = [], ys = [];
    ((pane && pane.data) || []).forEach(function (tr, i) {
      if (tr.meta !== 'rejected') return;
      var own = tr.y || [];
      var k = per && per.length ? own.length / per.length : 1;   // points per segment: 3 on the line, 1 marker
      idx.push(i);
      ys.push(own.map(function (v, j) {
        // the cn level has a y for every segment, so the trace's nulls are the line's separators
        if (per && v !== null && v !== undefined) v = per[Math.floor(j / k)];
        if (v === null || v === undefined) return null;
        return mode === 'linear' ? Math.pow(2, v) : v;
      }));
    });
    return {idx: idx, update: {y: ys, visible: idx.map(function () { return show; })}};
  }

  if (typeof module !== 'undefined' && module.exports) {
    module.exports = {pickLevel: pickLevel, tileRange: tileRange, decodeTile: decodeTile,
                      segmentVisible: segmentVisible, keepPoints: keepPoints, resolveChrom: resolveChrom,
                      rejectedUpdate: rejectedUpdate, breakGaps: breakGaps};
  }
  if (typeof document === 'undefined' || typeof window === 'undefined') return;

  var CFG = window.CNV_CONFIG || {};
  var MANIFEST = CFG.manifest || {chroms: [], bins: []};
  var BINS = MANIFEST.bins || [10000];
  var TILE_POINTS = MANIFEST.tile_points || 8192;
  var POINTS_PER_VIEW = CFG.pointsPerView || MANIFEST.points_per_view || 25000;
  var LOG2_FLOOR = (CFG.log2Floor === undefined) ? -8 : CFG.log2Floor;
  var DATA_ROOT = CFG.dataRoot || MANIFEST.data_root || 'data';
  var TILE_CACHE_MAX = 400;
  var allChroms = CFG.chroms || [];
  var segmentsByChrom = CFG.segments || {};
  // double-click is ours (homeView): Plotly's reset restores the newPlot-time log2 y range
  var plotlyConfig = {displayModeBar: false, responsive: true, doubleClick: false};
  var chromMeta = {};
  (MANIFEST.chroms || []).forEach(function (c) { chromMeta[c.name] = c; });

  // ── tile / panel loading ────────────────────────────────────────
  var inlineTiles = {};              // decoded single-file tiles, never evicted
  var tileCache = new Map();         // LRU of fetched tiles
  var pendingTiles = {};
  var panelCache = {};
  var pendingPanels = {};

  function inject(src, onerror) {
    var s = document.createElement('script');
    s.async = true;
    s.src = src;
    s.onerror = onerror;
    s.onload = function () { if (s.parentNode) s.parentNode.removeChild(s); };
    document.head.appendChild(s);
  }

  function emptyTile() { return {x: [], r: [], d: [], sd: null}; }

  window.__cnvOnTile = function (key, payload) {
    var dec = decodeTile(payload);
    tileCache.set(key, dec);
    while (tileCache.size > TILE_CACHE_MAX) {
      tileCache.delete(tileCache.keys().next().value);
    }
    var waiters = pendingTiles[key];
    if (!waiters) return;
    delete pendingTiles[key];
    waiters.forEach(function (f) { f(dec); });
  };

  window.__cnvOnPanel = function (chrom, spec) {
    panelCache[chrom] = spec;
    var waiters = pendingPanels[chrom];
    if (!waiters) return;
    delete pendingPanels[chrom];
    waiters.forEach(function (f) { f(spec); });
  };

  function loadTile(chrom, k, t) {
    var key = chrom + '/' + k + '/' + t;
    if (inlineTiles[key]) return Promise.resolve(inlineTiles[key]);
    var raw = window.CNV_INLINE && window.CNV_INLINE.tiles[key];
    if (raw) {
      inlineTiles[key] = decodeTile(raw);
      return Promise.resolve(inlineTiles[key]);
    }
    var hit = tileCache.get(key);
    if (hit) { tileCache.delete(key); tileCache.set(key, hit); return Promise.resolve(hit); }
    if (pendingTiles[key]) {
      return new Promise(function (res) { pendingTiles[key].push(res); });
    }
    pendingTiles[key] = [];
    return new Promise(function (res) {
      pendingTiles[key].push(res);
      inject(DATA_ROOT + '/' + chrom + '/L' + k + '/' + t + '.js', function () {
        var w = pendingTiles[key];
        if (!w) return;
        delete pendingTiles[key];
        console.warn('cnvscope_viz: tile load failed ' + key);
        w.forEach(function (f) { f(emptyTile()); });
      });
    });
  }

  function loadPanel(chrom) {
    if (panelCache[chrom]) return Promise.resolve(panelCache[chrom]);
    var inline = window.CNV_INLINE && window.CNV_INLINE.panels[chrom];
    if (inline) { panelCache[chrom] = inline; return Promise.resolve(inline); }
    if (pendingPanels[chrom]) {
      return new Promise(function (res) { pendingPanels[chrom].push(res); });
    }
    pendingPanels[chrom] = [];
    return new Promise(function (res) {
      pendingPanels[chrom].push(res);
      inject(DATA_ROOT + '/' + chrom + '/panel.js', function () {
        var w = pendingPanels[chrom];
        if (!w) return;
        delete pendingPanels[chrom];
        console.warn('cnvscope_viz: panel load failed ' + chrom);
        w.forEach(function (f) { f(null); });
      });
    });
  }

  // ── plot helpers ────────────────────────────────────────────────
  var state = {chrom: null, level: -1, t0: -1, t1: -1, bin: 0, pts: 0,
               token: 0, loading: false};
  var lastArrays = {x: [], r: [], d: [], sd: []};
  var activeSpec = null;
  var ready = false;
  var refreshTimer = null;
  var programmatic = false;
  var filters = readFilters();   // applied sidebar filters; edits wait for Apply
  var ideogramFull = null;       // unfiltered ideogram call arrays, copied before the first filter

  function panelOf(chrom) { return document.getElementById('chrom-' + chrom); }

  function paneGd(panel, track) {
    return panel && panel.querySelector('.track-pane[data-track="' + track + '"] .js-plotly-plot');
  }

  // Purged mounts keep the plotly class: only these still accept relayout/restyle.
  function plotted(gd) { return !!(gd && gd._fullLayout); }

  // Plotly ops are fire-and-forget here; a purge mid-flight rejects the promise.
  function pcall(p) { if (p && p.catch) p.catch(function () {}); }

  function traceIdx(gd, name) {
    if (!gd || !gd.data) return -1;
    for (var i = 0; i < gd.data.length; i++) {
      if (gd.data[i].meta === name) return i;
    }
    return -1;
  }

  function currentYMode() {
    var sel = document.getElementById('y-mode');
    return sel ? sel.value : 'linear';
  }

  function yFromR(r, mode) {
    var out = new Array(r.length);
    for (var i = 0; i < r.length; i++) {
      var v = r[i];
      if (v === null || v === undefined) out[i] = null;
      else if (mode === 'linear') out[i] = v;
      else out[i] = v > 0 ? Math.log2(v) : LOG2_FLOOR;
    }
    return out;
  }

  function probeHover(mode) {
    return mode === 'linear' ? 'pos=%{x:,.0f}<br>r=%{y:.3f}<extra></extra>'
                             : 'pos=%{x:,.0f}<br>log2r=%{y:.2f}<extra></extra>';
  }

  function fmtBin(bin) {
    if (bin >= 1e6) return (bin / 1e6) + ' Mb';
    if (bin >= 1000) return (bin / 1000) + ' kb';
    return bin + ' bp';
  }

  function setStatus(panel, bin, pts) {
    var el = panel && panel.querySelector('[data-role="status"]');
    if (!el) return;
    el.textContent = 'bin ' + fmtBin(bin) + ' · ' +
      (pts === null ? 'loading…' : pts.toLocaleString() + ' pts');
  }

  // ── level-of-detail refresh ─────────────────────────────────────
  function requestRefresh(chrom, x0, x1) {
    if (refreshTimer) clearTimeout(refreshTimer);
    refreshTimer = setTimeout(function () { refresh(chrom, x0, x1, false); }, 100);
  }

  function refresh(chrom, x0, x1, force) {
    var cm = chromMeta[chrom];
    var panel = panelOf(chrom);
    if (!cm || !panel) return;
    if (state.chrom && chrom !== state.chrom) return;  // stale debounce from a chrom we left
    if (!(x1 > x0)) { x0 = 0; x1 = cm.length; }
    var k = Math.min(pickLevel(BINS, x1 - x0, POINTS_PER_VIEW), cm.levels.length - 1);
    var lvl = cm.levels[k];
    var tr = tileRange(x0, x1, lvl.bin, TILE_POINTS, cm.length, lvl.n_tiles);
    if (!force && chrom === state.chrom && k === state.level &&
        tr[0] === state.t0 && tr[1] === state.t1) return;
    state.chrom = chrom; state.level = k; state.t0 = tr[0]; state.t1 = tr[1];
    state.bin = lvl.bin; state.loading = true;
    setStatus(panel, lvl.bin, null);
    var token = ++state.token;
    var wanted = [];
    for (var t = tr[0]; t <= tr[1]; t++) {
      if (lvl.counts[t] > 0) wanted.push(t);
    }
    Promise.all(wanted.map(function (t) { return loadTile(chrom, k, t); })).then(function (tiles) {
      if (token !== state.token) return;
      state.loading = false;
      applyTiles(panel, tiles, lvl.bin);
      [tr[0] - 1, tr[1] + 1].forEach(function (t) {
        if (t >= 0 && t < lvl.n_tiles && lvl.counts[t] > 0) loadTile(chrom, k, t);
      });
    });
  }

  function applyTiles(panel, tiles, bin) {
    var x = [], r = [], d = [], sd = [], haveSd = false;
    tiles.forEach(function (t) {
      x = x.concat(t.x); r = r.concat(t.r); d = d.concat(t.d);
      if (t.sd) { haveSd = true; sd = sd.concat(t.sd); }
      else { sd = sd.concat(new Array(t.x.length).fill(null)); }
    });
    lastArrays = {x: x, r: r, d: d, sd: haveSd ? sd : []};
    var mode = currentYMode();
    var gd = paneGd(panel, 'scatter');
    var pi = traceIdx(gd, 'probes');
    if (pi >= 0) {
      try {
        pcall(Plotly.restyle(gd, {x: [x], y: [yFromR(r, mode)], 'marker.color': [d]}, [pi]));
      } catch (e) { console.error('cnvscope_viz: restyle failed', e); }
    }
    var vgd = paneGd(panel, 'variance');
    var vi = traceIdx(vgd, 'variance');
    if (vi >= 0) {
      try { pcall(Plotly.restyle(vgd, {x: [haveSd ? x : []], y: [haveSd ? sd : []]}, [vi])); } catch (e) {}
    }
    var mgd = paneGd(panel, 'mq0');
    var mi = traceIdx(mgd, 'mq0');
    if (mi >= 0) {
      var dg = breakGaps(x, d, bin);
      try { pcall(Plotly.restyle(mgd, {x: [dg.x], y: [dg.y]}, [mi])); } catch (e) {}
    }
    state.pts = x.length;
    panel.dataset.lod = String(bin);
    setStatus(panel, bin, x.length);
    if (!ready) { ready = true; document.body.dataset.cnvReady = '1'; }
  }

  // ── y-axis mode ─────────────────────────────────────────────────
  // Home y of a pane: the depth track's per-Scale range, else its spec range, else autorange.
  function homeY(track, mode) {
    var lay = (activeSpec && activeSpec[track] && activeSpec[track].layout) || {};
    var r = track === 'scatter'
      ? (lay.meta || {})[mode === 'linear' ? 'yrange_linear' : 'yrange_log2']
      : lay.yaxis && lay.yaxis.range;
    // a copy: drag-zoom writes yaxis.range[0]/[1] in place and must not reach the spec
    return r ? {'yaxis.range': r.slice()} : {'yaxis.autorange': true};
  }

  function applyYMode() {
    var panel = panelOf(state.chrom);
    var gd = paneGd(panel, 'scatter');
    if (!plotted(gd) || !activeSpec) return;
    var mode = currentYMode();
    var meta = (activeSpec.scatter.layout && activeSpec.scatter.layout.meta) || {};
    var pi = traceIdx(gd, 'probes');
    if (pi >= 0) {
      try {
        pcall(Plotly.restyle(gd, {y: [yFromR(lastArrays.r, mode)], hovertemplate: probeHover(mode)}, [pi]));
      } catch (e) { console.error(e); }
    }
    drawCalls();
    drawRejected();
    var upd = homeY('scatter', mode);
    upd['yaxis.title.text'] = mode === 'linear' ? 'ratio' : 'log2 ratio';
    if (gd.layout && gd.layout.shapes && gd.layout.shapes.length) {
      var ny = mode === 'linear' ? Math.pow(2, meta.neutral_log2r) : meta.neutral_log2r;
      upd['shapes[0].y0'] = ny;
      upd['shapes[0].y1'] = ny;
    }
    try { pcall(Plotly.relayout(gd, upd)); } catch (e) {}
  }

  // ── call filters ────────────────────────────────────────────────
  function readFilters() {
    var sourceSel = document.getElementById('source-mode');
    return {minSize: parseInt(document.getElementById('min-size').value, 10) || 0,
            minQual: parseFloat(document.getElementById('min-qual').value) || 0,
            filterMode: document.getElementById('filter-mode').value,
            eventType: document.getElementById('event-type').value,
            sourceMode: sourceSel ? sourceSel.value : 'all'};
  }

  function keepMask(chrom) {
    return (segmentsByChrom[chrom] || []).map(function (s) { return segmentVisible(s, filters); });
  }

  // One restyle giving each call trace (has customdata) of gd the points maskOf(trace) keeps from `full`.
  function restyleCalls(gd, full, maskOf, mapY) {
    if (!plotted(gd) || !full) return;
    var idx = [], subs = [];
    full.data.forEach(function (tr, i) {
      if (!tr.customdata) return;
      var sub = keepPoints(tr, maskOf(tr));
      if (mapY && sub.y) sub.y = sub.y.map(mapY);
      idx.push(i);
      subs.push(sub);
    });
    if (!idx.length) return;
    var upd = {};
    CALL_KEYS.forEach(function (k) {
      // a trace without k gets undefined, which restyle leaves untouched
      if (subs.some(function (s) { return s[k]; })) upd[k] = subs.map(function (s) { return s[k]; });
    });
    try { pcall(Plotly.restyle(gd, upd, idx)); } catch (e) { console.error('cnvscope_viz: restyle failed', e); }
  }

  // Active chrom's calls from the full spec arrays: filtered subset, then the Scale's y transform.
  function drawCalls() {
    var panel = panelOf(state.chrom);
    if (!panel || !activeSpec) return;
    var keep = keepMask(state.chrom);
    var maskOf = function () { return keep; };
    var toLinear = function (v) { return v == null ? null : Math.pow(2, v); };
    restyleCalls(paneGd(panel, 'scatter'), activeSpec.scatter, maskOf,
                 currentYMode() === 'linear' ? toLinear : null);
    restyleCalls(paneGd(panel, 'ribbon'), activeSpec.ribbon, maskOf, null);
  }

  function showRejected() {
    var cb = document.getElementById('show-rejected');
    return !cb || cb.checked;
  }

  function rejectedLevel() {
    var sel = document.getElementById('rejected-level');
    return sel ? sel.value : 'cn';
  }

  function drawRejected() {
    var gd = paneGd(panelOf(state.chrom), 'scatter');
    if (!plotted(gd) || !activeSpec) return;
    var r = rejectedUpdate(activeSpec.scatter, rejectedLevel(), currentYMode(), showRejected());
    if (!r.idx.length) return;
    try { pcall(Plotly.restyle(gd, r.update, r.idx)); } catch (e) { console.error('cnvscope_viz: restyle failed', e); }
  }

  function captureIdeogram() {
    var gd = document.getElementById('ideogram-div');
    if (!gd || !gd.data) return;
    ideogramFull = {data: gd.data.map(function (tr) {
      var c = {meta: tr.meta};
      CALL_KEYS.forEach(function (k) {
        if (Array.isArray(tr[k])) c[k] = tr[k].slice();
      });
      return c;
    })};
  }

  function drawIdeogram() {
    var masks = {};
    restyleCalls(document.getElementById('ideogram-div'), ideogramFull, function (tr) {
      if (!masks[tr.meta]) masks[tr.meta] = keepMask(tr.meta);
      return masks[tr.meta];
    }, null);
  }

  // Grey out chroms in the selector with no non-neutral call passing the applied filters.
  function markChroms() {
    var f = filters;
    var filtering = (f.minSize > 0 || f.minQual > 0 || f.filterMode !== 'all' ||
                     f.eventType !== 'all' || f.sourceMode !== 'all');
    allChroms.forEach(function (c) {
      var opt = document.querySelector('#chrom-select option[value="' + c.name + '"]');
      if (!opt) return;
      var visible = (segmentsByChrom[c.name] || []).some(function (s) {
        return s.kind && segmentVisible(s, f);
      });
      opt.style.color = (filtering && !visible) ? '#bbb' : '';
    });
  }

  function showCalls() {
    drawCalls();
    drawIdeogram();
    markChroms();
  }

  // ── sidebar controls ────────────────────────────────────────────
  function applyFilters() {
    filters = readFilters();
    var yRange = parseFloat(document.getElementById('y-range').value) || 3;
    var gd = paneGd(panelOf(state.chrom), 'scatter');
    if (plotted(gd)) {
      var range = currentYMode() === 'linear'
        ? [Math.max(0, 1 - yRange), 1 + yRange]
        : [-yRange, yRange];
      try { pcall(Plotly.relayout(gd, {'yaxis.range': range})); } catch (e) {}
    }
    showCalls();
  }

  function applyBottomXAxis(panel) {
    // Only the bottom-most visible track shows x-axis tick labels.
    var visGd = [].slice.call(
      panel.querySelectorAll('.track-pane:not([data-hidden]) .js-plotly-plot')).filter(plotted);
    visGd.forEach(function (gd, i) {
      try { pcall(Plotly.relayout(gd, {'xaxis.showticklabels': i === visGd.length - 1})); } catch (e) {}
    });
  }

  function applyTrackVisibility() {
    document.querySelectorAll('.track-toggle').forEach(function (cb) {
      document.querySelectorAll('.track-pane[data-track="' + cb.dataset.track + '"]').forEach(function (pane) {
        if (cb.checked) pane.removeAttribute('data-hidden');
        else pane.setAttribute('data-hidden', '');
      });
    });
    // Plotly figures inside divs that were previously display:none need
    // resize() so they pick up the now-correct container width.
    requestAnimationFrame(function () {
      document.querySelectorAll('.track-pane:not([data-hidden]) .js-plotly-plot').forEach(function (g) {
        if (!plotted(g)) return;
        try { pcall(Plotly.Plots.resize(g)); } catch (e) {}
      });
      document.querySelectorAll('.chrom-panel').forEach(applyBottomXAxis);
    });
  }

  function reset() {
    document.getElementById('y-range').value = 3;
    document.getElementById('min-size').value = 0;
    document.getElementById('min-qual').value = 0;
    document.getElementById('filter-mode').value = 'pass';
    document.getElementById('event-type').value = 'all';
    var sourceSel = document.getElementById('source-mode');
    if (sourceSel) sourceSel.value = 'all';
    document.querySelectorAll('.track-toggle').forEach(function (cb) { cb.checked = true; });
    var rejectedBox = document.getElementById('show-rejected');
    if (rejectedBox) rejectedBox.checked = true;
    var levelSel = document.getElementById('rejected-level');
    if (levelSel) levelSel.value = 'cn';
    applyFilters();
    applyTrackVisibility();
    drawRejected();
  }

  // ── x-axis sync + navigation ────────────────────────────────────
  function setupXSync(panel) {
    var panes = panel.querySelectorAll('.track-pane .js-plotly-plot');
    if (panes.length === 0) return;
    var syncing = false;
    panes.forEach(function (gd) {
      gd.on('plotly_relayout', function (ed) {
        if (syncing || programmatic) return;
        var xr = ed['xaxis.range'];
        if (!xr && ed['xaxis.range[0]'] !== undefined) {
          xr = [ed['xaxis.range[0]'], ed['xaxis.range[1]']];
        }
        var autoRange = !xr && ed['xaxis.autorange'];
        if (!xr && !autoRange) return;
        recordRegion(state.chrom, xr, autoRange);
        syncing = true;
        panes.forEach(function (g2) {
          if (g2 === gd || !plotted(g2)) return;
          try {
            if (autoRange) pcall(Plotly.relayout(g2, {'xaxis.autorange': true}));
            else pcall(Plotly.relayout(g2, {'xaxis.range': xr}));
          } catch (e) {}
        });
        setTimeout(function () { syncing = false; }, 0);
        var cm = chromMeta[state.chrom];
        if (!cm) return;
        if (autoRange) requestRefresh(state.chrom, 0, cm.length);
        else requestRefresh(state.chrom, xr[0], xr[1]);
      });
    });
  }

  // Parse "chrom:start-end" / "chrom" into parts (commas in numbers allowed).
  function parseRegion(v) {
    var m = v.match(/^([^:\s]+)(?::([0-9,]+)(?:-([0-9,]+))?)?$/);
    if (!m) return null;
    return {chrom: m[1],
            start: m[2] ? m[2].replace(/,/g, '') : null,
            end: m[3] ? m[3].replace(/,/g, '') : null};
  }

  // Persist the current region to the URL (?region=) so refresh restores it.
  // replaceState (not the hash) keeps continuous zoom from spamming history.
  function setRegionUrl(region) {
    try {
      var u = new URL(window.location.href);
      u.searchParams.set('region', region);
      history.replaceState(null, '', u);
    } catch (e) {}
  }

  // Mirror a chrom's visible x-range into the jump box + URL. Called from the
  // relayout handlers on every zoom/pan; autoRange (zoom reset) => whole chrom.
  function recordRegion(chrom, xr, autoRange) {
    if (!chrom) return;
    var region = (autoRange || !xr)
      ? chrom
      : chrom + ':' + Math.max(0, Math.round(xr[0])) + '-' + Math.round(xr[1]);
    var ji = document.getElementById('jump-input');
    if (ji) ji.value = region;
    setRegionUrl(region);
  }

  function applyZoom(chrom, xr) {
    var panel = panelOf(chrom);
    var cm = chromMeta[chrom];
    if (!panel || !cm) return;
    // relayout fires plotly_relayout in a microtask: hold the guard past it
    programmatic = true;
    panel.querySelectorAll('.track-pane .js-plotly-plot').forEach(function (gd) {
      if (!plotted(gd)) return;
      try {
        pcall(Plotly.relayout(gd, {'xaxis.range': xr ? [xr[0], xr[1]] : [0, cm.length]}));
      } catch (e) {}
    });
    setTimeout(function () { programmatic = false; }, 0);
    recordRegion(chrom, xr, !xr);
    if (xr) refresh(chrom, xr[0], xr[1], true);
    else refresh(chrom, 0, cm.length, true);
  }

  // Double-click on any pane: whole chrom everywhere, every y axis back to its home range.
  function homeView(chrom) {
    var panel = panelOf(chrom);
    if (chrom !== state.chrom || !panel || !activeSpec) return;
    applyZoom(chrom, null);  // its programmatic guard also covers the y relayouts below
    var mode = currentYMode();
    panel.querySelectorAll('.track-pane').forEach(function (pane) {
      var gd = pane.querySelector('.js-plotly-plot');
      if (!plotted(gd)) return;
      try { pcall(Plotly.relayout(gd, homeY(pane.dataset.track, mode))); } catch (e) {}
    });
  }

  function goToRegion(chrom, startStr, endStr, scroll) {
    chrom = resolveChrom(chrom, chromMeta);
    if (!chrom) return false;
    var start = startStr ? parseInt(startStr, 10) : null;
    var zoom = null;
    if (start !== null) zoom = [start, endStr ? parseInt(endStr, 10) : start + 1000000];
    activateChrom(chrom, scroll, false, zoom);
    return true;
  }

  function jumpTo() {
    var v = document.getElementById('jump-input').value.trim();
    if (!v) return;
    var r = parseRegion(v);
    if (!r) return;
    if (!goToRegion(r.chrom, r.start, r.end, true)) alert('Chromosome not found: ' + r.chrom);
  }

  // ── chrom activation ────────────────────────────────────────────
  function activateChrom(name, scrollIntoView, force, zoomTo) {
    var panel = panelOf(name);
    if (!panel) return;
    if (name === state.chrom && !force) {
      if (zoomTo) applyZoom(name, zoomTo);
      if (scrollIntoView) panel.scrollIntoView({behavior: 'smooth'});
      return;
    }
    if (state.chrom && state.chrom !== name) {
      var gone = state.chrom;
      var prev = panelOf(gone);
      if (prev) {
        prev.setAttribute('hidden', '');
        // purge next task: Plotly chains auto-margin redraws in microtasks off the last op
        var mounts = prev.querySelectorAll('.plotly-mount');
        setTimeout(function () {
          if (state.chrom === gone) return;  // switched back before the purge ran
          mounts.forEach(function (m) { try { Plotly.purge(m); } catch (e) {} });
        }, 0);
      }
    }
    panel.removeAttribute('hidden');
    state.chrom = name;
    state.level = -1; state.t0 = -1; state.t1 = -1;
    activeSpec = null;
    lastArrays = {x: [], r: [], d: [], sd: []};
    updateChromUi(name);
    loadPanel(name).then(function (spec) {
      if (!spec || state.chrom !== name) return;
      activeSpec = spec;
      panel.querySelectorAll('.track-pane').forEach(function (pane) {
        var fig = spec[pane.dataset.track];
        var mount = pane.querySelector('.plotly-mount');
        if (!fig || !mount) return;
        var copy = JSON.parse(JSON.stringify(fig));
        try { pcall(Plotly.newPlot(mount, copy.data, copy.layout, plotlyConfig)); } catch (e) { console.error(e); }
      });
      setupXSync(panel);
      panel.querySelectorAll('.track-pane .js-plotly-plot').forEach(function (gd) {
        gd.on('plotly_doubleclick', function () { homeView(name); });
      });
      applyTrackVisibility();
      applyBottomXAxis(panel);
      applyYMode();
      if (zoomTo) applyZoom(name, zoomTo);
      else applyZoom(name, null);
      if (scrollIntoView) panel.scrollIntoView({behavior: 'smooth'});
    });
  }

  function updateChromUi(name) {
    var sel = document.getElementById('chrom-select');
    if (sel) sel.value = name;
    document.querySelectorAll('.chrom-jumpbar a').forEach(function (a) {
      a.classList.toggle('active', a.dataset.chrom === name);
    });
  }

  // ── wiring ──────────────────────────────────────────────────────
  document.getElementById('apply-btn').addEventListener('click', applyFilters);
  document.getElementById('reset-btn').addEventListener('click', reset);
  document.getElementById('jump-btn').addEventListener('click', jumpTo);
  document.getElementById('jump-input').addEventListener('keydown', function (e) {
    if (e.key === 'Enter') jumpTo();
  });
  document.querySelectorAll('.track-toggle').forEach(function (cb) {
    cb.addEventListener('change', applyTrackVisibility);
  });
  var rejectedBox = document.getElementById('show-rejected');
  if (rejectedBox) rejectedBox.addEventListener('change', drawRejected);
  var levelSel = document.getElementById('rejected-level');
  if (levelSel) levelSel.addEventListener('change', drawRejected);
  var yModeSel = document.getElementById('y-mode');
  if (yModeSel) yModeSel.addEventListener('change', applyYMode);
  var selectEl = document.getElementById('chrom-select');
  if (selectEl) {
    selectEl.addEventListener('change', function () { activateChrom(this.value, false); });
  }
  document.querySelectorAll('.chrom-jumpbar a').forEach(function (a) {
    a.addEventListener('click', function (ev) {
      ev.preventDefault();
      activateChrom(a.dataset.chrom, true);
    });
  });

  // Ideogram subplots are one per chrom, in order: map the clicked axis to a chrom.
  function wireIdeogram() {
    var gd = document.getElementById('ideogram-div');
    if (!gd || !gd.on) return;
    gd.on('plotly_click', function (ev) {
      var p = ev.points && ev.points[0];
      if (!p || !p.xaxis) return;
      var n = parseInt((p.xaxis._id || 'x').slice(1), 10) || 1;
      var c = allChroms[n - 1];
      if (c) activateChrom(c.name, true);
    });
  }

  window.cnvviz = {
    state: function () {
      return {chrom: state.chrom, level: state.level, bin: state.bin,
              t0: state.t0, t1: state.t1, points: state.pts, loading: state.loading,
              cacheSize: tileCache.size, inline: Object.keys(inlineTiles).length};
    },
    pickLevel: pickLevel,
    tileRange: tileRange,
    decodeTile: decodeTile,
    refresh: refresh,
  };

  function whenPlotlyReady(fn) {
    if (window.Plotly) return fn();
    setTimeout(function () { whenPlotlyReady(fn); }, 50);
  }

  function start() {
    wireIdeogram();
    captureIdeogram();
    showCalls();
    // Restore the region from ?region= (set on every zoom) so refresh stays put.
    var restored = false;
    try {
      var region = new URL(window.location.href).searchParams.get('region');
      if (region) {
        var r = parseRegion(region);
        if (r) restored = goToRegion(r.chrom, r.start, r.end, true);
      }
    } catch (e) {}
    if (!restored && allChroms.length > 0) activateChrom(allChroms[0].name, false);
  }

  if (document.readyState === 'complete') whenPlotlyReady(start);
  else window.addEventListener('load', function () { whenPlotlyReady(start); });
})();
"""


class _NumpyEncoder(json.JSONEncoder):
    def default(self, o):
        if isinstance(o, np.ndarray): return o.tolist()
        if isinstance(o, (np.integer,)): return int(o)
        if isinstance(o, (np.floating,)):
            v = float(o)
            return v if math.isfinite(v) else None
        return super().default(o)


DATA_MARKER = '<!--CNV_DATA-->'


REPORT_FILES = {'index.html', 'viewer.js', 'plotly.min.js', 'data'}


def prepare_out_dir(path, force):
    """Create the split-mode output dir, clearing a previous report's data tree."""
    if os.path.isdir(path):
        entries = set(os.listdir(path))
        # a crashed build leaves data/ without a manifest: still ours to overwrite
        ours = (os.path.isfile(os.path.join(path, 'data', 'manifest.json'))
                or entries <= REPORT_FILES)
        if not ours and not force:
            raise SystemExit(f'[cnvscope_viz] {path}: not empty and not a previous report '
                             f'(no data/manifest.json); use --force to overwrite')
        if ours and os.path.isdir(os.path.join(path, 'data')):
            shutil.rmtree(os.path.join(path, 'data'))
    os.makedirs(os.path.join(path, 'data'), exist_ok=True)


def serve_report(out_dir, port, open_browser):
    """Serve a split-mode report over HTTP (file:// works too; this is for sharing)."""
    import functools
    import http.server
    import webbrowser
    if not os.path.isfile(os.path.join(out_dir, 'index.html')):
        raise SystemExit(f'[cnvscope_viz] {out_dir}: no index.html — not a report directory')
    handler = functools.partial(http.server.SimpleHTTPRequestHandler, directory=out_dir)
    httpd = http.server.ThreadingHTTPServer(('127.0.0.1', port), handler)
    url = f'http://127.0.0.1:{port}/index.html'
    log(f'serving {out_dir} at {url} (ctrl-c to stop)')
    if open_browser:
        webbrowser.open(url)
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        log('stopped')


def write_chrom_data(c, segments_for_chrom, sample_cn_neutral, mode, has_sv, args,
                     bins, emit_panel, emit_tile, rejected=(), show_mapq=False):
    """Stream one chrom's tiles and panel spec through the emit_* callbacks.

    Returns (manifest level list, pane metadata list).
    """
    levels = []
    coarse_med = np.zeros(0)
    for k, b in enumerate(bins):
        counts, tiles, med = level_tiles(c, b, args.tile_points, args.smooth, bool(args.variance))
        for t, payload in tiles:
            emit_tile(k, t, payload)
        levels.append({'bin': b, 'n_tiles': int(counts.size),
                       'counts': [int(v) for v in counts]})
        coarse_med = med  # the last level is the coarsest: it fixes the y ranges
    neutral = math.log2(max(c['cn_neutral'], 1) / max(sample_cn_neutral, 1))
    yranges = panel_yranges(coarse_med, neutral, args.y_max)
    panes = fig_chrom_panel(c, segments_for_chrom, sample_cn_neutral, yranges,
                            show_posterior=args.posterior, show_variance=args.variance,
                            show_mq0=args.mq0, mode=mode, has_sv=has_sv, rejected=rejected,
                            show_mapq=show_mapq)
    spec = {}
    for p in panes:
        fig_dict = p['fig'].to_plotly_json()
        spec[p['track']] = {'data': fig_dict.get('data', []),
                            'layout': fig_dict.get('layout', {}),
                            'height': p['height']}
    emit_panel(json.dumps(spec, cls=_NumpyEncoder, separators=(',', ':')))
    return levels, [{'track': p['track'], 'label': p['label'], 'height': p['height']}
                    for p in panes]


def build_report(meta, chroms, segments, mode, ref_info, args, output, split):
    """Build the report at `output` (directory in split mode, .html file otherwise).

    Returns (n_tiles, data_bytes).
    """
    import plotly.io as pio
    import plotly.offline as poff
    import jinja2

    rederive_sv_cn(segments, chroms)
    # Rejected segments are not calls: they ship with the panel data only, never in CNV_CONFIG.segments
    calls = {name: [s for s in ss if not s.get('rejected')] for name, ss in segments.items()}
    rejected = {name: [s for s in ss if s.get('rejected')] for name, ss in segments.items()}

    visible_chroms = select_chroms(chroms, args.min_chrom_probes, args.chroms, args.all_chroms)
    if not visible_chroms:
        raise SystemExit('[cnvscope_viz] no chromosome matched --chroms / --min-chrom-probes')

    qc_totals = {'n_probes': 0, 'n_dark': 0, 'log2r_chunks': []}
    for c in visible_chroms:
        q = chrom_qc(c)
        qc_totals['n_probes'] += q['n_probes']
        qc_totals['n_dark'] += q['n_dark']
        finite = c['log2r'][np.isfinite(c['log2r'])]
        if finite.size:
            qc_totals['log2r_chunks'].append(finite)
    all_log2r = np.concatenate(qc_totals['log2r_chunks']) if qc_totals['log2r_chunks'] else np.zeros(0)
    median_log2r = float(np.median(all_log2r)) if all_log2r.size else float('nan')
    mapd_value = mapd(visible_chroms)
    summary = call_summary(visible_chroms, segments, meta['sample_cn_neutral'])

    visible_chrom_names = {c['name'] for c in visible_chroms}
    has_cn = any(s.get('cn') is not None
                 for chrom, ss in segments.items() if chrom in visible_chrom_names
                 for s in ss)
    has_log2r_only = any(s.get('cn') is None and s.get('log2r') is not None
                         for chrom, ss in segments.items() if chrom in visible_chrom_names
                         for s in ss)
    has_ml_prob = any(s.get('ml_prob') is not None
                      for chrom, ss in segments.items() if chrom in visible_chrom_names
                      for s in ss)
    has_hmm_features = any(s.get('gamma') is not None
                           for chrom, ss in segments.items() if chrom in visible_chrom_names
                           for s in ss)
    show_posterior_legend = bool(args.posterior and has_hmm_features)
    has_sv = any(s.get('sv')
                 for chrom, ss in segments.items() if chrom in visible_chrom_names
                 for s in ss)
    shown_rejected = [s for chrom, ss in rejected.items() if chrom in visible_chrom_names for s in ss]
    has_rejected = bool(shown_rejected)
    rejected_has_hmm = any(s.get('hmmcn') is not None for s in shown_rejected)
    # the mapping-quality track, in every report whose shown CNVscope records carry MQS
    has_mqs = any(s.get('mqs') is not None and not s.get('sv')
                  for chrom, ss in segments.items() if chrom in visible_chrom_names for s in ss)
    rec = summary['records']
    # the plain "N segments" tile and chips unless there is something to split
    split_counts = has_sv or rec['rejected'] > 0 or rec['cnv_filtered'] > 0

    max_len = max((chrom_length(c) for c in visible_chroms), default=1)
    width_chunks = [c['end'].astype(np.int64) - c['pos'].astype(np.int64)
                    for c in visible_chroms if c['pos'].size]
    probe_width = int(np.median(np.concatenate(width_chunks))) if width_chunks else 0
    bins = build_levels(args.bin_sizes, probe_width if args.raw_probes else 0,
                        max_len, args.points_per_view)

    # Data is written first: the manifest counts are only known once the tiles exist.
    data_root = os.path.join(output, 'data') if split else None
    tmp_data = None if split else tempfile.NamedTemporaryFile('w', suffix='.js', delete=False)
    n_tiles = 0
    data_bytes = 0
    chrom_divs = []
    manifest_chroms = []
    for c in visible_chroms:
        name = c['name']
        n_segs = len(segments.get(name, []))
        n_rej = len(rejected.get(name, []))
        vlog(f'building {name} ({c["pos"].size} probes, {n_segs} segments, {n_rej} rejected)')
        cname = json.dumps(name)
        if split:
            for k in range(len(bins)):
                os.makedirs(os.path.join(data_root, name, f'L{k}'), exist_ok=True)
        else:
            tmp_data.write('<script>\n')

        def sink(rel, body):
            nonlocal data_bytes
            if split:
                with open(os.path.join(data_root, rel), 'w') as fp:
                    fp.write(body)
            else:
                tmp_data.write(body)
            data_bytes += len(body)

        def emit_tile(k, t, payload, _n=name, _c=cname):
            nonlocal n_tiles
            n_tiles += 1
            sink(f'{_n}/L{k}/{t}.js', f'cnvTile({_c},{k},{t},{payload});\n')

        def emit_panel(spec, _n=name, _c=cname):
            sink(f'{_n}/panel.js', f'cnvPanel({_c},{spec});\n')

        levels, pane_meta = write_chrom_data(
            c, calls.get(name, []), meta['sample_cn_neutral'], mode, has_sv, args,
            bins, emit_panel, emit_tile, rejected=rejected.get(name, []), show_mapq=has_mqs)
        if not split:
            tmp_data.write('</script>\n')
        manifest_chroms.append({'name': name, 'length': chrom_length(c),
                                'cn_neutral': c['cn_neutral'], 'levels': levels})
        chrom_divs.append({
            'name': name,
            'cn_neutral': c['cn_neutral'],
            'n_segments': n_segs,
            'segment_lines': (segment_lines(record_counts(segments.get(name, [])), has_sv)
                              if split_counts else []),
            'panes': pane_meta,
        })

    manifest = {
        'version': 1,
        'sample': meta['sample'] or 'unknown',
        'tile_points': args.tile_points,
        'points_per_view': args.points_per_view,
        'bins': bins,
        'channels': ['x', 'r', 'd'] + (['sd'] if args.variance else []),
        'log2_floor': LOG2R_ZERO_FLOOR,
        'data_root': 'data',
        'chroms': manifest_chroms,
    }

    vlog(f'building ideogram across {len(visible_chroms)} chroms')
    fig_idea = fig_genome_ideogram(visible_chroms, calls, meta['sample_cn_neutral'])
    ideogram_div = pio.to_html(fig_idea, full_html=False, include_plotlyjs=False,
                               default_width='100%', default_height='180px',
                               div_id='ideogram-div')

    config = {
        'mode': 'split' if split else 'single',
        'dataRoot': 'data',
        'pointsPerView': args.points_per_view,
        'log2Floor': LOG2R_ZERO_FLOOR,
        'chroms': [{'name': c['name'], 'length': chrom_length(c), 'cn_neutral': c['cn_neutral']}
                   for c in visible_chroms],
        'segments': {
            c['name']: [
                {'start': s['start'], 'end': s['end'], 'cn': s['cn'],
                 'kind': segment_kind(s, c['cn_neutral']),
                 'log2r': (None if (s.get('log2r') is None or (isinstance(s.get('log2r'), float) and math.isnan(s['log2r']))) else round(s['log2r'], 3)),
                 'qual': (None if (s['qual'] is None or math.isnan(s['qual'])) else round(s['qual'], 1)),
                 'filter': s['filter'], 'sv': bool(s.get('sv'))}
                for s in calls.get(c['name'], [])
            ] for c in visible_chroms
        },
        'manifest': manifest,
    }

    if args.cdn:
        plotly_js = '<script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>'
    elif split:
        plotly_js = '<script src="plotly.min.js"></script>'
    else:
        plotly_js = f'<script type="text/javascript">{poff.get_plotlyjs()}</script>'

    tmpl = jinja2.Environment(autoescape=False).from_string(HTML_TEMPLATE)
    ref_gb = ref_info['total_length'] / 1e9 if ref_info['total_length'] else 0
    calls_per_mb = summary['calls_per_mb']
    html = tmpl.render(
        sample=meta['sample'] or 'unknown',
        sex=meta['sex'],
        sample_cn_neutral=meta['sample_cn_neutral'],
        mode=mode,
        assembly=ref_info['assembly'],
        n_contigs=f"{ref_info['n_contigs']:,}",
        ref_length_gb=f"{ref_gb:.2f}" if ref_gb else '—',
        calls_per_mb=f"{calls_per_mb:.2f}" if calls_per_mb else '—',
        cmdline=ref_info['cmdline'],
        n_probes=f"{qc_totals['n_probes']:,}",
        n_dark=f"{qc_totals['n_dark']:,}",
        dark_pct=f"{100 * qc_totals['n_dark'] / max(qc_totals['n_probes'], 1):.1f}",
        n_segments=f"{summary['n_segments']:,}",
        segment_lines=segment_lines(rec, has_sv) if split_counts else [],
        n_events=f"{summary['n_events']:,}",
        n_gain=f"{summary['n_gain']:,}",
        n_loss=f"{summary['n_loss']:,}",
        median_log2r=f"{median_log2r:.3f}" if not math.isnan(median_log2r) else '—',
        mapd=f"{mapd_value:.3f}" if not math.isnan(mapd_value) else '—',
        chroms=chrom_divs,
        ideogram_div=ideogram_div,
        plotly_js=plotly_js,
        config_json=json.dumps(config, cls=_NumpyEncoder, separators=(',', ':')),
        viewer_script=('<script src="viewer.js"></script>' if split
                       else f'<script type="text/javascript">{VIEWER_JS}</script>'),
        logo_uri=load_logo_data_uri(),
        support_email=SUPPORT_EMAIL,
        version=__version__,
        generated=time.strftime('%Y-%m-%d %H:%M:%S %Z'),
        show_posterior_legend=show_posterior_legend,
        show_variance_legend=bool(args.variance),
        show_mq0_legend=bool(args.mq0),
        show_mapq=has_mqs,
        has_cn=has_cn,
        has_log2r_only=has_log2r_only,
        has_ml_prob=has_ml_prob,
        has_sv=has_sv,
        has_rejected=has_rejected,
        rejected_has_hmm=rejected_has_hmm,
        internal=bool(args.internal),
    )
    head, tail = html.split(DATA_MARKER, 1)

    if split:
        with open(os.path.join(output, 'data', 'manifest.json'), 'w') as fp:
            json.dump(manifest, fp, separators=(',', ':'))
        with open(os.path.join(output, 'viewer.js'), 'w') as fp:
            fp.write(VIEWER_JS)
        if not args.cdn:
            with open(os.path.join(output, 'plotly.min.js'), 'w') as fp:
                fp.write(poff.get_plotlyjs())
        with open(os.path.join(output, 'index.html'), 'w') as fp:
            fp.write(head)
            fp.write(tail)
    else:
        tmp_data.close()
        with open(output, 'w') as fp:
            fp.write(head)
            with open(tmp_data.name) as src:
                shutil.copyfileobj(src, fp)
            fp.write(tail)
        os.unlink(tmp_data.name)
    return bins, n_tiles, data_bytes


def main():
    ap = argparse.ArgumentParser(
        description='Build an HTML CNVscope report (split directory, or one .html file).',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        usage='%(prog)s VCF PROBES OUTPUT [options] | %(prog)s --serve OUT_DIR',
    )
    ap.add_argument('vcf', nargs='?', help='CNVscope output VCF (.vcf.gz with .tbi)')
    ap.add_argument('probes', nargs='?', help='CNVscope --dump_probes binary')
    ap.add_argument('output', nargs='?',
                    help='output directory (split report) or .html file (single file)')
    ap.add_argument('--serve', metavar='OUT_DIR', default=None,
                    help='serve an existing split report over HTTP instead of building')
    ap.add_argument('--port', type=int, default=8765, help='--serve port')
    ap.add_argument('--no-browser', action='store_true', help='--serve without opening a browser')
    which = ap.add_mutually_exclusive_group()
    which.add_argument('--chroms', default=None,
                       help='comma-separated chroms to include, chr prefix optional '
                            '(default: the autosomes and sex chromosomes)')
    which.add_argument('--all-chroms', action='store_true', default=False,
                       help='include every contig: chrM, unplaced, alt, decoy')
    ap.add_argument('--min-chrom-probes', type=int, default=100,
                    help='skip chroms with fewer than this many probes')
    ap.add_argument('--bin-sizes', default='10000,1000',
                    help='comma-separated bin sizes in bp; sorted ascending into pyramid levels, '
                         'auto-extended x5 until the longest chrom fits --points-per-view')
    ap.add_argument('--points-per-view', type=int, default=25000,
                    help='point budget for the visible window; picks the level')
    ap.add_argument('--tile-points', type=int, default=8192, help='bins per tile file')
    ap.add_argument('--smooth', type=int, default=1,
                    help='rolling median window over binned probes (odd, >=1)')
    ap.add_argument('--y-max', type=float, default=None,
                    help='hard cap on log2r y-axis (default: auto)')
    ap.add_argument('--advanced', action='store_true', default=False,
                    help='add the rolling SD and dark fraction tracks to the log2 ratio scatter, '
                         'mapping quality and segment ribbon')
    # internal use only: --advanced plus the HMM views (posterior, HMM CN level, HMM card fields)
    ap.add_argument('--internal', action='store_true', default=False, help=argparse.SUPPRESS)
    ap.add_argument('--raw-probes', action='store_true', default=False,
                    help='add the median probe width as the finest pyramid level')
    ap.add_argument('--cdn', action='store_true', default=False,
                    help='load plotly.js from CDN instead of shipping it')
    ap.add_argument('--force', action='store_true', default=False,
                    help='overwrite a non-empty output directory that is not a previous report')
    ap.add_argument('--verbose', '-v', action='store_true',
                    help='print per-chromosome progress (off: only opening + closing summary)')
    ap.add_argument('--version', action='version', version=f'cnvscope_viz.py {__version__}')
    args = ap.parse_args()

    global VERBOSE
    VERBOSE = args.verbose
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO,
                        format='[%(asctime)s] %(message)s', datefmt='%H:%M:%S')

    if args.serve:
        serve_report(args.serve, args.port, not args.no_browser)
        return
    if not (args.vcf and args.probes and args.output):
        ap.error('VCF, PROBES and OUTPUT are required (or use --serve OUT_DIR)')

    try:
        args.bin_sizes = sorted({int(p) for p in args.bin_sizes.split(',') if p.strip()})
    except ValueError:
        ap.error(f'--bin-sizes must be a comma-separated list of bp sizes (got {args.bin_sizes!r})')
    if not args.bin_sizes or args.bin_sizes[0] <= 0:
        ap.error('--bin-sizes values must be > 0')
    if args.tile_points < 1 or args.points_per_view < 1:
        ap.error('--tile-points and --points-per-view must be >= 1')

    # --advanced expands into the track flags; --internal adds the HMM posterior
    args.advanced = args.advanced or args.internal
    args.posterior = args.internal
    args.variance = args.advanced
    args.mq0 = args.advanced

    t0 = time.time()
    split = not args.output.endswith('.html')
    if split:
        prepare_out_dir(args.output, args.force)

    meta, chroms = read_probes_binary(args.probes)
    chrom_names = [c['name'] for c in chroms]
    ref_info = read_vcf_reference_info(args.vcf)
    segments, mode = read_segments_vcf(args.vcf, chrom_names, hmm=args.internal)

    kept = {c['name'] for c in select_chroms(chroms, args.min_chrom_probes, args.chroms, args.all_chroms)}
    n_seg_kept = sum(len(v) for k, v in segments.items() if k in kept)
    n_sv_kept = sum(1 for k, v in segments.items() if k in kept for s in v if s.get('sv'))
    n_rej_kept = sum(1 for k, v in segments.items() if k in kept for s in v if s.get('rejected'))
    left = [v for k, v in segments.items() if k not in kept and v]
    notes = [f'{n} {what}' for n, what in ((n_rej_kept, 'rejected'), (n_sv_kept, 'SV-derived')) if n]
    seg_note = f' ({", ".join(notes)})' if notes else ''
    log(f'input: sample={meta["sample"]} sex={meta["sex"]} assembly={ref_info["assembly"]} mode={mode} · '
        f'cnvscope_viz.py {__version__}')
    log(f'       chroms={len(kept)} kept (of {len(chroms)}) · segments={n_seg_kept}{seg_note} · '
        f'output={"split" if split else "single file"} · '
        f'advanced_tracks={"on" if args.advanced else "off"}' + (' · internal' if args.internal else ''))
    if not (args.chroms or args.all_chroms):
        if kept and not any(is_main_chrom(n) for n in kept):
            log('       no autosome or sex chromosome names in the probe dump: showing every contig')
        elif left:
            log(f'       {sum(map(len, left))} segments on {len(left)} other contigs left out '
                f'(--all-chroms adds them)')

    bins, n_tiles, data_bytes = build_report(meta, chroms, segments, mode, ref_info, args,
                                             args.output, split)

    if split:
        total = sum(os.path.getsize(os.path.join(dp, f))
                    for dp, _, fs in os.walk(args.output) for f in fs)
    else:
        total = os.path.getsize(args.output)
    log(f'wrote {args.output} · levels={",".join(str(b) for b in bins)} · tiles={n_tiles} · '
        f'data={data_bytes / (1024 * 1024):.1f} MB · total={total / (1024 * 1024):.1f} MB · '
        f'{time.time() - t0:.1f}s')


if __name__ == '__main__':
    main()
