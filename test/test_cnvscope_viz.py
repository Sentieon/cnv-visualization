#!/usr/bin/env python3
"""Tests for the cnvscope_viz report builder."""

import base64
import importlib.util
import json
import math
import os
import re
import shutil
import struct
import subprocess
import sys

import numpy as np
import pytest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_spec = importlib.util.spec_from_file_location(
    'cnvscope_viz', os.path.join(REPO, 'scripts', 'cnvscope_viz.py'))
viz = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(viz)

NODE = shutil.which('node')
BGZIP = shutil.which('bgzip')
TABIX = shutil.which('tabix')


# ── reference implementations (the loops the vectorized code replaced) ──
def ref_bin_probes(pos, r, mq0, bin_size):
    bidx = pos.astype(np.int64) // bin_size
    out_b, out_m, out_d, out_n = [], [], [], []
    for b in sorted(set(bidx.tolist())):
        m = bidx == b
        vals = r[m]
        vals = vals[np.isfinite(vals)]
        out_b.append(b)
        out_m.append(float(np.median(vals)) if vals.size else float('nan'))
        out_d.append(float(np.mean(mq0[m] == 0)))
        out_n.append(int(m.sum()))
    return np.array(out_b, dtype=np.int64), np.array(out_m), np.array(out_d), np.array(out_n)


def ref_rolling_std(x, window):
    n = len(x)
    out = np.full(n, np.nan)
    h = window // 2
    for i in range(n):
        chunk = x[max(0, i - h):min(n, i + h + 1)]
        chunk = chunk[np.isfinite(chunk)]
        if chunk.size >= 3:
            out[i] = float(np.std(chunk))
    return out


# ── binning ────────────────────────────────────────────────────────────
@pytest.mark.parametrize('seed', [0, 1, 2, 3])
def test_bin_probes_matches_loop(seed):
    rng = np.random.default_rng(seed)
    n = 500
    pos = np.sort(rng.integers(0, 20000, n)).astype(np.uint32)
    r = rng.normal(1.0, 0.3, n)
    r[rng.random(n) < 0.15] = np.nan          # missing probes
    r[rng.random(n) < 0.10] = 0.0             # hom-del probes
    mq0 = (rng.random(n) < 0.2).astype(np.uint8) * 0 + (rng.random(n) > 0.2).astype(np.uint8)
    for bin_size in (100, 512, 4096):
        b, m, d, c = viz.bin_probes(pos, r, mq0, bin_size)
        rb, rm, rd, rc = ref_bin_probes(pos, r, mq0, bin_size)
        assert np.array_equal(b, rb)
        assert np.array_equal(c, rc)
        np.testing.assert_allclose(d, rd, rtol=0, atol=1e-12)
        np.testing.assert_allclose(m, rm, rtol=1e-12, atol=1e-12, equal_nan=True)


def test_bin_probes_edge_cases():
    b, m, d, c = viz.bin_probes(np.zeros(0, dtype=np.uint32), np.zeros(0),
                                np.zeros(0, dtype=np.uint8), 1000)
    assert b.size == 0 and m.size == 0 and d.size == 0 and c.size == 0
    # one all-NaN bin, one bin with an even count, non-contiguous bin indices
    pos = np.array([0, 10, 5000, 5100, 5200, 5300, 90000], dtype=np.uint32)
    r = np.array([np.nan, np.nan, 1.0, 2.0, 3.0, 4.0, 0.0])
    mq0 = np.array([0, 1, 1, 1, 0, 1, 0], dtype=np.uint8)
    b, m, d, c = viz.bin_probes(pos, r, mq0, 1000)
    assert b.tolist() == [0, 5, 90]
    assert np.isnan(m[0])
    assert m[1] == pytest.approx(2.5)          # even count -> mean of the two middles
    assert m[2] == 0.0                         # r=0 hom-del survives
    assert d.tolist() == [0.5, 0.25, 1.0]
    assert c.tolist() == [2, 4, 1]


def test_rolling_std_matches_loop():
    rng = np.random.default_rng(7)
    x = rng.normal(0, 1, 400)
    x[rng.random(400) < 0.2] = np.nan
    x[50:60] = np.nan                           # an all-NaN window
    for w in (3, 5, 11):
        got = viz.rolling_std(x, w)
        ref = ref_rolling_std(x, w)
        assert np.array_equal(np.isnan(got), np.isnan(ref))
        ok = ~np.isnan(ref)
        np.testing.assert_allclose(got[ok], ref[ok], rtol=1e-6, atol=1e-8)


def test_rolling_std_degenerate():
    assert viz.rolling_std(np.zeros(0), 5).size == 0
    assert np.all(np.isnan(viz.rolling_std(np.ones(10), 1)))
    assert viz.rolling_std(np.ones(10), 5)[5] == pytest.approx(0.0)


def test_smooth_median():
    x = np.array([1.0, 5.0, 1.0, 1.0, np.nan, 1.0])
    assert viz.smooth_median(x, 1) is x
    sm = viz.smooth_median(x, 3)
    assert sm[1] == pytest.approx(1.0)          # spike removed
    assert sm.size == x.size


# ── pyramid math ───────────────────────────────────────────────────────
def test_build_levels():
    assert viz.build_levels([10000, 1000], 0, 250000000, 25000) == [1000, 10000]
    assert viz.build_levels([1000, 1000, 5000], 0, 1000, 25000) == [1000, 5000]
    assert viz.build_levels([1000], 200, 1000, 25000) == [200, 1000]
    # extension: 250 Mb / 1000 = 250000 > 25000 -> x5 until it fits
    lv = viz.build_levels([1000], 0, 250000000, 25000)
    assert lv[0] == 1000 and lv[-1] >= 250000000 / 25000
    assert all(lv[i] < lv[i + 1] for i in range(len(lv) - 1))


def test_pick_level():
    bins = [200, 1000, 5000, 25000]
    assert viz.pick_level(bins, 100000, 25000) == 0
    assert viz.pick_level(bins, 50000000, 25000) == 2
    assert viz.pick_level(bins, 10 ** 12, 25000) == len(bins) - 1


def test_tile_range():
    n_tiles = viz.n_tiles_for(1000000, 1000, 100)
    assert n_tiles == 10
    assert viz.tile_range(0, 1000000, 1000, 100, 1000000, n_tiles) == (0, 9)
    assert viz.tile_range(250000, 260000, 1000, 100, 1000000, n_tiles) == (2, 2)
    assert viz.tile_range(-5000, 10 ** 9, 1000, 100, 1000000, n_tiles) == (0, 9)


@pytest.mark.skipif(NODE is None, reason='node not installed')
def test_js_matches_python(tmp_path):
    js = tmp_path / 'viewer.js'
    js.write_text(viz.VIEWER_JS)
    driver = tmp_path / 'drive.js'
    driver.write_text("""
const v = require('%s');
const cases = JSON.parse(process.argv[2]);
const out = cases.map(function (c) {
  return [v.pickLevel(c.bins, c.span, c.budget),
          v.tileRange(c.x0, c.x1, c.bin, c.tp, c.len, c.nt)];
});
console.log(JSON.stringify(out));
""" % js)
    rng = np.random.default_rng(11)
    bins = [200, 1000, 5000, 25000]
    cases, expect = [], []
    for _ in range(200):
        length = int(rng.integers(10 ** 5, 250 * 10 ** 6))
        span = float(rng.integers(100, length))
        x0 = float(rng.integers(0, length))
        x1 = x0 + span
        k = viz.pick_level(bins, span, 25000)
        nt = viz.n_tiles_for(length, bins[k], 8192)
        cases.append({'bins': bins, 'span': span, 'budget': 25000,
                      'x0': x0, 'x1': x1, 'bin': bins[k], 'tp': 8192,
                      'len': length, 'nt': nt})
        expect.append([k, list(viz.tile_range(x0, x1, bins[k], 8192, length, nt))])
    got = json.loads(subprocess.check_output(
        [NODE, str(driver), json.dumps(cases)], text=True))
    assert got == expect


@pytest.mark.skipif(NODE is None, reason='node not installed')
def test_js_decode_tile(tmp_path):
    js = tmp_path / 'viewer.js'
    js.write_text(viz.VIEWER_JS)
    driver = tmp_path / 'drive.js'
    driver.write_text("""
const v = require('%s');
const t = v.decodeTile({x: [100, 50, 50], r: [1.0, null, 0.5], d: [0, 0.5, 1]});
console.log(JSON.stringify([t.x, t.r, t.d, t.sd]));
""" % js)
    got = json.loads(subprocess.check_output([NODE, str(driver)], text=True))
    assert got == [[100, 150, 200], [1.0, None, 0.5], [0, 0.5, 1], None]


# ── tiling ─────────────────────────────────────────────────────────────
def _synthetic_chrom(name='chrT', length=3000000, gap=(1000000, 2000000)):
    pos = np.concatenate([np.arange(0, gap[0], 200),
                          np.arange(gap[1], length, 200)]).astype(np.uint32)
    rng = np.random.default_rng(3)
    r = rng.normal(1.0, 0.2, pos.size)
    r[::97] = np.nan
    r[::131] = 0.0
    return {'name': name, 'length': length, 'cn_neutral': 2, 'blocks': [],
            'pos': pos, 'end': (pos + 200).astype(np.uint32),
            'r': r, 'r0': r.copy(),
            'log2r': viz._r_to_log2r(r), 'log2r0': viz._r_to_log2r(r),
            'mq0': (rng.random(pos.size) > 0.1).astype(np.uint8),
            'flags': np.zeros(pos.size, dtype=np.uint8)}


def test_level_tiles_partition():
    c = _synthetic_chrom()
    bin_size, tile_points = 1000, 256
    counts, tiles, med = viz.level_tiles(c, bin_size, tile_points, 1, True)
    assert counts.size == viz.n_tiles_for(c['length'], bin_size, tile_points)
    bidx_all, _, _, _ = viz.bin_probes(c['pos'], c['r'], c['mq0'], bin_size)
    assert counts.sum() == bidx_all.size
    assert med.size == bidx_all.size
    seen = []
    for t, payload in tiles:
        d = json.loads(payload)
        assert 'NaN' not in payload and 'Infinity' not in payload
        assert len(d['x']) == len(d['r']) == len(d['d']) == len(d['sd']) == counts[t]
        x = np.cumsum(d['x'])
        lo, hi = t * tile_points, (t + 1) * tile_points
        assert np.all((x // bin_size >= lo) & (x // bin_size < hi))
        seen.append(x)
    x_all = np.concatenate(seen)
    expect = ((bidx_all.astype(np.float64) + 0.5) * bin_size).astype(np.int64)
    assert np.array_equal(x_all, expect)
    assert np.all(np.diff(x_all) > 0)


def test_level_tiles_skips_empty_tiles():
    c = _synthetic_chrom()
    counts, tiles, _ = viz.level_tiles(c, 1000, 256, 1, False)
    emitted = {t for t, _ in tiles}
    assert emitted == {t for t in range(counts.size) if counts[t] > 0}
    assert any(counts[t] == 0 for t in range(counts.size))  # the gap leaves empty tiles


# ── end to end ─────────────────────────────────────────────────────────
def _write_probe_dump(path, chroms, sample='TEST', sex=1, cn_neutral=2):
    with open(path, 'wb') as fp:
        fp.write(viz.DUMP_MAGIC)
        fp.write(struct.pack('<HH', 2, 0))
        fp.write(struct.pack('<BBH', sex, cn_neutral, 0))
        fp.write(struct.pack('<I', len(chroms)))
        name = sample.encode()
        fp.write(struct.pack('<H', len(name)) + name)
        for c in chroms:
            cn = c['name'].encode()
            fp.write(struct.pack('<H', len(cn)) + cn)
            fp.write(struct.pack('<IBBH', c['length'], c['cn_neutral'], 0, 0))
            fp.write(struct.pack('<I', c['pos'].size))
            recs = np.zeros(c['pos'].size, dtype=np.dtype([
                ('start', '<u4'), ('end', '<u4'), ('v', '<f4'), ('v0', '<f4'),
                ('mq0', 'u1'), ('flags', 'u1'), ('pad', '<u2')]))
            recs['start'] = c['pos']
            recs['end'] = c['end']
            recs['v'] = c['r']
            recs['v0'] = c['r']
            recs['mq0'] = c['mq0']
            fp.write(recs.tobytes())
        fp.write(struct.pack('<I', viz.EOF_MARKER))


VCF_TEXT = """##fileformat=VCFv4.2
##reference=/ref/hg38.fa
##contig=<ID=chrT,length=3000000,assembly=hg38>
##INFO=<ID=END,Number=1,Type=Integer,Description="End">
##INFO=<ID=SVLEN,Number=1,Type=Integer,Description="Length">
##INFO=<ID=SVTYPE,Number=1,Type=String,Description="Type">
##INFO=<ID=CN,Number=1,Type=Integer,Description="Copy number">
##INFO=<ID=HMMCN,Number=1,Type=Integer,Description="HMM copy number">
##FORMAT=<ID=GT,Number=1,Type=String,Description="Genotype">
##FORMAT=<ID=LR,Number=1,Type=Float,Description="Log ratio">
##FORMAT=<ID=GAMMA,Number=5,Type=Float,Description="Posterior">
#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\tTEST
chrT\t300000\t.\tN\t<DEL>\t44\tPASS\tEND=400000;SVLEN=100000;SVTYPE=DEL;CN=1;HMMCN=1\tGT:LR:GAMMA\t0/1:-0.92:0.01,0.9,0.07,0.01,0.01
chrT\t2200000\t.\tN\t<DUP>\t51\tPASS\tEND=2400000;SVLEN=200000;SVTYPE=DUP;CN=3;HMMCN=3\tGT:LR:GAMMA\t0/1:0.55:0.01,0.02,0.07,0.89,0.01
"""


@pytest.fixture(scope='module')
def inputs(tmp_path_factory):
    if not (BGZIP and TABIX):
        pytest.skip('bgzip/tabix not installed')
    d = tmp_path_factory.mktemp('viz_inputs')
    c = _synthetic_chrom()
    probes = d / 'test.probes.bin'
    _write_probe_dump(str(probes), [c])
    vcf = d / 'test.vcf'
    vcf.write_text(VCF_TEXT)
    subprocess.check_call([BGZIP, '-f', str(vcf)])
    subprocess.check_call([TABIX, '-f', '-p', 'vcf', str(vcf) + '.gz'])
    return str(vcf) + '.gz', str(probes)


def _run(args):
    return subprocess.run([sys.executable, os.path.join(REPO, 'scripts', 'cnvscope_viz.py')] + args,
                          capture_output=True, text=True, cwd=REPO)


def test_split_build(inputs, tmp_path):
    vcf, probes = inputs
    out = tmp_path / 'report'
    p = _run([vcf, probes, str(out), '--bin-sizes', '1000,10000', '--tile-points', '256', '--cdn'])
    assert p.returncode == 0, p.stderr
    assert (out / 'index.html').is_file()
    assert (out / 'viewer.js').is_file()
    assert not (out / 'plotly.min.js').exists()      # --cdn
    manifest = json.loads((out / 'data' / 'manifest.json').read_text())
    assert manifest['version'] == 1
    assert manifest['bins'] == [1000, 10000]
    assert manifest['tile_points'] == 256
    assert manifest['channels'] == ['x', 'r', 'd']
    assert [c['name'] for c in manifest['chroms']] == ['chrT']
    index = (out / 'index.html').read_text()
    assert 'cnvTile(' not in index
    assert '"mode":"split"' in index
    assert (out / 'data' / 'chrT' / 'panel.js').is_file()
    n_files = 0
    for k, level in enumerate(manifest['chroms'][0]['levels']):
        for t, n in enumerate(level['counts']):
            f = out / 'data' / 'chrT' / f'L{k}' / f'{t}.js'
            assert f.is_file() == (n > 0)
            if not n:
                continue
            n_files += 1
            body = f.read_text()
            assert body.startswith(f'cnvTile("chrT",{k},{t},')
            payload = json.loads(body[body.index('{'):body.rindex('}') + 1])
            assert len(payload['x']) == n
            assert 'NaN' not in body
    assert n_files > 1
    spec = (out / 'data' / 'chrT' / 'panel.js').read_text()
    panel = json.loads(spec[spec.index('{'):spec.rindex('}') + 1])
    probe_trace = [t for t in panel['scatter']['data'] if t.get('meta') == 'probes'][0]
    assert probe_trace['x'] == [] and probe_trace['y'] == []
    meta = panel['scatter']['layout']['meta']
    assert set(meta) == {'yrange_log2', 'yrange_linear', 'neutral_log2r'}
    assert meta['yrange_linear'][0] < 0 < meta['yrange_linear'][1]


def test_single_file_build(inputs, tmp_path):
    vcf, probes = inputs
    out = tmp_path / 'report.html'
    p = _run([vcf, probes, str(out), '--bin-sizes', '1000', '--tile-points', '256',
              '--cdn', '--advanced'])
    assert p.returncode == 0, p.stderr
    html = out.read_text()
    assert 'cnvTile("chrT",0,' in html
    assert 'cnvPanel("chrT",' in html
    assert '"mode":"single"' in html
    assert '"sd"' in html                       # --advanced adds the SD channel
    assert 'NaN' not in html
    assert not (tmp_path / 'report.html.data').exists()


def test_advanced_channels(inputs, tmp_path):
    vcf, probes = inputs
    out = tmp_path / 'adv'
    p = _run([vcf, probes, str(out), '--bin-sizes', '1000', '--tile-points', '256',
              '--cdn', '--advanced'])
    assert p.returncode == 0, p.stderr
    manifest = json.loads((out / 'data' / 'manifest.json').read_text())
    assert manifest['channels'] == ['x', 'r', 'd', 'sd']
    body = (out / 'data' / 'chrT' / 'L0' / '0.js').read_text()
    payload = json.loads(body[body.index('{'):body.rindex('}') + 1])
    assert len(payload['sd']) == len(payload['x'])
    spec = (out / 'data' / 'chrT' / 'panel.js').read_text()
    panel = json.loads(spec[spec.index('{'):spec.rindex('}') + 1])
    dark = panel['mq0']['data'][0]
    assert (dark['type'], dark['fill'], dark['meta']) == ('scattergl', 'tozeroy', 'mq0')  # one WebGL area


def test_raw_probes_level(inputs, tmp_path):
    vcf, probes = inputs
    out = tmp_path / 'raw'
    p = _run([vcf, probes, str(out), '--bin-sizes', '10000', '--raw-probes', '--cdn'])
    assert p.returncode == 0, p.stderr
    manifest = json.loads((out / 'data' / 'manifest.json').read_text())
    assert manifest['bins'][0] == 200            # the synthetic probe width


def test_overwrite_rules(inputs, tmp_path):
    vcf, probes = inputs
    out = tmp_path / 'report'
    args = [vcf, probes, str(out), '--bin-sizes', '10000', '--cdn']
    assert _run(args).returncode == 0
    stale = out / 'data' / 'chrT' / 'L0' / '999.js'
    stale.write_text('stale')
    assert _run(args).returncode == 0            # a previous report is refreshed
    assert not stale.exists()
    other = tmp_path / 'notareport'
    other.mkdir()
    (other / 'keep.txt').write_text('hi')
    p = _run([vcf, probes, str(other), '--bin-sizes', '10000', '--cdn'])
    assert p.returncode != 0 and '--force' in p.stderr
    assert (other / 'keep.txt').exists()
    p = _run([vcf, probes, str(other), '--bin-sizes', '10000', '--cdn', '--force'])
    assert p.returncode == 0
    assert (other / 'keep.txt').exists()         # --force never deletes other files
    assert (other / 'index.html').is_file()


def test_half_written_report_is_ours(inputs, tmp_path):
    vcf, probes = inputs
    half = tmp_path / 'half'
    (half / 'data').mkdir(parents=True)          # what a crashed build leaves behind
    p = _run([vcf, probes, str(half), '--bin-sizes', '10000', '--cdn'])
    assert p.returncode == 0, p.stderr
    assert (half / 'data' / 'manifest.json').is_file()


def test_no_matching_chroms(inputs, tmp_path):
    vcf, probes = inputs
    p = _run([vcf, probes, str(tmp_path / 'none'), '--chroms', 'chrNOPE', '--cdn'])
    assert p.returncode != 0
    assert 'no chromosome matched' in p.stderr


def test_serve_rejects_non_report(tmp_path):
    empty = tmp_path / 'empty'
    empty.mkdir()
    p = _run(['--serve', str(empty)])
    assert p.returncode != 0
    assert 'no index.html' in p.stderr


def test_removed_lod_flag(inputs, tmp_path):
    vcf, probes = inputs
    p = _run([vcf, probes, str(tmp_path / 'x'), '--lod-threshold-bins', '200'])
    assert p.returncode != 0
    assert 'unrecognized arguments' in p.stderr


def test_version():
    p = _run(['--version'])
    assert p.returncode == 0 and p.stdout.strip() == f'cnvscope_viz.py {viz.__version__}'


# ── report fixes: MAPD, segment log2r, call filters, ribbon, header scope ──
def test_mapd_neighbours_within_chroms():
    a = {'pos': np.array([0, 200, 400, 600, 800], dtype=np.uint32),
         'log2r': np.array([0.0, 0.1, np.nan, -0.1, 0.0], dtype=np.float32)}
    b = {'pos': np.array([0, 200, 400], dtype=np.uint32),
         'log2r': np.array([5.0, 5.4, 5.0], dtype=np.float32)}
    # pooled diffs 0.1, 0.2, 0.1 | 0.4, 0.4 -> 0.2; value-sorted would give 0.1, across chroms 0.3
    assert viz.mapd([a, b]) == pytest.approx(0.2, abs=1e-6)
    shuffled = {'pos': np.array([400, 0, 200], dtype=np.uint32),
                'log2r': np.array([0.5, 0.0, 0.1], dtype=np.float32)}
    assert viz.mapd([shuffled]) == pytest.approx(0.25, abs=1e-6)   # genomic order: 0.1, 0.4
    assert math.isnan(viz.mapd([{'pos': np.zeros(1, np.uint32), 'log2r': np.zeros(1, np.float32)}]))


def test_dps_log2r():
    assert viz.dps_log2r([0.875765, 0.15, 83.7, 18.8]) == pytest.approx(math.log2(0.875765))
    assert viz.dps_log2r([0.0, 0.0, 0.0, 0.0]) == viz.LOG2R_ZERO_FLOOR
    assert viz.dps_log2r('0.5,0.1') == pytest.approx(-1.0)          # undeclared FORMAT: raw string
    for missing in (None, '.', [], [None]):
        assert viz.dps_log2r(missing) is None


def _seg(start, end, cn, filt='PASS', sv=False, cn_neutral=None):
    return {'start': start, 'end': end, 'cn': cn, 'qual': 30.0, 'filter': filt, 'sv': sv,
            'cn_neutral': cn_neutral, 'log2r': None}


PANEL_SEGS = [_seg(100000, 300000, 1), _seg(300000, 900000, 2),
              _seg(900000, 950000, 3, 'SVdup'), _seg(2200000, 2400000, 3),
              _seg(2500000, 2600000, 1, sv=True)]


def _panel(segs, has_sv, rejected=(), mode=0, posterior=False, mapq=False):
    yranges = {'yrange_log2': [-2.0, 2.0], 'yrange_linear': [-0.06, 2.0], 'neutral_log2r': 0.0}
    panes = viz.fig_chrom_panel(_synthetic_chrom(), segs, 2, yranges, posterior, False, False, mode,
                                has_sv=has_sv, rejected=rejected, show_mapq=mapq)
    # through JSON, as the panel files carry it (numpy arrays become lists)
    return {p['track']: json.loads(json.dumps(p['fig'].to_plotly_json(), cls=viz._NumpyEncoder)) for p in panes}


def test_ribbon_is_overlaid_hbars():
    rib = _panel(PANEL_SEGS, has_sv=True)['ribbon']
    assert rib['layout']['barmode'] == 'overlay'
    bars = [t for t in rib['data'] if t['type'] == 'bar']
    marks = [t for t in rib['data'] if t['type'] == 'scatter']
    assert bars and len(bars) == len(marks)
    seen = []
    for b, m in zip(bars, marks):
        assert b['orientation'] == 'h' and b['meta'] in ('ribbon', 'ribbon_sv')
        idx = list(b['customdata'])
        assert len(b['base']) == len(b['x']) == len(b['y']) == len(b['hovertext']) == len(idx)
        for i, base, size in zip(idx, b['base'], b['x']):
            assert (base, base + size) == (PANEL_SEGS[i]['start'], PANEL_SEGS[i]['end'])
        failing = PANEL_SEGS[idx[0]]['filter'] != 'PASS'
        assert b['opacity'] == (0.3 if failing else 0.8)
        # hover handles: invisible midpoint markers with the same index and card
        assert m['meta'] == b['meta'] and m['marker']['opacity'] == 0
        assert list(m['customdata']) == idx and list(m['hovertext']) == list(b['hovertext'])
        assert list(m['x']) == [(PANEL_SEGS[i]['start'] + PANEL_SEGS[i]['end']) // 2 for i in idx]
        seen += idx
    assert sorted(seen) == [0, 2, 3, 4]                    # the CN=2 segment is neutral: no bar
    sv_bar = [b for b in bars if b['meta'] == 'ribbon_sv']
    assert len(sv_bar) == 1 and list(sv_bar[0]['customdata']) == [4]
    assert sv_bar[0]['y'][0] < min(b['y'][0] for b in bars if b['meta'] == 'ribbon')


def test_segment_lines_carry_segment_index():
    lines = [t for t in _panel(PANEL_SEGS, has_sv=True)['scatter']['data']
             if t.get('meta') == 'segments']
    assert len(lines) == 2                                  # CNVscope + dotted SV-derived
    seen = []
    for t in lines:
        cd = list(t['customdata'])
        assert len(cd) == len(t['x']) == len(t['y'])
        for j in range(0, len(cd), 3):
            i = cd[j]
            assert cd[j + 1] == i and cd[j + 2] is None
            assert (t['x'][j], t['x'][j + 1]) == (PANEL_SEGS[i]['start'], PANEL_SEGS[i]['end'])
            seen.append(i)
    assert sorted(seen) == [0, 1, 2, 3, 4]


def test_ideogram_bars_carry_chrom_and_index():
    fig = viz.fig_genome_ideogram([_synthetic_chrom()], {'chrT': PANEL_SEGS}, 2).to_plotly_json()
    bars = [t for t in fig['data'] if t['type'] == 'bar']
    assert bars and all(t['meta'] == 'chrT' for t in bars)
    for t in bars:
        assert len(t['customdata']) == len(t['x']) == len(t['width']) == len(t['hovertext'])
    assert sorted(i for t in bars for i in t['customdata']) == [0, 2, 3, 4]


def test_call_summary_counts_shown_chroms_only():
    t = _synthetic_chrom('chrT')
    u = _synthetic_chrom('chrU', length=2000000, gap=(800000, 1200000))
    u['cn_neutral'] = 1
    segs = {'chrT': [_seg(0, 100, 1), _seg(100, 200, 2), _seg(200, 300, 3, sv=True)],
            'chrU': [_seg(0, 100, 2), _seg(100, 200, 1), _seg(300, 400, 0, sv=True)]}
    s = viz.call_summary([t], segs, 2)
    assert (s['n_segments'], s['n_sv'], s['n_events'], s['n_gain'], s['n_loss']) == (3, 1, 2, 1, 1)
    assert s['calls_per_mb'] == pytest.approx(2 / 3.0)           # PASS calls that change CN: CN=2 is none
    s = viz.call_summary([t, u], segs, 2)
    assert (s['n_segments'], s['n_sv'], s['n_events'], s['n_gain'], s['n_loss']) == (6, 2, 4, 2, 2)
    assert s['calls_per_mb'] == pytest.approx(4 / 5.0)
    assert [c['name'] for c in viz.select_chroms([t, u], 100, 'chrU,chrNOPE')] == ['chrU']
    assert viz.select_chroms([t, u], 10 ** 9, None) == []


def _named(name, n=200):
    return {'name': name, 'pos': np.zeros(n, dtype=np.uint32)}


@pytest.mark.parametrize('names, main', [
    (['chr1', 'chr2', 'chr10', 'chrX', 'chrY', 'chrM', 'chr1_KI270706v1_random', 'chrUn_KI270302v1',
      'chr22_KI270879v1_alt', 'chrEBV', 'HLA-A*01:01:01:01'], ['chr1', 'chr2', 'chr10', 'chrX', 'chrY']),
    (['1', '2', '10', 'X', 'Y', 'MT', 'GL000192.1', 'NC_007605', 'hs37d5'], ['1', '2', '10', 'X', 'Y']),
])
def test_select_chroms_main_set(names, main):
    chroms = [_named(n) for n in names]
    def pick(*a, **k):
        return [c['name'] for c in viz.select_chroms(chroms, 100, *a, **k)]
    assert pick(None) == main                                   # autosomes + X/Y by default
    assert pick(None, all_chroms=True) == names                 # --all-chroms: every contig
    assert pick('1, X' if names[0].startswith('chr') else 'chr1,chrX') == [main[0], main[3]]  # prefix optional
    assert pick(names[5]) == [names[5]]                         # a listed chrM/MT is shown
    chroms[1]['pos'] = np.zeros(5, dtype=np.uint32)
    assert pick(None) == [main[0]] + main[2:]                   # --min-chrom-probes still applies


def test_select_chroms_without_main_names_keeps_every_contig():
    chroms = [_named('chrT'), _named('chrU'), _named('NC_000001.11')]
    assert [c['name'] for c in viz.select_chroms(chroms, 100, None)] == ['chrT', 'chrU', 'NC_000001.11']


def _run_node(tmp_path, body, *args):
    js = tmp_path / 'viewer.js'
    js.write_text(viz.VIEWER_JS)
    driver = tmp_path / 'drive.js'
    driver.write_text("const v = require('%s');\n%s" % (js, body))
    return json.loads(subprocess.check_output([NODE, str(driver)] + list(args), text=True))


@pytest.mark.skipif(NODE is None, reason='node not installed')
def test_js_segment_visible(tmp_path):
    segs = [{'start': 0, 'end': 1000, 'qual': 30, 'filter': 'PASS', 'kind': 'loss', 'sv': False},
            {'start': 0, 'end': 50000, 'qual': None, 'filter': 'SVdup', 'kind': 'hom_del', 'sv': False},
            {'start': 0, 'end': 5000, 'qual': 8, 'filter': 'PASS', 'kind': 'amp', 'sv': True},
            {'start': 0, 'end': 9000, 'qual': 50, 'filter': 'PASS', 'kind': None, 'sv': False}]
    cases = [({}, [True, True, True, True]),
             ({'filterMode': 'pass'}, [True, False, True, True]),
             ({'filterMode': 'failing'}, [False, True, False, False]),
             ({'eventType': 'gain'}, [False, False, True, False]),       # neutral hidden too
             ({'eventType': 'loss'}, [True, True, False, False]),
             ({'sourceMode': 'cnv'}, [True, True, False, True]),
             ({'sourceMode': 'sv'}, [False, False, True, False]),
             ({'minSize': 5000}, [False, True, True, True]),             # end - start >= value
             ({'minQual': 10}, [True, False, False, True])]              # missing QUAL counts as 0
    base = {'minSize': 0, 'minQual': 0, 'filterMode': 'all', 'eventType': 'all', 'sourceMode': 'all'}
    filters = [dict(base, **f) for f, _ in cases]
    got = _run_node(tmp_path, """
const [segs, filters] = JSON.parse(process.argv[2]);
console.log(JSON.stringify(filters.map(f => segs.map(s => v.segmentVisible(s, f)))));
""", json.dumps([segs, filters]))
    assert got == [want for _, want in cases]


@pytest.mark.skipif(NODE is None, reason='node not installed')
def test_js_break_gaps(tmp_path):
    got = _run_node(tmp_path, """
console.log(JSON.stringify(v.breakGaps([500, 1500, 2500, 6500, 7500], [0, 0.5, 1, 1, 0], 1000)));
""")
    assert got == {'x': [500, 1500, 2500, None, 6500, 7500], 'y': [0, 0.5, 1, None, 1, 0]}


@pytest.mark.skipif(NODE is None, reason='node not installed')
def test_js_keep_points(tmp_path):
    line = {'x': [0, 10, None, 20, 30, None, 40, 50, None], 'y': [1, 1, None, 2, 2, None, 3, 3, None],
            'customdata': [0, 0, None, 2, 2, None, 5, 5, None]}
    bar = {'base': [0, 100, 200], 'x': [10, 20, 30], 'y': [0.5, 0.5, 0.5], 'width': 0.2,
           'customdata': [0, 2, 5], 'hovertext': ['a', 'b', 'c']}
    keep = [True, False, False, False, False, True]
    got = _run_node(tmp_path, """
const [line, bar, keep] = JSON.parse(process.argv[2]);
console.log(JSON.stringify([v.keepPoints(line, keep), v.keepPoints(bar, keep)]));
""", json.dumps([line, bar, keep]))
    assert got[0] == {'x': [0, 10, None, 40, 50, None], 'y': [1, 1, None, 3, 3, None],
                      'customdata': [0, 0, None, 5, 5, None]}
    assert got[1] == {'base': [0, 200], 'x': [10, 30], 'y': [0.5, 0.5],      # scalar width untouched
                      'customdata': [0, 5], 'hovertext': ['a', 'c']}


@pytest.mark.skipif(NODE is None, reason='node not installed')
def test_js_resolve_chrom(tmp_path):
    got = _run_node(tmp_path, """
const prefixed = {chr7: {}, chrX: {}}, plain = {'7': {}, X: {}};
console.log(JSON.stringify([
  ['chr7', '7', 'chrX', 'X', 'chr8', 'M'].map(n => v.resolveChrom(n, prefixed)),
  ['7', 'chr7', 'CHR7', 'X', 'chrX', 'chrM'].map(n => v.resolveChrom(n, plain))]));
""")
    assert got == [['chr7', 'chr7', 'chrX', 'chrX', None, None], ['7', '7', '7', 'X', 'X', None]]


def test_chrom_panes_handle_doubleclick():
    js = viz.VIEWER_JS
    cfg = re.search(r'var plotlyConfig = (\{[^}]*\});', js).group(1)
    assert 'doubleClick: false' in cfg and 'displayModeBar: false' in cfg and 'responsive: true' in cfg
    assert 'Plotly.newPlot(mount, copy.data, copy.layout, plotlyConfig)' in js
    assert "gd.on('plotly_doubleclick', function () { homeView(name); })" in js
    assert 'function homeView(chrom)' in js and 'applyZoom(chrom, null)' in js


VCF2_TEXT = """##fileformat=VCFv4.2
##reference=/ref/hg38.fa
##contig=<ID=chrT,length=3000000,assembly=hg38>
##contig=<ID=chrU,length=2000000,assembly=hg38>
##FILTER=<ID=MLrejected,Description="ML rejected">
##INFO=<ID=END,Number=1,Type=Integer,Description="End">
##INFO=<ID=SVLEN,Number=1,Type=Integer,Description="Length">
##INFO=<ID=SVTYPE,Number=1,Type=String,Description="Type">
##INFO=<ID=CN,Number=1,Type=Integer,Description="Copy number">
##INFO=<ID=CN_NEUTRAL,Number=1,Type=Integer,Description="Neutral copy number">
##INFO=<ID=SOURCE,Number=1,Type=String,Description="Call source">
##FORMAT=<ID=GT,Number=1,Type=String,Description="Genotype">
##FORMAT=<ID=LR,Number=1,Type=Float,Description="Log ratio vs model mean (DP)">
##FORMAT=<ID=DPS,Number=4,Type=Float,Description="Normalized read depth statistics">
#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\tTEST
chrT\t300001\t.\tN\t<DEL>\t44\tPASS\tEND=400000;SVLEN=100000;SVTYPE=DEL;CN=1\tGT:LR:DPS\t0/1:0.753:0.875765,0.154467,83.72,18.82
chrT\t1100001\t.\tN\t<DEL>\t30\tPASS\tEND=1150000;SVLEN=50000;SVTYPE=DEL;CN=0\tGT:LR:DPS\t1/1:-3.1:0,0,0,0
chrT\t2200001\t.\tN\t<DUP>\t51\tPASS\tEND=2400000;SVLEN=200000;SVTYPE=DUP;CN=3\tGT:LR:DPS\t0/1:0.55:1.47,0.2,60,20
chrT\t2600001\t.\tN\t<DUP>\t12\tMLrejected\tEND=2700000;SVLEN=100000;SVTYPE=DUP;CN=3\tGT:LR:DPS\t0/1:0.4:.
chrT\t2800001\t.\tN\t<DEL>\t.\tPASS\tEND=2900000;SVLEN=100000;SVTYPE=DEL;CN=1;CN_NEUTRAL=2;SOURCE=SV_CNV\tGT\t0/1
chrU\t100001\t.\tN\t<DUP>\t20\tPASS\tEND=300000;SVLEN=200000;SVTYPE=DUP;CN=2;CN_NEUTRAL=2\tGT:LR:DPS\t0/1:0.1:1.02,0.1,50,10
chrU\t500001\t.\tN\t<DUP>\t25\tPASS\tEND=700000;SVLEN=200000;SVTYPE=DUP;CN=2\tGT:LR:DPS\t0/1:0.9:1.9,0.1,50,10
chrU\t900001\t.\tN\t<DEL>\t.\tPASS\tEND=950000;SVLEN=50000;SVTYPE=DEL;CN=0;CN_NEUTRAL=1;SOURCE=SV_CNV\tGT\t1
"""


@pytest.fixture(scope='module')
def inputs2(tmp_path_factory):
    """chrT (diploid) + chrU (haploid, with a PAR-like CN_NEUTRAL=2 call); LR and DPS both present."""
    if not (BGZIP and TABIX):
        pytest.skip('bgzip/tabix not installed')
    d = tmp_path_factory.mktemp('viz_inputs2')
    u = _synthetic_chrom('chrU', length=2000000, gap=(800000, 1200000))
    u['cn_neutral'] = 1
    probes = d / 'two.probes.bin'
    _write_probe_dump(str(probes), [_synthetic_chrom(), u])
    vcf = d / 'two.vcf'
    vcf.write_text(VCF2_TEXT)
    subprocess.check_call([BGZIP, '-f', str(vcf)])
    subprocess.check_call([TABIX, '-f', '-p', 'vcf', str(vcf) + '.gz'])
    return str(vcf) + '.gz', str(probes)


def test_segment_log2r_from_dps_not_lr(inputs2):
    vcf, _ = inputs2
    segs, _ = viz.read_segments_vcf(vcf, ['chrT', 'chrU'])
    t = segs['chrT']
    assert t[0]['lr'] == pytest.approx(0.753)
    assert t[0]['log2r'] == pytest.approx(math.log2(0.875765))     # about -0.19, not LR
    assert t[1]['log2r'] == viz.LOG2R_ZERO_FLOOR                    # DPS[0] = 0: hom-del floor
    assert t[3]['log2r'] is None and t[3]['lr'] == pytest.approx(0.4)   # no DPS: never from LR
    assert t[4]['log2r'] is None and t[4]['lr'] is None


def test_call_location_is_vcf_pos_end(inputs2):
    vcf, _ = inputs2
    segs, _ = viz.read_segments_vcf(vcf, ['chrT', 'chrU'])
    s = segs['chrT'][0]                                          # POS 300001, END 400000
    assert (s['start'], s['end']) == (300000, 400000)            # kept 0-based, like the probes
    hov = viz.segment_hover('chrT', s, 2)
    assert hov.startswith('chrT:300,001-400,000<br>') and 'len=100,000 bp' in hov
    fig = viz.fig_genome_ideogram([_synthetic_chrom()], {'chrT': [s]}, 2).to_plotly_json()
    tips = [h for t in fig['data'] if t.get('hovertext') for h in t['hovertext']]
    assert tips and tips[0].startswith('chrT:300,001-400,000<br>')


def _header(html):
    """Header tiles as {label: (value, [sub lines])}."""
    hdr = html[html.index('<header>'):html.index('</header>')]
    tiles = {}
    for f in hdr.split('<div class="field">')[1:]:
        label = re.search(r'<div class="label">([^<]+)</div>', f).group(1)
        value = re.search(r'<div class="value">([^<]+)</div>', f).group(1)
        tiles[label] = (value, re.findall(r'<div class="sub">([^<]+)</div>', f))
    return tiles


def _config(html):
    return json.loads(re.search(r'window\.CNV_CONFIG = (.*?);\n', html).group(1))


def test_header_counts_shown_chroms_only(inputs2, tmp_path):
    vcf, probes = inputs2
    p = _run([vcf, probes, str(tmp_path / 'all'), '--bin-sizes', '10000', '--cdn'])
    assert p.returncode == 0, p.stderr
    assert 'chroms=2 kept (of 2)' in p.stderr
    assert 'no autosome or sex chromosome names in the probe dump: showing every contig' in p.stderr
    h = _header((tmp_path / 'all' / 'index.html').read_text())
    assert h['Segments'] == ('8', ['CNVscope: 5 PASS calls &middot; 1 rejected', 'SV-derived: 2 PASS calls'])
    # PASS calls that change CN: not the MLrejected record, not the neutral PAR call on chrU
    assert h['Events'] == ('6', ['2 gain &middot; 4 loss'])
    assert h['Calls / Mb'][0] == '1.20'                          # 6 calls / 5 Mb of shown chroms
    p = _run([vcf, probes, str(tmp_path / 'one'), '--bin-sizes', '10000', '--cdn', '--chroms', 'chrT'])
    assert p.returncode == 0, p.stderr
    assert 'chroms=1 kept (of 2)' in p.stderr
    h = _header((tmp_path / 'one' / 'index.html').read_text())
    assert h['Segments'] == ('5', ['CNVscope: 3 PASS calls &middot; 1 rejected', 'SV-derived: 1 PASS call'])
    assert h['Events'] == ('4', ['1 gain &middot; 3 loss'])
    assert h['Calls / Mb'][0] == '1.33'                          # 4 calls / 3 Mb
    assert h['Reference'][1][0].startswith('2 contigs')          # still the whole reference


def test_config_segments_kind(inputs2, tmp_path):
    vcf, probes = inputs2
    out = tmp_path / 'report'
    p = _run([vcf, probes, str(out), '--bin-sizes', '10000', '--cdn'])
    assert p.returncode == 0, p.stderr
    cfg = _config((out / 'index.html').read_text())
    # the MLrejected record is no call: it is not in the config
    assert [s['kind'] for s in cfg['segments']['chrT']] == ['loss', 'hom_del', 'gain', 'loss']
    # haploid chrU: a CN=2 call with its own CN_NEUTRAL=2 (PAR) is neutral, a plain CN=2 is a gain
    assert [s['kind'] for s in cfg['segments']['chrU']] == [None, 'gain', 'hom_del']
    assert cfg['segments']['chrT'][0]['log2r'] == pytest.approx(math.log2(0.875765), abs=1e-3)
    spec = (out / 'data' / 'chrU' / 'panel.js').read_text()
    panel = json.loads(spec[spec.index('{'):spec.rindex('}') + 1])
    bars = [t for t in panel['ribbon']['data'] if t['type'] == 'bar']
    assert sorted(i for t in bars for i in t['customdata']) == [1, 2]


def test_logo_is_bundled(monkeypatch):
    uri = viz.load_logo_data_uri()
    assert uri.startswith('data:image/png;base64,')
    assert base64.b64decode(uri.split(',', 1)[1])[:8] == b'\x89PNG\r\n\x1a\n'
    assert not hasattr(viz, 'LOGO_URL')  # the build makes no network request
    monkeypatch.setattr(viz, 'LOGO_PATH', os.path.join(REPO, 'missing.png'))
    assert viz.load_logo_data_uri() == ''


VCF3_TEXT = """##fileformat=VCFv4.2
##reference=/ref/ref.fa
##contig=<ID={c1},length=600000>
##contig=<ID={cx},length=600000>
##contig=<ID={cm},length=600000>
##contig=<ID={cu},length=600000>
##INFO=<ID=END,Number=1,Type=Integer,Description="End">
##INFO=<ID=SVLEN,Number=1,Type=Integer,Description="Length">
##INFO=<ID=SVTYPE,Number=1,Type=String,Description="Type">
##INFO=<ID=CN,Number=1,Type=Integer,Description="Copy number">
##FORMAT=<ID=GT,Number=1,Type=String,Description="Genotype">
#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\tTEST
{c1}\t100001\t.\tN\t<DEL>\t40\tPASS\tEND=150000;SVLEN=50000;SVTYPE=DEL;CN=1\tGT\t0/1
{cx}\t100001\t.\tN\t<DUP>\t40\tPASS\tEND=150000;SVLEN=50000;SVTYPE=DUP;CN=2\tGT\t1
{cm}\t100001\t.\tN\t<DUP>\t40\tPASS\tEND=150000;SVLEN=50000;SVTYPE=DUP;CN=3\tGT\t0/1
{cu}\t100001\t.\tN\t<DUP>\t40\tPASS\tEND=150000;SVLEN=50000;SVTYPE=DUP;CN=3\tGT\t0/1
"""
NAMING = {'chr': ('chr1', 'chrX', 'chrM', 'chrUn_KI270302v1'), 'plain': ('1', 'X', 'MT', 'GL000192.1')}


@pytest.fixture(scope='module', params=sorted(NAMING))
def inputs3(request, tmp_path_factory):
    """Chroms 1 and X (haploid) plus chrM and an unplaced contig, all with calls; named with or without chr."""
    if not (BGZIP and TABIX):
        pytest.skip('bgzip/tabix not installed')
    names = NAMING[request.param]
    d = tmp_path_factory.mktemp('viz_inputs3_' + request.param)
    chroms = [_synthetic_chrom(n, length=600000, gap=(200000, 400000)) for n in names]
    chroms[1]['cn_neutral'] = 1
    probes = d / 'four.probes.bin'
    _write_probe_dump(str(probes), chroms)
    vcf = d / 'four.vcf'
    vcf.write_text(VCF3_TEXT.format(**dict(zip(('c1', 'cx', 'cm', 'cu'), names))))
    subprocess.check_call([BGZIP, '-f', str(vcf)])
    subprocess.check_call([TABIX, '-f', '-p', 'vcf', str(vcf) + '.gz'])
    return str(vcf) + '.gz', str(probes), names


def test_report_shows_main_chroms(inputs3, tmp_path):
    vcf, probes, names = inputs3

    def build(tag, *opts):
        out = tmp_path / tag
        p = _run([vcf, probes, str(out), '--bin-sizes', '10000', '--cdn'] + list(opts))
        assert p.returncode == 0, p.stderr
        man = json.loads((out / 'data' / 'manifest.json').read_text())
        return p.stderr, [c['name'] for c in man['chroms']], _header((out / 'index.html').read_text())

    err, shown, h = build('default')
    assert shown == list(names[:2])                              # chrM/MT and the unplaced contig left out
    assert h['Segments'][0] == '2'
    assert 'chroms=2 kept (of 4) · segments=2 ·' in err
    assert '2 segments on 2 other contigs left out (--all-chroms adds them)' in err
    err, shown, h = build('all', '--all-chroms')
    assert shown == list(names) and h['Segments'][0] == '4'
    assert 'left out' not in err
    err, shown, _ = build('one', '--chroms', 'X' if names[1].startswith('chr') else 'chrX')
    assert shown == [names[1]]                                   # the other naming finds the same chrom


def test_chroms_and_all_chroms_exclusive(tmp_path):
    p = _run(['in.vcf.gz', 'in.probes.bin', str(tmp_path / 'out'), '--chroms', 'chr1', '--all-chroms'])
    assert p.returncode == 2 and 'not allowed with argument' in p.stderr


# ── model-rejected segments (CNVModelApply --emit_conf 0) ──────────────
def test_parse_probs_and_p_neutral():
    assert viz.parse_floats([0.0, 0.1, 0.85, 0.05, 0.0]) == [0.0, 0.1, 0.85, 0.05, 0.0]
    assert viz.parse_floats('0.00,0.10,0.85,0.05,0.00') == [0.0, 0.1, 0.85, 0.05, 0.0]   # undeclared: raw string
    assert viz.parse_floats('0.2,.,0.8') == [0.2, None, 0.8]
    assert viz.parse_floats([0.2, float('nan'), '-nan', 'nan']) == [0.2, None, None, None]
    assert viz.parse_floats(None) is None and viz.parse_floats([]) is None
    s = {'probs': [0.0, 0.1, 0.85, 0.05, 0.0], 'cn_neutral': None}
    assert viz.p_neutral(s, 2) == 0.85                            # no CN_NEUTRAL: the chrom's
    assert viz.p_neutral(dict(s, cn_neutral=1), 2) == 0.1         # the record's CN_NEUTRAL wins (male chrX)
    assert viz.p_neutral(dict(s, cn_neutral=0), 2) == 0.0         # female chrY
    assert viz.p_neutral(dict(s, cn_neutral=5), 2) is None        # no such index
    assert viz.p_neutral({'probs': None, 'cn_neutral': None}, 2) is None


def _rej(start, end, hmmcn=None, log2r=None, cn_neutral=2, probs=(0.0, 0.1, 0.85, 0.05, 0.0)):
    return {'start': start, 'end': end, 'cn': cn_neutral, 'hmmcn': hmmcn, 'log2r': log2r, 'qual': 12.0,
            'filter': 'MLrejected', 'sv': False, 'rejected': True, 'cn_neutral': cn_neutral,
            'probs': list(probs), 'ml_prob': 0.15}


def _as_line(ys):
    """A rejected line trace's y for per-segment levels: [y, y, null] per segment."""
    return [v for y in ys for v in (y, y, None)]


@pytest.mark.parametrize('mode', [0, 1])
def test_rejected_trace_levels(mode):
    hmm = (lambda k: k) if mode else (lambda k: None)                  # only mode 1 records carry HMMCN
    rej = [_rej(100000, 200000, hmmcn=hmm(1), log2r=-0.94), _rej(400000, 500000, hmmcn=hmm(2), log2r=0.01),
           _rej(600000, 650000, hmmcn=hmm(0), log2r=viz.LOG2R_ZERO_FLOOR),
           _rej(700000, 750000, hmmcn=hmm(3), log2r=-1.02, cn_neutral=1),                 # a haploid block
           _rej(800000, 850000, hmmcn=hmm(0), log2r=-3.5, cn_neutral=0),                  # CN 0: hom-del floor
           dict(_rej(2200000, 2300000, hmmcn=hmm(3), log2r=0.56, cn_neutral=1), cn=None),  # no CN: its neutral CN
           _rej(2400000, 2500000, cn_neutral=None)]                                       # no CN, DPS or HMMCN
    hmm3 = round(math.log2(1.5), 4)
    want = {'cn': [0.0, 0.0, 0.0, -1.0, viz.HOMDEL_FLOOR, -1.0, 0.0],                     # the default, both modes
            'depth': [-0.94, 0.01, viz.LOG2R_ZERO_FLOOR, -1.02, -3.5, 0.56, None]}
    if mode:
        want['hmm'] = [-1.0, 0.0, viz.HOMDEL_FLOOR, hmm3, viz.HOMDEL_FLOOR, hmm3, None]
    calls = PANEL_SEGS[:2]
    panel = _panel(calls, has_sv=False, rejected=rej)
    data = panel['scatter']['data']
    metas = [t.get('meta') for t in data]
    line, marks = [t for t in data if t.get('meta') == 'rejected']
    assert metas.index('rejected') < metas.index('segments')       # drawn under the calls
    assert line['type'] == 'scatter' and line['mode'] == 'lines' and line['hoverinfo'] == 'skip'
    assert line['line']['color'] == viz.REJECTED_COLOR
    assert line['line']['width'] < [t for t in data if t.get('meta') == 'segments'][0]['line']['width']
    assert 'customdata' not in line and 'customdata' not in marks   # the call filters never touch them
    assert list(line['x']) == [v for s in rej for v in (s['start'], s['end'], None)]   # every record
    assert list(line['y']) == _as_line(want['cn']) and list(marks['y']) == want['cn']
    assert marks['type'] == 'scatter' and marks['mode'] == 'markers' and marks['marker']['opacity'] == 0
    assert marks['hoverinfo'] == 'text'
    assert list(marks['x']) == [(s['start'] + s['end']) // 2 for s in rej]
    assert [h.split('<br>')[:2] for h in marks['hovertext']] == [
        [viz.call_location('chrT', s), 'rejected by the model'] for s in rej]
    # the menu's other levels, one y per segment (null without DPS / HMMCN); HMM CN only with HMMCN
    meta = panel['scatter']['layout']['meta']
    assert meta['yrange_log2'] == [-2.0, 2.0]
    assert meta['rejected_levels'] == {k: ys for k, ys in want.items() if k != 'cn'}
    # not calls: no ribbon bar
    rib = panel['ribbon']['data']
    assert all(t.get('meta') in ('ribbon', 'ribbon_sv') for t in rib)
    assert sorted(set(i for t in rib for i in t['customdata'])) == [0]   # the one loss call
    plain = _panel(calls, has_sv=False)['scatter']                     # no rejected segments: no levels
    assert 'rejected_levels' not in plain['layout']['meta']
    assert not [t for t in plain['data'] if t.get('meta') == 'rejected']


def test_rejected_hover():
    s = dict(_rej(100000, 200000, hmmcn=1, log2r=-0.94), lr=-0.9, cdelta=0.5, entropy=0.2, ecost=1.5)
    assert viz.rejected_hover('chrT', s, 2).split('<br>') == [
        'chrT:100,001-200,000', 'rejected by the model', 'CN=2 (neutral 2) HMMCN=1',
        'P(neutral)=0.85 ML_PROB=0.15', 'log2r=-0.940', 'QUAL=12.0 len=100,000 bp',
        'LR=-0.900 CDELTA=0.500', 'ENTROPY=0.200 ECOST=1.500']
    # mode 0 on a male chrX: no HMMCN or HMM features, P(neutral) is PROBS[1], QUAL '.' left out
    s0 = dict(_rej(0, 5000, log2r=-1.02, cn_neutral=1, probs=(0.02, 0.9, 0.05, 0.03, 0.0)), qual=float('nan'))
    assert viz.rejected_hover('chrX', s0, 1).split('<br>') == [
        'chrX:1-5,000', 'rejected by the model', 'CN=1 (neutral 1)', 'P(neutral)=0.90 ML_PROB=0.15',
        'log2r=-1.020', 'len=5,000 bp']


def test_posterior_covers_rejected_segments():
    g = [0.1, 0.8, 0.1, 0.0, 0.0]
    calls = [dict(_seg(300000, 400000, 1), gamma=g)]
    rej = [dict(_rej(100000, 300000, hmmcn=1), gamma=g), dict(_rej(400000, 600000, hmmcn=2), gamma=g)]
    post = _panel(calls, has_sv=False, rejected=rej, mode=1, posterior=True)['posterior']['data'][0]
    assert list(post['x']) == [100000, 300000, 400000, 600000]      # contiguous segments: shared edges
    assert all(len(row) == 3 for row in post['z'])


def test_counts_exclude_non_pass_and_rejected():
    t = _synthetic_chrom('chrT')
    rej = dict(_seg(0, 100, 3, 'MLrejected'), rejected=True)          # a rejected record that kept a gain CN
    segs = {'chrT': [rej, _seg(100, 200, 1), _seg(200, 300, 3, 'SVdup'), _seg(300, 400, 1, sv=True),
                     _seg(400, 500, 3, 'NoDepth', sv=True), _seg(500, 600, 2)]}
    s = viz.call_summary([t], segs, 2)
    # events: the PASS loss and the PASS SV-derived loss; not SVdup, NoDepth, MLrejected or the CN=2 record
    assert (s['n_segments'], s['n_events'], s['n_gain'], s['n_loss']) == (6, 2, 0, 2)
    assert s['calls_per_mb'] == pytest.approx(2 / 3.0)
    assert s['records'] == {'cnv_pass': 2, 'cnv_filtered': 1, 'rejected': 1, 'sv_pass': 1, 'sv_filtered': 1}
    assert viz.segment_lines(s['records'], True) == [
        'CNVscope: 2 PASS calls &middot; 1 filtered &middot; 1 rejected',
        'SV-derived: 1 PASS call &middot; 1 filtered']
    assert viz.segment_lines(dict(s['records'], cnv_pass=1207, cnv_filtered=0, rejected=23462),
                             False) == ['1,207 PASS calls &middot; 23,462 rejected']


# (chrom, POS, END, FILTER, CN, HMMCN, PROBS, DPS[0]); chrU is haploid
CNV4 = [('chrT', 100001, 300000, 'MLrejected', 2, 1, '0.00,0.10,0.85,0.05,0.00', 0.52),
        ('chrT', 300001, 400000, 'PASS', 1, 1, '0.01,0.90,0.08,0.01,0.00', 0.49),
        ('chrT', 400001, 600000, 'MLrejected', 2, 2, '0.00,0.00,1.00,0.00,0.00', 1.01),
        ('chrT', 600001, 650000, 'MLrejected', 2, 0, '0.30,0.10,0.60,0.00,0.00', 0.0),
        ('chrT', 2200001, 2400000, 'PASS', 3, 3, '0.00,0.01,0.09,0.90,0.00', 1.48),
        ('chrT', 2400001, 2500000, 'MLrejected', 2, 3, '0.00,0.00,0.70,0.30,0.00', 1.46),
        ('chrU', 100001, 300000, 'MLrejected', 1, 1, '0.05,0.90,0.05,0.00,0.00', 0.49),
        ('chrU', 500001, 700000, 'PASS', 2, 2, '0.00,0.05,0.90,0.05,0.00', 0.98)]
# combined VCF only: a CNVscope call the merge filtered, a converted call and a filtered converted call
MERGE4 = [('chrT', 2500001, 2600000, 'SVdup', 3, 3, '0.00,0.01,0.19,0.80,0.00', 1.45),
          ('chrT', 2800001, 2900000, 'PASS', 1, None, None, None),
          ('chrT', 2900001, 2950000, 'NoDepth', 3, None, None, None)]
VCF4_HEADER = """##fileformat=VCFv4.2
##reference=/ref/hg38.fa
##contig=<ID=chrT,length=3000000,assembly=hg38>
##contig=<ID=chrU,length=2000000,assembly=hg38>
##FILTER=<ID=MLrejected,Description="Rejected by machine learning model">
##FILTER=<ID=SVdup,Description="CNVscope call repeating a converted SV call">
##FILTER=<ID=NoDepth,Description="Long converted gain over which CNVscope saw no gain">
##INFO=<ID=END,Number=1,Type=Integer,Description="End">
##INFO=<ID=SVLEN,Number=1,Type=Integer,Description="Length">
##INFO=<ID=SVTYPE,Number=1,Type=String,Description="Type">
##INFO=<ID=CN,Number=1,Type=Integer,Description="Copy number">
##INFO=<ID=CN_NEUTRAL,Number=1,Type=Integer,Description="Neutral copy number">
##INFO=<ID=HMMCN,Number=1,Type=Integer,Description="HMM copy number">
##INFO=<ID=ML_PROB,Number=1,Type=Float,Description="Machine learning reject probability">
##INFO=<ID=PROBS,Number=5,Type=Float,Description="Machine learning probabilities">
##INFO=<ID=SOURCE,Number=1,Type=String,Description="Call source">
##INFO=<ID=DEDUP,Number=1,Type=String,Description="Matching converted interval">
##FORMAT=<ID=GT,Number=1,Type=String,Description="Genotype">
##FORMAT=<ID=LR,Number=1,Type=Float,Description="Log ratio">
##FORMAT=<ID=GAMMA,Number=5,Type=Float,Description="Posterior">
##FORMAT=<ID=DPS,Number=4,Type=Float,Description="Normalized read depth statistics">
##FORMAT=<ID=MQS,Number=1,Type=Float,Description="Mapping quality statistics">
#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\tTEST
"""
# FORMAT/MQS of the CNVscope records, as CNVscope writes it; the converted calls have none
MQS4 = {('chrT', 100001): '59.0', ('chrT', 300001): '60.0', ('chrT', 400001): '12.0', ('chrT', 600001): '.',
        ('chrT', 2200001): '60.0', ('chrT', 2400001): '33.5', ('chrT', 2500001): '41.0',
        ('chrU', 100001): '0.0', ('chrU', 500001): '58.0'}


def _vcf4(mode, combined):
    """CNVModelApply --emit_conf 0 output for chrT (diploid) and chrU (haploid), mode 0 or 1, optionally merged."""
    lines = []
    for chrom, pos, end, filt, cn, hmm, probs, dps in sorted(CNV4 + (MERGE4 if combined else [])):
        cn0 = 1 if chrom == 'chrU' else 2
        alt = '.' if filt == 'MLrejected' else ('<DEL>' if cn < cn0 else '<DUP>')
        info = f'END={end};SVLEN={end - pos + 1};SVTYPE=CNV;CN={cn};CN_NEUTRAL={cn0}'
        if probs is None:          # a converted call
            lines.append(f'{chrom}\t{pos}\t.\tN\t{alt}\t.\t{filt}\t{info};SOURCE=SV_CNV\tGT\t0/1')
            continue
        info += f';ML_PROB={1 - max(map(float, probs.split(","))):.2f};PROBS={probs}'
        fmt, vals = 'GT:DPS', [f'{dps},0.1,50,10']
        if mode == 1:
            info += f';HMMCN={hmm}'
            fmt, vals = 'GT:LR:GAMMA:DPS', [f'{dps - 1:.2f}', probs] + vals
        fmt, vals = fmt + ':MQS', vals + [MQS4[(chrom, pos)]]
        if combined:
            info += ';SOURCE=CNV' + (f';DEDUP={chrom}:{pos}-{end}' if filt == 'SVdup' else '')
        gt = '0/0' if filt == 'MLrejected' else '0/1'
        lines.append(f'{chrom}\t{pos}\t.\tN\t{alt}\t12\t{filt}\t{info}\t{fmt}\t' + ':'.join([gt] + vals))
    return VCF4_HEADER + '\n'.join(lines) + '\n'


def _write_inputs4(d, mode, combined):
    """The probe dump and the bgzipped, indexed _vcf4() VCF in directory d."""
    u = _synthetic_chrom('chrU', length=2000000, gap=(800000, 1200000))
    u['cn_neutral'] = 1
    probes = d / 'rej.probes.bin'
    _write_probe_dump(str(probes), [_synthetic_chrom(), u])
    vcf = d / 'rej.vcf'
    vcf.write_text(_vcf4(mode, combined))
    subprocess.check_call([BGZIP, '-f', str(vcf)])
    subprocess.check_call([TABIX, '-f', '-p', 'vcf', str(vcf) + '.gz'])
    return str(vcf) + '.gz', str(probes)


@pytest.fixture(scope='module', params=['mode1_combined', 'mode0_apply'])
def inputs4(request, tmp_path_factory):
    """CNVModelApply --emit_conf 0 output: mode 1 merged with converted SV calls, and plain mode 0."""
    if not (BGZIP and TABIX):
        pytest.skip('bgzip/tabix not installed')
    d = tmp_path_factory.mktemp('viz_inputs4_' + request.param)
    vcf, probes = _write_inputs4(d, 1 if request.param.startswith('mode1') else 0, 'combined' in request.param)
    return vcf, probes, request.param


def test_rejected_record_without_cn(tmp_path):
    if not (BGZIP and TABIX):
        pytest.skip('bgzip/tabix not installed')
    vcf = tmp_path / 'nocn.vcf'
    vcf.write_text(VCF4_HEADER + 'chrT\t100001\t.\tN\t.\t12\tMLrejected\tEND=200000;SVTYPE=CNV;CN_NEUTRAL=2;'
                   'HMMCN=1;PROBS=0.00,0.10,0.85,0.05,0.00\tGT:DPS\t0/0:0.52,0.1,50,10\n')
    subprocess.check_call([BGZIP, '-f', str(vcf)])
    subprocess.check_call([TABIX, '-f', '-p', 'vcf', str(vcf) + '.gz'])
    s, = viz.read_segments_vcf(str(vcf) + '.gz', ['chrT'])[0]['chrT']
    assert s['rejected'] and s['cn'] is None and s['hmmcn'] == 1       # HMMCN is not the model's CN
    assert viz.rejected_y(s, 'cn', 2, 2) == 0.0 and viz.rejected_y(s, 'hmm', 2, 2) == -1.0
    assert viz.rejected_hover('chrT', s, 2).split('<br>')[2] == 'HMMCN=1'


def _ideogram(html):
    i = html.index('"ideogram-div",') + len('"ideogram-div",')
    return json.JSONDecoder().raw_decode(html[i:].lstrip())[0]


def _panel_js(text, chrom):
    """The cnvPanel(chrom, spec) spec in a panel.js or a single-file report."""
    i = text.index('cnvPanel(%s,' % json.dumps(chrom)) + len('cnvPanel(%s,' % json.dumps(chrom))
    return json.JSONDecoder().raw_decode(text[i:])[0]


def test_read_rejected_records(inputs4):
    vcf, _, kind = inputs4
    segs, mode = viz.read_segments_vcf(vcf, ['chrT', 'chrU'])
    assert mode == (1 if kind.startswith('mode1') else 0)
    t = segs['chrT']
    assert [s['rejected'] for s in t][:6] == [True, False, True, True, False, True]
    assert t[0]['probs'] == [0.0, 0.1, 0.85, 0.05, 0.0] and t[0]['cn'] == 2 and t[0]['filter'] == 'MLrejected'
    assert t[0]['hmmcn'] == (1 if mode else None)
    if kind == 'mode1_combined':      # SOURCE=CNV stays a CNVscope record; converted and SVdup calls are no rejects
        assert [(s['filter'], s['sv'], s['rejected']) for s in t[6:]] == [
            ('SVdup', False, False), ('PASS', True, False), ('NoDepth', True, False)]
    assert viz.p_neutral(segs['chrU'][0], 1) == 0.9                # haploid: PROBS[CN_NEUTRAL=1]


@pytest.mark.parametrize('flag', ['--advanced', '--internal'])
def test_rejected_segments_report(inputs4, tmp_path, flag):
    vcf, probes, kind = inputs4
    combined, mode1 = 'combined' in kind, kind.startswith('mode1')
    hmm = mode1 and flag == '--internal'           # the HMM views: --internal, and a VCF with HMM output
    out = tmp_path / 'report'
    p = _run([vcf, probes, str(out), '--bin-sizes', '10000', '--cdn', flag])
    assert p.returncode == 0, p.stderr
    assert ('advanced_tracks=on · internal' in p.stderr) == (flag == '--internal')
    assert ('segments=11 (5 rejected, 2 SV-derived)' if combined else 'segments=8 (5 rejected)') in p.stderr
    html = (out / 'index.html').read_text()
    # index.html holds calls only: the config, the ideogram, the counts
    cfg = _config(html)
    assert [s['filter'] for s in cfg['segments']['chrT']] == (
        ['PASS', 'PASS', 'SVdup', 'PASS', 'NoDepth'] if combined else ['PASS', 'PASS'])
    assert [s['filter'] for s in cfg['segments']['chrU']] == ['PASS']
    assert '<br>rejected by the model<br>' not in html
    ideo = _ideogram(html)
    for name in ('chrT', 'chrU'):
        assert sorted(i for t in ideo if t.get('meta') == name for i in t['customdata']) == [
            i for i, s in enumerate(cfg['segments'][name]) if s['kind']]
    h = _header(html)
    if combined:
        assert h['Segments'] == ('11', ['CNVscope: 3 PASS calls &middot; 1 filtered &middot; 5 rejected',
                                        'SV-derived: 1 PASS call &middot; 1 filtered'])
        assert h['Events'] == ('4', ['2 gain &middot; 2 loss'])      # PASS calls only: not SVdup, NoDepth
        assert h['Calls / Mb'][0] == '0.80'                          # 4 calls / 5 Mb
    else:
        assert h['Segments'] == ('8', ['3 PASS calls &middot; 5 rejected'])
        assert h['Events'] == ('3', ['2 gain &middot; 1 loss'])
        assert h['Calls / Mb'][0] == '0.60'
    chips = re.findall(r'<span class="chip">([^<]*)</span>', html)
    assert ('CNVscope: 2 PASS calls &middot; 1 filtered &middot; 4 rejected' if combined
            else '2 PASS calls &middot; 4 rejected') in chips
    assert '<input type="checkbox" id="show-rejected" checked>' in html
    assert ('<span class="internal-tag">Internal</span>' in html) == (flag == '--internal')
    # the level menu sits under the checkbox: model CN first and selected, HMM CN in internal reports only
    menu = re.search(r'<select id="rejected-level">(.*?)</select>', html, re.S).group(1)
    assert html.index('id="show-rejected"') < html.index('id="rejected-level"')
    assert re.findall(r'<option value="(\w+)"', menu) == ['cn', 'depth'] + (['hmm'] if hmm else [])
    assert '<option value="cn" selected>model CN</option>' in menu
    legend = re.search(r'<dt>Rejected segments \(grey\)</dt>\s*<dd>(.*?)</dd>', html, re.S).group(1)
    assert '<em>model CN</em> (the default)' in legend and '<em>measured depth</em>' in legend
    assert ('<em>HMM CN</em>' in legend) == hmm                             # the levels the menu offers
    assert ('<dt>HMM posterior</dt>' in html) == ('data-track="posterior"' in html) == hmm
    # the rejected segments ship with each chrom's panel data: y at the model CN, every level in layout.meta
    levels = {'chrT': {'cn': [0.0] * 4,
                       'depth': [round(math.log2(v), 4) for v in (0.52, 1.01)] + [viz.LOG2R_ZERO_FLOOR,
                                                                                  round(math.log2(1.46), 4)],
                       'hmm': [-1.0, 0.0, viz.HOMDEL_FLOOR, round(math.log2(1.5), 4)]},
              'chrU': {'cn': [-1.0], 'depth': [round(math.log2(0.49), 4)], 'hmm': [-1.0]}}   # haploid
    for name, want in levels.items():
        if not hmm:
            del want['hmm']
        panel = _panel_js((out / 'data' / name / 'panel.js').read_text(), name)
        line, marks = [t for t in panel['scatter']['data'] if t.get('meta') == 'rejected']
        assert line['y'][::3] == want['cn'] and marks['y'] == want['cn']
        assert panel['scatter']['layout']['meta']['rejected_levels'] == {
            k: ys for k, ys in want.items() if k != 'cn'}
        rib = [i for t in panel['ribbon']['data'] for i in t['customdata']]
        assert sorted(set(rib)) == [i for i, s in enumerate(cfg['segments'][name]) if s['kind']]
    t0 = _panel_js((out / 'data' / 'chrT' / 'panel.js').read_text(), 'chrT')
    card = [t for t in t0['scatter']['data'] if t.get('meta') == 'rejected'][1]['hovertext'][0].split('<br>')
    assert card[:4] == ['chrT:100,001-300,000', 'rejected by the model',
                        'CN=2 (neutral 2)' + (' HMMCN=1' if hmm else ''), 'P(neutral)=0.85 ML_PROB=0.15']
    assert any(line.startswith('LR=') for line in card) == hmm                   # the HMM features
    u0 = _panel_js((out / 'data' / 'chrU' / 'panel.js').read_text(), 'chrU')
    assert 'P(neutral)=0.90' in [t for t in u0['scatter']['data'] if t.get('meta') == 'rejected'][1]['hovertext'][0]
    # internal mode 1: the HMM posterior covers the rejected segments too
    assert ('posterior' in t0) == hmm


def test_rejected_segments_single_file(inputs4, tmp_path):
    vcf, probes, _ = inputs4
    out = tmp_path / 'report.html'
    p = _run([vcf, probes, str(out), '--bin-sizes', '10000', '--cdn'])
    assert p.returncode == 0, p.stderr
    html = out.read_text()
    assert all(s['filter'] != 'MLrejected' for ss in _config(html)['segments'].values() for s in ss)
    marks = [t for t in _panel_js(html, 'chrT')['scatter']['data'] if t.get('meta') == 'rejected'][1]
    assert len(marks['hovertext']) == 4 and 'NaN' not in html


def test_older_vcf_has_no_rejected_controls(inputs, tmp_path):
    vcf, probes = inputs
    out = tmp_path / 'report'
    p = _run([vcf, probes, str(out), '--bin-sizes', '10000', '--cdn', '--advanced'])
    assert p.returncode == 0, p.stderr
    assert 'segments=2 ·' in p.stderr
    html = (out / 'index.html').read_text()
    assert 'show-rejected' not in html and 'rejected-level' not in html and 'Rejected segments' not in html
    assert 'data-track="mapq"' not in html and 'Mapping quality' not in html     # no MQS in this VCF
    assert _header(html)['Segments'] == ('2', [])                  # nothing to split: the plain tile
    assert '<span class="chip">2 segments</span>' in html
    panel = _panel_js((out / 'data' / 'chrT' / 'panel.js').read_text(), 'chrT')
    assert not [t for t in panel['scatter']['data'] if t.get('meta') == 'rejected']
    assert 'rejected_levels' not in panel['scatter']['layout']['meta']
    assert 'mapq' not in panel


@pytest.mark.skipif(NODE is None, reason='node not installed')
def test_js_rejected_update(tmp_path):
    # the traces' own y is the cn level; the other levels have one y per segment (the second: no DPS)
    levels = {'depth': [-0.94, None], 'hmm': [-1, -10]}
    traces = [{'meta': 'probes', 'x': [], 'y': []},
              {'meta': 'rejected', 'x': [0, 10, None, 20, 30, None], 'y': [0, 0, None, -1, -1, None]},
              {'meta': 'rejected', 'x': [5, 25], 'y': [0, -1]},
              {'meta': 'segments', 'x': [0, 10, None], 'y': [1, 1, None], 'customdata': [0, 0, None]}]
    pane = {'data': traces, 'layout': {'meta': {'neutral_log2r': 0, 'rejected_levels': levels}}}
    no_hmm = {'data': traces, 'layout': {'meta': {'rejected_levels': {'depth': levels['depth']}}}}
    got = _run_node(tmp_path, """
const [pane, noHmm] = JSON.parse(process.argv[2]);
const calls = [[pane, 'cn', 'linear', true], [pane, 'depth', 'log2', true], [pane, 'hmm', 'linear', false],
               [pane, 'hmm', 'log2', true], [noHmm, 'hmm', 'log2', true],
               [{data: [pane.data[0], pane.data[3]], layout: {}}, 'cn', 'linear', true]];
console.log(JSON.stringify(calls.map(c => v.rejectedUpdate(...c)).concat([pane.layout.meta.rejected_levels])));
""", json.dumps([pane, no_hmm]))
    cn, depth, hmm_lin, hmm_log2, missing, none, kept = got
    rnd = lambda ys: [[None if v is None else round(v, 3) for v in y] for y in ys]
    assert cn['idx'] == [1, 2] and cn['update']['visible'] == [True, True]
    assert rnd(cn['update']['y']) == [[1.0, 1.0, None, 0.5, 0.5, None], [1.0, 0.5]]      # Scale linear: 2^y
    assert depth['update']['y'] == [[-0.94, -0.94, None, None, None, None], [-0.94, None]]   # log2, trace shapes
    assert rnd(hmm_lin['update']['y']) == [[0.5, 0.5, None, 0.001, 0.001, None], [0.5, 0.001]]
    assert hmm_lin['update']['visible'] == [False, False]                                # box unticked
    assert hmm_log2['update']['y'] == [[-1, -1, None, -10, -10, None], [-1, -10]]
    assert missing['idx'] == [1, 2] and missing['update']['y'] == [[None] * 6, [None] * 2]   # level not in panel
    assert none == {'idx': [], 'update': {'y': [], 'visible': []}}
    assert kept == levels and traces[1]['y'] == [0, 0, None, -1, -1, None]    # the spec arrays stay as they are


def test_viewer_wires_rejected_toggle():
    js = viz.VIEWER_JS
    assert "document.getElementById('show-rejected')" in js
    assert "rejectedBox.addEventListener('change', drawRejected)" in js
    assert "levelSel.addEventListener('change', drawRejected)" in js            # the level menu redraws
    draw = js[js.index('function rejectedLevel()'):js.index('function captureIdeogram()')]
    assert "document.getElementById('rejected-level')" in draw and "sel ? sel.value : 'cn'" in draw
    assert 'rejectedUpdate(activeSpec.scatter, rejectedLevel(), currentYMode(), showRejected())' in draw
    apply_y = js[js.index('function applyYMode()'):js.index('// ── call filters')]
    assert 'drawRejected();' in apply_y              # each chrom opening and Scale switch, at the menu's level
    reset = js[js.index('function reset()'):js.index('// ── x-axis sync')]
    assert 'rejectedBox.checked = true' in reset and "levelSel.value = 'cn'" in reset   # Reset: model CN
    assert 'drawRejected();' in reset


# CNVModelApply's own number text, as combine_cnv.py v2.2.0 copies its rejected records: 0.00, -nan, -0.000000, .
VCF5_TEXT = """##fileformat=VCFv4.2
##contig=<ID=chrT,length=3000000>
##FILTER=<ID=MLrejected,Description="Rejected by machine learning model">
##INFO=<ID=END,Number=1,Type=Integer,Description="End">
##INFO=<ID=SVTYPE,Number=1,Type=String,Description="Type">
##INFO=<ID=CN,Number=1,Type=Integer,Description="Copy number">
##INFO=<ID=CN_NEUTRAL,Number=1,Type=Integer,Description="Neutral copy number">
##INFO=<ID=HMMCN,Number=1,Type=Integer,Description="HMM copy number">
##INFO=<ID=ML_PROB,Number=1,Type=Float,Description="Machine learning reject probability">
##INFO=<ID=PROBS,Number=5,Type=Float,Description="Machine learning probabilities">
##INFO=<ID=SOURCE,Number=1,Type=String,Description="Call source">
##FORMAT=<ID=GT,Number=1,Type=String,Description="Genotype">
##FORMAT=<ID=LR,Number=1,Type=Float,Description="Log ratio">
##FORMAT=<ID=CDELTA,Number=1,Type=Float,Description="HMM confidence delta">
##FORMAT=<ID=GAMMA,Number=5,Type=Float,Description="Posterior">
##FORMAT=<ID=DPS,Number=4,Type=Float,Description="Normalized read depth statistics">
#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\tTEST
chrT\t100001\t.\tN\t.\t-nan\tMLrejected\tCN=2;CN_NEUTRAL=2;END=200000;HMMCN=1;ML_PROB=-nan;PROBS=0.00,-nan,0.97,0.03,0.00;SVTYPE=CNV;SOURCE=CNV\tGT:LR:CDELTA:GAMMA:DPS\t0/0:-nan:.:0.000000,-nan,1.000000,0.000000,0.000000:-nan,-0.000000,1.000000,1.000000
chrT\t200001\t.\tN\t.\t15.45\tMLrejected\tCN=2;CN_NEUTRAL=2;END=300000;HMMCN=3;ML_PROB=0.03;PROBS=0.00,0.00,0.97,0.03,0.00;SVTYPE=CNV;SOURCE=CNV\tGT:LR:CDELTA:GAMMA:DPS\t0/0:0.189544:nan:0.000000,0.000000,0.484690,0.295146,0.000000:1.373747,0.217363,48.640568,13.548934
chrT\t300001\t.\tN\t<DEL>\t5.26\tPASS\tCN=1;CN_NEUTRAL=2;END=400000;HMMCN=1;ML_PROB=0.41;PROBS=0.11,0.59,0.30,0.01,0.00;SVTYPE=CNV;SOURCE=CNV\tGT:LR:CDELTA:GAMMA:DPS\t0/1:-0.500000:0.920918:0.000000,1.000000,0.000000,0.000000,0.000000:0.520000,0.100000,50.000000,10.000000
"""


def test_reader_handles_cnvmodelapply_number_text(tmp_path):
    if not (BGZIP and TABIX):
        pytest.skip('bgzip/tabix not installed')
    vcf = tmp_path / 'text.vcf'
    vcf.write_text(VCF5_TEXT)
    subprocess.check_call([BGZIP, '-f', str(vcf)])
    subprocess.check_call([TABIX, '-f', '-p', 'vcf', str(vcf) + '.gz'])
    segs, mode = viz.read_segments_vcf(str(vcf) + '.gz', ['chrT'])   # a '.' or -nan float used to raise
    assert mode == 1
    a, b, c = segs['chrT']
    assert math.isnan(a['qual']) and a['ml_prob'] is None and a['lr'] is None and a['cdelta'] is None
    assert a['probs'] == [0.0, None, 0.97, 0.03, 0.0] and a['gamma'] == [0.0, None, 1.0, 0.0, 0.0]
    assert a['log2r'] is None and a['rejected']                     # DPS[0] -nan: no measured depth
    assert b['probs'] == [0.0, 0.0, 0.97, 0.03, 0.0] and b['ml_prob'] == 0.03 and b['cdelta'] is None
    assert b['lr'] == pytest.approx(0.189544) and b['log2r'] == pytest.approx(math.log2(1.373747))
    assert c['ml_prob'] == 0.41 and c['cdelta'] == pytest.approx(0.920918) and not c['rejected']
    assert viz.p_neutral(a, 2) == 0.97 and viz.p_neutral(b, 2) == 0.97
    card = viz.rejected_hover('chrT', a, 2)
    assert 'nan' not in card.lower() and 'P(neutral)=0.97' in card and 'QUAL' not in card
    assert 'nan' not in viz.segment_hover('chrT', c, 2).lower()
    # the whole report: strict JSON throughout, no nan in any card
    probes = tmp_path / 'text.probes.bin'
    _write_probe_dump(str(probes), [_synthetic_chrom()])
    out = tmp_path / 'report.html'
    p = _run([str(vcf) + '.gz', str(probes), str(out), '--bin-sizes', '10000', '--cdn', '--internal'])
    assert p.returncode == 0, p.stderr
    html = out.read_text()
    assert 'NaN' not in html
    panel = _panel_js(html, 'chrT')
    cards = [h for t in panel['scatter']['data'] + panel['ribbon']['data'] for h in (t.get('hovertext') or [])]
    assert len(cards) == 2 + 2 and not any('nan' in h.lower() for h in cards)   # 2 rejected + bar and marker of the call
    assert panel['posterior']['data'][0]['z'][1][0] is None                     # GAMMA -nan: an empty cell


# ── mapping quality (FORMAT/MQS) ───────────────────────────────────────
def test_read_mqs(tmp_path):
    assert viz._first_num('59.0') == 59.0 and viz._first_num('59.0,1') == 59.0      # undeclared: raw text
    assert viz._first_num([58.0]) == 58.0 and viz._first_num(60) == 60.0
    for missing in (None, '.', '-nan', [], [None]):
        assert viz._first_num(missing) is None
    if not (BGZIP and TABIX):
        pytest.skip('bgzip/tabix not installed')
    vcf = tmp_path / 'mqs.vcf'
    vcf.write_text(VCF4_HEADER +
                   'chrT\t100001\t.\tN\t.\t12\tMLrejected\tEND=200000;SVTYPE=CNV;CN=2;CN_NEUTRAL=2\tGT:DPS:MQS\t'
                   '0/0:1.0,0.1,50,10:59.0\n'
                   'chrT\t200001\t.\tN\t<DEL>\t30\tPASS\tEND=300000;SVTYPE=CNV;CN=1;CN_NEUTRAL=2\tGT:DPS:MQS\t'
                   '0/1:0.5,0.1,50,10:.\n'
                   'chrT\t300001\t.\tN\t<DUP>\t30\tPASS\tEND=400000;SVTYPE=CNV;CN=3;CN_NEUTRAL=2\tGT:DPS\t'
                   '0/1:1.5,0.1,50,10\n'
                   'chrT\t400001\t.\tN\t<DEL>\t.\tPASS\tEND=450000;SVTYPE=DEL;CN=1;CN_NEUTRAL=2;SOURCE=SV_CNV\tGT\t0/1\n')
    subprocess.check_call([BGZIP, '-f', str(vcf)])
    subprocess.check_call([TABIX, '-f', '-p', 'vcf', str(vcf) + '.gz'])
    segs = viz.read_segments_vcf(str(vcf) + '.gz', ['chrT'])[0]['chrT']
    assert [s['mqs'] for s in segs] == [59.0, None, None, None]     # '59.0', '.', no MQS, SV-derived


def _steps(x, y):
    """The horizontal pieces (x0, x1, y) of an 'hv' step line whose runs are split by nulls."""
    return [(x[i], x[i + 1], y[i]) for i in range(len(x) - 1) if x[i] is not None and x[i + 1] is not None]


def test_mapq_pane():
    calls = [dict(_seg(100000, 300000, 1), mqs=59.0), dict(_seg(300000, 400000, 3, 'SVdup'), mqs=12.0),
             dict(_seg(400000, 500000, 2), mqs=None),                                     # no MQS: no line
             dict(_seg(500000, 600000, 1, sv=True), mqs=60.0),                            # SV-derived: not drawn
             dict(_seg(2200000, 2400000, 3), mqs=33.5)]
    rej = [dict(_rej(0, 100000), mqs=60.0), dict(_rej(2400000, 2500000), mqs=0.0)]
    want = [(0, 100000, 60), (100000, 300000, 59), (300000, 400000, 12), (2200000, 2400000, 33.5),
            (2400000, 2500000, 0)]                       # calls of any FILTER and rejected segments, by start
    panel = _panel(calls, has_sv=True, rejected=rej, mapq=True)
    assert list(panel) == ['scatter', 'mapq', 'ribbon']                     # directly below the depth track
    line, marks = panel['mapq']['data']
    assert line['meta'] == marks['meta'] == 'mapq' and line['hoverinfo'] == 'skip'
    assert line['mode'] == 'lines' and line['line']['shape'] == 'hv' and line['line']['color'] == viz.MAPQ_COLOR
    assert list(line['x']) == [0, 100000, 300000, 400000, None, 2200000, 2400000, 2500000]   # null at the gap
    assert _steps(line['x'], line['y']) == want
    assert 'customdata' not in line
    assert marks['mode'] == 'markers' and marks['marker']['opacity'] == 0          # one hover point per record
    assert list(marks['x']) == [(s + e) // 2 for s, e, _ in want] and list(marks['y']) == [v for _, _, v in want]
    assert [list(c) for c in marks['customdata']] == [[s + 1, e] for s, e, _ in want]       # POS, END
    assert marks['hovertemplate'] == 'chrT:%{customdata[0]:,d}-%{customdata[1]:,d}<br>MAPQ=%{y:.1~f}<extra></extra>'
    lay = panel['mapq']['layout']
    assert lay['yaxis']['range'] == [-3.0, 66.0] and list(lay['yaxis']['tickvals']) == [0, 20, 40, 60]
    assert lay['yaxis']['title']['text'] == 'MAPQ' and lay['height'] == 100
    assert 'mapq' not in _panel(calls, has_sv=True, rejected=rej)                # show_mapq off: no pane
    empty = _panel(PANEL_SEGS[:2], has_sv=False, mapq=True)['mapq']              # a chrom with no MQS: empty pane
    assert list(empty['data']) == [] and empty['layout']['yaxis']['range'] == [-3.0, 66.0]


def test_hover_cards_show_mapq():
    call = dict(_seg(300000, 400000, 1), log2r=-0.19, mqs=59.0)
    assert 'log2r=-0.190 MAPQ=59' in viz.segment_hover('chrT', call, 2).split('<br>')
    rej = dict(_rej(100000, 200000, hmmcn=1), mqs=33.5)                          # no measured depth: MAPQ alone
    assert viz.rejected_hover('chrT', rej, 2).split('<br>')[4] == 'MAPQ=33.5'
    assert 'MAPQ' not in viz.segment_hover('chrT', _seg(0, 10, 1), 2) + viz.rejected_hover('chrT', _rej(0, 10), 2)


def test_mapq_track_report(inputs4, tmp_path):
    vcf, probes, kind = inputs4
    combined = 'combined' in kind
    out = tmp_path / 'report'
    p = _run([vcf, probes, str(out), '--bin-sizes', '10000', '--cdn'])           # every report, --advanced or not
    assert p.returncode == 0, p.stderr
    html = (out / 'index.html').read_text()
    assert '<input type="checkbox" class="track-toggle" data-track="mapq" checked>mapping quality' in html
    assert '<dt>Mapping quality</dt>' in html
    want = {'chrT': [(100000, 300000, 59), (300000, 400000, 60), (400000, 600000, 12), (2200000, 2400000, 60),
                     (2400000, 2500000, 33.5)] + ([(2500000, 2600000, 41)] if combined else []),   # the SVdup call
            'chrU': [(100000, 300000, 0), (500000, 700000, 58)]}          # MQS '.' and converted calls: no line
    for name, recs in want.items():
        # a track pane like the others, so x sync, double-click and the bottom tick labels include it
        assert re.findall(r'<div class="track-pane" data-track="(\w+)">\s*<div class="plotly-mount" id="chrom-%s-'
                          % name, html) == ['scatter', 'mapq', 'ribbon']
        panel = _panel_js((out / 'data' / name / 'panel.js').read_text(), name)
        line, marks = panel['mapq']['data']
        assert _steps(line['x'], line['y']) == recs
        assert marks['y'] == [v for _, _, v in recs] and marks['customdata'] == [[s + 1, e] for s, e, _ in recs]
    t = _panel_js((out / 'data' / 'chrT' / 'panel.js').read_text(), 'chrT')
    card = [tr for tr in t['scatter']['data'] if tr.get('meta') == 'rejected'][1]['hovertext'][0].split('<br>')
    assert card[4] == f'log2r={math.log2(0.52):.3f} MAPQ=59'                     # the rejected chrT:100,001-300,000
    calls = [h for tr in t['ribbon']['data'] for h in tr['hovertext']]
    assert any(h.startswith('chrT:300,001-400,000<br>') and 'MAPQ=60' in h for h in calls)
    assert all('MAPQ' not in h for h in calls if 'SV-derived call' in h)


# ── customer and internal reports (--internal adds the HMM views) ─────
HMM_MARKERS = ('HMMCN', 'HMM CN', 'LR=', 'CDELTA', 'ENTROPY', 'ECOST', 'MAXCONF', 'MINCONF', 'raw HMM', 'GAMMA',
               'posterior', '"hmm"')


def test_help_hides_internal():
    p = _run(['--help'])
    assert p.returncode == 0 and '--advanced' in p.stdout
    assert '--internal' not in p.stdout and 'posterior' not in p.stdout.lower()


def _report_files(out):
    """index.html and every data file of a split report, as text."""
    files = [out / 'index.html'] + sorted(f for f in (out / 'data').rglob('*') if f.is_file())
    return {str(f.relative_to(out)): f.read_text() for f in files}


@pytest.mark.parametrize('flags', [[], ['--advanced']])
def test_customer_reports_carry_no_hmm_output(inputs4, tmp_path, flags):
    vcf, probes, kind = inputs4
    out = tmp_path / 'report'
    p = _run([vcf, probes, str(out), '--bin-sizes', '10000', '--cdn'] + flags)
    assert p.returncode == 0, p.stderr
    hits = [(name, m) for name, text in _report_files(out).items() for m in HMM_MARKERS if m in text]
    assert hits == []                                     # index.html and data/: no HMM output, mode 0 or 1
    html = (out / 'index.html').read_text()
    assert 'internal' not in p.stderr and 'internal-tag' not in html
    keys = {k for ss in _config(html)['segments'].values() for s in ss for k in s}
    assert keys == {'start', 'end', 'cn', 'kind', 'log2r', 'qual', 'filter', 'sv'}


def test_internal_mode0_equals_advanced(tmp_path):
    if not (BGZIP and TABIX):
        pytest.skip('bgzip/tabix not installed')
    vcf, probes = _write_inputs4(tmp_path, 0, True)       # mode 0 merged with converted calls
    built = {}
    for flag in ('--advanced', '--internal'):
        out = tmp_path / flag.strip('-')
        p = _run([vcf, probes, str(out), '--bin-sizes', '10000', '--cdn', flag])
        assert p.returncode == 0, p.stderr
        files = _report_files(out)
        files['index.html'] = re.sub(r'\d{4}-\d\d-\d\d \d\d:\d\d:\d\d \w*', 'TIME', files['index.html'])
        built[flag] = files
    tag = '<span class="internal-tag">Internal</span>'
    assert tag in built['--internal']['index.html'] and 'internal-tag' not in built['--advanced']['index.html']
    # the tag and its style rule
    built['--internal']['index.html'] = re.sub(r'\n  \.topbar \.internal-tag \{[^}]*\}', '',
                                               built['--internal']['index.html'].replace(tag, ''))
    assert built['--internal'] == built['--advanced']      # a mode 0 VCF has no HMM output: only the tag differs
