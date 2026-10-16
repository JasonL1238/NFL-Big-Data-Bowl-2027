"""Check every number quoted in writeup/draft.md (v2.1, Combine Game Equivalents) against its committed source.

For each quoted number this script:
  1. recomputes it from its source JSON key (experiments/results/E0xx.json, E007_leaderboard.json,
     experiments/posthoc_writeup_v1/*.json) and compares at the quoted rounding. A few numbers are arithmetic
     on committed keys (sums, ratios, a Spearman-Brown inversion, a binomial chance); the formula is written
     next to each such check and in the draft's src comment;
  2. asserts the quoted string (the exact phrase carrying the number) is present in the draft text
     (HTML comments stripped).
Then it fails on any numeric token in the text that no check (or the short allowlist of non-result
constants below) covers, on any number-bearing paragraph without a "<!-- src" comment nearby, and on the
budget: <= 1,950 words (captions and table cells count; comments, image links and [[TBD...]] do not) and
<= 8 figures + tables.

Run: .venv/bin/python writeup/check_numbers.py [path/to/draft.md]   (exit code 1 on any failure)
On a comment-stripped copy (the text pasted into Kaggle) the src-comment check is skipped; every number is still checked.
"""
import json
import math
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
R = ROOT / 'experiments' / 'results'
PH = ROOT / 'experiments' / 'posthoc_writeup_v1'
DRAFT = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / 'writeup' / 'draft.md'

load = lambda p: json.loads(p.read_text())
e7lb = load(R / 'E007_leaderboard.json')
e8 = load(R / 'E008.json')
e9 = load(R / 'E009.json')
e10 = load(R / 'E010.json')
e11 = load(R / 'E011.json')
e12 = load(R / 'E012.json')
e12p = load(R / 'E012_prereg.json')
ph_pairs = load(PH / 'ph_pairs.json')

raw = DRAFT.read_text()
text = re.sub(r'<!--.*?-->', lambda m: ' ' * len(m.group(0)), raw, flags=re.S)  # keep offsets aligned with raw

MPH = 3600 / 1760  # yd/s -> mph
fails, covered = [], []


def cover(quoted):
    """Record every occurrence of a quoted string as covered; return True if it occurs at all."""
    found = False
    for m in re.finditer(re.escape(quoted), text):
        covered.append((m.start(), m.end()))
        found = True
    return found


def chk(label, got, want, nd=2, quoted=None):
    """want: a number compared at nd decimals, or a (lo, hi) tuple for a range check, or a bool."""
    if isinstance(want, tuple):
        ok = want[0] <= got <= want[1]
    elif isinstance(want, bool):
        ok = bool(got) == want
    else:
        ok = round(got, nd) == round(want, nd)
    if quoted is not None and not cover(quoted):
        ok = False
        label += f' [quoted string {quoted!r} not in text]'
    print(('OK  ' if ok else 'FAIL'), label, '|', got, '->', want)
    if not ok:
        fails.append(label)


def allow(quoted, reason):
    """A non-result constant (protocol constant, data scope, definition or label). Must be in the text."""
    ok = cover(quoted)
    print(('OK  ' if ok else 'FAIL'), f'allow {quoted!r} ({reason})')
    if not ok:
        fails.append(f'allow {quoted!r} not in text')


# ---------------------------------------------------------------- lookups
ct = {r['trait']: r for r in e12['cge_table']}
tr = e12['traits']
wc = e12['deliverables']['weight_curves']
case = [c for c in e12['deliverables']['case_studies'] if c['nfl_id'] == 59278 and c['trait'] == 'T1'][0]
wr_cases = [c for c in e12['deliverables']['case_studies'] if c['trait'] == 'T1']
bond = [p for p in e7lb['players'] if p['nfl_id'] == 59278][0]
V2 = e12['validation_2']
PR = V2['primary']
SPT = V2['secondary_per_trait']
PHR = e12['post_hoc_review']
PHV2 = PHR['validation_2']
QS = PHV2['qualifying_share_2025_rookies']
ident = [t for t, r in ct.items() if r['identified']]
notid = [t for t, r in ct.items() if not r['identified']]
post_hoc_ids = {p['id'] for p in e12['post_hoc']}
tm8 = {r['id']: r for r in e8['transfer_map']}
tm9 = {r['id']: r for r in e9['transfer_map']}
S8 = e8['summary']
T8 = S8['transfer']
dp = e8['drill_profile']
hw = {t: tr[t]['half_weight_game']['C2']['est'] for t in tr}  # = lambda + CGE (internal fit)
lam = {t: ct[t]['lambda']['est'] for t in ct}

# ---------------------------------------------------------------- Section 1: hook (E012 case study, E007 leaderboard)
chk('A official 40', bond['official_forty'], 4.39, quoted='4.39 forty at the 2025 Combine')
chk('A draft year', bond['draft_year'], 2025, 0)
chk('A is the T1 case study', case['nfl_id'], 59278, 0)
chk('A last-20 speed (flying 20, mph)', bond['flying20_mph'], 23.2, 1, quoted='tracking clocked his last 20 yards at 23.2 mph')
chk('A finishing-speed rank', bond['rank'], 1, 0, quoted='the fastest finish for his 40 time of 91 Combine receivers')
chk('A rank (ph_pairs)', ph_pairs['hook_fast_rank_hidden_of_91'], 1, 0)
chk('rank is by hidden speed', max(p['hidden_speed_mph'] for p in e7lb['players'] if p['hidden_speed_mph'] is not None),
    bond['hidden_speed_mph'], 6)
chk('91 Combine WRs', e7lb['meta']['n'], 91, 0)
chk('2023-24 fit (design label)', 'fit on the 2023-24 seasons only' in V2['design_label'], True,
    quoted='A forecast from his Combine results and draft slot, fit on 2023–24')
chk('T1 per-game value is q90', tr['T1']['per_game_value'] == 'q90', True,
    quoted='(the 90th percentile of his per-play peaks, well below a sprint)')
chk('A C2 prior mph', case['prior_C2'] * MPH, 17.6, 1, quoted='at 17.6 mph')
chk('A position average mph', case['prior_C0'] * MPH, 17.5, 1, quoted='the 17.5 receiver average')
chk('A game 1 mph', case['game_values'][0] * MPH, 18.3, 1, quoted='He hit 18.3 in his first qualifying game')
chk('A season mean mph', case['season_mean'] * MPH, 18.9, 1, quoted='averaged 18.9 over 15')
chk('A n games', case['n_games'], 15, 0)
chk('A z predictive ~ 2', case['z_predictive'], 2, 0, quoted='two forecast standard deviations high')
chk('A largest WR gap (absolute)', case['abs_gap'] == max(c['abs_gap'] for c in wr_cases), True,
    quoted='the largest gap among 2025 rookie receivers with ten or more qualifying games')
chk('A rule largest', case['rule'].startswith('largest'), True)
chk('case-study rule fixed in prereg', '2 largest and 2 smallest' in e12p['deliverables'][2], True,
    quoted='(a rule fixed in advance picked him)')
chk('rule universe >= 10 games', 'among players with >= 10 games' in e12p['deliverables'][2], True)
chk('A has >= 10 games', case['n_games'] >= 10, True)
chk('2023-24 CGE', case['cge_C2_2023_24_fit'], 0.14, 2, quoted='in that fit the forecast was worth only 0.14 games')
chk('T1 prior not flat (so the text must not say flat)', PHV2['per_trait']['T1']['c2_prior_flat'], False)
chk('case CGE = cge_table 2023-24 fit', case['cge_C2_2023_24_fit'], ct['T1']['cge_C2_2023_24_seasons_fit'], 9)
p_two_sd = 2 * (1 - 0.5 * (1 + math.erf(2 / math.sqrt(2))))  # P(|z| >= 2), z standard normal
n_wr = PR['H1']['n_by_trait']['T1']
chk('15 test receivers', n_wr, 15, 0,
    quoted='calibrated forecasts miss at least one of 15 receivers by two standard deviations about half the time')
chk('about half: 1 - (1 - P(|z|>=2))^15', 1 - (1 - p_two_sd) ** n_wr, (0.45, 0.55))
spl = e7lb['meta']['post_hoc_official_splits']
chk('official splits do not corroborate', spl['corr_tracked_vs_official_10_40_time_beyond_forty'] < 0
    and 'not corroborated' in spl['note'], True,
    quoted='official electronic splits do not corroborate tracked finishing speed')
t1c3 = SPT['T1']['1']['blend_C3_vs_blend_C2']
chk('T1 tracking ratio', t1c3['ratio'], 0.96, 2, quoted='left the one-game top-speed error within noise (ratio 0.96, 95% CI 0.80–1.09)')
chk('T1 tracking lo', t1c3['ratio_ci95'][0], 0.80, 2)
chk('T1 tracking hi', t1c3['ratio_ci95'][1], 1.09, 2)
chk('T1 tracking CI spans 1', t1c3['ratio_ci95'][0] < 1 < t1c3['ratio_ci95'][1], True)
chk('A weight > .5 at game 2 (2023-24 fit)', case['weight_C2'][1] > 0.5, True,
    quoted='by his second game his own record outweighed the forecast')
chk('T1 half-weight < 2 (internal fit)', hw['T1'] < 2, True)
chk('Table 1 has 12 traits', len(tr), 12, 0, quoted='Table 1 gives that moment for 12 traits')
allow('tracked 40', 'name of the drill: the 40-yard dash')

# ---------------------------------------------------------------- Section 2: framework
chk('debut 18.3 mph', case['game_values'][0] * MPH, 18.3, 1, quoted='an 18.3 mph debut')
allow('tracked at 10 Hz', 'data spec (AGENTS.md)')
chk('12 traits', len(tr), 12, 0, quoted='For 12 game traits at WR, DL, OL and DB')
chk('positions', sorted({t['position'] for t in tr.values()}) == ['DB', 'DL', 'OL', 'WR'], True)
chk('T1 qualifying = 10 plays', tr['T1']['min_plays_per_game'], 10, 0, quoted='(WR top speed: 10 tracked plays)')
chk('A1 weight form', 'k / (k + lambda + CGE_C)' in e12['amendments'][0]['finding'], True,
    quoted='w = k / (k + λ + CGE)')
chk('C0 half-weight = lambda', max(abs(tr[t]['half_weight_game']['C0']['est'] - lam[t]) for t in tr), 0, 9)
chk('worked example lambda', lam['T1'], 1.5, 1, quoted='For WR top speed (λ 1.5, CGE 0.5), four games give w = 0.67: 0.67 × his average + 0.33 × the prior')
chk('worked example CGE', ct['T1']['cge_C2']['est'], 0.5, 1)
chk('worked example w4', wc['T1']['C2']['w'][4], 0.67, 2)
chk('worked example 1 - w4', 1 - wc['T1']['C2']['w'][4], 0.33, 2)
chk('w4 = 4 / (4 + lambda + CGE)', 4 / (4 + hw['T1']), wc['T1']['C2']['w'][4], 9)
sig_mph = math.sqrt(tr['T1']['variance']['sigma2']['est']) * MPH
se4 = sig_mph / math.sqrt(4 + hw['T1'])  # posterior SD = sqrt(sigma2 / (k + lambda + CGE))
chk('SE after 4 games (mph)', se4, 0.33, 2, quoted='with standard error σ / √(k + λ + CGE) = 0.33 mph (σ = 0.8 mph, the single-game noise)')
chk('single-game noise sigma (mph)', sig_mph, 0.8, 1)
chk('identification rule', all(PHR['per_trait'][t]['identification_rule'].startswith('rel >= 0.5 and odd/even split-half >= 0.5')
                                for t in tr) and 'PH7' in post_hoc_ids, True,
    quoted='reliability or odd/even split-half below 0.5 (post hoc)')
allow('tracked-40 speed profile', 'name of the C3 feature set')
chk('A1 before real data', 'before any CGE' in e12['amendments'][0]['status'], True,
    quoted='one amendment, made before any real-data run, fixed the weight formula')

# ---------------------------------------------------------------- 3.1 and Table 1
chk('nine identified', len(ident), 9, 0, quoted='For the nine reliably measured traits')
chk('hw = lambda + CGE', max(abs(hw[t] - lam[t] - ct[t]['cge_C2']['est']) for t in ident), 0, 9)
chk('hw = weight-curve half_weight', max(abs(hw[t] - wc[t]['C2']['half_weight_game']) for t in ident), 0, 9)
hwi = {t: hw[t] for t in ident}
chk('hw min (T1)', min(hwi.values()), 2.0, 1, quoted='after 2.0 (WR top speed) to 6.8 games')
chk('hw min is T1', min(hwi, key=hwi.get) == 'T1', True)
chk('hw max (T2)', max(hwi.values()), 6.8, 1)
chk('hw max is T2', max(hwi, key=hwi.get) == 'T2', True)
chk('heading 2 to 7', (round(min(hwi.values())), round(max(hwi.values()))) == (2, 7), True,
    quoted='within two to seven games')
chk('lambda min (identified)', min(lam[t] for t in ident), 1.5, 1,
    quoted='the position average is worth 1.5 to 6.8 games, the Combine and draft 0 to 1.2 more')
chk('lambda max (identified)', max(lam[t] for t in ident), 6.8, 1)
chk('CGE min (identified)', min(ct[t]['cge_C2']['est'] for t in ident), 0, 1)
chk('CGE max (identified)', max(ct[t]['cge_C2']['est'] for t in ident), 1.2, 1)
qs = [v['share'] for v in QS.values()]
chk('qualifying share min', 100 * min(qs), 47, 0, quoted='(47–92% of appearances qualified), that is three to 12 appearances')
chk('qualifying share max', 100 * max(qs), 92, 0)
app = {t: (hw[t] if t in ident else lam[t]) / QS[t]['share'] for t in tr}  # qualifying games -> appearances
chk('appearances min (identified) = three', round(min(app[t] for t in ident)), 3, 0)
chk('appearances max (identified) = 12', round(max(app[t] for t in ident)), 12, 0)
lam_ni = sorted(lam[t] for t in notid)
chk('not-identified lambda lo', lam_ni[0], 11, 0, quoted='half signal only after 11–23 qualifying games, about 14–39 appearances')
chk('not-identified lambda hi', lam_ni[-1], 23, 0)
app_ni = sorted(app[t] for t in notid)
chk('not-identified appearances lo', app_ni[0], 14, 0)
chk('not-identified appearances hi', app_ni[-1], 39, 0)
chk('DL pressure appearances', app['T8'], 22, 0, quoted="a rookie pass rusher's pressure rate needs about 22")
allow('fit on 2023–25', 'data scope of the internal fit (weight_curves[T].fit)')
chk('weight curves internal fit', all('2023-25' in wc[t]['fit'] for t in tr), True)
allow('Weight after 1 / 4 / 8', 'table design: the k values shown')
allow('(95% CI)', 'CI level (Table 1 header)')
TABLE = [  # (trait, label, plays, mark, lambda, cge, hw-or-lambda, lo, hi, appearances, weights or None)
    ('T1', 'WR top speed', '≥ 10 plays', '', '1.5', '0.5', '2.0', '1.2', '2.7', '3', ('0.34', '0.67', '0.80')),
    ('T7', 'DL get-off', '≥ 5 rushes', '', '3.3', '0.2', '3.5', '2.4', '5.8', '5', ('0.22', '0.53', '0.69')),
    ('T3', 'WR HITCH braking', '≥ 2 breaks', '', '4.6', '0', '4.6', '3.2', '9.0', '10', ('0.18', '0.46', '0.63')),
    ('T6', 'DL first-second rush distance', '≥ 5 rushes', '', '3.6', '1.2', '4.8', '2.7', '6.9', '6', ('0.17', '0.45', '0.62')),
    ('T5', 'WR yards per route over expected', '≥ 10 routes', '', '4.8', '0', '4.8', '3.1', '13.2', '8', ('0.17', '0.45', '0.62')),
    ('T12', 'DB top speed', '≥ 10 plays', '', '4.8', '0.3', '5.1', '3.1', '9.7', '6', ('0.16', '0.44', '0.61')),
    ('T2', 'WR speed kept through OUT/IN breaks', '≥ 2 breaks', '', '6.8', '0', '6.8', '4.5', '14.0', '12', ('0.13', '0.37', '0.54')),
    ('T9', 'OL pass-set acceleration', '≥ 10 sets', '†', '5.8', '0', '5.8', '3.7', '12.3', '7', None),
    ('T11', 'DB retreat speed', '≥ 3 plays', '†', '5.8', '0', '5.8', '4.3', '9.0', '6', None),
    ('T10', 'OL pressure allowed over expected', '≥ 15 snaps', '‡', '10.8', '—', '10.8', '7.7', '21.1', '14', ('0.08', '0.27', '0.42')),
    ('T8', 'DL pressure over expected', '≥ 10 rushes', '‡', '11.2', '—', '11.2', '6.5', '44.1', '22', ('0.08', '0.26', '0.42')),
    ('T4', 'WR separation over expected', '≥ 3 routes', '‡', '22.8', '—', '22.8', '13.2', '97.8', '39', ('0.04', '0.15', '0.26')),
]
chk('table covers every trait', sorted(r[0] for r in TABLE) == sorted(tr), True)
chk('identified rows sorted by half-weight', [r[0] for r in TABLE[:9] if r[3] == ''] ==
    sorted([t for t in ident if t not in ('T9', 'T11')], key=lambda t: (round(hw[t], 1), t != 'T6')), True)
chk('not-identified rows sorted by lambda', [r[0] for r in TABLE[9:]] == sorted(notid, key=lam.get), True)
chk('dagger rows = non-stationary T9, T11', all('non-stationary' in ' '.join(ct[t]['notes']) for t in ('T9', 'T11')), True)
for t, lab, plays, mark, l_, c_, h_, lo_, hi_, ap_, ws in TABLE:
    wtxt = ' / '.join(ws) if ws else '—'
    line = f'| {lab} ({plays}){mark} | {l_} + {c_} | {h_} ({lo_}–{hi_}) | {ap_} | {wtxt} |'
    chk(f'table {t} min plays', tr[t]['min_plays_per_game'], int(re.search(r'\d+', plays).group()), 0, quoted=line)
    chk(f'table {t} lambda', lam[t], float(l_), 1)
    if t in ident:
        chk(f'table {t} CGE', ct[t]['cge_C2']['est'], float(c_), 1)
        chk(f'table {t} half-weight', hw[t], float(h_), 1)
        ci = tr[t]['half_weight_game']['C2']['ci95_prereg_bootstrap']
    else:
        chk(f'table {t} not identified', ct[t]['identified'], False)
        chk(f'table {t} position-average half-weight = lambda', wc[t]['C0']['half_weight_game'], float(h_), 1)
        ci = ct[t]['lambda']
    chk(f'table {t} CI lo', ci['lo'], float(lo_), 1)
    chk(f'table {t} CI hi', ci['hi'], float(hi_), 1)
    chk(f'table {t} appearances', app[t], float(ap_), 0)
    if ws:
        prior = 'C2' if t in ident else 'C0'
        for k, w in zip((1, 4, 8), ws):
            chk(f'table {t} weight k={k} ({prior})', wc[t][prior]['w'][k], float(w), 2)
zero_t = [t for t in ident if ct[t]['cge_C2']['est'] == 0]
chk('zeros are clipped (CV R2 <= 0)', all(ct[t]['r2_cv']['C2'] <= 0 for t in zero_t), True,
    quoted='0: no out-of-sample gain (clipped)')
shift = max(abs(v['mean_tau']) for t in ('T9', 'T11')
            for v in PHR['per_trait'][t]['stationarity_and_game_unit']['season_change_within_player'].values())
chk('season shift up to 2.2 SD', shift, 2.2, 1, quoted='Levels shift between seasons (up to 2.2 between-player SD): no weights')

# ---------------------------------------------------------------- 3.2 CGE
chk('T6 CGE', ct['T6']['cge_C2']['est'], 1.2, 1, quoted='worth 1.2 games for DL first-second rush distance')
chk('T1 CGE', ct['T1']['cge_C2']['est'], 0.5, 1, quoted='0.5 for WR top speed')
chk('T7 CGE', ct['T7']['cge_C2']['est'], 0.2, 1, quoted='0.2–0.3 for DL get-off and DB top speed')
chk('T12 CGE', ct['T12']['cge_C2']['est'], 0.3, 1)
chk('zero for five', len(zero_t), 5, 0, quoted='and 0 for the other five')
chk('every bootstrap interval includes 0', all(r['cge_C2']['lo'] <= 0 for r in e12['cge_table']), True,
    quoted='The preregistered bootstrap intervals all include 0 but are biased low')
chk('biased low: most T1 draws below the estimate', tr['T1']['cge']['C2']['ci95_prereg_bootstrap']['share_draws_below_est'] > 0.5, True)
chk('biased low: every simulated miss is low (T1, T6)', all(abs(PHR['simulation']['interval_coverage'][t]['b1']['share_upper_below_truth'] - (1 - PHR['simulation']['interval_coverage'][t]['b1']['coverage'])) < 1e-9 for t in ('T1', 'T6')), True)
cov = [PHR['simulation']['interval_coverage'][t]['b1']['coverage'] for t in ('T1', 'T6')]
chk('coverage about 80%', min(cov), (0.75, 0.85), quoted='covering about 80% in simulation')
chk('coverage about 80% (hi)', max(cov), (0.75, 0.85))
chk('T1 calibrated lower', ct['T1']['cge_C2_calibrated_lower95'], 0.08, 2,
    quoted='put WR top speed at 0.08–1.6 games, DL rush distance at 0.10–3.9, and the rest below 1.7–3.6')
chk('T1 calibrated upper', ct['T1']['cge_C2_calibrated_upper95'], 1.6, 1)
chk('T6 calibrated lower', ct['T6']['cge_C2_calibrated_lower95'], 0.10, 2)
chk('T6 calibrated upper', ct['T6']['cge_C2_calibrated_upper95'], 3.9, 1)
rest = [t for t in ident if t not in ('T1', 'T6')]
chk('rest calibrated lower = 0', all(ct[t]['cge_C2_calibrated_lower95'] == 0 for t in rest), True)
chk('rest calibrated upper min', min(ct[t]['cge_C2_calibrated_upper95'] for t in rest), 1.7, 1)
chk('rest calibrated upper max', max(ct[t]['cge_C2_calibrated_upper95'] for t in rest), 3.6, 1)
chk('calibrated bounds post hoc (PH6)', 'PH6' in post_hoc_ids, True)
ns = [r['n_players'] for r in e12['cge_table']]
chk('n players lo', min(ns), 62, 0, quoted='(the Combine as learned from 62–85 players, negative fits clipped to 0)')
chk('n players hi', max(ns), 85, 0)
vs0 = {t: ct[t]['cge_C2_vs_C0_same_cv'] for t in ident}
big = [v for v in vs0.values() if v >= 0.4]
chk('vs C0 same CV: five >= 0.4', len(big), 5, 0,
    quoted='against the position average under the same cross-validation, five traits reach 0.4–1.7 games (post hoc)')
chk('vs C0 same CV min of five', min(big), 0.4, 1)
chk('vs C0 same CV max', max(big), 1.7, 1)
chk('vs C0 same CV post hoc (PH8)', 'PH8' in post_hoc_ids, True)
chk('T6 role reference', ct['T6']['cge_role_reference'], 2.4, 1,
    quoted='roster position alone is worth 2.4 games, the Combine −0.25 beyond it')
chk('T6 beyond role', ct['T6']['cge_C2_beyond_role'], -0.25, 2)
chk('T1 2023-24 fit', ct['T1']['cge_C2_2023_24_seasons_fit'], 0.14, 2,
    quoted='WR top speed falls to 0.14 in the 2023–24 fit and to 0 without one draft class')
chk('T1 LOCO min 0', ct['T1']['cge_C2_leave_one_class_out_range'][0], 0, 2)
r1 = PHR['per_trait']['T1']['references']['role']
chk('T1 role = deep share', r1['reference'].startswith('deep-route share'), True)
chk('T1 r with deep share', r1['r_trait_deep_share'], 0.61, 2,
    quoted='(r 0.61 with deep-route share), though the forecast adds 1.0 game beyond deep share')
chk('T1 Combine beyond deep share', r1['combine_beyond_ref'], 1.0, 1)
chk('T1 beyond deep share = cge_table', r1['combine_beyond_ref'], ct['T1']['cge_C2_beyond_role'], 9)
rel_ni = sorted(ct[t]['rel_player_mean'] for t in notid)
sh_ni = sorted(ct[t]['split_half'] for t in notid)
chk('not-identified rel lo', rel_ni[0], 0.34, 2, quoted='reliability 0.34–0.53, split-half 0.20–0.28')
chk('not-identified rel hi', rel_ni[-1], 0.53, 2)
chk('not-identified split-half lo', sh_ni[0], 0.20, 2)
chk('not-identified split-half hi', sh_ni[-1], 0.28, 2)
chk('T8 raw CV R2', ct['T8']['r2_cv']['C2'], 0.21, 2,
    quoted='the forecast fit DL pressure (cross-validated R² 0.21) about as well as WR top speed (0.22)')
chk('T1 raw CV R2', ct['T1']['r2_cv']['C2'], 0.22, 2)
chk('T8 exposure r', ct['T8']['exposure_r_trait_log_games'], 0.40, 2, quoted='(r +0.40 with qualifying games)')

# ---------------------------------------------------------------- 3.3 tracking
d = [ct[t]['delta_cge_tracking'] for t in ident]
chk('dCGE min', min(x['est'] for x in d), -0.24, 2, quoted='by −0.24 to +0.18 games across the nine traits, every interval including 0')
chk('dCGE max', max(x['est'] for x in d), 0.18, 2)
chk('dCGE intervals include 0', all(x['lo'] <= 0 <= x['hi'] for x in d), True)
chk('H3 is the estimate after one game', 'at k = 1' in PR['H3']['claim'], True,
    quoted='it made the estimate after one game 2.1% worse (95% CI 0.4% to 3.9%; preregistered, 12 traits pooled)')
chk('H3 % worse', 100 * (PR['H3']['ratio'] - 1), 2.1, 1)
chk('H3 lo', 100 * (PR['H3']['ratio_ci95'][0] - 1), 0.4, 1)
chk('H3 hi', 100 * (PR['H3']['ratio_ci95'][1] - 1), 3.9, 1)
chk('H3 not passed', PR['H3']['pass'], False)
chk('H3 pooled over 12 traits', len(PR['H3']['n_by_trait']), 12, 0)

# ---------------------------------------------------------------- 3.4 2025 rookies
allow('we fit on 2023–24', 'data scope of the validation-2 fit')
chk('H1 n players', PR['H1']['n_players'], 81, 0, quoted='(81 rookies, 222 player-traits)')
chk('H1 n player-traits', PR['H1']['n_player_traits'], 222, 0)
chk('pseudo-prospective label', V2['design_label'].startswith('pseudo-prospective'), True,
    quoted='The test is pseudo-prospective: trait definitions were chosen after earlier analyses had used 2025 games')
chk('H1 72%', 100 * (1 - PR['H1']['ratio']), 72, 0, quoted='cut squared error 72% against his game alone')
chk('H1 ratio', PR['H1']['ratio'], 0.28, 2, quoted='(error ratio 0.28, 95% CI 0.20–0.40; 68% on the nine priced traits)')
chk('H1 lo', PR['H1']['ratio_ci95'][0], 0.20, 2)
chk('H1 hi', PR['H1']['ratio_ci95'][1], 0.40, 2)
pool = lambda ts, key: sum(SPT[t]['1'][key] for t in ts) / len(ts)  # H1's pooling: equal-weight mean over traits
chk('H1 pooling reproduced (a)', pool(list(tr), 'se_blend_C2'), PR['H1']['pooled_a'], 9)
chk('H1 pooling reproduced (b)', pool(list(tr), 'se_games'), PR['H1']['pooled_b'], 9)
chk('H1 identified-only cut 68%', 100 * (1 - pool(ident, 'se_blend_C2') / pool(ident, 'se_games')), 68, 0)
chk('H2 52%', 100 * (1 - PR['H2']['ratio']), 52, 0, quoted='after two 52% (both preregistered)')
chk('H1 H2 pass', PR['H1']['pass'] and PR['H2']['pass'], True)
calib = [V2['calibration_80pct']['C2'][str(k)]['pooled_coverage'] for k in range(5)]
chk('80% coverage min', 100 * min(calib), 77, 0, quoted="Its 80% intervals held a rookie's remaining-games average 77–81% of the time (k = 0 to 4)")
chk('80% coverage max', 100 * max(calib), 81, 0)
chk('shrinkage share 99%', 100 * PHV2['shrinkage_share_of_blend_gain']['1'], 99, 0, quoted='captures 99% of that gain')
k0 = V2['post_hoc_combine_prior_vs_position_mean']['k0']['comb_C2_vs_C0']
chk('k0 5%', 100 * (1 - k0['ratio']), 5, 0,
    quoted='Before game 1 the Combine + draft forecast beats the position average by 5% (95% CI 0.3% to 9.6%)')
chk('k0 CI 0.3%', 100 * (1 - k0['ratio_ci95'][1]), 0.3, 1)
chk('k0 CI 9.6%', 100 * (1 - k0['ratio_ci95'][0]), 9.6, 1)
drv = PHV2['ph3_drivers']['k0']
top2 = sorted((k for k in drv if k.startswith('T')), key=lambda k: drv[k]['contribution_to_pooled'])[:2]
chk('drivers T1, T4', sorted(top2) == ['T1', 'T4'], True, quoted='mostly via WR top speed and the unpriced separation trait')
k0i = PHV2['ph3_drivers']['k0_identified_traits_only']
chk('identified only 3.5%', 100 * (1 - k0i['ratio']), 3.5, 1, quoted='on priced traits by 3.5%; after one game by 2.5%')
k1 = V2['post_hoc_combine_prior_vs_position_mean']['k1']['blend_C2_vs_C0']
chk('k1 2.5%', 100 * (1 - k1['ratio']), 2.5, 1)
chk('none survives multiplicity (min BH q > 0.05)', min(PHV2['ph3_bh_q_over_10'].values()) > 0.05, True,
    quoted='All are post hoc; none survives multiplicity control')
chk('PH3 / PH10 post hoc', {'PH3', 'PH10'} <= post_hoc_ids, True)

# ---------------------------------------------------------------- 4. Why (E008, E009)
chk('18 links', T8['n_links'], 18, 0, quoted='Of 18 links from a Combine tracking signature')
chk('16 nulls beyond B2', T8['n_nulls'], 16, 0,
    quoted='16 show nothing detectable beyond the stopwatch and draft slot, and the same 16 beyond the stopwatch alone')
chk('16 nulls beyond B1', e9['summary']['b1_n_nulls'], 16, 0)
chk('same 16', sorted(e9['summary']['b1_passes_c1']) == sorted(T8['passes_c1']), True)
chk('draft delta median', e9['summary']['draft_delta_abs_median'], 0.01, 2, quoted='by a median of 0.01 (at most 0.14; intervals allow 0.37)')
chk('draft delta max', max(abs(x) for x in e9['summary']['draft_delta_signed_range']), 0.14, 2)
chk('draft delta interval limit', e9['summary']['draft_delta_largest_upper_limit']['jackknife'][1], 0.37, 2)
chk('ceiling lo', T8['ceiling_per_link_95'][0], 0.09, 2, quoted='excludes partial correlations above +0.09 to +0.38 (detectable effect 0.31–0.38 at 60–80 players)')
chk('ceiling hi', T8['ceiling_per_link_95'][1], 0.38, 2)
single = [r for r in e8['transfer_map'] if '+' not in r['positions']]
chk('n lo', min(r['n'] for r in single), (60, 62))
chk('n hi', max(r['n'] for r in single), (78, 82))
chk('MDE lo', min(r['mde80_one_sided'] for r in single), 0.31, 2)
chk('MDE hi', max(r['mde80_one_sided'] for r in single), 0.38, 2)
chk('error-free >= .5 for six', len(T8['corrected_beyond_b2_ge_05']), 6, 0, quoted='effects of 0.5 or more remain possible for six links')
g = S8['drill_profile']['by_group_kind']
chk('40 metrics n', g['40-yd dash | repeat attempts']['n'], 28, 0, quoted='All 28 tracked-40 metrics are stable traits (reliability ≥ 0.5; top speed 0.95–0.98)')
chk('40 metrics >= .5', g['40-yd dash | repeat attempts']['n_ge_05'], 28, 0)
chk('top speed rel lo', S8['drill_profile']['forty']['top_speed']['rel'][0], 0.95, 2)
chk('top speed rel hi', S8['drill_profile']['forty']['top_speed']['rel'][1], 0.98, 2)
rep = [r for r in dp if r['kind'] == 'repeat attempts' and not r['scope'].startswith('40')]
chk('repeat drill metrics', len(rep), 22, 0, quoted='Of 22 position-drill metrics with repeat reps, 2 are')
chk('repeat drill >= .5', sum(r['rel_round1_def'] >= 0.5 for r in rep), 2, 0)
chk('cross-drill n', g['position drill | cross-drill']['n'], 68, 0, quoted='of 68 composites across drills that target one skill, 10 are')
chk('cross-drill >= .5', g['position drill | cross-drill']['n_ge_05'], 10, 0)
ws = [r['share_session'] for r in dp if r['metric'] in ('wave rev_accel', 'wave peak_decel', 'wave peak_reaccel', 'wave exit_speed')]
chk('wave rows', len(ws), 4, 0)
chk('wave session lo', 100 * min(ws), 55, 0, quoted='55–63% of OL wave-drill variance lies between Combine sessions')
chk('wave session hi', 100 * max(ws), 63, 0)

# ---------------------------------------------------------------- 5. What tracking does show (E008, E009, E010)
w = tm8['E006-P1']
chk('WR signal r', w['partial_r'], 0.48, 2, quoted='(partial r +0.48, 95% CI +0.20 to +0.68; +0.42 beyond the stopwatch alone)')
chk('WR signal lo', w['ci_lo'], 0.20, 2)
chk('WR signal hi', w['ci_hi'], 0.68, 2)
chk('WR signal beyond B1', tm9['E006-P1']['b1_partial_r'], 0.42, 2)
chk('WR drop3', w['drop3_partial_r'], 0.31, 2, quoted='it falls to +0.31 (−0.01 to +0.57)')
chk('WR drop3 lo', w['drop3_ci_lo'], -0.01, 2)
chk('WR drop3 hi', w['drop3_ci_hi'], 0.57, 2)
mul = S8['multiplicity']['E006-P1']
chk('round-1 tests', mul['m_round1_tests'], 90, 0, quoted='across about 90 first-round tests (q 0.06)')
chk('round-1 q', mul['bh_q_all_round1_tests'], 0.06, 2)
allow('his 40 peak', 'definition: the tracked-40 peak speed')
chk('E010 reliability', e10['combine']['stop_rule']['reliability'], 0.74, 2, quoted='a stable trait (0.74)')
P = e10['game']['primaries']
chk('E010 P1 r', P['P1']['partial_r'], 0.00, 2, quoted='(partial r 0.00 and −0.04; preregistered)')
chk('E010 P2 r', P['P2']['partial_r'], -0.04, 2)
chk('E010 CIs include 0', all(P[k]['partial_r_ci95'][0] < 0 < P[k]['partial_r_ci95'][1] for k in ('P1', 'P2')), True)

# ---------------------------------------------------------------- 6. Recommendations
chk('C0 weight k=1 (T1)', wc['T1']['C0']['w'][1], 0.40, 2, quoted="WR top speed's weights become 0.40 / 0.73 / 0.84 after 1 / 4 / 8 games")
chk('C0 weight k=4 (T1)', wc['T1']['C0']['w'][4], 0.73, 2)
chk('C0 weight k=8 (T1)', wc['T1']['C0']['w'][8], 0.84, 2)
chk('C0 weight = k / (k + lambda)', abs(wc['T1']['C0']['w'][4] - 4 / (4 + lam['T1'])), 0, 9)
r6 = PHR['per_trait']['T6']['references']['role']
chk('T6 role = NFL roster position', r6['reference'].startswith('NFL roster position') and 'latest roster designation' in r6['reference'], True,
    quoted='(post hoc; roles may postdate the draft)')
chk('role beats the Combine (T6)', r6['cge_ref'] > ct['T6']['cge_C2']['est'] and r6['cge_ref_plus_C2'] < r6['cge_ref'], True,
    quoted='roster role (DE, DT, NT, OLB) beats the Combine: worth 2.4 games, outweighed after 6.0 games')
chk('role worth', r6['cge_ref'], 2.4, 1)
chk('role half-weight = lambda + CGE_role', lam['T6'] + ct['T6']['cge_role_reference'], 6.0, 1)
chk('raw first game 3.5x', PR['H1']['pooled_b'] / PR['H1']['pooled_a'], 3.5, 1,
    quoted='had 3.5 times the forecast error of the shrunk estimate')
chk('point estimates from 0', min(ct[t]['cge_C2']['est'] for t in ident), 0, 1,
    quoted='the stopwatch and draft slot, 0 to 1.2 (point estimates), with more than about 4 ruled out')
chk('point estimates to 1.2', max(ct[t]['cge_C2']['est'] for t in ident), 1.2, 1)
chk('more than about 4 ruled out', max(ct[t]['cge_C2_calibrated_upper95'] for t in ident), 4, 0)
dl40 = S8['drill_profile']['forty']['dl_starts']['rel']
chk('40 first-step rel lo', dl40[0], 0.66, 2, quoted='stable traits in the 40 (reliability 0.66–0.79) but mostly not in the DL pass-rush drill (0.23–0.51)')
chk('40 first-step rel hi', dl40[1], 0.79, 2)
pr = [r for r in dp if r['scope'] == 'pass-rush drill attempts']
chk('DL pass-rush rows = the 6 first-step metrics', sorted(r['metric'] for r in pr) == sorted(['d05', 'd10', 's05', 's10', 'amax06', 't3']), True)
chk('DL pass-rush rel lo', min(r['rel_round1_def'] for r in pr), 0.23, 2)
chk('DL pass-rush rel hi', max(r['rel_round1_def'] for r in pr), 0.51, 2)
chk('mostly not: at most 1 of 6 reaches 0.5', sum(r['rel_round1_def'] >= 0.5 for r in pr) <= 1, True)
variants = [r for r in rep if 'reaction' in r['scope'] or 'opposite sides' in r['scope']]
chk('most repeats are mirror-side or reaction reps', len(variants) > len(rep) / 2, True,
    quoted='Most existing repeats are mirror-side or reaction reps')
chk('16 of 22 variants', len(variants), 16, 0)
icc1 = [r['icc_unit'] for r in pr]
chk('single-rep reliability lo', min(icc1), 0.15, 2, quoted='at single-rep reliabilities of 0.15–0.39, two to six identical reps reach 0.5')
chk('single-rep reliability hi', max(icc1), 0.39, 2)
reps_needed = sorted(math.ceil((1 - i) / i) for i in icc1)  # Spearman-Brown: n = (R/(1-R)) (1-icc)/icc, R = 0.5
chk('reps needed two to six', (reps_needed[0], reps_needed[-1]) == (2, 6), True)
chk('wave session effects dominate (> 50%)', min(ws) > 0.5, True, quoted='Session effects dominate the OL wave drill')
chk('2025 comeback', any('2025 COMEBACK is a different movement' in c for c in e8['links']['E002-P3']['caveats']), True,
    quoted='the 2025 comeback drill was a different movement')
chk('E011 2025 only', e11['data']['draft_years'] == [2025], True, quoted='The 2025 ball data (54 receivers, 18 drills, no height)')
chk('E011 players', e11['data']['players'], 54, 0)
chk('E011 drills', e11['data']['drill_names'], 18, 0)
chk('E011 within one frame', 100 * e11['detection']['contact_time_check']['share_within_0.1s'], 87, 0,
    quoted='time 87% of reached balls within one frame')
rel11 = [v['cross_drill']['reliability'] for v in e11['reliability'].values()
         if v.get('cross_drill', {}).get('reliability') is not None]
chk('E011 best reliability', max(rel11), 0.42, 2, quoted='no catch mechanic is yet a stable trait (best 0.42)')
dn = e11['reliability']['catch_offset']['drills_needed_range_from_icc_ci']['for_0.5']
chk('catch drills needed lo', dn[0], 6, 0, quoted='needs about 6–46 catching drills per player')
chk('catch drills needed hi', dn[1], 46, 0)

# ---------------------------------------------------------------- 7. Limitations
chk('n players lo (limits)', min(ns), 62, 0, quoted='62–85 players per trait')
chk('three classes', sorted(tr['T1']['by_class']) == ['2023', '2024', '2025'], True, quoted='three classes, one season for 2025 rookies')
pop = max(r['cge_C2_population_signal_upper95'] for r in e12['cge_table'] if r['cge_C2_population_signal_upper95'] is not None)
chk('population-signal bound', pop, 15, 0, quoted='could be worth up to 15 games in simulation')
nr = [PHR['per_trait'][t]['stationarity_and_game_unit']['noise_ratio_lowest_vs_highest_plays_quartile'] for t in tr]
nr2 = [x for x in nr if x >= 2]
chk('noise ratio >= 2 for 9 of 12', len(nr2), 9, 0, quoted='low-play games are 2.0–3.5 times noisier (9 of 12 traits)')
chk('noise ratio lo', min(nr2), 2.0, 1)
chk('noise ratio hi', max(nr2), 3.5, 1)
roles = ' '.join(PHR['per_trait'][t]['references']['role']['reference'] for t in ('T6', 'T12'))
chk('roles pooled (DE/DT, CB/FS)', all(x in roles for x in ('DE', 'DT', 'CB', 'FS')), True,
    quoted='position averages pool roles (edge with interior, corner with safety)')
# spelled-out numbers (the uncovered-number scan below sees digits only)
chk('heading: fifteen games', case['n_games'], 15, 0, quoted='One rookie, fifteen games')
chk('four priors = C0-C3', len([k for k in e12p['information_sets'] if re.fullmatch(r'C\d', k)]), 4, 0, quoted='We price four priors')
chk('three most supportive receivers', w['n'] - w['drop3_n'], 3, 0, quoted='without its three most supportive receivers')
chk('heading: about one game (max CGE)', max(ct[t]['cge_C2']['est'] for t in ident), 1, 0, quoted='a fraction of a game to about one')
allow('2025 Combine', 'data scope')
allow('2025 rookie', 'data scope')
allow('2025 rookies', 'data scope')
allow('2025 games', 'data scope')
allow('the 2025 test', 'data scope')
allow('game 1', 'illustrative game number')
allow('(95% CIs)', 'CI level (Figure 4 caption)')

# ---------------------------------------------------------------- coverage of every numeric token
spans = sorted(covered)


def is_covered(a, b):
    return any(s <= a and b <= e for s, e in spans)


struct = []  # numbers that are structure, not results
for m in re.finditer(r'^(#+ )([\d.]+)', text, flags=re.M):  # section numbers
    struct.append((m.start(2), m.end(2)))
for m in re.finditer(r'^(\d+)\. ', text, flags=re.M):  # list numbers
    struct.append((m.start(1), m.end(1)))
for m in re.finditer(r'(?:Figure|Table|Section|figures/fig)\s*(\d+)', text):  # figure, table and section labels
    struct.append((m.start(1), m.end(1)))
for m in re.finditer(r'figures/fig\d_[\w.]+', text):  # image paths
    struct.append((m.start(), m.end()))
uncovered = []
for m in re.finditer(r'\d+(?:[.,]\d+)*', text):
    a, b = m.start(), m.end()
    if is_covered(a, b) or any(s <= a and b <= e for s, e in struct):
        continue
    line = text[text.rfind('\n', 0, a) + 1:text.find('\n', b)]
    uncovered.append(f'{m.group()!r} in: {line.strip()[:110]}')
for u in uncovered:
    print('FAIL uncovered number', u)
    fails.append('uncovered ' + u)

# ---------------------------------------------------------------- every number-bearing block has a src comment nearby
blocks = re.split(r'\n\s*\n', raw)
if '<!--' not in raw:  # a comment-stripped copy (the Kaggle paste): numbers are still checked, sources cannot be
    print('\nstripped copy (no HTML comments): src-comment check skipped')
    blocks = []
for i, blk in enumerate(blocks):
    body = re.sub(r'<!--.*?-->', ' ', blk, flags=re.S)
    if blk.lstrip().startswith(('#', '![', '**Figure', '**Table', '**Code', '**Clubs', '**The league')):
        continue
    body_nums = re.sub(r'(?:Figure|Table|Section)\s*\d+|^\d+\. |^#+.*$|game [13]\b|\b40\b|10 Hz', ' ', body, flags=re.M)
    if not re.search(r'\d', body_nums):
        continue
    if not any('<!-- src' in b for b in blocks[i:i + 3]):
        print('FAIL no src comment near block', i, body.strip()[:80])
        fails.append(f'no src comment near block {i}')

# ---------------------------------------------------------------- budget
plain = re.sub(r'<!--.*?-->', ' ', raw, flags=re.S)
plain = re.sub(r'!\[[^\]]*\]\([^)]*\)', ' ', plain)
plain = re.sub(r'\[\[TBD[^\]]*\]\]', ' ', plain)
words = [t for t in re.split(r'\s+', plain) if re.search(r'\w', t)]  # \w is Unicode: λ and σ tokens are rendered words
items = sorted(set(re.findall(r'\*\*(Figure \d+|Table \d+)\.\*\*', raw)))
print(f'\nwords: {len(words)} (limit 1,950); figures + tables: {len(items)} {items} (limit 8)')
if len(words) > 1950:
    fails.append(f'word count {len(words)} > 1,950')
if len(items) > 8:
    fails.append(f'{len(items)} figures + tables > 8')
tbd = re.findall(r'\[\[TBD[^\]]*\]\]', raw)
print('open placeholders:', tbd)

print(f'\n{len(fails)} failures')
for f in fails:
    print('  -', f)
sys.exit(1 if fails else 0)
