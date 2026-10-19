"""Check every number quoted in writeup/draft.md (v2.6, Combine Game Equivalents) against its committed source.

For each quoted number this script:
  1. recomputes it from its source JSON key (experiments/results/E0xx.json (E007-E013), E007_leaderboard.json,
     experiments/posthoc_writeup_v1/*.json, experiments/posthoc_writeup_v2/e012_case_priors.json) and compares
     at the quoted rounding. A few numbers are arithmetic
     on committed keys (sums, ratios, a Spearman-Brown inversion, a binomial chance); the formula is written
     next to each such check and in the draft's src comment;
  2. asserts the quoted string (the exact phrase carrying the number) is present in the draft text
     (HTML comments stripped).
Then it fails on any numeric token in the text that no check (or the short allowlist of non-result
constants below) covers, on any number-bearing paragraph without a "<!-- src" comment nearby, and on the
budget (HTML comments and image links removed; captions and table cells count):
  - rendered words <= 1,900: tokens with a letter or digit in the rendered text (table pipes become cell
    breaks; heading, quote and bullet markers and the table separator row are dropped);
  - whitespace tokens <= 1,950, counted both on the pasted markdown and on the rendered text, so any word
    counter a judge uses stays well under the 2,000 limit;
  - <= 8 figures + tables.
It also checks the named players (user decision 2026-10-18; writeup/outline.md section 4): each name in NAMED
matches every committed record for his nfl_id (E007.json, E007_leaderboard.json, E012.json), no other player in
those files or in E013.json is named, no "Receiver A/B" placeholder is left, each named player's first mention is his
full name, and no draft-status, off-field, draft-slot or stopwatch-only reference appears in a sentence about a named
player: in the text (a sentence that names him, or one with he/his/him in a section that names him) and in the HTML
comments (a sentence that names him or his nfl_id, or one with he/his/him in a comment that does).

Run: .venv/bin/python writeup/check_numbers.py [path/to/draft.md]   (exit code 1 on any failure)
On a comment-stripped copy (the text pasted into Kaggle) the src-comment check is skipped; every number is still checked.
"""
import difflib
import json
import math
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
R = ROOT / 'experiments' / 'results'
PH = ROOT / 'experiments' / 'posthoc_writeup_v1'
PH2 = ROOT / 'experiments' / 'posthoc_writeup_v2'
DRAFT = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / 'writeup' / 'draft.md'

load = lambda p: json.loads(p.read_text())
e7 = load(R / 'E007.json')
e7lb = load(R / 'E007_leaderboard.json')
e8 = load(R / 'E008.json')
e9 = load(R / 'E009.json')
e10 = load(R / 'E010.json')
e11 = load(R / 'E011.json')
e12 = load(R / 'E012.json')
e12p = load(R / 'E012_prereg.json')
e13 = load(R / 'E013.json')  # role-aware re-fit (v2.4: Section 3.2 correction and DL wording, Section 3.4 note)
e13p = load(R / 'E013_prereg.json')
ph_pairs = load(PH / 'ph_pairs.json')
cp = load(PH2 / 'e012_case_priors.json')  # post hoc export of stored E012 forecasts (case-study priors C0-C3, pools)

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
S8 = e8['summary']
T8 = S8['transfer']
dp = e8['drill_profile']
hw = {t: tr[t]['half_weight_game']['C2']['est'] for t in tr}  # = lambda + CGE (internal fit)
lam = {t: ct[t]['lambda']['est'] for t in ct}

# ---------------------------------------------------------------- Section 1: hook (E012 case study, E007 leaderboard)
chk('A official 40', bond['official_forty'], 4.39, quoted='4.39 forty at the 2025 Combine')
chk('A Combine class 2025', bond['draft_year'], 2025, 0)
chk('A is the T1 case study', case['nfl_id'], 59278, 0)
chk('A last-20 speed (flying 20, mph)', bond['flying20_mph'], 23.2, 1, quoted='tracking clocked his last 20 yards at 23.2 mph')
chk('A finishing-speed rank', bond['rank'], 1, 0, quoted='the fastest finish for his 40 time of 91 Combine receivers')
chk('A rank (ph_pairs)', ph_pairs['hook_fast_rank_hidden_of_91'], 1, 0)
chk('rank is by hidden speed', max(p['hidden_speed_mph'] for p in e7lb['players'] if p['hidden_speed_mph'] is not None),
    bond['hidden_speed_mph'], 6)
chk('91 Combine WRs', e7lb['meta']['n'], 91, 0)
chk('2023-24 fit (design label)', 'fit on the 2023-24 seasons only' in V2['design_label'], True,
    quoted='A Combine + draft forecast, fit on 2023–24')
chk('T1 per-game value is q90', tr['T1']['per_game_value'] == 'q90', True,
    quoted='put his typical game top speed (90th-percentile play peak, well below a sprint)')
chk('A C2 prior mph', case['prior_C2'] * MPH, 17.6, 1, quoted='at 17.6 mph')
chk('A position average mph', case['prior_C0'] * MPH, 17.5, 1, quoted='the 17.5 receiver average')
chk('A game 1 mph', case['game_values'][0] * MPH, 18.3, 1, quoted='He hit 18.3 in his first qualifying game')
chk('A season mean mph', case['season_mean'] * MPH, 18.9, 1, quoted='averaged 18.9 over 15')
chk('A n games', case['n_games'], 15, 0)
chk('A z predictive ~ 2', case['z_predictive'], 2, 0, quoted='two forecast standard deviations high')
pool1 = cp['pools']['T1']
chk('T1 test receivers (k = 0 pool)', pool1['n_test_players_k0'], 15, 0)
chk('T1 test receivers with 10+ games', pool1['n_with_10plus_games'], 11, 0,
    quoted='the largest gap among the 11 of 15 rookie receivers in our 2025 test with ten or more qualifying games')
chk('A largest WR gap (absolute)', case['abs_gap'] == max(c['abs_gap'] for c in wr_cases), True)
chk('A rule largest', case['rule'].startswith('largest'), True)
chk('case-study rule fixed in prereg', '2 largest and 2 smallest' in e12p['deliverables'][2], True,
    quoted='(a rule fixed in advance chose him)')
chk('rule universe >= 10 games', 'among players with >= 10 games' in e12p['deliverables'][2], True,
    quoted='ten or more qualifying games')
chk('A has >= 10 games', case['n_games'] >= 10, True)
chk('2023-24 CGE', case['cge_C2_2023_24_fit'], 0.14, 2, quoted='in that fit the forecast was worth only 0.14 games')
chk('T1 prior not flat (so the text must not say flat)', PHV2['per_trait']['T1']['c2_prior_flat'], False)
chk('case CGE = cge_table 2023-24 fit', case['cge_C2_2023_24_fit'], ct['T1']['cge_C2_2023_24_seasons_fit'], 9)
p_two_sd = 2 * (1 - 0.5 * (1 + math.erf(2 / math.sqrt(2))))  # P(|z| >= 2), z standard normal
chk('15 = H1 T1 test receivers', PR['H1']['n_by_trait']['T1'], pool1['n_test_players_k0'], 0)
chk('15 = T1 tracking-test receivers', SPT['T1']['1']['n'], pool1['n_test_players_k0'], 0)
n_sel = pool1['n_with_10plus_games']  # the chance argument uses the pool the rule chose from
chk('about 40%: 1 - (1 - P(|z|>=2))^11', 100 * (1 - (1 - p_two_sd) ** n_sel), 40, 0,
    quoted='among 11 receivers a calibrated forecast misses at least one by two standard deviations about 40% of the time')
cpa = [c for c in cp['cases'] if c['nfl_id'] == 59278 and c['trait'] == 'T1'][0]
for key in ('prior_C0', 'prior_C2', 'season_mean'):  # the export matches E012.json's committed case study
    chk(f'case-priors export {key} = E012.json', cpa[key], case[key], 9)
chk('tracking shift = C3 - C2', cpa['tracking_shift_C3_minus_C2'], cpa['prior_C3'] - cpa['prior_C2'], 9)
chk('A tracking nudge (mph)', cpa['tracking_shift_C3_minus_C2'] * MPH, 0.3, 1,
    quoted='Combine tracking nudged his forecast 0.3 mph toward his season average')
chk('nudge is toward his season (C2 < C3 < season)', cpa['prior_C2'] < cpa['prior_C3'] < cpa['season_mean'], True)
chk('with tracking, still far below his season (> 1 mph)', (cpa['season_mean'] - cpa['prior_C3']) * MPH > 1, True)
spl = e7lb['meta']['post_hoc_official_splits']
chk('official splits do not corroborate', spl['corr_tracked_vs_official_10_40_time_beyond_forty'] < 0
    and 'not corroborated' in spl['note'], True,
    quoted='official electronic splits do not corroborate tracked finishing speed')
t1c3 = SPT['T1']['1']['blend_C3_vs_blend_C2']
chk('T1 tracking ratio', t1c3['ratio'], 0.96, 2, quoted='across all 15 it left error after one game within noise (ratio 0.96, 95% CI 0.80–1.09)')
chk('T1 tracking test n', t1c3['n_players'], 15, 0)
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
chk('T1 qualifying = 10 plays (Table 1: ≥10 plays)', tr['T1']['min_plays_per_game'], 10, 0)
chk('A1 weight form', 'k / (k + lambda + CGE_C)' in e12['amendments'][0]['finding'], True,
    quoted='w = k/(k + λ + CGE)')
chk('C0 half-weight = lambda', max(abs(tr[t]['half_weight_game']['C0']['est'] - lam[t]) for t in tr), 0, 9)
chk('worked example lambda', lam['T1'], 1.5, 1, quoted='For WR top speed (λ 1.5, CGE 0.5), four games give w = 0.67')
chk('worked example CGE', ct['T1']['cge_C2']['est'], 0.5, 1)
chk('worked example w4', wc['T1']['C2']['w'][4], 0.67, 2)
chk('w4 = 4 / (4 + lambda + CGE)', 4 / (4 + hw['T1']), wc['T1']['C2']['w'][4], 9)
sig_mph = math.sqrt(tr['T1']['variance']['sigma2']['est']) * MPH
se4 = sig_mph / math.sqrt(4 + hw['T1'])  # posterior SD = sqrt(sigma2 / (k + lambda + CGE))
chk('SE after 4 games (mph)', se4, 0.33, 2, quoted='with standard error σ/√(k + λ + CGE) = 0.33 mph (σ = 0.8 mph, the single-game noise)')
chk('single-game noise sigma (mph)', sig_mph, 0.8, 1)
chk('identification rule', all(PHR['per_trait'][t]['identification_rule'].startswith('rel >= 0.5 and odd/even split-half >= 0.5')
                                for t in tr) and 'PH7' in post_hoc_ids, True,
    quoted='reliability or odd/even split-half below 0.5 (post hoc)')
allow('tracked-40 speed profile', 'name of the C3 feature set')
chk('scope: the estimand is the in-game trait level of a playing rookie (per qualifying game)', all(t['min_plays_per_game'] > 0 for t in tr.values()), True,
    quoted="It prices forecasts of a playing rookie's trait level, not who plays or is drafted")
chk('worth scales with lambda: CGE = lambda R2 / (1 - R2) (framework)', 'lambda' in json.dumps(e12p['framework']) and 'R2' in json.dumps(e12p['framework']), True,
    quoted='worth scales with λ, so priors matter least for fast-stabilizing traits')
chk('A1 before real data', 'before any CGE' in e12['amendments'][0]['status'], True,
    quoted='one amendment before any real-data run fixed the weight formula')

# ---------------------------------------------------------------- 3.1 and Table 1
chk('nine identified', len(ident), 9, 0, quoted='For the nine priced traits')
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
chk('qualifying share min (not quoted since v2.5)', 100 * min(qs), 47, 0)
chk('qualifying share max', 100 * max(qs), 92, 0)
app = {t: (hw[t] if t in ident else lam[t]) / QS[t]['share'] for t in tr}  # qualifying games -> appearances
chk('appearances min (identified) = three', round(min(app[t] for t in ident)), 3, 0)
chk('appearances max (identified) = 12', round(max(app[t] for t in ident)), 12, 0,
    quoted="At 2025 rookies' qualifying rates, that is three to 12 appearances")
lam_ni = sorted(lam[t] for t in notid)
chk('not-identified lambda lo (1 dp, as Table 1)', lam_ni[0], 10.8, 1, quoted='half signal only after 10.8–22.8 qualifying games, about 14–39 appearances')
chk('not-identified lambda hi (1 dp, as Table 1)', lam_ni[-1], 22.8, 1)
app_ni = sorted(app[t] for t in notid)
chk('not-identified appearances lo', app_ni[0], 14, 0)
chk('not-identified appearances hi', app_ni[-1], 39, 0)
chk('DL pressure appearances (Table 1)', app['T8'], 22, 0)
allow('fit on 2023–25', 'data scope of the internal fit (weight_curves[T].fit)')
chk('weight curves internal fit', all('2023-25' in wc[t]['fit'] for t in tr), True)
allow('Weight after 1/4/8', 'table design: the k values shown')
allow('(95% CI)', 'CI level (Table 1 header)')
TABLE = [  # (trait, label, plays, mark, lambda, cge, hw-or-lambda, lo, hi, appearances, weights or None)
    ('T1', 'WR top speed', '≥10 plays', '', '1.5', '0.5', '2.0', '1.2', '2.7', '3', ('0.34', '0.67', '0.80')),
    ('T7', 'DL get-off', '≥5 rushes', '', '3.3', '0.2', '3.5', '2.4', '5.8', '5', ('0.22', '0.53', '0.69')),
    ('T3', 'WR HITCH braking', '≥2 breaks', '', '4.6', '0', '4.6', '3.2', '9.0', '10', ('0.18', '0.46', '0.63')),
    ('T6', 'DL first-second rush distance', '≥5 rushes', '', '3.6', '1.2', '4.8', '2.7', '6.9', '6', ('0.17', '0.45', '0.62')),
    ('T5', 'WR yards per route over expected', '≥10 routes', '', '4.8', '0', '4.8', '3.1', '13.2', '8', ('0.17', '0.45', '0.62')),
    ('T12', 'DB top speed', '≥10 plays', '', '4.8', '0.3', '5.1', '3.1', '9.7', '6', ('0.16', '0.44', '0.61')),
    ('T2', 'WR speed kept through OUT/IN breaks', '≥2 breaks', '', '6.8', '0', '6.8', '4.5', '14.0', '12', ('0.13', '0.37', '0.54')),
    ('T9', 'OL pass-set acceleration', '≥10 sets', '†', '5.8', '0', '5.8', '3.7', '12.3', '7', None),
    ('T11', 'DB retreat speed', '≥3 plays', '†', '5.8', '0', '5.8', '4.3', '9.0', '6', None),
    ('T10', 'OL pressure allowed over expected', '≥15 snaps', '‡', '10.8', '—', '10.8', '7.7', '21.1', '14', ('0.08', '0.27', '0.42')),
    ('T8', 'DL pressure over expected', '≥10 rushes', '‡', '11.2', '—', '11.2', '6.5', '44.1', '22', ('0.08', '0.26', '0.42')),
    ('T4', 'WR separation over expected', '≥3 routes', '‡', '22.8', '—', '22.8', '13.2', '97.8', '39', ('0.04', '0.15', '0.26')),
]
chk('table covers every trait', sorted(r[0] for r in TABLE) == sorted(tr), True)
chk('identified rows sorted by half-weight', [r[0] for r in TABLE[:9] if r[3] == ''] ==
    sorted([t for t in ident if t not in ('T9', 'T11')], key=lambda t: (round(hw[t], 1), t != 'T6')), True)
chk('not-identified rows sorted by lambda', [r[0] for r in TABLE[9:]] == sorted(notid, key=lam.get), True)
chk('dagger rows = non-stationary T9, T11', all('non-stationary' in ' '.join(ct[t]['notes']) for t in ('T9', 'T11')), True)
for t, lab, plays, mark, l_, c_, h_, lo_, hi_, ap_, ws in TABLE:
    wtxt = '/'.join(ws) if ws else '—'
    line = f'|{lab} ({plays}){mark}|{l_}+{c_}|{h_} ({lo_}–{hi_})|{ap_}|{wtxt}|'  # compact separators (word budget)
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
chk('Table 1 CIs = preregistered bootstrap, CGE part biased low (draws below the estimate; every simulated miss low)',
    all('ci95_prereg_bootstrap' in tr[t]['half_weight_game']['C2'] for t in ident)
    and tr['T1']['cge']['C2']['ci95_prereg_bootstrap']['share_draws_below_est'] > 0.5, True,
    quoted='CIs: preregistered bootstrap, CGE part biased low')
shift = max(abs(v['mean_tau']) for t in ('T9', 'T11')
            for v in PHR['per_trait'][t]['stationarity_and_game_unit']['season_change_within_player'].values())
chk('season shift up to 2.2 SD', shift, 2.2, 1, quoted='Levels shift between seasons (up to 2.2 between-player SD): no weights')

# ---------------------------------------------------------------- 3.2 CGE
chk('T6 CGE', ct['T6']['cge_C2']['est'], 1.2, 1, quoted='DL rush distance (1.2)')
chk('T1 CGE', ct['T1']['cge_C2']['est'], 0.5, 1, quoted='WR top speed (0.5) falls to')
chk('T6, T1 are the two largest', sorted(ident, key=lambda t: -ct[t]['cge_C2']['est'])[:2] == ['T6', 'T1'], True,
    quoted='The two largest values are soft (post hoc)')
chk('zero for five (Table 1 zeros)', len(zero_t), 5, 0)
chk('worth from 0 (min)', min(ct[t]['cge_C2']['est'] for t in ident), 0, 1, quoted='the Combine + draft forecast is worth 0 to 1.2 games (Figure 2)')
chk('runs to 1.2 (max)', max(ct[t]['cge_C2']['est'] for t in ident), 1.2, 1)
# v2.5 (judge 2 item 4): the stopwatch and measurements alone (information set C1, preregistered), pooled by trait;
# never for a named player (outline section 4). The draft slot's increment = CGE(C2) - CGE(C1), point estimates.
c1 = {t: ct[t]['cge_C1']['est'] for t in ident}
slot_inc = {t: ct[t]['cge_C2']['est'] - c1[t] for t in ident}
chk('C1 is a preregistered information set (stopwatch and measurements)', 'traditional Combine' in e12p['information_sets']['C1'], True,
    quoted='without the draft slot, nearly the same (it adds at most 0.1)')
chk('C1 worth max (nearly the same: 1.2)', max(c1.values()), 1.2, 1)
chk('C1 worth min 0', min(c1.values()), 0, 1)
chk('draft slot adds at most 0.1 (max over priced traits)', max(slot_inc.values()), 0.1, 1)
chk('draft slot never lowers the worth (increments >= 0)', min(slot_inc.values()) >= -1e-9, True)
chk('every bootstrap interval includes 0', all(r['cge_C2']['lo'] <= 0 for r in e12['cge_table']), True,
    quoted='The preregistered bootstrap intervals all include 0 but are biased low')
chk('biased low: most T1 draws below the estimate', tr['T1']['cge']['C2']['ci95_prereg_bootstrap']['share_draws_below_est'] > 0.5, True)
chk('biased low: every simulated miss is low (T1, T6)', all(abs(PHR['simulation']['interval_coverage'][t]['b1']['share_upper_below_truth'] - (1 - PHR['simulation']['interval_coverage'][t]['b1']['coverage'])) < 1e-9 for t in ('T1', 'T6')), True)
cov = [PHR['simulation']['interval_coverage'][t]['b1']['coverage'] for t in ('T1', 'T6')]
chk('coverage about 80%', min(cov), (0.75, 0.85), quoted='covering about 80% in simulation')
chk('coverage about 80% (hi)', max(cov), (0.75, 0.85))
CB = PHR['simulation']['calibrated_bounds']['traits']
chk('calibrated bounds are two one-sided 95% bounds = a 90% two-sided interval', all('calibrated_one_sided_95' in CB[t] for t in ident), True,
    quoted='Simulation-calibrated 90% bounds (post hoc) put WR top speed at 0.08–1.6 games, DL rush distance at 0.10–3.9, and the rest below 1.7–3.6')
chk('calibrated bounds = cge_table keys', max(abs(CB[t]['calibrated_one_sided_95']['upper'] - ct[t]['cge_C2_calibrated_upper95']) for t in ident), 0, 9)
chk('T1 calibrated lower', ct['T1']['cge_C2_calibrated_lower95'], 0.08, 2)
chk('T1 calibrated upper', ct['T1']['cge_C2_calibrated_upper95'], 1.6, 1)
chk('T6 calibrated lower', ct['T6']['cge_C2_calibrated_lower95'], 0.10, 2)
chk('T6 calibrated upper', ct['T6']['cge_C2_calibrated_upper95'], 3.9, 1)
rest = [t for t in ident if t not in ('T1', 'T6')]
chk('rest calibrated lower = 0', all(ct[t]['cge_C2_calibrated_lower95'] == 0 for t in rest), True)
chk('rest calibrated upper min', min(ct[t]['cge_C2_calibrated_upper95'] for t in rest), 1.7, 1)
chk('rest calibrated upper max', max(ct[t]['cge_C2_calibrated_upper95'] for t in rest), 3.6, 1)
chk('calibrated bounds post hoc (PH6)', 'PH6' in post_hoc_ids, True)
ns = [r['n_players'] for r in e12['cge_table']]
chk('n players lo', min(ns), 62, 0, quoted='Learned from 62–85 players per trait')
chk('n players hi', max(ns), 85, 0)
# v2.4 correction (E013 review): E012's cge_C2_vs_C0_same_cv mixed ridge shrinkage of the class means into the Combine
# value ("five traits reach 0.4-1.7"); the reference is now a ridge on the draft-year dummies alone (E013 PH8).
S13 = e13['post_hoc_review']['summary']
I13 = e13['post_hoc_review']['internal']
ph13 = {p['id']: p['what'] for p in e13['post_hoc']}
dr = S13['e012_C2_vs_dummies_ridge_same_cv']
chk('same-CV reference = ridge on the draft-year dummies alone (E013 PH8, post hoc)', 'draft-year dummies alone' in ph13['PH8'], True,
    quoted='The estimate may be conservative: against draft-class averages under the same cross-validation, WR top speed reaches 0.63 games and DL rush distance 1.68 (intervals include 0), the rest under 0.4;')
chk('vs class averages, same CV: WR top speed', dr['T1'], 0.63, 2)
chk('vs class averages, same CV: DL rush distance', dr['T6'], 1.68, 2)
chk('summary = per-trait key (T1, T6)', max(abs(dr[t] - I13[t]['same_cv_vs_dummies_ridge']['e012_C2']['est']) for t in ('T1', 'T6')), 0, 9)
chk('over the 9 priced traits', sorted(dr) == sorted(ident), True)
chk('only WR top speed and DL rush distance reach 0.4 (the rest under 0.4)', sorted(t for t, v in dr.items() if v >= 0.4) == ['T1', 'T6']
    and sorted(S13['e012_C2_vs_dummies_ridge_ge_0.4']) == ['T1', 'T6'], True)
chk('both intervals include 0', all(I13[t]['same_cv_vs_dummies_ridge']['e012_C2']['lo'] <= 0 for t in ('T1', 'T6')), True)
chk('"may be conservative": both exceed the main estimate', all(dr[t] > ct[t]['cge_C2']['est'] for t in ('T1', 'T6')), True)
pops = [ct[t]['cge_C2_population_signal_upper95'] for t in ident]
chk('perfectly learned signal: population-signal upper bounds, min (priced traits)', min(pops), 4, 0,
    quoted='a perfectly learned Combine signal could be worth up to 4–15 games (both post hoc)')
chk('perfectly learned signal: max', max(pops), 15, 0)
chk('population-signal estimand = perfectly learned linear signal (PH6)', 'perfectly learned' in PHR['simulation']['calibrated_bounds']['estimand'], True)
chk('the contaminated "five traits" claim is gone', 'five traits reach' not in text, True)
chk('T6 role reference', ct['T6']['cge_role_reference'], 2.4, 1,
    quoted='designation alone (which may postdate the draft) is worth 2.4 games, the Combine + draft forecast −0.25 beyond it')
chk('T6 role reference = the four-level NFL roster position, possibly post-draft', PHR['per_trait']['T6']['references']['role']['reference'].startswith('NFL roster position')
    and all(x in PHR['per_trait']['T6']['references']['role']['reference'] for x in ('DE', 'DT', 'NT', 'OLB', 'latest roster designation')), True)
chk('T6 beyond role', ct['T6']['cge_C2_beyond_role'], -0.25, 2)
rd = I13['T6']['roster_designation']
ols = rd['ols_hc1_role_weight_olb']
pe = rd['pure_edge_olb_vs_de']
r2s = rd['r2_in_sample']
chk('DL: tracks body type and roster designation (in-sample R2: designation > weight > 0.1; descriptive)', r2s['roster_position'] > r2s['weight'] > 0.1
    and ols['olb']['p'] < 0.05, True,
    quoted='DL rush distance (1.2) tracks body type and roster designation (descriptive)')
# v2.5 (judges): "not edge alignment" and "OLB vs DE" never explain this trait's 1.2 or 2.4 games (the label is four-level).
# v2.6 (final checker F2): "OLB vs DE" is allowed once, in the Roles note's PH10 sentence (edge share given designation).
sec32 = text[text.index('### 3.2'):text.index('### 3.3')]
chk('the v2.4 "not edge alignment" overreach is gone (judges, v2.5); "OLB vs DE" only in the Roles note (v2.6)',
    'not edge alignment' not in text and 'OLB vs DE' not in sec32 and text.count('OLB vs DE') == 1
    and text.index('OLB vs DE') > text.index('**Roles.**'), True)
chk('DL: among pure edge players OLB > DE (Welch p < 0.05; all three classes)', pe['mean_olb'] > pe['mean_de'] and pe['welch_p_two_sided'] < 0.05
    and all(v['olb'] > v['de'] for v in pe['by_class'].values()), True)
chk('roster designation analysis is post hoc and descriptive (E013 PH10)', 'possibly the latest roster label' in ph13.get('PH10', '')
    and 'descriptive' in ph13['PH10'], True)
chk('T1 2023-24 fit', ct['T1']['cge_C2_2023_24_seasons_fit'], 0.14, 2,
    quoted='falls to 0.14 in the 2023–24 fit and to 0 without one draft class')
chk('T1 LOCO min 0', ct['T1']['cge_C2_leave_one_class_out_range'][0], 0, 2)
r1 = PHR['per_trait']['T1']['references']['role']
chk('T1 role = deep share', r1['reference'].startswith('deep-route share'), True)
chk('T1 r with deep share', r1['r_trait_deep_share'], 0.61, 2,
    quoted='(r 0.61 with deep-route share), though the forecast adds 1.0 game beyond deep share')
chk('T1 Combine beyond deep share', r1['combine_beyond_ref'], 1.0, 1)
chk('T1 beyond deep share = cge_table', r1['combine_beyond_ref'], ct['T1']['cge_C2_beyond_role'], 9)
rel_ni = sorted(ct[t]['rel_player_mean'] for t in notid)
sh_ni = sorted(ct[t]['split_half'] for t in notid)
chk('three not priced', len(notid), 3, 0, quoted='The three outcome traits cannot be priced')
chk('not-identified rel lo', rel_ni[0], 0.34, 2,
    quoted='at reliability 0.34–0.53 and split-half 0.20–0.28, simulation cannot bound their worth above (post hoc)')
chk('no calibrated upper bound for any not-priced trait (not bounded within the grid)', all(ct[t]['cge_C2_calibrated_upper95'] is None
    and CB[t]['calibrated_one_sided_95']['upper'] is None for t in notid), True)
chk('E012 notes: do not interpret the not-priced CGE (none quoted)', all(any('do not interpret' in n for n in ct[t]['notes']) for t in notid), True)
chk('not-identified rel hi', rel_ni[-1], 0.53, 2)
chk('not-identified split-half lo', sh_ni[0], 0.20, 2)
chk('not-identified split-half hi', sh_ni[-1], 0.28, 2)
chk('T8 raw CV R2', ct['T8']['r2_cv']['C2'], 0.21, 2,
    quoted='the forecast fit DL pressure (cross-validated R² 0.21) about as well as WR top speed (0.22)')
chk('T1 raw CV R2', ct['T1']['r2_cv']['C2'], 0.22, 2)
chk('T8 exposure r', ct['T8']['exposure_r_trait_log_games'], 0.40, 2, quoted='(r +0.40 with qualifying games)')

# ---------------------------------------------------------------- 3.3 tracking
# v2.5: the dCGE range (-0.24 to +0.18) and "every interval including 0" are no longer quoted (judge 2: degenerate B1 intervals)
chk('dCGE range no longer quoted', 'every interval including 0' not in text and '−0.24 to +0.18' not in text, True)
chk('H3 is the estimate after one game', 'at k = 1' in PR['H3']['claim'], True,
    quoted='Combine tracking made the estimate after one game 2.1% worse (95% CI 0.4% to 3.9%; preregistered, 12 traits pooled)')
chk('H3 % worse', 100 * (PR['H3']['ratio'] - 1), 2.1, 1)
chk('H3 lo', 100 * (PR['H3']['ratio_ci95'][0] - 1), 0.4, 1)
chk('H3 hi', 100 * (PR['H3']['ratio_ci95'][1] - 1), 3.9, 1)
chk('H3 not passed', PR['H3']['pass'], False)
chk('H3 pooled over 12 traits', len(PR['H3']['n_by_trait']), 12, 0)
c3set = e12p['information_sets']['C3']
sig_n = {sum(t.get('matched_signature') is not None for t in e12p['traits'] if t['id'] == k) for k in tr}
chk('tracking set = five tracked-40 features plus a matched drill signature if any (five or six)', sum(f in c3set for f in ('c40_smax', 'c40_v_20_40', 'c40_vmax', 'c40_tau', 'c40_a0')) == 5
    and sig_n == {0, 1}, True, quoted='That describes our model (five or six tracked features, 62–85 players)')
chk("Section 4's two WR links: the two non-null links, both WR, single-feature, in sample", sorted(T8['passes_c1']) == ['E001-WR', 'E006-P1']
    and all(tm8[i]['positions'] == 'WR' and isinstance(tm8[i]['feature'], str) for i in T8['passes_c1']), True,
    quoted="Section 4's two WR links are single-feature, in-sample correlations")

# ---------------------------------------------------------------- 3.4 2025 rookies
allow('we fit on 2023–24', 'data scope of the validation-2 fit')
chk('H1 n players', PR['H1']['n_players'], 81, 0, quoted='(81 rookies, 222 player-traits)')
chk('H1 n player-traits', PR['H1']['n_player_traits'], 222, 0)
chk('pseudo-prospective label', V2['design_label'].startswith('pseudo-prospective'), True,
    quoted='The test is pseudo-prospective: trait definitions were chosen after earlier analyses had used 2025 games')
chk('H1 72%', 100 * (1 - PR['H1']['ratio']), 72, 0, quoted='cut squared error 72% against his game alone')
chk('H1 CI: 60% (1 - upper ratio)', 100 * (1 - PR['H1']['ratio_ci95'][1]), 60, 0, quoted='(95% CI 60–80%; 68% on the nine priced traits)')
chk('H1 CI: 80% (1 - lower ratio)', 100 * (1 - PR['H1']['ratio_ci95'][0]), 80, 0)
pool = lambda ts, key: sum(SPT[t]['1'][key] for t in ts) / len(ts)  # H1's pooling: equal-weight mean over traits
chk('H1 pooling reproduced (a)', pool(list(tr), 'se_blend_C2'), PR['H1']['pooled_a'], 9)
chk('H1 pooling reproduced (b)', pool(list(tr), 'se_games'), PR['H1']['pooled_b'], 9)
chk('H1 identified-only cut 68%', 100 * (1 - pool(ident, 'se_blend_C2') / pool(ident, 'se_games')), 68, 0)
chk('H2 52%', 100 * (1 - PR['H2']['ratio']), 52, 0, quoted='after two 52% (both preregistered)')
chk('H1 H2 pass', PR['H1']['pass'] and PR['H2']['pass'], True)
calib = [V2['calibration_80pct']['C2'][str(k)]['pooled_coverage'] for k in range(5)]
chk('80% coverage min', 100 * min(calib), 77, 0, quoted="Its 80% intervals held 77–81% of rookies' remaining-games averages")
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
chk('identified only 3.5%', 100 * (1 - k0i['ratio']), 3.5, 1, quoted='on priced traits by 3.5%, and after one game by 2.5% (both intervals include 0)')
k1 = V2['post_hoc_combine_prior_vs_position_mean']['k1']['blend_C2_vs_C0']
chk('k1 2.5%', 100 * (1 - k1['ratio']), 2.5, 1)
chk('both intervals include 0 (ratio 1)', k0i['ratio_ci95'][0] < 1 < k0i['ratio_ci95'][1] and k1['ratio_ci95'][0] < 1 < k1['ratio_ci95'][1], True)
chk('none survives multiplicity (min BH q > 0.05)', min(PHV2['ph3_bh_q_over_10'].values()) > 0.05, True,
    quoted='All are post hoc; none survives multiplicity control')
chk('PH3 / PH10 post hoc', {'PH3', 'PH10'} <= post_hoc_ids, True)

# ---------------------------------------------------------------- 3.4 role-aware note (E013; v2.4, rewritten in v2.5 and v2.6)
# v2.5 (both judges; LESSONS 2026-10-19): the shares' intervals and selection, and the preregistered H1 interval instead of
# the floored "beyond" range and the 80%-power detectable effect. v2.6 (final checker F1-F4): the shares are labelled post
# hoc (PH6); the DL sentence says what PH10 shows (edge share adds nothing detectable once roster designation and weight
# are controlled); the out-of-fold alignment worth is no longer quoted (DL: edge-share worth that tracks designation;
# hitch braking: set against a floored 0, and its prospective counterpart went the other way); the null is scoped to
# alignment measured from a rookie's first games, so that it is not read as "role does nothing".
def sentences(s):
    s = re.sub(r'!\[[^\]]*\]\([^)]*\)', ' ', s).replace('**', '').replace('|', ' ')
    return [' '.join(x.split()) for x in re.split(r'(?<=[.!?])\s+|\n+', s) if x.strip()]


text_s = sentences(text)
roles_s = sentences(text[text.index('**Roles.**'):text.index('## 4.')])
A13 = S13['anova_share']
sig13 = ['T3', 'T6', 'T12']
roles13 = {t: e13['internal'][t]['role'] for t in sig13}
chk('alignment = slot share (WR), edge share (DL), corner share (DB)', roles13 == {'T3': 'slot share', 'T6': 'edge share', 'T12': 'corner share'}
    and all(x in e13p['role_definition']['role_share'][pos] for pos, x in (('WR', 'slot share'), ('DL', 'edge share'), ('DB', 'corner share'))), True,
    quoted='Alignment (slot, edge or corner share) explains 29%, 21% and 10% of between-player differences in WR hitch braking, DL rush distance and DB top speed')
chk('role share T3 (WR hitch braking)', 100 * A13['T3'], 29, 0)
chk('role share T6 (DL rush distance)', 100 * A13['T6'], 21, 0)
chk('role share T12 (DB top speed)', 100 * A13['T12'], 10, 0)
chk('trait names and positions', [(e13['internal'][t]['position'], e13['internal'][t]['name']) for t in sig13] ==
    [('WR', 'HITCH peak braking'), ('DL', 'First-second rush distance'), ('DB', 'Game top speed')], True)
chk('summary share = per-trait ANOVA key', max(abs(A13[t] - I13[t]['anova']['share_of_between_player_variance']['est']) for t in sig13), 0, 9)
chk('the only three of nine passing false-discovery control (BH q < 0.05)', sorted(S13['anova_share_bh_q_lt_0.05']) == sorted(sig13)
    and all(S13['anova_bh_q'][t] < 0.05 for t in sig13) and len(S13['anova_bh_q']) == 9
    and sum(q < 0.05 for q in S13['anova_bh_q'].values()) == 3, True, quoted='the only three of nine passing false-discovery control')
sh_ci = {t: I13[t]['anova']['share_of_between_player_variance'] for t in sig13}
for t, lo_, hi_ in (('T3', 4, 57), ('T6', 4, 44), ('T12', -1, 28)):
    chk(f'share interval {t} lo', 100 * sh_ci[t]['lo'], lo_, 0, quoted='(95% CIs 4–57%, 4–44%, −1% to 28%; post hoc)')
    chk(f'share interval {t} hi', 100 * sh_ci[t]['hi'], hi_, 0)
s_share = [x for x in roles_s if '29%' in x]
chk('F4: the shares sentence (29/21/10%, selection) is labelled post hoc (E013 PH6)', len(s_share) == 1 and 'post hoc' in s_share[0]
    and 'share of between-player variance' in ph13['PH6'] and 'permutation' in ph13['PH6'], True)
# F2 (v2.6): for DL the signal tracks roster designation and body type, not edge share (E013 PH10; descriptive, in sample)
r2gain = max(r2s['roster_position_and_role_share'] - r2s['roster_position'], r2s['weight_and_role_share'] - r2s['weight'])
DL_SENT = ('For DL, edge share adds nothing detectable once roster designation (OLB vs DE) and weight are controlled: the signal '
           'tracks designation and body type (descriptive; the label may postdate the draft)')
chk('F2: edge share adds nothing detectable once designation and weight are controlled (OLS HC1: edge share p > 0.05, OLB > 0 '
    'with p < 0.001; edge share adds < 0.01 R2 beyond either; weight R2 > 0.1; pure-edge OLB > DE, p < 0.05, every class)',
    ols['edge_share']['p'] > 0.05 and ols['olb']['p'] < 0.001 and ols['olb']['coef'] > 0 and r2gain < 0.01 and r2s['weight'] > 0.1
    and pe['mean_olb'] > pe['mean_de'] and pe['welch_p_two_sided'] < 0.05 and all(v['olb'] > v['de'] for v in pe['by_class'].values())
    and 'descriptive' in ph13['PH10'] and 'possibly the latest roster label' in ph13['PH10'], True, quoted=DL_SENT)
# Mutation guards for the edge-share attribution (F2): the DL share is followed at once by the designation sentence; no
# other sentence mentions edge share or edge alignment; no sentence gives rush distance a games value from alignment.
i21 = [k for k, x in enumerate(roles_s) if '21%' in x]
edge_s = [x for x in text_s if re.search(r'\bedge[- ](?:share|alignment)', x, re.I)]
dl_games = [x for x in text_s if 'rush distance' in x and re.search(r'\balignment\b|\bedge\b', x, re.I) and re.search(r'\bgames?\b', x)]
chk('F2 guard: the 21% DL share is followed at once by the designation sentence', len(i21) == 1 and i21[0] + 1 < len(roles_s)
    and roles_s[i21[0] + 1].startswith(DL_SENT), True)
chk(f'F2 guard: the only sentence about edge share is the designation sentence {[x[:60] for x in edge_s]}',
    len(edge_s) == 1 and edge_s[0].startswith(DL_SENT), True)
chk(f'F2 guard: no games value for rush distance from alignment or edge share {[x[:60] for x in dl_games]}', dl_games == [], True)
# F1/F2 (v2.6): the out-of-fold alignment worth is not quoted. T6's is edge-share worth (above); T3's set alignment against
# the forecast's floored 0, and E013's prospective hitch-braking ratio is above 1 under every variant.
at3, at6 = I13['T3']['attribution_C2'], I13['T6']['attribution_C2']
rwd = S13['role_worth_vs_dummies_ridge_same_cv']
p13t3 = {v: r['T3'] for v, r in e13['post_hoc_review']['prospective']['per_trait_h1_ratio_k1'].items()}
sec13 = e13['prospective']['tests']['secondary_family']['per_trait:T3:C0_role_vs_C0_k1']
chk(f'premise: hitch-braking alignment worth (0.24 / 0.90) is in sample and its prospective ratio is > 1 under every variant {p13t3}',
    at3['role_alone']['est'] > 0 and rwd['T3'] > 0 and len(p13t3) == 4 and min(p13t3.values()) > 1
    and abs(sec13['ratio'] - p13t3['prereg']) < 1e-9, True)
chk("premise: the forecast's 0 for hitch braking is floored (cross-validated R2 <= 0, clipped)",
    ct['T3']['cge_C2']['est'] == 0 and ct['T3']['r2_cv']['C2'] <= 0 and at3['c_alone']['est'] == 0, True)
chk('F1/F2 guard: the out-of-fold alignment worth is not quoted (0.2–0.9, 0.8–1.1)', '0.2–0.9' not in text and '0.8–1.1' not in text, True)
hb_cmp = [x for x in text_s if re.search(r'\balignment\b|\bslot share\b|\brole\b', x, re.I) and 'hitch braking' in x
          and re.search(r'forecast|Combine|outvalu|worth more|beats?\b', x)]
chk(f'F1 guard: no sentence compares alignment with the Combine + draft forecast for hitch braking {[x[:60] for x in hb_cmp]}',
    hb_cmp == [] and not re.search(r'outvalu', text, re.I), True)
# F3 (v2.6): the log's "adds at most 0.16 games beyond the Combine and draft" is a difference of floored fits for 5 of 9
# traits (hitch braking among them); its "could not detect" overstates the test's power at the best case (about 0.59).
fl = sorted(t for t, v in I13.items() if 'attribution_C2' in v and (v['attribution_C2']['both']['est'] == 0 or v['attribution_C2']['c_alone']['est'] == 0))
chk(f'premise: role beyond the Combine + draft forecast involves a floored fit for 5 of 9 traits {fl}', fl == ['T11', 'T2', 'T3', 'T5', 'T9'], True)
chk('F3 guard: no floored "beyond" value (−0.06 to +0.16, at most 0.16)', '−0.06 to +0.16' not in text
    and not re.search(r'(?:at most|up to|adds?|beyond)[^.;|]{0,30}\b0\.16\b|\b0\.16 games', text) and 'detectable effect (4%)' not in text, True)
pw13 = e13['post_hoc_review']['prospective']['power']
nd = lambda z: 0.5 * (1 + math.erf(z / math.sqrt(2)))
power_best = nd(abs(math.log(pw13['best_case_from_internal_fit']['H1']['ratio_vs_observed_position_average_error'])) / pw13['primary']['H1']['sd_log_ratio'] - 1.6449)
chk(f'premise: power at the best case is about 0.59 (not 0, so "could not detect" overstates; not 0.8 either) {power_best:.2f}',
    0.5 < power_best < 0.8 and pw13['primary']['H1']['mde_ratio_one_sided_80pct_power'] < pw13['best_case_from_internal_fit']['H1']['ratio_vs_observed_position_average_error'], True)
nothing = re.search(r"\b(?:role|alignment)\b[^.;:]{0,30}\b(?:does not matter|doesn't matter|does nothing|is worth nothing|worth nothing|adds nothing|is useless|is worthless|worthless|is irrelevant|irrelevant|is meaningless|useless)", text, re.I)
chk(f'F3 guard: nothing says role or alignment does nothing {nothing.group() if nothing else ""}', nothing is None
    and 'could not detect' not in text and 'excludes even' not in text, True)
P13 = e13['prospective']['tests']['primary']
pre13 = {h['id']: h for h in e13p['prospective_test']['primary_tests']}
h1 = P13['H1']
best13 = 1 - e13['post_hoc_review']['prospective']['power']['best_case_from_internal_fit']['H1']['ratio_vs_observed_position_average_error']
chk('preregistered H1/H2: alignment share from his first 1 / 2 games beats the position-average blend', pre13['H1']['claim'].endswith('at k = 1')
    and pre13['H2']['claim'].endswith('at k = 2') and (h1['k'], P13['H2']['k']) == (1, 2), True,
    quoted="Still, alignment measured from a rookie's first games did not improve 2025 forecasts (preregistered): error rose 2.8%, and its 95% CI (−0.4% to +7.0%) excludes the 3% cut exactly known alignment would bring (post hoc); a post hoc precision correction agrees")
chk('H1 error rose 2.8%', 100 * (h1['ratio'] - 1), 2.8, 1)
chk('H1 CI lo -0.4%', 100 * (h1['ratio_ci95'][0] - 1), -0.4, 1)
chk('H1 CI hi +7.0%', 100 * (h1['ratio_ci95'][1] - 1), 7.0, 1)
chk('H1, H2 not supported (pass false, ratio > 1)', not h1['pass'] and not P13['H2']['pass'] and h1['ratio'] > 1 and P13['H2']['ratio'] > 1, True)
chk('3% cut if alignment were known exactly (best case, internal fit; PH7)', 100 * best13, 3, 0)
chk('the H1 interval excludes the best case', h1['ratio_ci95'][0] > 1 - best13, True)
chk('E013 test is pseudo-prospective on 2025 rookies', e13['prospective']['design_label'].startswith('pseudo-prospective'), True)
cal13 = e13['post_hoc_review']['prospective']['tests']['calibrated:H1']
chk('precision-corrected H1 agrees: not better, interval also excludes the best case (post hoc PH4)', cal13['ratio'] > 1 and cal13['p_one_sided'] > 0.05
    and cal13['ratio_ci95'][0] > 1 - best13 and 'PH4' in ph13 and 'precision mismatch' in ph13['PH4'], True)

# ---------------------------------------------------------------- 4. Why (E008, E009)
chk('18 links', T8['n_links'], 18, 0, quoted='Of 18 links from a Combine tracking signature')
chk('16 nulls beyond B2', T8['n_nulls'], 16, 0,
    quoted='16 show nothing detectable beyond the stopwatch and draft slot, and the same 16 beyond the stopwatch alone')
chk('16 nulls beyond B1', e9['summary']['b1_n_nulls'], 16, 0)
chk('same 16', sorted(e9['summary']['b1_passes_c1']) == sorted(T8['passes_c1']), True)
chk('the other two are WR speed links, both fragile (E006-P1: drop-3 CI includes 0 and round-1 q > 0.05; E001-WR: fails BH and leave-3-out)',
    all(tm8[i]['positions'] == 'WR' and tm8[i]['construct'] == 'Sprint speed' for i in T8['passes_c1'])
    and tm8['E006-P1']['drop3_ci_lo'] < 0 < S8['multiplicity']['E006-P1']['bh_q_all_round1_tests'] - 0.05
    and tm8['E001-WR']['verdict'] == 'p < .05; fails BH + leave-3-out', True,
    quoted='the other two, both WR speed links, are fragile (Section 5)')
chk('ceiling lo', T8['ceiling_per_link_95'][0], 0.09, 2, quoted='each null excludes only partial correlations above +0.09 to +0.38')
chk('ceiling hi', T8['ceiling_per_link_95'][1], 0.38, 2)
chk('error-free >= .5 for six', len(T8['corrected_beyond_b2_ge_05']), 6, 0, quoted='for error-free measures six links could reach 0.5 or more')
nulls = [r for r in e8['transfer_map'] if r['id'] not in T8['passes_c1']]
chk('16 null links in the map', len(nulls), 16, 0)
chk('13 of 16 nulls pass the reliability gate', T8['nulls_passing_c5'], 13, 0,
    quoted='13 of the 16 pass the reliability gate (both measures ≥ 0.5)')
chk('gate = Combine feature and game target both >= 0.5', sum(r['rel_x'] >= 0.5 and r['rel_y'] >= 0.5 for r in nulls), 13, 0)
chk('gate flag agrees', sum(bool(r['c5_reliability_ge_05']) for r in nulls), 13, 0)
g = S8['drill_profile']['by_group_kind']
chk('40 metrics n', g['40-yd dash | repeat attempts']['n'], 28, 0, quoted='All 28 tracked-40 metrics are repeatable within a session (reliability ≥ 0.5; top speed 0.95–0.98)')
chk('40 metrics >= .5', g['40-yd dash | repeat attempts']['n_ge_05'], 28, 0)
chk('top speed rel lo', S8['drill_profile']['forty']['top_speed']['rel'][0], 0.95, 2)
chk('top speed rel hi', S8['drill_profile']['forty']['top_speed']['rel'][1], 0.98, 2)
rep = [r for r in dp if r['kind'] == 'repeat attempts' and not r['scope'].startswith('40')]
chk('repeat drill metrics', len(rep), 22, 0, quoted='Of 22 position-drill metrics with repeat reps, 16 of them reaction or opposite-side reps, 2 are')
chk('16 of the 22 are reaction or opposite-side reps', sum(r['scope'].startswith(('45-degree reaction', 'pass-rush-drop attempts (opposite sides)')) for r in rep), 16, 0)
chk('repeat drill >= .5', sum(r['rel_round1_def'] >= 0.5 for r in rep), 2, 0)
chk('cross-drill n', g['position drill | cross-drill']['n'], 68, 0, quoted='of 68 composites across drills that target one skill, 10 are')
chk('cross-drill >= .5', g['position drill | cross-drill']['n_ge_05'], 10, 0)
ws = [r['share_session'] for r in dp if r['metric'] in ('wave rev_accel', 'wave peak_decel', 'wave peak_reaccel', 'wave exit_speed')]
chk('wave rows', len(ws), 4, 0)
chk('wave session lo', 100 * min(ws), 55, 0, quoted='55–63% of OL wave-drill variance lies between Combine sessions')
chk('wave session hi', 100 * max(ws), 63, 0)

# ---------------------------------------------------------------- 5. What tracking does show (E008, E009, E010)
w = tm8['E006-P1']
chk("the stronger of Section 4's two WR links", w['partial_r'] > tm8['E001-WR']['partial_r'], True,
    quoted="Of Section 4's two WR links, the stronger remains suggestive")
chk('WR signal r', w['partial_r'], 0.48, 2, quoted='(extrapolated from in-game accelerations; partial r +0.48, 95% CI +0.20 to +0.68)')
chk('WR signal lo', w['ci_lo'], 0.20, 2)
chk('WR signal hi', w['ci_hi'], 0.68, 2)
chk('WR drop3', w['drop3_partial_r'], 0.31, 2, quoted='it falls to +0.31 (−0.01 to +0.57)')
chk('WR drop3 lo', w['drop3_ci_lo'], -0.01, 2)
chk('WR drop3 hi', w['drop3_ci_hi'], 0.57, 2)
mul = S8['multiplicity']['E006-P1']
chk('round-1 tests', mul['m_round1_tests'], 90, 0, quoted='across about 90 first-round tests (q 0.06)')
chk('round-1 q', mul['bh_q_all_round1_tests'], 0.06, 2)
# v2.5: the drill-intensity sentence (E010) is cut for words; Figure 5 shows it ("later test; no game link")

# ---------------------------------------------------------------- 6. Recommendations
chk('deployment before game 1 is untested (E013 prereg: no role prior at k = 0)', 'No role prior at k = 0' in e13p['role_definition']['in_season'], True,
    quoted='Intended deployment, known before game 1, is a prior we did not test')
# v2.6 (final checker F1): v2.5's "in our 2023-25 fit, alignment outvalues the Combine + draft forecast for WR hitch
# braking" is gone (prospectively worse under every variant; set against a floored 0; premises in the Roles block).
clubs = text[text.index('**Clubs, week to week**'):text.index("**The league's")]
al_clubs = re.findall(r'\balignment\b|\b(?:slot|edge|corner) share\b|\bhitch\b|outvalu', clubs, re.I)
chk(f'F1 guard: no alignment advice in the Clubs items {al_clubs}', al_clubs == [], True)
r6 = PHR['per_trait']['T6']['references']['role']
chk('T6 designation = NFL roster position', r6['reference'].startswith('NFL roster position') and 'latest roster designation' in r6['reference'], True,
    quoted='(post hoc; designations may postdate the draft)')
chk('designation prior worth more than the Combine + draft forecast (T6)', r6['cge_ref'] > ct['T6']['cge_C2']['est'] and r6['cge_ref_plus_C2'] < r6['cge_ref'], True,
    quoted='for DL rush distance, a roster-designation prior is outweighed after 6.0 games, not 4.8')
chk('not 4.8 = Table 1 half-weight with the Combine + draft forecast (T6)', hw['T6'], 4.8, 1)
chk('role worth', r6['cge_ref'], 2.4, 1)
chk('role half-weight = lambda + CGE_role', lam['T6'] + ct['T6']['cge_role_reference'], 6.0, 1)
chk('raw first game 3.5x', PR['H1']['pooled_b'] / PR['H1']['pooled_a'], 3.5, 1,
    quoted='had 3.5 times the forecast error of the shrunk estimate')
chk('Clubs item 3: as modeled, Combine tracking made estimates slightly worse (H3 ratio > 1)', PR['H3']['ratio'] > 1, True,
    quoted="Don't weight Combine tracking** as we modeled it: it made estimates slightly worse")
dl40 = S8['drill_profile']['forty']['dl_starts']['rel']
chk('40 first-step rel lo', dl40[0], 0.66, 2, quoted='repeatable in the 40 (reliability 0.66–0.79) but mostly not in the DL pass-rush drill (0.23–0.51)')
chk('40 first-step rel hi', dl40[1], 0.79, 2)
pr = [r for r in dp if r['scope'] == 'pass-rush drill attempts']
chk('DL pass-rush rows = the 6 first-step metrics', sorted(r['metric'] for r in pr) == sorted(['d05', 'd10', 's05', 's10', 'amax06', 't3']), True)
chk('DL pass-rush rel lo', min(r['rel_round1_def'] for r in pr), 0.23, 2)
chk('DL pass-rush rel hi', max(r['rel_round1_def'] for r in pr), 0.51, 2)
chk('mostly not: at most 1 of 6 reaches 0.5', sum(r['rel_round1_def'] >= 0.5 for r in pr) <= 1, True)
icc1 = [r['icc_unit'] for r in pr]
chk('single-rep reliability lo', min(icc1), 0.15, 2, quoted='At single-rep reliabilities of 0.15–0.39, two to six identical reps reach 0.5')
chk('single-rep reliability hi', max(icc1), 0.39, 2)
reps_needed = sorted(math.ceil((1 - i) / i) for i in icc1)  # Spearman-Brown: n = (R/(1-R)) (1-icc)/icc, R = 0.5
chk('reps needed two to six', (reps_needed[0], reps_needed[-1]) == (2, 6), True)
chk('wave session effects dominate (> 50%)', min(ws) > 0.5, True, quoted='Session effects dominate the OL wave drill')
chk('2025 comeback', any('2025 COMEBACK is a different movement' in c for c in e8['links']['E002-P3']['caveats']), True,
    quoted='the 2025 comeback drill was a different movement')
chk('E011 2025 only', e11['data']['draft_years'] == [2025], True,
    quoted='In the 2025 data no catch mechanic is yet a stable trait (best 0.42; 6–46 catching drills needed)')
rel11 = [v['cross_drill']['reliability'] for v in e11['reliability'].values()
         if v.get('cross_drill', {}).get('reliability') is not None]
chk('E011 best reliability', max(rel11), 0.42, 2)
dn = e11['reliability']['catch_offset']['drills_needed_range_from_icc_ci']['for_0.5']
chk('catch drills needed lo', dn[0], 6, 0)
chk('catch drills needed hi', dn[1], 46, 0)

# ---------------------------------------------------------------- 7. Limitations
chk('n players lo (limits)', min(ns), 62, 0, quoted='62–85 players per trait')
chk('three classes', sorted(tr['T1']['by_class']) == ['2023', '2024', '2025'], True, quoted='three classes, one season for 2025 rookies')
# v2.5: the Limitations estimand bullet moved to Section 3.2 (population-signal bounds 4-15)
nr = [PHR['per_trait'][t]['stationarity_and_game_unit']['noise_ratio_lowest_vs_highest_plays_quartile'] for t in tr]
nr2 = [x for x in nr if x >= 2]
chk('noise ratio >= 2 for 9 of 12', len(nr2), 9, 0, quoted='low-play games are 2.0–3.5 times noisier (9 of 12 traits)')
chk('noise ratio lo', min(nr2), 2.0, 1)
chk('noise ratio hi', max(nr2), 3.5, 1)
roles = ' '.join(PHR['per_trait'][t]['references']['role']['reference'] for t in ('T6', 'T12'))
chk('roles pooled (DE/DT, CB/FS)', all(x in roles for x in ('DE', 'DT', 'CB', 'FS')), True,
    quoted='position averages pool roles (Section 3.4)')
# spelled-out numbers (the uncovered-number scan below sees digits only)
chk('heading: fifteen games', case['n_games'], 15, 0, quoted='One rookie, fifteen games')
chk('four priors = C0-C3', len([k for k in e12p['information_sets'] if re.fullmatch(r'C\d', k)]), 4, 0, quoted='We price four priors')
chk('three most supportive receivers', w['n'] - w['drop3_n'], 3, 0, quoted='Without its three most supportive receivers')
chk('heading: about one game (max CGE)', max(ct[t]['cge_C2']['est'] for t in ident), 1, 0, quoted='a fraction of a game to about one')
allow('2025 Combine', 'data scope')
allow('2025 rookie', 'data scope')
allow('2025 rookies', 'data scope')
allow('2025 games', 'data scope')
allow('the 2025 test', 'data scope')
allow('game 1', 'illustrative game number')
allow('a typical game-1 top-speed play', 'Figure 1 panel: a play from his first qualifying game (figure_plan selection rule)')
allow('40 vs DL pass-rush drill', 'Figure 5 panel: drill names (the 40-yard dash)')
chk('Figure 3 coverage panel is the 80% interval', 'C2' in V2['calibration_80pct'], True, quoted='80% interval coverage.')
# v2.6: the Figure 3 caption names the forecast by its information set (outline section 4: never "the Combine" for C2
# in a comparison); the figure's title and rows say "Combine + draft forecast" too (make_figures.py).
cap3 = re.search(r'\*\*Figure 3\.\*\*[^\n]*', text).group()
chk('Figure 3 caption: "the Combine + draft forecast", not "the Combine"', re.search(r'\bthe Combine\b(?! \+ draft)', cap3) is None, True,
    quoted='shrinkage, the Combine + draft forecast, tracking and alignment')
# v2.6: Figure 1's band is the rule's 80% posterior interval for his level (prior SD x sqrt(1 - w); make_figures.py), not
# the calibrated 80% interval for a rookie's remaining-games average that Section 3.4 and Figure 3 evaluate. No text
# sentence about Figure 1, its band or the hook may call it calibrated or cite its coverage.
sec1 = text[text.index('## 1.'):text.index('## 2.')]
f1_s = [x for x in text_s if 'Figure 1' in x or re.search(r'\bband\b', x, re.I)] + [x for x in sentences(sec1) if re.search(r'\binterval', x)]
f1_bad = [x[:60] for x in f1_s if re.search(r'calibrat|coverage|held \d', x, re.I)]
chk(f'Figure 1 band never called the calibrated interval {f1_bad}', f1_bad == [], True)
allow('calibrated 90% bounds bold, 95% bootstrap thin', 'interval levels (Figure 2 caption): two one-sided 95% bounds = 90%; B1 95%')
allow('(95% CIs)', 'CI level (Figure 4 caption)')

# ---------------------------------------------------------------- named players (user decision 2026-10-18; outline.md section 4)
# Real names are allowed. A named player's draft status, draft position and off-field matters never are, and neither
# the draft slot nor the stopwatch-only forecast is tied to a named player anywhere, comments included (v2.4).
NAMED = {59278: 'Isaiah Bond',  # the hook: E012's preregistered WR case study
         58274: 'Kyle Williams'}  # Figure 1's same-40 comparison (E007 concept_pair); figure only


def name_records(o, out):
    """Every (nfl_id, name) pair in a committed results file (key 'name' or 'display_name')."""
    if isinstance(o, dict):
        nm = o.get('name', o.get('display_name'))
        if 'nfl_id' in o and isinstance(nm, str):
            out.append((o['nfl_id'], nm))
        for v in o.values():
            name_records(v, out)
    elif isinstance(o, list):
        for v in o:
            name_records(v, out)
    return out


recs = name_records([e7, e7lb, e12, e13], [])
for nid, nm in NAMED.items():
    got = sorted({n for i, n in recs if i == nid})
    chk(f'nfl_id {nid} is {nm!r} in all {sum(i == nid for i, _ in recs)} committed records (found {got})', got == [nm], True)
chk('the hook is the E012 case study, named', case['name'] == NAMED[59278] and bond['name'] == NAMED[59278], True,
    quoted='Isaiah Bond ran a 4.39 forty at the 2025 Combine')
chk('Figure 1 caption names him', case['name'] == NAMED[59278], True, quoted='**Figure 1.** Isaiah Bond (animated)')
cpair = e7['concept_pair']
chk("Kyle Williams is Figure 1's same-40 comparison (E007 concept pair)",
    (cpair['fast']['nfl_id'], cpair['slow']['nfl_id']) == (59278, 58274)
    and abs(cpair['fast']['forty'] - cpair['slow']['forty']) <= cpair['tolerance_s'] + 1e-9, True)
universe = sorted({n for _, n in recs})
others = [n for n in universe if n not in NAMED.values() and re.search(rf'\b{re.escape(n)}\b', raw)]
chk(f'no other player named in the text or comments (of {len(universe)} named in the committed results) {others}', others == [], True)
chk('no "Receiver A/B" placeholder left', re.search(r'\bReceiver [AB]\b', text) is None, True)
parts = {x for nm in NAMED.values() for x in nm.split()}
near = sorted({w for w in re.findall(r'\b[A-Z][a-z]+\b', text) if w not in parts
               and any(difflib.SequenceMatcher(None, w, x).ratio() >= 0.75 for x in parts)})
chk(f'no misspelled first name or surname of a named player {near}', near == [], True)
for nid, nm in NAMED.items():
    sur = nm.split()[-1]
    full = [m.start() for m in re.finditer(rf'\b{re.escape(nm)}\b', text)]
    bare = [m.start() for m in re.finditer(rf'\b{re.escape(sur)}\b', text)]
    first_is_full = (not bare) or (bool(full) and bare[0] == full[0] + len(nm) - len(sur))
    chk(f'{nm}: first mention in the text is the full name ({len(bare)} mentions)', first_is_full, True)
# A sentence is about a named player if it names him, or if it has he/his/him in a section (heading to heading) that
# names him: in the hook every sentence after the first refers to him by pronoun. "Combine + draft forecast" (the
# forecast's name) is allowed; every pattern below is not. "select..." counts only next to a draft word (a "selection
# rule" is legitimate); "draft forecast" is not a draft word.
DRAFT_WORDS = re.compile(r'\b(?:un)?drafted\b|\bdraftees?\b|\bdraft (?:status|position|round|pick|slot)s?\b|\brounds?\b'
                         r'|\bpick\w*|\bsigned\b|\bfree[- ]agen(?:t|ts|cy)\b|\bUDFA\b'
                         r'|\bselect\w*(?=[^.;]{0,40}\b(?:draft(?! forecast)\w*|rounds?|overall)\b)'
                         r'|\b(?:draft(?! forecast)\w*|rounds?|overall)\b[^.;]{0,40}\bselect\w*', re.I)
OFF_FIELD = re.compile(r'\boff[- ]field\b|\barrest\w*|\balleg\w*|\bcharged\b|\bsuspen\w*|\blawsuits?\b|\bpolice\b'
                       r'|\blegal\b|\binvestigat\w*|\baccus\w*|\bassault\w*|\bmedical\w*', re.I)
STOPWATCH_ONLY = re.compile(r'\bstopwatch[- ]only\b|\bstopwatch (?:and|\+) measurements\b|\bprior_C1\b|\bC1\b', re.I)
GUARD = (DRAFT_WORDS, OFF_FIELD, STOPWATCH_ONLY)
PRONOUN = re.compile(r'\b(?:he|his|him|himself)\b', re.I)
NAME_RE = re.compile('|'.join(rf'\b{re.escape(x)}\b' for nm in NAMED.values() for x in (nm, nm.split()[-1])))
plain = re.sub(r'!\[[^\]]*\]\([^)]*\)', ' ', text).replace('**', '').replace('|', ' ')
about = []
for sec in re.split(r'(?m)^(?=#+ )', plain):
    if NAME_RE.search(sec):
        about += [' '.join(x.split()) for x in re.split(r'(?<=[.!?])\s+|\n+', sec)
                  if NAME_RE.search(x) or PRONOUN.search(x)]
chk('scope covers the hook and the Figure 1 caption', any(x.startswith('Isaiah Bond ran') for x in about)
    and any(x.startswith('Isaiah Bond (animated)') for x in about), True)
hits = [(m.group(), x[:80]) for x in about for rx in GUARD for m in rx.finditer(x)]
chk(f'no draft-status, off-field or stopwatch-only words in the {len(about)} sentences about named players {hits}', hits == [], True)
# The same scan over the HTML comments: a comment sentence is about a named player if it names him or his nfl_id, or
# if it has he/his/him in a comment that does.
ID_RE = re.compile('|'.join(rf'\b{i}\b' for i in NAMED))
about_c = []
for cm in re.findall(r'<!--(.*?)-->', raw, flags=re.S):
    if NAME_RE.search(cm) or ID_RE.search(cm):
        about_c += [' '.join(x.split()) for x in re.split(r'(?<=[.!?])\s+', cm)
                    if NAME_RE.search(x) or ID_RE.search(x) or PRONOUN.search(x)]
hits_c = [(m.group(), x[:100]) for x in about_c for rx in GUARD for m in rx.finditer(x)]
chk(f'no draft-status, off-field or stopwatch-only words in the {len(about_c)} comment sentences about named players {hits_c}',
    hits_c == [], True)
# The stopwatch-only forecast (information set C1) appears nowhere in a text section or a comment that names a player.
C1_RE = re.compile(r'\bprior_C1\b|\bC1\b|\b[Ss]topwatch[- ]only\b')
scopes = [sec for sec in re.split(r'(?m)^(?=#+ )', text) if NAME_RE.search(sec)]
scopes += [cm for cm in re.findall(r'<!--(.*?)-->', raw, flags=re.S) if NAME_RE.search(cm) or ID_RE.search(cm)]
hits_c1 = [m.group() for sc in scopes for m in C1_RE.finditer(sc)]
chk(f'no stopwatch-only forecast in the {len(scopes)} sections and comments that name a player {hits_c1}', hits_c1 == [], True)

# ---------------------------------------------------------------- coverage of every numeric token
spans = sorted(covered)


def is_covered(a, b):
    return any(s <= a and b <= e for s, e in spans)


struct = []  # numbers that are structure, not results
for m in re.finditer(r'^(#+ )([\d.]+)', text, flags=re.M):  # section numbers
    struct.append((m.start(2), m.end(2)))
for m in re.finditer(r'^(\d+)\. ', text, flags=re.M):  # list numbers
    struct.append((m.start(1), m.end(1)))
for m in re.finditer(r'(?:Figure|Table|Section|figures/fig)\s*(\d+(?:\.\d+)?)', text):  # figure, table and section labels (3.4)
    struct.append((m.start(1), m.end(1)))
for m in re.finditer(r'figures/fig\d_[\w.]+', text):  # image paths
    struct.append((m.start(), m.end()))
for m in re.finditer(r'https?://\S+', text):  # links (e.g. the notebook URL; usernames may contain digits)
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
    body_nums = re.sub(r'(?:Figure|Table|Section)\s*\d+(?:\.\d+)?|^\d+\. |^#+.*$|game [13]\b|\b40\b|10 Hz', ' ', body, flags=re.M)
    if not re.search(r'\d', body_nums):
        continue
    if not any('<!-- src' in b for b in blocks[i:i + 3]):
        print('FAIL no src comment near block', i, body.strip()[:80])
        fails.append(f'no src comment near block {i}')

# ---------------------------------------------------------------- budget
pasted = re.sub(r'<!--.*?-->', ' ', raw, flags=re.S)  # the markdown pasted into Kaggle
pasted = re.sub(r'!\[[^\]]*\]\([^)]*\)', ' ', pasted)  # images carry no words


def render(md):
    """Approximate the text a reader sees: table pipes become cell breaks; markup-only tokens are dropped."""
    out = []
    for ln in md.split('\n'):
        if re.fullmatch(r'\s*\|?\s*:?-+:?\s*(\|\s*:?-+:?\s*)*\|?\s*', ln):  # table separator row
            continue
        ln = re.sub(r'^\s*#+\s+', '', ln)  # heading markers
        ln = re.sub(r'^\s*>\s?', '', ln)  # blockquote marker
        ln = re.sub(r'^\s*[-*]\s+', '', ln)  # bullet markers
        if ln.lstrip().startswith('|'):
            ln = ln.replace('|', ' ')  # table cells
        out.append(ln.replace('**', ''))
    return '\n'.join(out)


rendered = render(pasted)
# [[TBD...]] placeholders count: each becomes link text at submission
words = [t for t in rendered.split() if re.search(r'\w', t)]  # \w is Unicode: λ and σ tokens are words
tok_md, tok_rendered = len(pasted.split()), len(rendered.split())
items = sorted(set(re.findall(r'\*\*(Figure \d+|Table \d+)\.\*\*', raw)))
print(f'\nrendered words: {len(words)} (limit 1,900); whitespace tokens: {tok_md} pasted markdown, {tok_rendered} rendered '
      f'(limit 1,950); figures + tables: {len(items)} {items} (limit 8)')
if len(words) > 1900:
    fails.append(f'rendered word count {len(words)} > 1,900')
if max(tok_md, tok_rendered) > 1950:
    fails.append(f'whitespace tokens {max(tok_md, tok_rendered)} > 1,950')
if len(items) > 8:
    fails.append(f'{len(items)} figures + tables > 8')
tbd = re.findall(r'\[\[TBD[^\]]*\]\]', raw)
print('open placeholders:', tbd)

print(f'\n{len(fails)} failures')
for f in fails:
    print('  -', f)
sys.exit(1 if fails else 0)
