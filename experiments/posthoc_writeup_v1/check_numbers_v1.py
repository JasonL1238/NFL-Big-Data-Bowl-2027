"""Check every number quoted in writeup/draft.md v1 against the results JSONs (and the ph_*.json post hoc writeup
analyses), at the quoted rounding. Also asserts each quoted string is actually present in the draft text."""
import json
import re
from pathlib import Path

R = Path('/Users/jasonli/Documents/GitHub/NFL-Big-Data-Bowl-2027/experiments/results/')
H = Path(__file__).parent
E = {k: json.loads((R / f'{k}.json').read_text()) for k in ['E002', 'E006', 'E007', 'E008', 'E007_leaderboard']}
PH = {k: json.loads((H / f'ph_{k}.json').read_text()) for k in ['pairs', 'inseason', 'rev3']}
draft = Path('/Users/jasonli/Documents/GitHub/NFL-Big-Data-Bowl-2027/writeup/draft.md').read_text()
text = re.sub(r'<!--.*?-->', ' ', draft, flags=re.S)
e8, e7, lb = E['E008'], E['E007'], E['E007_leaderboard']
tm = {r['id']: r for r in e8['transfer_map']}
dp = e8['drill_profile']
S = e8['summary']
T = S['transfer']
fails = []


def row(metric, scope_sub=''):
    rs = [r for r in dp if r['metric'] == metric and scope_sub in r['scope']]
    assert len(rs) == 1, (metric, scope_sub, len(rs))
    return rs[0]


def pl(i):
    return [p for p in lb['players'] if p['nfl_id'] == i][0]


def chk(label, got, want, nd=2, quoted=None):
    ok = round(got, nd) == want if not isinstance(want, tuple) else (want[0] <= got <= want[1])
    if quoted is not None and quoted not in text:
        ok = False
        label += f' [quoted string {quoted!r} not in text]'
    print(('OK  ' if ok else 'FAIL'), label, got, want)
    if not ok:
        fails.append(label)


# ---- Section 1 (hook, framing, answer)
chk('hook 40 fast', e7['concept_pair']['fast']['forty'], 4.39, quoted='4.39 and 4.40')
chk('hook 40 slow', e7['concept_pair']['slow']['forty'], 4.40)
chk('hook flying20 fast', pl(59278)['flying20_mph'], 23.2, 1, quoted='23.2 mph')
chk('hook flying20 slow', pl(58274)['flying20_mph'], 22.2, 1, quoted='the other 22.2')
chk('hook pairs', e7['concept_pair']['post_hoc_all_matched_pairs']['n_pairs'], 172, 0, quoted='172 same-40 receiver pairs')
chk('hook = largest gap', float(PH['pairs']['hook_pair_is_largest_hidden_gap']), 1.0, quoted='the largest gap among 172')
chk('hook game fast', pl(59278)['game_top_speed_mph'], 19.7, 1, quoted='19.7 mph')
chk('hook game slow', pl(58274)['game_top_speed_mph'], 19.3, 1, quoted='the other 19.3')
chk('hook fewer yds/route', float(pl(59278)['yprr_oe'] < pl(58274)['yprr_oe']), 1.0, quoted='fewer yards per route')
chk('n links', T['n_links'], 18, 0, quoted='18 Combine-to-game links')
chk('r2 flying20 on 40', e7['leaderboard']['r2_flying20_on_forty_year_centered'], 0.81, quoted='81% of the flying 20')
chk('16 of 18', T['n_nulls'], 16, 0, quoted='Sixteen of 18 links')
chk('about +0.3 (re-test)', e7['primary']['P1']['seeds']['0']['partial_r'], (0.25, 0.35), quoted='partial r about +0.3')
chk('about +0.3 (E001)', tm['E001-WR']['partial_r'], (0.25, 0.35))
# ---- Table 1
chk('top speed rel lo', S['drill_profile']['forty']['top_speed']['rel'][0], 0.95, quoted='0.95–0.98 / 0.29–0.80')
chk('top speed rel hi', S['drill_profile']['forty']['top_speed']['rel'][1], 0.98)
res40 = [tm[k]['rel_x_b2'] for k in ('E001-WR', 'E001-DB', 'E001-DL', 'E001-OL', 'E006-P1')] + \
        [e7['primary']['P1']['disattenuated']['feature_reliability_beyond_B2']]
chk('40 top resid lo', min(res40), 0.29); chk('40 top resid hi', max(res40), 0.80)
uf = {u['metric']: u['rel_round1_def'] for u in S['drill_profile']['used_features']}
chk('d10 rel', uf['d10'], 0.73, quoted='0.73 / 0.59'); chk('d10 resid', tm['E003-P1']['rel_x_b2'], 0.59)
chk('a0 rel', uf['a0'], 0.71, quoted='0.71 / 0.56'); chk('a0 resid', tm['E006-P2']['rel_x_b2'], 0.56)
chk('route lo', uf['stop peak_decel'], 0.44, quoted='0.44–0.56 / 0.38–0.63'); chk('route hi', uf['turn retention'], 0.56)
rr = [tm[k]['rel_x_b2'] for k in ('E002-P1', 'E002-P2', 'E002-P3')]
chk('route resid lo', min(rr), 0.38); chk('route resid hi', max(rr), 0.63)
chk('mirror rel', uf['mirror rev_accel'], 0.73, quoted='0.73* / 0.69–0.70*')
mr = [tm[k]['rel_x_b2'] for k in ('E004-P1', 'E004-P2', 'E004-P3')]
chk('mirror resid lo', min(mr), 0.69); chk('mirror resid hi', max(mr), 0.70)
chk('DB dv rel', uf['dv_10 | entry+turn adj.'], 0.55, quoted='0.55 / 0.57–0.58'); chk('DB bp rel', uf['backpedal bp_peak'], 0.55)
chk('DB resid lo', tm['E005-P1']['rel_x_b2'], 0.57); chk('DB resid hi', tm['E005-P2']['rel_x_b2'], 0.58)
chk('wave rel', row('wave rev_accel')['rel_round1_def'], 0.02, quoted='0.02; 0.38')
chk('PR drill d10 rel', row('d10', 'pass-rush drill attempts')['rel_round1_def'], 0.38)
chk('WR re-test', e7['primary']['P1']['seeds']['0']['partial_r'], 0.31, quoted='WR game top speed +0.31, +0.32; capacity +0.48')
chk('WR E001', tm['E001-WR']['partial_r'], 0.32)
chk('WR capacity', tm['E006-P1']['partial_r'], 0.48)
lls = [tm['E001-WR']['ci_lo'], tm['E006-P1']['ci_lo'], e7['primary']['P1']['seeds']['0']['partial_r_ci95'][0]]
chk('WR lower lim lo', min(lls), 0.00, quoted='lower limits 0.00 to +0.20'); chk('WR lower lim hi', max(lls), 0.20)
ceil = lambda ks, f='ceiling': max(tm[k][f] for k in ks)
chk('other pos', ceil(['E001-DB', 'E001-DL', 'E001-OL']), 0.38, quoted='other positions ≤ +0.38 (≤ +0.56)')
chk('other pos dis', ceil(['E001-DB', 'E001-DL', 'E001-OL'], 'ceiling_dis_b2'), 0.56)
chk('DL burst', ceil(['E003-P1', 'E003-P2', 'E003-P3']), 0.38, quoted='quick pressure ≤ +0.38 (≤ +0.56)')
chk('DL burst dis', ceil(['E003-P1', 'E003-P2', 'E003-P3'], 'ceiling_dis_b2'), 0.56)
chk('a0', tm['E006-P2']['ceiling'], 0.09, quoted='≤ +0.09 (≤ +0.14)'); chk('a0 dis', tm['E006-P2']['ceiling_dis_b2'], 0.14)
chk('route', ceil(['E002-P1', 'E002-P2', 'E002-P3']), 0.28, quoted='separation ≤ +0.28 (≤ +0.73)')
chk('route dis', ceil(['E002-P1', 'E002-P2', 'E002-P3'], 'ceiling_dis_b2'), 0.73)
chk('mirror', ceil(['E004-P1', 'E004-P2', 'E004-P3']), 0.38, quoted='pressure allowed ≤ +0.38 (≤ +0.57)')
chk('mirror dis', ceil(['E004-P1', 'E004-P2', 'E004-P3'], 'ceiling_dis_b2'), 0.57)
chk('DB', ceil(['E005-P1', 'E005-P2']), 0.35, quoted='redirection ≤ +0.35 (≤ +0.59)')
chk('DB dis', ceil(['E005-P1', 'E005-P2'], 'ceiling_dis_b2'), 0.59)
# ---- Section 2
chk('40 attempts', E['E006']['combine_qc']['attempts'], 793, 0, quoted='763 of 793')
chk('40 qc', E['E006']['combine_qc']['qc_pass'], 763, 0)
mov = [abs(r['r_target_log_exposure']) for r in e8['transfer_map'] if not r['outcome_target']]
out = [abs(r['r_target_log_exposure']) for r in e8['transfer_map'] if r['outcome_target']]
chk('exposure movement', max(mov), 0.32, quoted='at most 0.32 (movement) and 0.40 (outcomes)'); chk('exposure outcomes', max(out), 0.40)
chk('n_perm_fl', e8['settings']['n_perm_fl'], 10000, 0, quoted='Freedman–Lane permutation (10,000)')
chk('tests >120', S['multiplicity']['E006-P1']['m_round1_tests'] + e7['bh_all_tests']['n_tests'], (121, 10**6), quoted='more than 120')
# ---- 3.1
g = S['drill_profile']['by_group_kind']
chk('40 n', g['40-yd dash | repeat attempts']['n'], 28, 0, quoted='All 28 tracked-40 metrics')
chk('40 n>=.5', g['40-yd dash | repeat attempts']['n_ge_05'], 28, 0)
chk('start lo', S['drill_profile']['forty']['start_phase']['rel'][0], 0.67, quoted='start phase 0.67–0.81')
chk('start hi', S['drill_profile']['forty']['start_phase']['rel'][1], 0.81)
wod = [tm[k]['rel_x_b2'] for k in ('E001-WR', 'E001-DB', 'E001-OL', 'E006-P1')] + [e7['primary']['P1']['disattenuated']['feature_reliability_beyond_B2']]
chk('resid WR/OL/DB lo', min(wod), 0.62, quoted='0.62–0.80 for WR, OL and DB but 0.29 for DL'); chk('resid WR/OL/DB hi', max(wod), 0.80)
chk('resid DL', tm['E001-DL']['rel_x_b2'], 0.29)
chk('burst 0.73 -> 0.59', tm['E003-P1']['rel_x'], 0.73, quoted='falls from 0.73 to 0.59')
rep = [r for r in dp if r['kind'] == 'repeat attempts' and not r['scope'].startswith('40')]
chk('rep n', len(rep), 22, 0, quoted='2 of 22 repeated-rep metrics'); chk('rep pass', sum(r['rel_round1_def'] >= 0.5 for r in rep), 2, 0)
chk('reaction/opposite', sum(('reaction' in r['scope']) or ('opposite' in r['scope']) for r in rep), 16, 0, quoted='16 of the 22')
pr = [r for r in rep if r['scope'] == 'pass-rush drill attempts']
chk('PR n', len(pr), 6, 0, quoted='1 of 6 does'); chk('PR pass', sum(r['rel_round1_def'] >= 0.5 for r in pr), 1, 0)
chk('PR pass is s10', float([r for r in pr if r['rel_round1_def'] >= 0.5][0]['metric'] == 's10'), 1.0)
chk('d10 PR 0.38 vs 40 0.73', row('d10', 'pass-rush drill attempts')['rel_round1_def'], 0.38, quoted='0.38 against 0.73')
chk('d10 40', row('d10', '40-yd dash attempts')['rel_round1_def'], 0.73)
chk('xd n', g['position drill | cross-drill']['n'], 68, 0, quoted='10 of 68'); chk('xd pass', g['position drill | cross-drill']['n_ge_05'], 10, 0)
chk('WR retest', S['drill_profile']['wr_route_same_drill_retest']['retest_icc_median'], 0.70, quoted='median 0.70')
chk('drill entry vs top', E['E002']['diagnostics']['combine']['c_turn_entry_speed~combine_top_speed']['r'], 0.01, quoted='(r = 0.01)')
chk('game entry vs top', E['E002']['diagnostics']['game']['g_turn_entry_speed~game_top_speed']['r'], 0.47, quoted='unlike in games (0.47)')
ws = [row(m)['share_session'] for m in ['wave rev_accel', 'wave peak_decel', 'wave peak_reaccel', 'wave exit_speed']]
chk('wave lo', min(ws), 0.55, quoted='55–63%'); chk('wave hi', max(ws), 0.63)
# ---- 3.2
chk('null min p', T['nulls_min_fl_p'], 0.21, quoted='p ≥ 0.21')
chk('null r lo', T['nulls_signed_partial_r'][0], -0.15, quoted='partial r −0.15 to +0.12'); chk('null r hi', T['nulls_signed_partial_r'][1], 0.12)
chk('ceil lo', T['ceiling_per_link_95'][0], 0.09, quoted='above +0.09 to +0.38'); chk('ceil hi', T['ceiling_per_link_95'][1], 0.38)
nulls = [r for r in e8['transfer_map'] if r['id'] not in ('E006-P1', 'E001-WR')]
chk('six <= .22', sum(r['ceiling'] <= 0.22 for r in nulls), 6, 0, quoted='above +0.22 for six links')
chk('others >= .24', min(r['ceiling'] for r in nulls if r['ceiling'] > 0.22), (0.235, 1))
chk('joint lo', T['ceiling_joint_bonferroni18'][0], 0.15, quoted='+0.15 to +0.48 jointly'); chk('joint hi', T['ceiling_joint_bonferroni18'][1], 0.48)
chk('dis lo', T['ceiling_corrected_beyond_b2'][0], 0.14, quoted='+0.14 to +0.73'); chk('dis hi', T['ceiling_corrected_beyond_b2'][1], 0.73)
chk('six >=.5', len(T['corrected_beyond_b2_ge_05']), 6, 0, quoted='0.5 or more for six links')
single = [r['mde80_one_sided'] for r in e8['transfer_map'] if '+' not in r['positions']]
chk('mde lo', min(single), 0.31, quoted='0.31–0.38'); chk('mde hi', max(single), 0.38)
ns = [r['n'] for r in e8['transfer_map'] if '+' not in r['positions']]
chk('n lo', min(ns), (60, 61), quoted='about 60–80 players'); chk('n hi', max(ns), (80, 82))
chk('no2025 outcome', min(r['no2025_fl_p'] for r in e8['transfer_map'] if r['outcome_target']), 0.48, quoted='p ≥ 0.48')
# ---- 3.3
w = tm['E001-WR']
chk('E001 r', w['partial_r'], 0.32, quoted='+0.32 [+0.03, +0.56], n = 64'); chk('E001 lo', w['ci_lo'], 0.03); chk('E001 hi', w['ci_hi'], 0.56); chk('E001 n', w['n'], 64, 0)
w = tm['E006-P1']
chk('E006 r', w['partial_r'], 0.48, quoted='+0.48 [+0.20, +0.68], one-sided p = 0.0007'); chk('E006 lo', w['ci_lo'], 0.20); chk('E006 hi', w['ci_hi'], 0.68)
chk('E006 p', w['fl_p'], 0.0007, 4)
chk('E006 R1000', e8['post_hoc_e006_r1000']['E006-P1']['result']['partial_r'], 0.51, quoted='+0.51 with a less noisy estimator')
chk('E006 q14 (crit 4)', w['bh_q14'], (0, 0.05)); chk('E006 seeds', w['seeds_passing'], 3, 0); chk('E006 drop3 p (crit 6)', w['drop3_fl_p'], (0, 0.05))
chk('E006 crit5 rel', min(w['rel_x'], w['rel_y']), (0.5, 1))
A = PH['rev3']['A_E006_P1_r20']
for sd in ('0', '1', '2'):
    c = A[sd]['checks']
    chk(f'rev3 seed {sd} c1-3', float(c['fl_p_lt_05'] and c['sign_matches'] and c['excess_lb_gt_0'] and c['loco_classes_pass'] in ('2/3', '3/3')), 1.0,
        quoted='passes go criteria 1–6 under our current rules')
chk('q all r1', S['multiplicity']['E006-P1']['bh_q_all_round1_tests'], 0.06, quoted='(q ≈ 0.06)')
chk('m r1', S['multiplicity']['E006-P1']['m_round1_tests'], 90, 0, quoted='~90 round-one tests')
p1 = e7['primary']['P1']['seeds']['0']
chk('E007 r', p1['partial_r'], 0.31, quoted='+0.31 [0.00, +0.56], p = 0.025, q = 0.075'); chk('E007 lo', p1['partial_r_ci95'][0], 0.00)
chk('E007 hi', p1['partial_r_ci95'][1], 0.56); chk('E007 p', p1['fl_p'], 0.025, 3); chk('E007 q', e7['primary']['P1']['bh_q'], 0.075, 3)
chk('cv B2', p1['cv_r2']['B2'], 0.18, quoted='from 0.18 to 0.23'); chk('cv full', p1['cv_r2']['full'], 0.23)
chk('excess', p1['delta_r2_vs_B2']['excess'], 0.045, 3, quoted='+0.045 ± 0.029 SE'); chk('excess se', p1['delta_r2_vs_B2']['se'], 0.029, 3)
ws2 = S['wr_speed_signal']
chk('shared', ws2['n_shared'], 61, 0, quoted='same 61 receivers'); chk('feat r', ws2['r_feature_b2_resid'], 0.77, quoted='features correlate 0.77, targets 0.83')
chk('targ r', ws2['r_target_b2_resid'], 0.83)
l3 = e7['robustness']['leave_3_out']['P1']
chk('l3 r', l3['result']['partial_r'], 0.12, quoted='re-test +0.12 (p = 0.24)'); chk('l3 p', l3['result']['fl_p'], 0.24)
chk('l3 all 2024', float(all(d['draft_year'] == 2024 for d in l3['dropped'])), 1.0, quoted='all 2024 draftees')
chk('E006 drop3 r', tm['E006-P1']['drop3_partial_r'], 0.31, quoted='primary +0.31 [−0.01, +0.57]')
chk('E006 drop3 lo', tm['E006-P1']['drop3_ci_lo'], -0.01); chk('E006 drop3 hi', tm['E006-P1']['drop3_ci_hi'], 0.57)
bc = p1['by_class']
chk('c23', bc['2023']['partial_r'], -0.09, quoted='−0.09, +0.54, +0.41 (2023–25)'); chk('c24', bc['2024']['partial_r'], 0.54); chk('c25', bc['2025']['partial_r'], 0.41)
n25 = e7['robustness']['without_2025']['P1']
chk('no25 r', n25['partial_r'], 0.24, quoted='+0.24 [−0.16, +0.57]'); chk('no25 lo', n25['partial_r_ci95'][0], -0.16); chk('no25 hi', n25['partial_r_ci95'][1], 0.57)
ds = e7['post_hoc_review']['tests']['P1_plus_deep_share']
chk('deep r', ds['partial_r'], 0.21, quoted='leaves +0.21 [−0.10, +0.49]'); chk('deep lo', ds['partial_r_ci95'][0], -0.10); chk('deep hi', ds['partial_r_ci95'][1], 0.49)
cc = e7['post_hoc_review']['covariate_correlations_with_B2_residuals']['P1']['deep_share']
chk('deep share with target > with feature > 0', float(cc['with_target_res'] > cc['with_feature_res'] > 0), 1.0, quoted='which faster finishers run slightly more')
chk('timing r', e7['leaderboard']['post_hoc_official_splits']['corr_tracked_vs_official_10_40_time_beyond_forty'], -0.11, quoted='(r = −0.11)')
chk('l3 power at .3', e7['post_hoc_review']['P1_power_and_heterogeneity']['leave_3_out_power']['by_true_partial_r']['0.3']['p_leave_3_out_given_criterion_1'], (0.30, 0.36), quoted='one time in three')
p3 = e7['primary']['P3']['seeds']['0']
chk('P3 r', p3['partial_r'], -0.31, quoted='−0.31 [−0.58, +0.02]'); chk('P3 lo', p3['partial_r_ci95'][0], -0.58); chk('P3 hi', p3['partial_r_ci95'][1], 0.02)
ub = e7['post_hoc_review']['upper_95_bounds']['P3']
chk('P3 bounds max', max(ub[k] for k in ub if k != 'reverse_leave_3_out_fisher'), 0.10, quoted='above about +0.10')
chk('lb rel', lb['meta']['post_hoc_uncertainty']['reliability_of_hidden'], 0.72, quoted='reliability 0.72')
ar = lb['meta']['post_hoc_attempt_replication']
chk('top15', ar['n_top_quintile_by_attempt1'], 15, 0, quoted='8 of 15 top-fifth'); chk('stay8', ar['n_still_top_quintile_by_attempt2'], 8, 0)
# ---- 3.4
chk('c5 nulls', T['nulls_passing_c5'], 13, 0, quoted='13 of 16 nulls')
chk('c5 resid', sum(r['rel_x_b2'] >= 0.5 and r['rel_y_b2'] >= 0.5 for r in nulls), 12, 0, quoted='(12 on the residual)')
pdf = [u['rel_round1_def'] for u in S['drill_profile']['used_features'] if u['metric'] in
       ('stop peak_decel', 'turn retention', 'mirror rev_accel', 'dv_10 | entry+turn adj.', 'backpedal bp_peak')]
chk('drill feats lo', min(pdf), 0.44, quoted='(0.44–0.73)'); chk('drill feats hi', max(pdf), 0.73)
chk('route shape r', E['E002']['diagnostics']['game']['g_turn_retention~g_turn_turn_radius']['r'], 0.88, quoted='r = 0.88 with turn radius')
cu = 1 - e7['post_hoc_review']['alignment']['cushion_oe_align_eb']['tau2'] / e7['targets']['eb_diagnostics']['cushion_oe']['tau2']
chk('cushion share removed', cu, (0.70, 0.80), quoted='about three quarters')
chk('control cols', tm['E002-P1']['n'] - tm['E002-P1']['resid_df'], (19, 23), quoted='about 20 control columns')
# ---- 4
tr = e8['target_reliability_by_position']
chk('WR top', tr['E001-WR']['WR'], 0.91, quoted='Receiver top speed (0.91)'); chk('DL burst', tr['E003-P1']['DL'], 0.83, quoted='DL burst (0.83)')
chk('getoff', tr['E003-P2']['DL'], 0.82, quoted='get-off (0.82)')
I = PH['inseason']
chk('inseason m1', I['game_first_m']['1']['r'], 0.56, quoted='(r 0.56 vs 0.57)'); chk('inseason best combine', I['combine'][I['best_combine']]['r'], 0.57)
chk('best combine is max', float(I['combine'][I['best_combine']]['r'] == max(v['r'] for v in I['combine'].values())), 1.0)
later = [I['game_first_m'][str(m)]['r'] for m in range(2, 9)]
chk('inseason m2-8 about 0.77', sum(later) / len(later), (0.765, 0.775), quoted='about 0.77')
chk('inseason m2-8 all > combine', float(min(later) > I['combine'][I['best_combine']]['r']), 1.0, quoted='two or more games, probably better')
# ---- 5
a = e7['attrition']
chk('16 lack 40', a['combine_wr'] - a['with_tracked_40'], 16, 0, quoted='of 107 Combine receivers, 16 lack a tracked 40 and 30 more')
chk('30 more', a['with_tracked_40'] - a['P1_ge_50_plays'], 30, 0)
# ---- post hoc values used only in comments / figure footers
chk('pairs share', PH['pairs']['share'], 0.59); chk('pairs ci lo', PH['pairs']['share_ci95_receiver_bootstrap'][0], 0.43)
chk('pairs ci hi', PH['pairs']['share_ci95_receiver_bootstrap'][1], 0.73)
chk('mean game gap', e7['concept_pair']['post_hoc_all_matched_pairs']['mean_game_gap_mph'], 0.20)
print('\nN checks:', sum(1 for _ in open(__file__) if 'chk(' in _), '| FAILS:', fails)
