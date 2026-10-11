"""Recompute paired confirmation statistics from published fields, without model calls."""
from pathlib import Path
import csv, hashlib, itertools, json
from collections import defaultdict
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
INPUT = ROOT / 'artifacts/tables/confirmation_records.json'
OUT = ROOT / 'artifacts/tables/confirmation_statistics'
SEED = 20260825
SAMPLES = 10000


def holm(values):
    order = np.argsort(values, kind='stable')
    adjusted = [0.0] * len(values)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (len(values) - rank) * values[i]))
        adjusted[i] = running
    return adjusted


def inference(effects, seed):
    effects = np.asarray(effects, dtype=float)
    rng = np.random.default_rng(seed)
    samples = rng.choice(effects, (SAMPLES, len(effects)), replace=True).mean(axis=1)
    lo, hi = np.quantile(samples, [0.025, 0.975])
    observed = abs(effects.mean())
    if len(effects) <= 12:
        signs = np.asarray(list(itertools.product([-1, 1], repeat=len(effects))))
        null = (signs * effects).mean(axis=1)
        p = np.count_nonzero(abs(null) >= observed - 1e-12) / len(null)
        exact = True
    else:
        signs = rng.choice([-1, 1], (SAMPLES, len(effects)))
        null = (signs * effects).mean(axis=1)
        p = (1 + np.count_nonzero(abs(null) >= observed - 1e-12)) / (SAMPLES + 1)
        exact = False
    return float(effects.mean()), float(lo), float(hi), float(p), exact, len(null)


def eligible(row, metric):
    if metric == 'clean_success':
        return row['condition'] == 'clean' and row['success'] is not None
    if not row['applied'] or row['condition'] == 'clean':
        return False
    field = {'fault_success': 'success', 'mas_consequence': 'M', 'recorded_recovery': 'raw_recovery_detected'}[metric]
    return row.get(field) is not None


def value(row, metric):
    return bool(row[{'clean_success': 'success', 'fault_success': 'success', 'mas_consequence': 'M', 'recorded_recovery': 'raw_recovery_detected'}[metric]])


def run():
    rows = json.loads(INPUT.read_text())
    assert len(rows) == 18900
    identities = [(r['domain'], r['task'], r['model'], r['topology'], r['condition'], r['rep']) for r in rows]
    assert len(set(identities)) == len(rows), 'Duplicate scheduled identities'
    results = []
    for axis in ['model', 'topology']:
        strata = defaultdict(list)
        for r in rows:
            strata[(r['domain'], r['model'] if axis == 'topology' else '')].append(r)
        for (domain, fixed_model), group in sorted(strata.items()):
            indices = defaultdict(dict)
            for r in group:
                key = (r['task'], r['condition'], r['rep'], r['topology'] if axis == 'model' else '')
                assert key not in indices[r[axis]]
                indices[r[axis]][key] = r
            for metric in ['clean_success', 'fault_success', 'mas_consequence', 'recorded_recovery']:
                for left, right in itertools.combinations(sorted(indices), 2):
                    common = sorted(indices[left].keys() & indices[right].keys())
                    clusters = defaultdict(list)
                    cells = defaultdict(int)
                    for key in common:
                        a, b = indices[left][key], indices[right][key]
                        if not (eligible(a, metric) and eligible(b, metric)):
                            continue
                        av, bv = value(a, metric), value(b, metric)
                        clusters[a['task']].append(int(bv) - int(av))
                        cells[f'{int(av)}{int(bv)}'] += 1
                    if not clusters:
                        continue
                    effects = [np.mean(clusters[k]) for k in sorted(clusters)]
                    seed = SEED + len(results)
                    estimate, lo, hi, p, exact, n = inference(effects, seed)
                    results.append(dict(axis=axis, domain=domain, fixed_model=fixed_model, metric=metric,
                        left=left, right=right, effect_right_minus_left=estimate, lower_95=lo, upper_95=hi,
                        p_value_raw=p, task_count=len(clusters), eligible_pair_count=sum(cells.values()),
                        common_design_pair_count=len(common), pairs_00=cells['00'], pairs_01=cells['01'],
                        pairs_10=cells['10'], pairs_11=cells['11'], sign_flip_exact=exact,
                        permutations=n, bootstrap_samples=SAMPLES, seed=seed))
    families = defaultdict(list)
    for i, r in enumerate(results):
        families[(r['axis'], r['domain'], r['metric'])].append(i)
    for family, indices in families.items():
        adjusted = holm([results[i]['p_value_raw'] for i in indices])
        for i, p in zip(indices, adjusted):
            results[i]['p_value_holm'] = p
            results[i]['holm_family'] = '|'.join(family)
            results[i]['holm_family_size'] = len(indices)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT/'paired_comparisons.json').write_text(json.dumps(results, indent=2)+'\n')
    with (OUT/'paired_comparisons.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(results[0]), lineterminator="\n"); writer.writeheader(); writer.writerows(results)
    protocol = dict(input='artifacts/tables/confirmation_records.json', input_sha256=hashlib.sha256(INPUT.read_bytes()).hexdigest(),
        scheduled_records=len(rows), resamples=SAMPLES, seed=SEED, algorithm='NumPy default_rng; deterministic row-order seed offsets',
        effect='Right minus left, equal weight per task after averaging eligible repeated condition pairs within task',
        pairing='domain/task/condition/repetition plus topology for model comparisons; fixed model for topology comparisons',
        eligibility='Clean success uses known clean outcomes; other metrics require both faults applied and both metric fields present. Recovery missingness is not treated as false.',
        ci='Task-cluster percentile bootstrap, 95%; repeated conditions remain together within each task.',
        p_value='Two-sided task sign-flip, exact at <=12 tasks, otherwise 10,000 Monte Carlo samples with plus-one correction.',
        holm='Separate model and topology families, each within domain and metric. Topology family pools the three fixed-model strata.',
        scope='Recomputed supplement for the published confirmation projection. Does not include position-extension records, A-label contrasts absent from the projection, or unrecorded detection/recovery information.',
        causal_boundary='Deployed model and workflow comparisons are configuration-level associations. MAS consequence labels are post-exposure and do not establish causal effects on failure.')
    (OUT/'protocol.json').write_text(json.dumps(protocol, indent=2)+'\n')
    print(f'Published {len(results)} paired comparisons in {len(families)} Holm families from {len(rows)} records.')


if __name__ == '__main__':
    run()
