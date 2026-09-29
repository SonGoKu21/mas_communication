"""Offline paired RQ4 analysis; no model or environment imports."""
import argparse
import csv
import json
import random
from collections import defaultdict
from pathlib import Path
from statistics import mean

ARMS = ('baseline', 'action_only', 'semantic_only', 'combined')


def cluster_interval(values, repeats=4000):
    if not values:
        return None
    means = [mean(v) for _, v in sorted(values.items())]
    rng = random.Random(20260914)
    draws = sorted(mean(rng.choices(means, k=len(means))) for _ in range(repeats))
    return dict(estimate=mean(means), low=draws[int(.025 * repeats)],
                high=draws[int(.975 * repeats)], clusters=len(means),
                pairs=sum(len(v) for v in values.values()))


def factorial(rows, matched=False):
    clean = {(r['task_id'], r['topology'], r['repeat_index'], r['arm']): r['final_task_success']
             for r in rows if r['experiment'] == 'main' and r['condition'] == 'clean'}
    pairs = defaultdict(dict)
    for row in rows:
        if row['experiment'] != 'main' or row['condition'] == 'clean':
            continue
        pairs[(row['task_id'], row['topology'], row['condition'], row['repeat_index'])][row['arm']] = row
    groups = defaultdict(lambda: defaultdict(list))
    for (task, topology, condition, repeat), arms in pairs.items():
        if set(arms) != set(ARMS):
            continue
        if matched and not all(clean.get((task, topology, repeat, a)) is True for a in ARMS):
            continue
        b, a, s, c = [int(arms[k]['final_task_success']) for k in ARMS]
        effects = dict(action_without_semantic=a-b, semantic_without_action=s-b,
                       action_with_semantic=c-s, semantic_with_action=c-a,
                       factorial_action=((a-b)+(c-s))/2,
                       factorial_semantic=((s-b)+(c-a))/2, interaction=c-a-s+b,
                       combined_vs_baseline=c-b)
        cluster = arms['baseline']['product_cluster']
        for metric, value in effects.items():
            groups[(topology, condition, metric)][cluster].append(value)
    return [dict(topology=k[0], condition=k[1], contrast=k[2], matched_clean=matched,
                 **cluster_interval(v)) for k, v in sorted(groups.items())]


def diagnostic_pairs(rows):
    results = []
    main = {(r['task_id'], r['topology'], r['condition'], r['repeat_index'], r['variant']): r
            for r in rows if r['experiment'] == 'main'}
    stress = defaultdict(lambda: defaultdict(list))
    for r in rows:
        if r['experiment'] != 'stress':
            continue
        key = (r['task_id'], r['topology'], r['condition'], r['repeat_index'], r['variant'])
        original = main.get(key)
        if original is not None:
            stress[(r['topology'], r['condition'], r['variant'])][r['product_cluster']].append(
                int(r['final_task_success']) - int(original['final_task_success']))
    for (topology, condition, arm), vals in sorted(stress.items()):
        results.append(dict(experiment='stress', topology=topology, condition=condition,
                            contrast=arm + ':finite2_minus_single', **cluster_interval(vals)))
    path = defaultdict(dict)
    for r in rows:
        if r['experiment'] == 'path_diagnostic':
            path[(r['task_id'], r['condition'], r['repeat_index'])][r['variant']] = r
    groups = defaultdict(lambda: defaultdict(list))
    for (_, condition, _), arms in path.items():
        for left, right in [('duplicate_forwarding', 'single_path'),
                            ('independent_observation', 'single_path'),
                            ('independent_observation', 'duplicate_forwarding')]:
            if left in arms and right in arms:
                groups[(condition, left + '_minus_' + right)][arms[left]['product_cluster']].append(
                    int(arms[left]['final_task_success']) - int(arms[right]['final_task_success']))
    for (condition, contrast), vals in sorted(groups.items()):
        results.append(dict(experiment='path_diagnostic', topology='fixed', condition=condition,
                            contrast=contrast, **cluster_interval(vals)))
    return results


def write_csv(path, rows):
    if not rows:
        path.write_text('')
        return
    fields = list(dict.fromkeys(k for r in rows for k in r))
    with path.open('w') as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def analyze(source, output):
    manifest = json.loads((source / 'matrix_manifest.json').read_text())
    rows = [json.loads(s) for s in (source / 'main_runs.jsonl').read_text().splitlines()]
    expected = {j['job_key'] for j in manifest['config']['jobs']}
    if len({r['job_key'] for r in rows}) != len(rows) or len({r['run_id'] for r in rows}) != len(rows):
        raise ValueError('duplicate result')
    if any(r['job_key'] not in expected for r in rows):
        raise ValueError('unknown result')
    output.mkdir(parents=True, exist_ok=False)
    effects = factorial(rows) + factorial(rows, matched=True)
    diagnostics = diagnostic_pairs(rows)
    grouped = defaultdict(list)
    for r in rows:
        grouped[(r['experiment'], r['topology'], r['condition'], r['variant'])].append(r)
    rates = []
    for (experiment, topology, condition, variant), group in sorted(grouped.items()):
        rates.append(dict(experiment=experiment, topology=topology, condition=condition, variant=variant,
                          runs=len(group), success=sum(r['final_task_success'] for r in group),
                          actual_exposure=sum(len(r.get('fault_events', [])) for r in group),
                          scheduled_exposure=sum(r['planned_exposures'] for r in group),
                          fault_triggered=sum(bool(r.get('fault_events')) for r in group),
                          mean_model_calls=mean(r.get('model_calls', 0) for r in group),
                          mean_known_tokens=mean(r.get('known_total_tokens', 0) for r in group),
                          mean_latency_ms=mean(r['latency_ms'] for r in group)))
    summary = dict(completed=len(rows), planned=len(expected), complete=len(rows) == len(expected),
                   factorial_effects=effects, diagnostics=diagnostics, groups=rates,
                   inference='task/product-cluster percentile bootstrap, 4000 draws; descriptive small-cluster intervals; no equivalence claim')
    (output / 'summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2))
    write_csv(output / 'factorial_effects.csv', effects)
    write_csv(output / 'diagnostic_effects.csv', diagnostics)
    write_csv(output / 'group_results.csv', rates)
    lines = ['# RQ4 机制实验', '', f"完成 {len(rows)}/{len(expected)}。" + ('最终统计。' if summary['complete'] else '中间检查点，不作最终结论。'), '',
             '主实验检验两个正交保护开关；stress比较有限两次与单次；路径诊断比较同源复制与重新观测。',
             '三次重复不是API可控seed。主实验6个商品聚类、stress3个、路径5个，置信区间仅作小样本描述，不证明等效。', '',
             '## 分组结果', '', '|部分|拓扑|故障|配置|成功/运行|实际/计划注入|', '|---|---|---|---|---:|---:|']
    for r in rates:
        lines.append(f"|{r['experiment']}|{r['topology']}|{r['condition']}|{r['variant']}|{r['success']}/{r['runs']}|{r['actual_exposure']}/{r['scheduled_exposure']}|")
    lines += ['', '## 解释边界', '', '单次主矩阵的恢复通信豁免；stress对同一受影响路径的前两次交付注入，独立观测旁路不共享该故障源。',
              '语义机制使用本次初始真实观测固定商品身份。该信任假设不能推广到初始观测也受损的情况。',
              '成功包含已记录完整购物车约束，不覆盖库存、价格和订单。缺少实际触发不能视为恢复。',
              '同样成功率不等于统计等效；路径诊断没有额外clean，不单独声称完整matched-clean因果结果。']
    (output / 'report_zh.md').write_text('\n'.join(lines) + '\n')
    seen = set()
    with (output / 'representative_traces.jsonl').open('w') as f:
        for r in rows:
            if r['condition'] == 'clean':
                continue
            key = (r['experiment'], r['condition'], r['variant'], r['final_task_success'])
            if key in seen:
                continue
            seen.add(key)
            keep = ('run_id', 'task_id', 'experiment', 'topology', 'condition', 'variant', 'final_task_success',
                    'fault_events', 'exposure_deliveries', 'semantic_contract_events', 'recovery_events',
                    'observed_M_consequence', 'whole_budget', 'recorded_cart_audit')
            f.write(json.dumps({k: r.get(k) for k in keep}, ensure_ascii=False) + '\n')
    print(json.dumps(dict(completed=len(rows), effects=len(effects), diagnostics=len(diagnostics))))


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('source', type=Path)
    p.add_argument('output', type=Path)
    a = p.parse_args()
    analyze(a.source, a.output)
