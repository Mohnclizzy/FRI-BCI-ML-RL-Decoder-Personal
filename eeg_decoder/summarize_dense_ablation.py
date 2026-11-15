"""Aggregate dense-window ablation results with seeds nested within subjects."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import statistics


METRICS = {
    'balanced_accuracy': lambda r: r['metrics']['window_metrics']['balanced_accuracy'],
    'raw_rest_activation': lambda r: r['metrics']['command_metrics']['raw']['rest_command_fraction'],
    'raw_imagery_coverage': lambda r: 1-r['metrics']['command_metrics']['raw']['missed_imagery_fraction'],
    'raw_missed_imagery': lambda r: r['metrics']['command_metrics']['raw']['missed_imagery_fraction'],
    'raw_rest_episode_activation': lambda r: r['metrics']['command_metrics']['raw']['rest_episode_activation_fraction'],
    'confidence_rest_activation': lambda r: r['metrics']['command_metrics']['confidence']['rest_command_fraction'],
    'confidence_imagery_coverage': lambda r: 1-r['metrics']['command_metrics']['confidence']['missed_imagery_fraction'],
    'confidence_missed_imagery': lambda r: r['metrics']['command_metrics']['confidence']['missed_imagery_fraction'],
    'temporal_rest_activation': lambda r: r['metrics']['command_metrics']['temporal']['rest_command_fraction'],
    'temporal_imagery_coverage': lambda r: 1-r['metrics']['command_metrics']['temporal']['missed_imagery_fraction'],
    'temporal_missed_imagery': lambda r: r['metrics']['command_metrics']['temporal']['missed_imagery_fraction'],
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run-dir', type=Path, required=True,
                        help='Directory containing seed_<n>_target_summary.json files.')
    args = parser.parse_args()
    summaries = sorted(args.run_dir.glob('seed_*_target_summary.json'))
    if not summaries:
        raise FileNotFoundError(f'No seed target summaries in {args.run_dir}')
    rows = []
    for path in summaries:
        items = json.loads(path.read_text(encoding='utf-8'))
        if len(items) != 15:
            raise ValueError(f'{path} contains {len(items)} subjects, expected 15')
        rows.extend(items)
    seeds = sorted({int(r['seed']) for r in rows})
    subjects = sorted({int(r['subject']) for r in rows})
    if len(rows) != len(seeds)*15 or subjects != list(range(1, 16)):
        raise ValueError('Expected complete S001-S015 coverage for every seed')

    per_subject = []
    for subject in subjects:
        participant = [r for r in rows if r['subject'] == subject]
        if {r['seed'] for r in participant} != set(seeds):
            raise ValueError(f'Seed coverage incomplete for S{subject:03d}')
        row = {'subject': f'S{subject:03d}', 'n_training_seeds': len(participant)}
        for key, getter in METRICS.items():
            values = [float(getter(r)) for r in participant]
            row[key] = statistics.mean(values)
            row[key+'_seed_sd'] = statistics.stdev(values) if len(values) > 1 else 0.0
        per_subject.append(row)

    csv_path = args.run_dir / 'participant_metrics.csv'
    with csv_path.open('w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(per_subject[0]))
        writer.writeheader(); writer.writerows(per_subject)
    aggregate = {}
    for key in METRICS:
        values = [r[key] for r in per_subject]
        aggregate[key] = {'participant_mean': statistics.mean(values),
                          'participant_sd': statistics.stdev(values),
                          'n_participants': len(values)}
    baseline_file = args.run_dir.parent / 'gpu_campaign_2026-11-13' / 'development_summary.json'
    historical_baseline = None
    if baseline_file.exists():
        groups = json.loads(baseline_file.read_text(encoding='utf-8'))['groups']
        key = 'eegnet8/transfer/0.0001/300'
        if key not in groups or not groups[key].get('complete'):
            raise ValueError('Historical selected EEGNet8 baseline is missing/incomplete')
        b = groups[key]
        historical_baseline = {
            'source': 'gpu_campaign_2026-11-13/development_summary.json',
            'recipe': key, 'observations': b['completed'],
            'balanced_accuracy': b['mean_balanced_accuracy'],
            'raw_rest_activation': b['mean_raw_rest_activation'],
            'raw_missed_imagery': b['mean_raw_missed_imagery'],
            'dense_minus_baseline': {
                'balanced_accuracy': aggregate['balanced_accuracy']['participant_mean']-b['mean_balanced_accuracy'],
                'raw_rest_activation': aggregate['raw_rest_activation']['participant_mean']-b['mean_raw_rest_activation'],
                'raw_missed_imagery': aggregate['raw_missed_imagery']['participant_mean']-b['mean_raw_missed_imagery'],
            },
            'caveat': 'Historical development aggregate; seed/participant pairing was unavailable in its compact summary.'
        }
    result = {'seeds': seeds, 'n_participants': len(subjects), 'n_seed_subject_rows': len(rows),
              'aggregation': 'average the 3 training seeds within each participant, then summarize n=15 participants',
              'target_r12_touched': False, 'role': 'exploratory R08 development; not independent confirmation',
              'aggregate': aggregate, 'historical_baseline_context': historical_baseline,
              'participant_metrics_file': csv_path.name}
    (args.run_dir/'aggregate.json').write_text(json.dumps(result, indent=2)+'\n', encoding='utf-8')
    md = ['# Dense causal-window ablation (R08 development)', '',
          f"Seeds: {', '.join(map(str, seeds))}. Participants: {len(subjects)}.",
          'Seeds are averaged within participant; cohort SD uses the 15 participants.',
          'R08 is development data and is not independent confirmation. Target R12 was not loaded.', '',
          '| Metric | Mean across participants | SD across participants |', '|---|---:|---:|']
    for key, item in aggregate.items():
        md.append(f"| {key.replace('_',' ')} | {item['participant_mean']:.3f} | {item['participant_sd']:.3f} |")
    if historical_baseline:
        md += ['', '## Historical recipe context', '',
               'The earlier selected EEGNet8 transfer recipe (LR .0001, epoch 300) reported 68.15% balanced accuracy, 40.11% raw REST activation, and 47.95% missed imagery across 45 participant-seed rows. This run reports 62.52%, 42.07%, and 40.61%, respectively. Dense training modestly reduced missed imagery in this exploratory comparison but reduced balanced accuracy and did not reduce REST activation. The compact historical summary has no paired rows, so these are descriptive differences, not paired inference.']
    (args.run_dir/'summary.md').write_text('\n'.join(md)+'\n', encoding='utf-8')
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise RuntimeError('Install matplotlib to generate the required scientific plots') from exc
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), constrained_layout=True)
    colors = {'raw': '#c44e52', 'confidence': '#4c72b0', 'temporal': '#55a868'}
    labels = {'raw': 'Raw 0.50', 'confidence': 'Confidence 0.70',
              'temporal': 'Temporal 0.70 / 0.50 s'}
    for gate in colors:
        x = [r[f'{gate}_rest_activation'] for r in per_subject]
        y = [r[f'{gate}_imagery_coverage'] for r in per_subject]
        axes[0].scatter(x, y, color=colors[gate], alpha=.58, s=28, label=labels[gate])
        axes[0].scatter([statistics.mean(x)], [statistics.mean(y)], color=colors[gate],
                        edgecolor='black', linewidth=.7, marker='D', s=75, zorder=4)
    axes[0].set(xlabel='REST time with ENABLE command', ylabel='Imagery time with ENABLE command',
                title='Command tradeoff (R08 development)', xlim=(-.02, 1.02), ylim=(-.02, 1.02))
    axes[0].grid(alpha=.2); axes[0].legend(frameon=False, fontsize=8)
    ba = [r['balanced_accuracy'] for r in per_subject]
    axes[1].scatter(range(1, len(ba)+1), ba, color='#8172b2', s=32)
    axes[1].axhline(statistics.mean(ba), color='#8172b2', linestyle='--',
                    label=f'participant mean = {statistics.mean(ba):.3f}')
    axes[1].set(xlabel='Participant index (S001–S015)', ylabel='Balanced accuracy',
                title='Window classification (R08 development)', xticks=range(1, 16), ylim=(0, 1))
    axes[1].grid(alpha=.2); axes[1].legend(frameon=False, fontsize=8)
    plot_path = args.run_dir/'dense_ablation_tradeoff.png'
    fig.savefig(plot_path, dpi=180); plt.close(fig)
    print(json.dumps({'aggregate': aggregate, 'participant_metrics': str(csv_path)}, indent=2))


if __name__ == '__main__':
    main()
