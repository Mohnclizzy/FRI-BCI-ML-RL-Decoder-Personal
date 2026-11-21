import json, pathlib
base = pathlib.Path('results/robomimic_controls_s001r12_2026-11-06')
def label(s):
    if s.get('terminal_retained'): return 'complete_retained'
    if s.get('first_acquisition_step') is None: return 'contact_none'
    if s.get('post_acquisition_loss'): return 'contact_lost'
    return 'complete_unretained'
out = []
for d in sorted(p for p in base.iterdir() if (p/'summary.json').exists()):
    s = json.loads((d/'summary.json').read_text())
    out.append({'condition': s['condition'], 'seed': s['seed'],
                'label': label(s), 'terminal_retained': bool(s.get('terminal_retained')),
                'first_acquisition_step': s.get('first_acquisition_step'),
                'max_rise_m': s.get('max_rise_m'), 'checkpoint_sha256': s.get('checkpoint_sha256')})
out.sort(key=lambda r: (r['condition'], r['seed']))
with (base/'controls.tsv').open('w') as f:
    f.write('condition\tseed\tcompletion_state\tterminal_retained\tfirst_acquisition_step\tmax_rise_m\n')
    for r in out:
        f.write('\t'.join(str(r[k]) for k in ['condition','seed','label','terminal_retained','first_acquisition_step','max_rise_m'])+'\n')
print('\n'.join('\t'.join(str(r[k]) for k in ['condition','seed','label','terminal_retained','first_acquisition_step','max_rise_m']) for r in out))
print('checkpoint', out[0]['checkpoint_sha256'] if out else None)
