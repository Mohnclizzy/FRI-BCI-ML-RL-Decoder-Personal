"""Create a 30 Hz compact trace from the selected-field 120 Hz trace."""
import csv
from pathlib import Path
p=Path('results/unilateral_2026-11-03')
with (p/'trace_120hz.csv').open(newline='') as inp, (p/'trace_30hz.csv').open('x',newline='') as out:
    r=csv.reader(inp); w=csv.writer(out); w.writerow(next(r)); counter={}
    for row in r:
        seed=row[0]; counter[seed]=counter.get(seed,0)+1
        if counter[seed]%4==0: w.writerow(row)
