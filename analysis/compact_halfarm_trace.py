"""Create 30 Hz compact output from selected-field 120 Hz evaluation data."""
import csv
from pathlib import Path
p=Path('results/halfarm_2026-11-03')
with (p/'trace_120hz.csv').open(newline='') as inp, (p/'trace_30hz.csv').open('x',newline='') as out:
 r=csv.reader(inp);w=csv.writer(out);w.writerow(next(r));count={}
 for row in r:
  seed=row[0];count[seed]=count.get(seed,0)+1
  if count[seed]%4==0:w.writerow(row)
