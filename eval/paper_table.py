#!/usr/bin/env python
"""Collect paper_metrics.json files (paper_eval.py) into one table; missing values print as NaN.

  python cad_eval/paper_table.py cad_eval/runs/*_7962/paper_metrics.json baselines/*/paper_metrics.json \
      [--csv table.csv] [--latex] [--paper-latex]

--paper-latex emits the body of the paper's table: metrics as rows, systems as columns, every value at
three decimals, with the Hausdorff distance in units of 1e-2 (the raw values round to the same three
decimals for the strongest systems).  The JSON keeps raw, unscaled numbers -- the scaling lives here,
in the presentation, so nothing downstream can mistake a scaled value for a measured one.
"""
import argparse
import csv
import json

COLS = ['System', 'n', 'IR (%)', 'CD median', 'CD mean', 'HD median', 'HD mean', 'NC median', 'NC mean', 'ECD',
        'IoU', 'Euler match (%)']


def cell(v):
    return 'NaN' if v is None else (f'{v:g}' if isinstance(v, float) else str(v))


# paper table: (label, path in paper_metrics.json['metrics'], multiplier)
PAPER_ROWS = [('IR (\\%) $\\downarrow$', ('IR_pct',), 1.0),
              ('CD $\\downarrow$ \\textit{mean}', ('CD', 'mean'), 1.0),
              ('CD $\\downarrow$ \\textit{median}', ('CD', 'median'), 1.0),
              ('HD ($\\times 10^{-2}$) $\\downarrow$ \\textit{mean}', ('HD', 'mean'), 100.0),
              ('HD ($\\times 10^{-2}$) $\\downarrow$ \\textit{median}', ('HD', 'median'), 100.0),
              ('NC $\\uparrow$ \\textit{mean}', ('NC', 'mean'), 1.0),
              ('NC $\\uparrow$ \\textit{median}', ('NC', 'median'), 1.0),
              ('ECD $\\downarrow$', ('ECD', 'median'), 1.0),
              ('IoU $\\uparrow$', ('IoU', 'mean'), 1.0),
              ('Euler match (\\%) $\\uparrow$', ('euler_match_pct',), 1.0)]
HIGHER_IS_BETTER = {'NC', 'IoU', 'Euler'}


def dig(metrics, path):
    v = metrics
    for k in path:
        v = v.get(k) if isinstance(v, dict) else None
        if v is None:
            return None
    return float(v)


def paper_latex(files, decimals=3):
    """Body of the paper table: one row per metric, one column per system, best value in bold."""
    docs = [json.load(open(f)) for f in files]
    names = [d['table_row']['System'] for d in docs]
    print('% columns: ' + ' | '.join(names))
    for label, path, mult in PAPER_ROWS:
        vals = [None if (v := dig(d['metrics'], path)) is None else v * mult for d in docs]
        good = [v for v in vals if v is not None]
        if good:
            up = any(label.startswith(k) for k in HIGHER_IS_BETTER)
            best = max(good) if up else min(good)
        cells = []
        for v in vals:
            if v is None:
                cells.append('--')
            else:
                t = f'{v:.{decimals}f}'
                cells.append(f'\\textbf{{{t}}}' if good and v == best else t)
        print(f'{label}\n& ' + '\n& '.join(cells) + ' \\\\')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('files', nargs='+')
    ap.add_argument('--csv', default=None)
    ap.add_argument('--latex', action='store_true')
    ap.add_argument('--paper-latex', action='store_true',
                    help="the paper's table body: metrics as rows, 3 decimals, HD in 1e-2")
    ap.add_argument('--decimals', type=int, default=3)
    a = ap.parse_args()
    if a.paper_latex:
        paper_latex(a.files, a.decimals)
        return
    rows = [json.load(open(f))['table_row'] for f in a.files]
    table = [[cell(r.get(c)) for c in COLS] for r in rows]
    if a.latex:
        print(' & '.join(COLS) + r' \\ \hline')
        for t in table:
            print(' & '.join(t) + r' \\')
    else:
        print('| ' + ' | '.join(COLS) + ' |')
        print('|' + '---|' * len(COLS))
        for t in table:
            print('| ' + ' | '.join(t) + ' |')
    if a.csv:
        with open(a.csv, 'w', newline='') as f:
            w = csv.writer(f)
            w.writerow(COLS)
            w.writerows(table)


if __name__ == '__main__':
    main()
