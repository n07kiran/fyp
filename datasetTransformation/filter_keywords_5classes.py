import csv
import ast
import json

INPUT = 'all_morphology_keywords.csv'
OUTPUT = 'all_morphology_keywords_5classes.csv'

# map by substring to canonical class names
MAPPINGS = [
    ('micro', 'Microcytes'),
    ('macro', 'Macrocytes'),
    ('normo', 'Normocytes'),
    ('ellipt', 'Elliptocytes'),
    ('target', 'Target Cells'),
]

def map_terms(terms):
    out = []
    for t in terms:
        s = t.strip().lower()
        for k, name in MAPPINGS:
            if k in s:
                if name not in out:
                    out.append(name)
                break
    return out

with open(INPUT, newline='') as inf, open(OUTPUT, 'w', newline='') as outf:
    reader = csv.DictReader(inf)
    fieldnames = ['cohort', 'file_name', 'classes']
    writer = csv.DictWriter(outf, fieldnames=fieldnames)
    writer.writeheader()
    for row in reader:
        raw = row.get('keywords', '')
        try:
            terms = ast.literal_eval(raw)
        except Exception:
            # fallback: try splitting
            terms = [x.strip() for x in raw.strip('[]').split(',') if x.strip()]
        mapped = map_terms(terms)
        writer.writerow({
            'cohort': row.get('cohort',''),
            'file_name': row.get('file_name',''),
            'classes': json.dumps(mapped)
        })

print('Wrote', OUTPUT)
