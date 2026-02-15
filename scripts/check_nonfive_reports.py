#!/usr/bin/env python3
import csv,ast,glob,os,re
base=os.path.dirname(os.path.dirname(__file__))
csvf=os.path.join(base,'all_morphology_keywords.csv')
five_map={
    'normocytic':['normocytic'],
    'microcytic':['microcytic'],
    'macrocytic':['macrocytic'],
    'elliptocytes':['elliptocyte','elliptocytes'],
    'target cells':['target cell','target cells','codocyte','codocytes']
}
five=set(five_map.keys())
rows=[]
with open(csvf) as fp:
    r=csv.DictReader(fp)
    for row in r:
        try:
            kws=ast.literal_eval(row['keywords'])
        except Exception:
            kws=[]
        kws=[k.strip().lower() for k in kws]
        if not any(k in five for k in kws):
            rows.append((row['cohort'],row['file_name'],kws))

results=[]
for cohort,fn,kws in rows:
    # search for morphology report first
    pattern=os.path.join(base,'AneRBC_dataset','**','Morphology_reports', fn + '.txt')
    matches=glob.glob(pattern, recursive=True)
    if not matches:
        matches=glob.glob(os.path.join(base,'AneRBC_dataset','**', fn + '.txt'), recursive=True)
    path=''
    content='(not found)'
    if matches:
        path=matches[0]
        try:
            with open(path) as m:
                content=m.read()
        except Exception as e:
            content=f'(error reading: {e})'
    text=content.lower()
    found_terms=[]
    for canonical,variants in five_map.items():
        for v in variants:
            if re.search(r"\b"+re.escape(v)+r"\b", text):
                found_terms.append(canonical)
                break
    results.append({'cohort':cohort,'file_name':fn,'parser_keywords':"; ".join(kws),'report_path':path,'report_found_terms':"; ".join(found_terms),'report_excerpt':content[:400].replace('\n',' ')})

# print concise summary
print('checked', len(results), 'files')
for r in results:
    print(r['file_name'], '| report_terms=', r['report_found_terms'] or '(none)', '| report_path=', r['report_path'] or '(none)')

# write CSV
outf=os.path.join(base,'nonfive_report_check.csv')
with open(outf,'w') as w:
    writer=csv.DictWriter(w, fieldnames=['cohort','file_name','parser_keywords','report_path','report_found_terms','report_excerpt'])
    writer.writeheader()
    for r in results:
        writer.writerow(r)
print('wrote', outf)
