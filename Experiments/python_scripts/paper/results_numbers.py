"""Every number the results section of the paper quotes, recomputed from a results CSV.

    python3 results_numbers.py Experiments/results/results.csv > results_numbers.md
"""
import sys, pandas as pd, numpy as np
d = pd.read_csv(sys.argv[1], low_memory=False)
d['sys'] = d.experiment_name.str.split('_').str[0]
d['algo'] = d.experiment_name.str.split('_').str[1]
d = d[d.sys != 'Pytrich-NOVELTY']
BASE = {'Panda-rc2ff','Panda-rc2lmc','Panda-rc2add','Panda-lmcountlmtypebid','Pytrich-TDG','Pytrich-LMCOUNT'}
llm = d[~d.sys.isin(BASE)]
DOM = ['Barman-BDI','Blocksworld-GTOHP','Depots','Robot','Rover-GTOHP','Towers','Lamps']
key = ['domain_name','problem_name']
def vb(df, col='expanded_nodes'):  # virtual best per instance: min nodes
    return df.groupby(key)[col].min()
def bestlen(df):  # plan length of the min-node run
    i = df.groupby(key)['expanded_nodes'].idxmin(); return df.loc[i].set_index(key)['solution_size']
out = []
P = lambda *a: out.append(' '.join(str(x) for x in a))

P("# Results numbers on the D9 record\n")
P("## 1. Coverage (virtual best over algorithms)")
cov = pd.DataFrame({s: d[d.sys==s].groupby('domain_name').problem_name.nunique() for s in sorted(d.sys.unique())}).fillna(0).astype(int).T
cov = cov[DOM]; cov['total'] = cov.sum(1); cov['six'] = cov[DOM[:6]].sum(1)
llmvb = llm.groupby('domain_name').problem_name.nunique().reindex(DOM).fillna(0).astype(int); cov.loc['LLM-VB'] = list(llmvb)+[llmvb.sum(), llmvb[:6].sum()]
P(cov.to_markdown()); P()

def paired(a, b, name):
    sh = pd.concat([a.rename('a'), b.rename('b')], axis=1, join='inner')
    sh['ratio'] = sh.a/sh.b; sh['dom'] = sh.index.get_level_values(0)
    rows = []
    for dm, g in list(sh.groupby('dom')) + [('ALL', sh), ('SIX', sh[sh.dom!='Lamps'])]:
        w = g[g.a<g.b]; rows.append(dict(domain=dm, shared=len(g), wins=len(w), ties=int((g.a==g.b).sum()), losses=int((g.a>g.b).sum()),
            median_ratio=round(g.ratio.median(),2), median_pct_fewer=round(100*(1-g.ratio.median()),0),
            mean_pct_fewer_when_win=round(100*(1-w.ratio).mean(),0) if len(w) else None,
            all_wins=len(w)==len(g)))
    P(f"### {name}"); P(pd.DataFrame(rows).to_markdown(index=False)); P()

P("## 2. Expanded nodes, paired per instance, LLM-VB vs baseline (virtual best over algorithms on both sides)")
for s in ['Pytrich-TDG','Pytrich-LMCOUNT','Panda-rc2ff']:
    paired(vb(llm), vb(d[d.sys==s]), f"LLM-VB vs {s}")

P("## 3. Cactus facts: instances solved within N expanded nodes (virtual best per instance)")
rows=[]
for s, df in [('LLM-VB', llm), ('Panda-rc2ff', d[d.sys=='Panda-rc2ff']), ('Pytrich-TDG', d[d.sys=='Pytrich-TDG'])]:
    v = vb(df)
    for dm in DOM:
        x = v[v.index.get_level_values(0)==dm]
        rows.append(dict(system=s, domain=dm, solved=len(x), within_1000=int((x<=1000).sum()), within_10000=int((x<=10000).sum()), within_100000=int((x<=1e5).sum())))
P(pd.DataFrame(rows).pivot(index='domain', columns='system', values=['within_1000','within_10000','solved']).reindex(DOM).to_markdown()); P()

P("## 4. Plan length, paired per instance (length of the min-node run on each side)")
def plen(a, b, name):
    sh = pd.concat([a.rename('a'), b.rename('b')], axis=1, join='inner'); sh['dom']=sh.index.get_level_values(0)
    rows=[]
    for dm, g in list(sh.groupby('dom'))+[('ALL',sh),('SIX',sh[sh.dom!='Lamps'])]:
        w=g[g.a<g.b]; rows.append(dict(domain=dm, shared=len(g), shorter=len(w), equal=int((g.a==g.b).sum()), longer=int((g.a>g.b).sum()),
            median_ratio=round((g.a/g.b).median(),2), mean_pct_shorter_when_shorter=round(100*(1-w.a/w.b).mean(),0) if len(w) else None))
    P(f"### LLM-VB vs {name}"); P(pd.DataFrame(rows).to_markdown(index=False)); P()
for s in ['Pytrich-TDG','Pytrich-LMCOUNT','Panda-rc2ff']: plen(bestlen(llm), bestlen(d[d.sys==s]), s)

P("## 5. Models")
P("### 5a. Coverage per model (virtual best over its candidates and algorithms)")
mc = pd.DataFrame({m: llm[llm.sys==m].groupby('domain_name').problem_name.nunique() for m in sorted(llm.sys.unique())}).fillna(0).astype(int).T[DOM]; mc['total']=mc.sum(1)
P(mc.to_markdown()); P()
P("### 5b. Median expanded nodes per model, plot definition (median over every solved run of every candidate)")
ms = llm.groupby('sys').expanded_nodes.agg(['median','count']).sort_values('median'); P(ms.to_markdown()); P()
P("### 5c. Median expanded nodes per model, virtual best per instance (one value per solved instance)")
mv = pd.DataFrame({m: vb(llm[llm.sys==m]).agg(['median','count']) for m in sorted(llm.sys.unique())}).T.sort_values('median'); P(mv.to_markdown()); P()
P("### 5d. Robot: instances unsolved by any LLM heuristic")
rob = set(d[d.domain_name=='Robot'].problem_name); P(sorted(rob - set(llm[llm.domain_name=='Robot'].problem_name))); P()
P("### 5e. Viable (model, domain, heuristic) triples in the record (solved at least one instance)")
tri = llm.groupby(['sys','domain_name']).heuristic_name.nunique().unstack().fillna(0).astype(int)[DOM]; tri['total']=tri.sum(1); P(tri.to_markdown()); P("Total viable triples:", int(llm.groupby(['sys','domain_name','heuristic_name']).ngroups)); P()

P("## 6. Algorithms (plot definition: median over every solved LLM run)")
P(llm.groupby('algo').expanded_nodes.agg(['median','count']).to_markdown()); P()
pa = llm.groupby(['sys','algo']).expanded_nodes.median().unstack(); pa['A*/GBFS']=(pa['StandardA']/pa['GBFS']).round(1); P(pa.round(0).to_markdown()); P()
P("### 6b. Per model, virtual best per instance under each algorithm, paired wins GBFS vs A* and GBFS vs WA*")
rows=[]
for m in sorted(llm.sys.unique()):
    g = {a: vb(llm[(llm.sys==m)&(llm.algo==a)]) for a in ['GBFS','StandardA','WA']}
    for a in ['StandardA','WA']:
        sh = pd.concat([g['GBFS'].rename('g'), g[a].rename('o')],axis=1,join='inner')
        rows.append(dict(model=m, vs=a, shared=len(sh), gbfs_wins=int((sh.g<sh.o).sum()), ties=int((sh.g==sh.o).sum()), median_ratio=round((sh.g/sh.o).median(),2)))
P(pd.DataFrame(rows).to_markdown(index=False))
if len(sys.argv) > 2:
    open(sys.argv[2], 'w').write('\n'.join(out))
print('\n'.join(out))
