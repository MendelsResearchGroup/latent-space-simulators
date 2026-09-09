"""Read-only presentation of the selected joint-Adam development results."""
from pathlib import Path
import base64, json, importlib.util, html
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import HTML, display
ROOT=Path(__file__).resolve().parents[2]
REID_CONFIG='lr0.08_b0.25_p50_mask_blocking_nodes'
IDS=[93,84,96,97,52,59,57,53,2,74]

def tables():
    r=pd.read_csv(ROOT/'notebooks/results/reid_joint_sweep/study_v1/results.csv')
    r=r[r.config_id.eq(REID_CONFIG)&r.kind.eq('ml')].set_index('index').reindex(IDS)
    assert len(r)==10 and r.status.eq('completed').all()
    reid=r[['original_p_ratio','physical_p_ratio']].rename(columns={'original_p_ratio':'original','physical_p_ratio':'optimized'})
    t=pd.read_csv(ROOT/'notebooks/results/thermal_adam30/study_v1/thermal_results.csv')
    t=t[t.kind.eq('original')|(t.kind.eq('ml')&t.arm.eq('A'))]
    assert len(t)==180 and t.status.eq('completed').all()
    a=t.groupby(['index','temperature','kind']).physical_p_ratio.agg(['mean','std','count']).unstack('kind')
    assert len(a)==30 and (a['count']==3).all().all()
    thermal=pd.DataFrame({'original':a['mean']['original'],'optimized':a['mean']['ml'],'original_sd':a['std']['original'],'optimized_sd':a['std']['ml']}).reset_index().set_index('index')
    return reid,thermal

def overview():
    reid,thermal=tables(); rows=[]
    for name,d in [('Reid',reid),('dePablo mixed-T',thermal)]:
        positive=d.original>0
        rows.append({'Dataset':name,'Networks':len(d),'Lower p-ratio':f'{int((d.optimized<d.original).sum())}/{len(d)}','Positive → negative':f'{int((positive&(d.optimized<0)).sum())}/{int(positive.sum())}','Mean original':d.original.mean(),'Mean optimized':d.optimized.mean()})
    return pd.DataFrame(rows).set_index('Dataset').style.format({'Mean original':'{:+.3f}','Mean optimized':'{:+.3f}'})

def plot_results():
    spec=importlib.util.spec_from_file_location('editorial',ROOT/'src/lss/plotting.py');style=importlib.util.module_from_spec(spec);spec.loader.exec_module(style);style.apply_editorial_style()
    reid,thermal=tables();fig,axes=plt.subplots(1,2,figsize=(12,4.2),layout='constrained')
    ax=axes[0];x=np.arange(10)
    ax.vlines(x,reid.original,reid.optimized,color='#c7beb0',lw=2)
    ax.scatter(x,reid.original,color='#8a8175',s=35,label='Original',zorder=3)
    ax.scatter(x,reid.optimized,color='#146b66',s=38,label='Adam optimized',zorder=3)
    ax.axhline(0,color='#38332d',lw=.8);ax.set(xticks=x,xticklabels=IDS,xlabel='Reid network',ylabel='Physical p-ratio',title='Reid · all 10 networks');ax.legend(frameon=False,fontsize=9)
    ax=axes[1];lo=min(thermal.original.min(),thermal.optimized.min())-.08;hi=max(thermal.original.max(),thermal.optimized.max())+.08
    colors=['#146b66','#428d9e','#7b77a7','#b77955','#ac5265','#b19742']
    for color,(T,g) in zip(colors,thermal.groupby('temperature')):
        ax.errorbar(g.original,g.optimized,xerr=g.original_sd,yerr=g.optimized_sd,fmt='o',color=color,ms=4,capsize=2,alpha=.85,label=f'{T:g} K')
    ax.plot([lo,hi],[lo,hi],color='#8a8175',lw=.9,ls='--');ax.axhline(0,color='#38332d',lw=.6);ax.axvline(0,color='#38332d',lw=.6)
    ax.set(xlim=(lo,hi),ylim=(lo,hi),xlabel='Original p-ratio',ylabel='Optimized p-ratio',title='Mixed-T · all 30 networks')
    ax.text(.04,.94,'Below diagonal = improved',transform=ax.transAxes,va='top',fontsize=9)
    ax.legend(frameon=False,fontsize=8,ncol=2,loc='lower right');plt.show()

def show_gifs(source):
    index=ROOT/'notebooks/results/adam_showcase/index.json'
    entries=[r for r in json.loads(index.read_text())['gifs'] if r['source']==source]
    cards=[]
    for r in entries:
        p=Path(r['path']);p=p if p.is_absolute() else ROOT/p
        data=base64.b64encode(p.read_bytes()).decode('ascii')
        label=f"Reid {r['index']}" if source=='reid' else f"dePablo {r['index']} · {r['temperature']:g} K"
        caption=f"{label}: p = {r['original_p']:+.3f} → {r['optimized_p']:+.3f}"
        cards.append(f'<div><p style="margin:6px 0;font-size:14px">{html.escape(caption)}</p><img alt="{html.escape(caption)}; original left and Adam optimized right" src="data:image/gif;base64,{data}" style="width:100%;height:auto"/></div>')
    display(HTML('<div style="display:grid;grid-template-columns:repeat(auto-fit,minmax(360px,1fr));gap:12px;max-width:1000px">'+''.join(cards)+'</div>'))
