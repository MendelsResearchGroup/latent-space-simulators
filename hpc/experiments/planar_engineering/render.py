#!/usr/bin/env python3
"""Render each fixed original/final pair; never select a physical winner."""
from pathlib import Path
import argparse
import json
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.collections import LineCollection
from matplotlib.patches import Rectangle
from PIL import Image
from lss.engineering.animation import _segments
from lss.engineering.evaluation import sha256


def main(output,cohort,index):
    output=Path(output);case=output/'cases'/f'{cohort}_{index}'
    declared=json.loads((case/'outcome.json').read_text())
    destination=Path('/rg/mendels_prj/alexander.z/latent-space-simulators/notebooks/engineering/Planar Adam engineering')
    destination.mkdir(parents=True,exist_ok=True)
    rows=pd.read_csv(case/'final_verification/results.csv')
    seed=0 if cohort=='same10_reid' else 786
    shown=rows[rows.thermal_seed.eq(seed)].set_index('kind').loc[['original','optimized']]
    record=dict(source='reid' if cohort=='same10_reid' else 'depablo_mixed_temp',index=index,temperature=declared['temperature'],thermal_seed=seed,status='unavailable')
    if not shown.status.eq('completed').all():
        record['reason']='A predeclared shown-seed simulation failed; no replacement seed selected.'
        (case/'gif.json').write_text(json.dumps(record,indent=2)+'\n');return
    tracks=[np.load(path) for path in shown.trajectory_path]
    graph=torch.load(shown.graph_path.iloc[0],weights_only=False,map_location='cpu')
    edges=np.unique(np.sort(graph.edge_index.numpy(),axis=0),axis=1)
    count=len(tracks[0]['positions']);assert len(tracks[1]['positions'])==count
    boxes=[t['boxes'][:,:2,:2] for t in tracks]
    lower=np.min([b[:,:,0].min(0) for b in boxes],axis=0);upper=np.max([b[:,:,1].max(0) for b in boxes],axis=0)
    padding=(upper-lower)*.035
    fig,axes=plt.subplots(1,2,figsize=(8,4.2),dpi=80,facecolor='#f7f4ee');canvas=FigureCanvasAgg(fig)
    fig.subplots_adjust(left=.035,right=.985,bottom=.09,top=.78,wspace=.08)
    title=f"Reid {index}" if cohort=='same10_reid' else f"dePablo {index} · {declared['temperature']:g} K · seed786"
    fig.suptitle(title+' · planar Adam',fontsize=13,color='#38332d',y=.98)
    contact_counts=shown.trajectory_contact_frames.astype(int).tolist()
    fig.text(.5,.89,f"Frames with spring contacts: original {contact_counts[0]}/{count}, optimized {contact_counts[1]}/{count}",ha='center',fontsize=9,color='#a63f3f' if any(contact_counts) else '#146b66')
    progress=fig.text(.5,.025,'',ha='center',fontsize=10,color='#6e655b');artists=[]
    for ax,bounds,kind,color,(_,row) in zip(axes,boxes,['Original','Optimized'],['#8a8175','#146b66'],shown.iterrows()):
        ax.set_facecolor('#f7f4ee');ax.set_aspect('equal');ax.set_xlim(lower[0]-padding[0],upper[0]+padding[0]);ax.set_ylim(lower[1]-padding[1],upper[1]+padding[1]);ax.set_xticks([]);ax.set_yticks([])
        for spine in ax.spines.values():spine.set_visible(False)
        ax.set_title(f"{kind} · p-ratio {row.physical_p_ratio:+.3f}",fontsize=11,color=color)
        box=bounds[0];rect=Rectangle(box[:,0],*(box[:,1]-box[:,0]),fill=False,edgecolor=color,lw=.7);ax.add_patch(rect)
        lines=LineCollection([],colors=color,linewidths=.45,alpha=.65);ax.add_collection(lines)
        points=ax.scatter([],[],s=5,color=color,linewidths=0);artists.append((rect,lines,points))
    images=[]
    for frame in range(count):
        for track,bounds,(rect,lines,points) in zip(tracks,boxes,artists):
            box=bounds[frame];wrapped,segments=_segments(track['positions'][frame],box,edges)
            rect.set_bounds(*box[:,0],*(box[:,1]-box[:,0]));lines.set_segments(segments);points.set_offsets(wrapped)
        progress.set_text(f'Trajectory frame {frame}/{count-1}')
        canvas.draw();images.append(Image.fromarray(np.asarray(canvas.buffer_rgba()).copy()).convert('RGB'))
    plt.close(fig)
    palette=images[0].convert('P',palette=Image.Palette.ADAPTIVE,colors=64)
    encoded=[im.quantize(palette=palette,dither=Image.Dither.NONE) for im in images]
    durations=(np.diff(np.rint(np.linspace(0,400,count+1))).astype(int)*10).tolist()
    path=destination/f"{record['source']}_{index}.gif"
    encoded[0].save(path,save_all=True,append_images=encoded[1:],duration=durations,loop=0,disposal=1,optimize=False)
    with Image.open(path) as image:
        elapsed=0
        for frame in range(image.n_frames):image.seek(frame);elapsed+=image.info['duration']
        assert image.n_frames==count and elapsed==4000
    record.update(status='completed',path=str(path),sha256=sha256(path),frames=count,duration_ms=4000,
                  trajectory_contact_frames_original=contact_counts[0],trajectory_contact_frames_optimized=contact_counts[1],
                  renderer_sha256=sha256(Path(__file__)),original_p_ratio=float(shown.physical_p_ratio.iloc[0]),optimized_p_ratio=float(shown.physical_p_ratio.iloc[1]))
    (case/'gif.json').write_text(json.dumps(record,indent=2)+'\n');print(path,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--cohort',required=True);p.add_argument('--index',type=int,required=True)
    a=p.parse_args();main(a.output,a.cohort,a.index)
