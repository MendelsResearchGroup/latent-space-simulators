#!/usr/bin/env python3
"""Render frozen physical trajectories with explicit GIF timing and periodic bonds."""
import argparse,hashlib,json,sys,os,importlib.util
from pathlib import Path
DONOR=Path('/rg/mendels_prj/alexander.z/DL-course-project')
sys.path.insert(0,str(DONOR.parent/'MetaForge/src'))
import auxetic
import numpy as np,torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.patches import Rectangle
from PIL import Image

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def segments(pos,box,edge):
    length=box[:,1]-box[:,0]
    wrapped=box[:,0]+np.mod(pos-box[:,0],length)
    a=wrapped[edge[0]];delta=wrapped[edge[1]]-a;delta-=np.round(delta/length)*length
    parts=[]
    for dx in [-1,0,1]:
        for dy in [-1,0,1]:
            shift=np.array([dx,dy])*length;parts.append(np.stack([a+shift,a+delta+shift],axis=1))
    return wrapped,np.concatenate(parts)
def render(out,key):
    theme_path=Path('/rg/mendels_prj/alexander.z/latent-space-simulators/src/lss/plotting.py')
    spec=importlib.util.spec_from_file_location('editorial',theme_path);theme=importlib.util.module_from_spec(spec);spec.loader.exec_module(theme);theme.apply_editorial_style()
    case=json.loads((out/'inputs.json').read_text())['render'][key]
    dependencies=json.loads((out/'provenance.json').read_text())['dependencies']
    if case['source']=='reid':
        paths=[out/'assets'/f"reid_{case['index']}_{v}.npz" for v in ['original','optimized']]
    else:paths=[Path(case[v+'_trajectory']) for v in ['original','optimized']]
    for path in paths:
        if str(path) in dependencies and sha(path)!=dependencies[str(path)]:raise RuntimeError('trajectory hash mismatch')
    tracks=[np.load(p) for p in paths];n=len(tracks[0]['positions']);assert n==len(tracks[1]['positions'])==(121 if case['source']=='reid' else 200)
    graph_path=Path(case['original_graph']);assert sha(graph_path)==dependencies[str(graph_path)]
    graph=torch.load(graph_path,weights_only=False,map_location='cpu');edge=graph.edge_index.numpy();edge=np.unique(np.sort(edge,axis=0),axis=1)
    box_arrays=[v['boxes'][:,:2,:2] for v in tracks]
    low=np.min([b[:,:,0].min(0) for b in box_arrays],axis=0);high=np.max([b[:,:,1].max(0) for b in box_arrays],axis=0);pad=(high-low)*.04
    fig,axes=plt.subplots(1,2,figsize=(8,4.2),dpi=100,facecolor='#f7f4ee')
    fig.subplots_adjust(left=.035,right=.985,bottom=.085,top=.82,wspace=.08)
    title=f"Reid {case['index']}" if case['source']=='reid' else f"dePablo {case['index']} · {case['temperature']:g} K"
    fig.suptitle(title,fontsize=15,color='#38332d',y=.98)
    progress=fig.text(.5,.025,'Compression 0%',ha='center',fontsize=10,color='#6e655b')
    artists=[]
    for ax,track,bounds,label,color,p in zip(axes,tracks,box_arrays,['Original','Adam optimized'],['#8a8175','#146b66'],[case['original_p'],case['optimized_p']]):
        ax.set_facecolor('#f7f4ee');ax.set_aspect('equal');ax.set_xlim(low[0]-pad[0],high[0]+pad[0]);ax.set_ylim(low[1]-pad[1],high[1]+pad[1]);ax.set_xticks([]);ax.set_yticks([])
        for spine in ax.spines.values():spine.set_visible(False)
        ax.set_title(f'{label}\np-ratio {p:+.3f}',fontsize=12,color=color,pad=8)
        box=bounds[0];patch=Rectangle(box[:,0],*(box[:,1]-box[:,0]),fill=False,edgecolor=color,lw=.8);ax.add_patch(patch)
        ax.add_patch(Rectangle(box[:,0],*(box[:,1]-box[:,0]),fill=False,edgecolor='#b6ad9e',lw=.65,ls='--',zorder=1))
        lines=LineCollection([],colors=color,linewidths=.45,alpha=.60);ax.add_collection(lines);lines.set_clip_path(patch)
        dots=ax.scatter([],[],s=5,color=color,linewidths=0);artists.append((patch,lines,dots))
    images=[];palette=None
    for i in range(n):
        for track,bounds,(patch,lines,dots) in zip(tracks,box_arrays,artists):
            box=bounds[i];positions,segs=segments(track['positions'][i],box,edge)
            patch.set_bounds(*box[:,0],*(box[:,1]-box[:,0]));lines.set_segments(segs);dots.set_offsets(positions)
        progress.set_text(f'Compression {100*i/(n-1):.0f}%');fig.canvas.draw()
        im=Image.fromarray(np.asarray(fig.canvas.buffer_rgba()).copy()).convert('RGB')
        if palette is None:palette=im.convert('P',palette=Image.Palette.ADAPTIVE,colors=256)
        images.append(im.quantize(palette=palette))
    plt.close(fig)
    durations=(np.diff(np.rint(np.linspace(0,400,n+1))).astype(int)*10).tolist()
    path=out/'assets'/case['filename'];images[0].save(path,save_all=True,append_images=images[1:],duration=durations,loop=0,disposal=2,optimize=False)
    with Image.open(path) as gif:
        total=0;frames=gif.n_frames
        for i in range(frames):gif.seek(i);total+=gif.info.get('duration',0)
    assert total==4000
    item={k:case[k] for k in ['source','index','temperature','original_p','optimized_p','thermal_seed'] if k in case}
    item.update(path=str(path.resolve()),gif_sha256=sha(path),source_frames=n,gif_frames=frames,duration_ms=total,pbs_job_id=os.environ.get('PBS_JOBID'),render_sha256=sha(__file__),trajectory_sha256={str(p):sha(p) for p in paths},deformation_scale=1)
    (out/'assets'/f'{key}_gif.json').write_text(json.dumps(item,indent=2)+'\n')
def collect(out):
    cases=json.loads((out/'inputs.json').read_text())['render'];rows=[]
    for key in cases:
        row=json.loads((out/'assets'/f'{key}_gif.json').read_text());assert sha(row['path'])==row['gif_sha256'] and row['duration_ms']==4000;rows.append(row)
    assert len(rows)==7
    (out/'index.json').write_text(json.dumps({'gifs':rows,'total':7,'duration_seconds':4},indent=2)+'\n')
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--key');p.add_argument('--collect',action='store_true');a=p.parse_args();collect(a.output) if a.collect else render(a.output,a.key)
