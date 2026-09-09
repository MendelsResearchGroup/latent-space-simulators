"""Compact presentation assets without changing physical frames or playback time."""
from pathlib import Path
import hashlib,json
from PIL import Image
ROOT=Path('/rg/mendels_prj/alexander.z/latent-space-simulators')
out=ROOT/'notebooks/results/adam_showcase'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
 index=json.loads((out/'index.json').read_text());records=[]
 for r in index['gifs']:
  p=Path(r['path']);prior_hash=sha(p);images=[];durations=[]
  with Image.open(p) as im:
   for i in range(im.n_frames):
    im.seek(i);images.append(im.convert('RGB').resize((640,336),Image.Resampling.LANCZOS));durations.append(im.info['duration'])
  palette=images[0].quantize(colors=64)
  frames=[im.quantize(palette=palette,dither=Image.Dither.NONE) for im in images]
  tmp=p.with_suffix('.compact.gif');frames[0].save(tmp,save_all=True,append_images=frames[1:],duration=durations,loop=0,disposal=1,optimize=True)
  with Image.open(tmp) as im:
   total=0
   for i in range(im.n_frames):im.seek(i);total+=im.info.get('duration',0)
   assert total==4000
   r['gif_frames']=im.n_frames
  tmp.replace(p);r.update(gif_sha256=sha(p),precompression_gif_sha256=prior_hash,display_dimensions=[640,336],encoding='64-color palette, no dithering, differential frames; all physical frame times retained')
  records.append(r);key=str(r['index']) if r['source']=='reid' else f"thermal_{r['index']}"
  (out/'assets'/f'{key}_gif.json').write_text(json.dumps(r,indent=2)+'\n')
  print(p.name,round(p.stat().st_size/1e6,2),'MB',flush=True)
 index['gifs']=records;(out/'index.json').write_text(json.dumps(index,indent=2)+'\n')
if __name__=='__main__':main()
