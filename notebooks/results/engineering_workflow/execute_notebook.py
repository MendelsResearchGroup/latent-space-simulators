from pathlib import Path
import json, os
import nbformat
from nbclient import NotebookClient
from jupyter_client.kernelspec import KernelSpecManager
root=Path('/rg/mendels_prj/alexander.z/latent-space-simulators')
path=root/'notebooks/engineering/02_engineer_one_network.ipynb'
ks=Path('/tmp/lss-final-kernels/lss-donor');ks.mkdir(parents=True,exist_ok=True)
(ks/'kernel.json').write_text(json.dumps({'argv':['/rg/mendels_prj/alexander.z/DL-course-project/.venv/bin/python','-m','ipykernel_launcher','-f','{connection_file}'],'display_name':'LSS donor Python','language':'python'}))
manager=KernelSpecManager(kernel_dirs=[str(ks.parent)])
nb=nbformat.read(path,as_version=4)
client=NotebookClient(nb,timeout=1800,kernel_name='lss-donor',kernel_manager_class=__import__('jupyter_client').KernelManager,resources={'metadata':{'path':str(path.parent)}})
client.km=client.create_kernel_manager();client.km.kernel_spec_manager=manager
try:
 client.execute()
finally:
 nbformat.write(nb,path)
print('COMPLETED',path,flush=True)
