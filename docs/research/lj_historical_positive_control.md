# Reconciling historical working LJ reconstruction

Notebook08 saved cell20 reports LJ AE endpoint response R2 .6296 on30 evaluation
networks at frame100, versus cell10 .4681 on110 networks. These are different
populations; neither number is a current replay. Notebook09 cell4 independently
records .468 on110 at frame100 with ordinary AE reconstruction selection.
Thus expert response selection is not the only historical path to positive LJ.

Current notebook08 configuration: seed456456; retained Reid/lowT/LJ train30/30/60,
validation30 each; latent6,width192,decoder tokens46,150 training frames,
max50/patience8,batch32,lr1e-4,wd1e-5,node-pooled mean objective,5 edge channels
with LJ graph-distance3 augmentation,physical reference context,normalized_delta.
AE checkpoint uses retained-source response at50/150 (historical only, incompatible
with current selection requirement). Current notebook09: seed123,train30/30/60,
latent6,width128,tokens32,100frames,max50/patience8,mean objective,5 augmented
edge channels,ordinary reconstruction selection. Notebook text/config and saved
outputs are not guaranteed synchronized. No corresponding08/09 artifact directory
or checkpoint was found in the current notebooks/results top-level inventory.

Recent capacity test used train20/20/20,latent6/8,width96,tokens32,101frames,
source-balanced objective,worst-source reconstruction selection,4 edge channels,
fixed current split,40epochs/patience8. It tests dimension under that recipe,
not reproducibility of08/09. More training examples, more updates, width, edges,
selection and evaluation populations are confounded. No single cause is established.

Priority: locate the exact working example user means. If checkpoint exists,
re-evaluate its coordinate reconstruction and response on unchanged evaluation
identities without refitting, recording provenance. If absent, build a clearly
labeled reconstruction of the historical recipe using current reserved-split
rules; do not touch final test or call it exact replay. Prefer09 as a compliant
ordinary-reconstruction positive control, with08 model/data budget a second
hypothesis but reconstruction-only checkpoint selection. Bridge differences by
matched data/update-budget, width/decoder, weighting and edge contrasts, not a
new undifferentiated architecture search. Never infer global AE impossibility
from the failed recent recipe or conflate coordinate reconstruction with p-ratio.
