#!/usr/bin/env python3
"""Generate a minimal editable architecture diagram and matching SVG."""
from pathlib import Path
import html
import xml.etree.ElementTree as ET

OUT = Path(__file__).resolve().parents[2] / 'docs/figures'
W, H = 2030, 730
INK = '#334155'
BLUE, BLUE_FILL = '#48678C', '#E5EEFF'
PURPLE, PURPLE_FILL = '#78639C', '#F0E9FF'
ORANGE, ORANGE_FILL = '#94632B', '#FFF2D9'
TEAL, TEAL_FILL = '#3D796D', '#E3F5ED'
# id, x, y, width, height, label, shape, stroke, fill
nodes = [
    ('embedding', 35, 225, 175, 90, 'Node embeddings\nhᵢ⁽⁰⁾', 'round', BLUE, BLUE_FILL),
    ('messages', 260, 225, 190, 90, 'Message passing\nhᵢ⁽⁰⁾ → hᵢ⁽²⁾', 'round', BLUE, BLUE_FILL),
    ('encoder', 510, 195, 180, 150, 'Encoder', 'compress', BLUE, BLUE_FILL),
    ('latent', 745, 230, 85, 80, 'zₜ', 'ellipse', PURPLE, PURPLE_FILL),
    ('propagator', 890, 490, 220, 120, 'Propagator\nPredicts Δẑₜ', 'round', ORANGE, ORANGE_FILL),
    ('future', 1175, 510, 85, 80, 'ẑₜ₊₁', 'ellipse', PURPLE, PURPLE_FILL),
    ('decoder', 1320, 195, 190, 150, 'Decoder', 'expand', BLUE, BLUE_FILL),
    ('output', 1570, 225, 190, 90, 'Displacement\nûₜ or ûₜ₊₁', 'data', INK, '#FFFFFF'),
    ('context', 880, 65, 245, 65, 'Static node embeddings\nH_ref', 'round', TEAL, TEAL_FILL),
]
# id, source, target, waypoints (including endpoints), color, dashed
edges = [
    ('e0','embedding','messages',[(210,270),(260,270)],INK,False),
    ('e1','messages','encoder',[(450,270),(510,270)],INK,False),
    ('e2','encoder','latent',[(690,270),(745,270)],INK,False),
    ('reconstruct','latent','decoder',[(830,270),(1320,270)],INK,False),
    ('e3','latent','propagator',[(787,310),(787,550),(890,550)],ORANGE,False),
    ('e4','propagator','future',[(1110,550),(1175,550)],ORANGE,False),
    ('e5','future','decoder',[(1260,550),(1415,550),(1415,345)],ORANGE,False),
    ('e6','decoder','output',[(1510,270),(1570,270)],INK,False),
    ('c1','context','embedding',[(880,97),(122,97),(122,225)],TEAL,True),
    ('c2','context','propagator',[(1000,130),(1000,490)],TEAL,True),
    ('c3','context','decoder',[(1125,97),(1415,97),(1415,195)],TEAL,True),
    ('repeat','future','propagator',[(1217,590),(1217,655),(1000,655),(1000,610)],ORANGE,False),
]
labels = [
    ('reference_note',1002,40,'From reference structure x_ref',17,TEAL),
    ('position_note',1665,355,'x̂ = x_ref + û',21,BLUE),
    ('ae_stage',410,155,'1 · Train the autoencoder',21,BLUE),
    ('dynamics_stage',1120,455,'2 · Train propagator · AE frozen',20,ORANGE),
    ('reconstruction_label',1135,245,'Reconstruction',17,BLUE),
    ('context_pool_note',1105,410,'Pooled context c',16,TEAL),
    ('decoder_context_note',1420,75,'H_ref',18,TEAL),
    ('features_note',122,352,'Displacement features +',16,BLUE),
    ('edges_note',122,375,'adjacent-edge embeddings',16,BLUE),
    ('messages_note',355,352,'2 local update rounds',16,BLUE),
    ('encoder_note',600,375,'Global attention',17,BLUE),
    ('update_note',1200,625,'ẑₜ₊₁ = zₜ + Δẑₜ',19,ORANGE),
    ('repeat_note',1108,685,'Repeat for rollout',17,ORANGE),
]
# Make the observed displacement input explicit without expanding block text.
nodes = [(id, x + 220, y, w, h, label, shape, stroke, fill)
         for id, x, y, w, h, label, shape, stroke, fill in nodes]
edges = [(id, source, target, [(x + 220, y) for x, y in points], color, dashed)
         for id, source, target, points, color, dashed in edges]
labels = [(id, x + 220, y, label, size, color) for id, x, y, label, size, color in labels]
nodes.insert(0, ('input', 30, 225, 180, 90, 'Displacement\nuₜ = xₜ − x_ref', 'data', BLUE, BLUE_FILL))
edges.insert(0, ('input_embedding', 'input', 'embedding', [(210, 270), (255, 270)], INK, False))
# Message passing and global pooling are stages of the complete encoder.
font_sizes = {n[0]: 26 for n in nodes}
font_sizes.update(input=20, embedding=20, messages=20, latent=23, future=23,
                  propagator=23, output=20, context=20)
by_id = {n[0]: n for n in nodes}
mx = ET.Element('mxfile',host='app.diagrams.net')
diagram = ET.SubElement(mx,'diagram',id='shared-main-flow',name='Main architecture')
model = ET.SubElement(diagram,'mxGraphModel',grid='1',gridSize='10',page='1',pageWidth=str(W),pageHeight=str(H),math='0',shadow='0',background='#FFFFFF')
root = ET.SubElement(model,'root')
ET.SubElement(root,'mxCell',id='0')
ET.SubElement(root,'mxCell',id='1',parent='0')
styles = {'data':'shape=parallelogram;','compress':'shape=trapezoid;direction=west;','expand':'shape=trapezoid;direction=east;','ellipse':'ellipse;','round':'rounded=1;arcSize=15;'}
for id,x,y,w,h,label,shape,stroke,fill in nodes:
    style = styles[shape] + f'whiteSpace=wrap;html=1;fillColor={fill};strokeColor={stroke};strokeWidth=2;fontFamily=Helvetica;fontSize={font_sizes[id]};fontColor={INK};align=center;verticalAlign=middle;'
    cell = ET.SubElement(root,'mxCell',id=id,value=html.escape(label).replace('\n','<br>'),style=style,vertex='1',parent='1')
    ET.SubElement(cell,'mxGeometry',x=str(x),y=str(y),width=str(w),height=str(h),**{'as':'geometry'})
for id,source,target,points,color,dashed in edges:
    style=f'edgeStyle=orthogonalEdgeStyle;rounded=0;jumpStyle=arc;strokeColor={color};strokeWidth=2;endArrow=block;endFill=1;dashed={int(dashed)};'
    for end,key,p in [('exit',source,points[0]),('entry',target,points[-1])]:
        _,x,y,w,h,*_ = by_id[key]
        style += f'{end}X={(p[0]-x)/w};{end}Y={(p[1]-y)/h};{end}Perimeter=0;'
    cell=ET.SubElement(root,'mxCell',id=id,style=style,edge='1',parent='1',source=source,target=target)
    geom=ET.SubElement(cell,'mxGeometry',relative='1',**{'as':'geometry'})
    if len(points)>2:
        arr=ET.SubElement(geom,'Array',**{'as':'points'})
        for x,y in points[1:-1]: ET.SubElement(arr,'mxPoint',x=str(x),y=str(y))
for id,x,y,label,size,color in labels:
    cell=ET.SubElement(root,'mxCell',id=id,value=label,style=f'text;html=1;strokeColor=none;fillColor=none;fontSize={size};fontFamily=Helvetica;fontColor={color};align=center;',vertex='1',parent='1')
    ET.SubElement(cell,'mxGeometry',x=str(x-160),y=str(y-14),width='320',height='28',**{'as':'geometry'})
OUT.mkdir(parents=True,exist_ok=True)
ET.indent(mx)
(OUT/'shared_ae_propagator.drawio').write_bytes(ET.tostring(mx,encoding='utf-8',xml_declaration=True))
svg=[f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}">', '<title>Encoder, latent propagator, and decoder</title>', '<rect width="100%" height="100%" fill="white"/>','<defs>']
for color in (INK,TEAL,ORANGE):
    svg.append(f'<marker id="a{color[1:]}" markerWidth="8" markerHeight="8" refX="7" refY="4" orient="auto"><path d="M0,0 L8,4 L0,8 Z" fill="{color}"/></marker>')
svg.append('</defs>')
for _,_,_,points,color,dashed in edges:
    coords=' '.join(f'{x},{y}' for x,y in points)
    dash=' stroke-dasharray="7 6"' if dashed else ''
    if dashed:
        svg.append(f'<polyline points="{coords}" fill="none" stroke="white" stroke-width="7"/>')
    svg.append(f'<polyline points="{coords}" fill="none" stroke="{color}" stroke-width="2"{dash} marker-end="url(#a{color[1:]})"/>')
def text(label,x,y,size,color):
    lines=label.split('\n')
    for i,line in enumerate(lines):
        yy=y+(i-(len(lines)-1)/2)*size*1.25
        svg.append(f'<text x="{x}" y="{yy}" text-anchor="middle" dominant-baseline="middle" font-family="Arial, Helvetica, sans-serif" font-size="{size}" fill="{color}">{html.escape(line)}</text>')
for id,x,y,w,h,label,shape,stroke,fill in nodes:
    common=f'fill="{fill}" stroke="{stroke}" stroke-width="2"'
    if shape=='ellipse':
        svg.append(f'<ellipse cx="{x+w/2}" cy="{y+h/2}" rx="{w/2}" ry="{h/2}" {common}/>')
    elif shape=='round':
        svg.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="12" {common}/>')
    else:
        if shape=='compress': points=[(x,y),(x+w,y+30),(x+w,y+h-30),(x,y+h)]
        elif shape=='expand': points=[(x,y+30),(x+w,y),(x+w,y+h),(x,y+h-30)]
        else: points=[(x+12,y),(x+w,y),(x+w-12,y+h),(x,y+h)]
        coords=' '.join(f'{a},{b}' for a,b in points)
        svg.append(f'<polygon points="{coords}" {common}/>')
    text(label,x+w/2,y+h/2,font_sizes[id],INK)
for _,x,y,label,size,color in labels: text(label,x,y,size,color)
svg.append('</svg>')
(OUT/'shared_ae_propagator.svg').write_text('\n'.join(svg))
for name in ('shared_ae_propagator.drawio','shared_ae_propagator.svg'):
    ET.parse(OUT/name)
    print(OUT/name)
