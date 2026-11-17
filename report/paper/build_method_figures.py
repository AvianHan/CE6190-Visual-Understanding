"""Publication schematics with exact archived ICS inputs and output masks.

Feature grids and module icons are schematic, not measured activations. Images,
masks and the example concept are exact local evidence; see assets/provenance.
No model execution, external download, or query-GT use is involved.
"""
from pathlib import Path
import hashlib
import json

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle
from matplotlib.path import Path as MplPath
import numpy as np
from PIL import Image

HERE = Path(__file__).resolve().parent
OUT = HERE / 'figures'
ASSETS = OUT / 'assets'
plt.rcParams.update({'font.family':'STIXGeneral','font.size':8.1,
    'mathtext.fontset':'stix','pdf.fonttype':42,'ps.fonttype':42,'svg.fonttype':'none'})
PALETTE = {'blue':'#466F87','cyan':'#78B6C8','gold':'#F2D98E',
           'coral':'#E48578','olive':'#98AF1E'}
INK, LINE = '#22282C', '#727A7D'
LAYOUT_AUDIT = {}

def tint(c, a):
    return tuple((1-a)+a*int(c[i:i+2],16)/255 for i in (1,3,5))

def canvas():
    f=plt.figure(figsize=(6.875,3.0))
    ax=f.add_axes([0,0,1,1]);ax.set(xlim=(0,6.875),ylim=(0,3));ax.axis('off')
    return f,ax

def text(ax,x,y,s,size=8.1,weight='normal',color=INK,ha='center',va='center',**kw):
    return ax.text(x,y,s,fontsize=size,fontweight=weight,color=color,ha=ha,va=va,zorder=8,**kw)

def panel(ax,x,y,w,h,label,color='blue'):
    ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=0,rounding_size=.085',
                  lw=0,fc=tint(PALETTE[color],.135 if color=='blue' else .11),zorder=0))
    text(ax,x+.13,y+h-.18,label,size=9.4,weight='bold',color=PALETTE['blue'],ha='left')

def module(ax,x,y,w,h,lines,accent=None,size=8.2):
    ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=0,rounding_size=.035',
                 lw=.65,ec=tint(PALETTE[accent],.65) if accent else '#9DA7AB',
                 fc='white',zorder=4))
    if isinstance(lines,str):lines=[lines]
    step=.15 if len(lines)>1 else 0
    for i,s in enumerate(lines):
        text(ax,x+w/2,y+h/2+step*((len(lines)-1)/2-i),s,size=size,
             weight='bold' if i==0 and len(lines)>1 else 'normal')

def arrow(ax,pts,color=LINE,dashed=False,head=True):
    p=MplPath(pts,[MplPath.MOVETO]+[MplPath.LINETO]*(len(pts)-1))
    ax.add_patch(FancyArrowPatch(path=p,arrowstyle='-|>' if head else '-',
          mutation_scale=6.5,lw=.75,color=color,zorder=7,
          linestyle=(0,(2.5,2)) if dashed else '-',joinstyle='round',capstyle='round'))

def raster(ax,filename,x,y,w,h,mask=False,overlay=False):
    a=np.array(Image.open(ASSETS/filename).convert('RGB'))
    if mask:
        a=np.where(a[:,:,:1]>0,np.array([70,111,135]),np.array([255,255,255])).astype('uint8')
    if overlay:
        m=np.array(Image.open(ASSETS/'reference_mask.png'))>0
        c=np.array([120,182,200],float)
        a=a.astype(float);a[m]=.38*a[m]+.62*c;a=a.astype('uint8')
    ih,iw=a.shape[:2];scale=min(w/iw,h/ih);ww,hh=iw*scale,ih*scale
    xx,yy=x+(w-ww)/2,y+(h-hh)/2
    ax.imshow(a,extent=(xx,xx+ww,yy,yy+hh),aspect='auto',zorder=5,interpolation='nearest' if mask else 'antialiased')
    ax.add_patch(Rectangle((xx,yy),ww,hh,fc='none',ec='#9DA7AB',lw=.45,zorder=6))
    return xx,yy,ww,hh

def grid(ax,x,y,w=.4,h=.28,mode='features'):
    colors=([PALETTE['blue'],PALETTE['cyan'],tint(PALETTE['cyan'],.32),tint(PALETTE['blue'],.3)]
            if mode=='features' else [PALETTE['blue'],PALETTE['coral'],PALETTE['gold'],PALETTE['olive']])
    a=([[0,0,1,2],[0,1,1,2],[3,3,1,2]] if mode=='clusters'
       else [[0,2,1,3],[1,3,0,2],[2,0,3,1]])
    for r in range(3):
        for c in range(4):
            ax.add_patch(Rectangle((x+c*w/4,y+(2-r)*h/3),w/4-.008,h/3-.008,
                         fc=colors[a[r][c]],ec='none',zorder=6))

def save(fig,stem):
    fig.canvas.draw();renderer=fig.canvas.get_renderer();page=fig.bbox
    texts=[t for ax in fig.axes for t in ax.texts]
    clipped=[t.get_text() for t in texts if not page.contains(*t.get_window_extent(renderer).get_points()[0])
             or not page.contains(*t.get_window_extent(renderer).get_points()[1])]
    if clipped:raise RuntimeError(f'{stem}: text outside canvas {clipped}')
    for ext in ('pdf','svg','png'):
        kw={'dpi':260} if ext=='png' else {}
        fig.savefig(OUT/f'{stem}.{ext}',**kw)
    LAYOUT_AUDIT[stem]={'width_inches':6.875,'height_inches':3.0,
        'minimum_font_pt':min(t.get_fontsize() for t in texts),
        'maximum_font_pt':max(t.get_fontsize() for t in texts),
        'text_outside_canvas':clipped,'text_count':len(texts),
        'raster_inputs':'Exact archived case06 images and predictions',
        'vector_elements':'All text, arrows, panels, feature/cluster icons, and module outlines',
        'schematic_not_measured':'Feature grids and query-box icon'}
    plt.close(fig)

def insid3():
    fig,ax=canvas()
    panel(ax,.04,.04,2.52,2.92,'(a) Visual features')
    panel(ax,2.78,1.56,4.05,1.40,'(b) Reference correspondence')
    panel(ax,2.78,.04,4.05,1.32,'(c) Seed and region aggregation')
    for x,name,label,mask in [(.19,'reference.jpg',r'Reference $I_r$',False),
                              (.97,'query.jpg',r'Query $I_q$',False),
                              (1.75,'reference_mask.png',r'Mask $M_r$',True)]:
        text(ax,x+.34,2.57,label,size=8)
        raster(ax,name,x,1.94,.68,.51,mask=mask)
    module(ax,.22,1.21,2.18,.40,'Frozen DINOv3',accent='blue',size=8.8)
    arrow(ax,[(.53,1.94),(.53,1.64)])
    arrow(ax,[(1.31,1.94),(1.31,1.64)])
    arrow(ax,[(1.31,1.21),(1.31,.88)])
    text(ax,.79,1.04,r'Original $F_q$',size=8,color=PALETTE['blue'])
    module(ax,.22,.20,2.18,.66,[],accent='blue')
    text(ax,1.31,.69,'Query clustering',size=8.5,weight='bold')
    grid(ax,.54,.32,mode='features');arrow(ax,[(1.05,.46),(1.52,.46)])
    grid(ax,1.65,.32,mode='clusters')
    text(ax,1.29,.32,r'$\{G_k\}$',size=8)
    # Only image features go to debiasing; the reference mask bypasses DINOv3.
    arrow(ax,[(2.40,1.41),(2.60,1.41),(2.60,2.32),(2.94,2.32)])
    module(ax,2.94,2.10,1.48,.43,['Positional debiasing',r'$F\!\left(I-BB^{\mathsf{T}}\right)$'],accent='cyan',size=8.3)
    arrow(ax,[(4.44,2.32),(4.69,2.32)])
    module(ax,4.70,2.10,1.91,.43,['Reference matching','Prototype + two-way matches'],accent='cyan',size=8.1)
    arrow(ax,[(2.09,1.94),(2.09,1.80),(5.35,1.80),(5.35,2.07)],color=PALETTE['blue'])
    text(ax,3.88,1.80,r'Reference mask $M_r$',size=8,color=PALETTE['blue'],
         bbox=dict(facecolor=tint(PALETTE['blue'],.135),edgecolor='none',pad=1))
    # Reference evidence guides seed selection and cluster scoring.
    arrow(ax,[(6.04,2.08),(6.04,1.48),(2.69,1.48),(2.69,.80),(2.94,.80)])
    arrow(ax,[(5.56,1.48),(5.56,.85),(5.46,.85)])
    text(ax,5.20,1.48,r'$C,\ p_r$',size=8,
         bbox=dict(facecolor='white',edgecolor='none',pad=.7))
    arrow(ax,[(2.40,.53),(2.83,.53),(2.83,.61),(2.94,.61)])
    module(ax,2.96,.48,1.12,.43,['Select seed $G_*$', 'Match-guided'],accent='gold',size=8.1)
    arrow(ax,[(4.10,.70),(4.28,.70)])
    module(ax,4.30,.48,1.14,.43,['Merge clusters',r'$s_k=a_k b_k w_k$'],accent='gold',size=8.1)
    text(ax,4.87,.27,r'$s_k>\theta$; upsample',size=8)
    arrow(ax,[(5.46,.70),(5.92,.70)])
    raster(ax,'insid3_prediction.png',5.97,.21,.66,.77,mask=True)
    text(ax,6.30,1.10,r'Prediction $\widehat M_q$',size=8.1)
    save(fig,'insid3_pipeline')

def cgics():
    fig,ax=canvas()
    panel(ax,.04,.04,1.25,2.92,'(a) Inputs')
    panel(ax,1.50,1.58,3.56,1.38,'(b) Concept guidance',color='gold')
    panel(ax,1.50,.04,3.56,1.34,'(c) Visual transfer',color='cyan')
    panel(ax,5.28,.04,1.55,2.92,'(d) Prediction')
    text(ax,.66,2.62,'Reference + mask',size=7.8)
    raster(ax,'reference.jpg',.18,1.82,.96,.72,overlay=True)
    text(ax,.66,1.53,r'Query $I_q$',size=8.1)
    raster(ax,'query.jpg',.27,.31,.78,1.13)
    # Reference target drives concept generation; both images drive the visual route.
    arrow(ax,[(1.14,2.30),(1.67,2.30)])
    arrow(ax,[(1.39,2.30),(1.39,.74),(1.67,.74)])
    arrow(ax,[(1.05,.90),(1.39,.90)],head=False)
    module(ax,1.68,2.07,.80,.48,['MLLM','Concepts'],accent='coral',size=8.3)
    arrow(ax,[(2.50,2.31),(2.70,2.31)])
    # Three actual candidates displayed; the full five-candidate list remains in provenance.
    for y,lab,chosen in [(2.47,'stop sign',False),(2.25,'red octagon',True),(2.03,'traffic sign',False)]:
        ax.add_patch(FancyBboxPatch((2.71,y-.095),1.11,.19,
              boxstyle='round,pad=0,rounding_size=.025',lw=.45,
              ec=PALETTE['coral'] if chosen else '#BBC0C2',
              fc=tint(PALETTE['coral'],.27) if chosen else 'white',zorder=4))
        text(ax,3.265,y,lab,size=7.9,weight='bold' if chosen else 'normal')
    arrow(ax,[(3.84,2.31),(4.05,2.31)])
    module(ax,4.06,2.02,.82,.58,['SAM 3','Score + search',r'$(I_r,M_r,I_q)$'],accent='coral',size=7.9)
    arrow(ax,[(4.47,2.00),(4.47,1.88),(2.08,1.88),(2.08,2.05)],dashed=True,color=PALETTE['coral'])
    text(ax,3.32,1.71,r'$s(c)=s_r(c)\,s_q(c)$',size=8.2)
    # Reference-query stitching; image pixels are exact archived inputs.
    raster(ax,'reference.jpg',1.68,.45,.70,.53,overlay=True)
    raster(ax,'query.jpg',2.39,.45,.36,.53)
    text(ax,2.22,.26,r'Stitch $[I_r\mid I_q]$',size=8.1)
    arrow(ax,[(2.80,.72),(3.03,.72)])
    module(ax,3.05,.49,.80,.45,['SAM 3','Visual prompts'],accent='cyan',size=8)
    arrow(ax,[(3.87,.72),(4.07,.72)])
    # Symbolic prompt boxes, deliberately not inferred from the final prediction mask.
    ax.add_patch(Rectangle((4.12,.50),.69,.45,fc='white',ec='#9DA7AB',lw=.6,zorder=4))
    ax.add_patch(Rectangle((4.29,.57),.30,.27,fc=tint(PALETTE['cyan'],.30),ec=PALETTE['blue'],lw=.75,zorder=5))
    text(ax,4.46,.28,r'Query boxes $V_q$',size=8.1)
    # The example concept and output prediction are archived observations.
    arrow(ax,[(4.90,2.31),(5.47,2.31)],color=PALETTE['coral'])
    module(ax,5.48,2.13,1.15,.36,'red octagon',accent='coral',size=8.3)
    text(ax,6.05,1.92,r'Selected concept $c^*$',size=7.9)
    arrow(ax,[(6.05,1.80),(6.05,1.63)],color=PALETTE['coral'])
    arrow(ax,[(4.84,.72),(5.16,.72),(5.16,1.40),(5.48,1.40)],color=PALETTE['blue'])
    module(ax,5.49,1.17,1.14,.45,['Joint SAM 3',r'$(I_q,c^*,V_q)$'],accent='olive',size=8.3)
    arrow(ax,[(6.05,1.15),(6.05,.96)])
    raster(ax,'cgics_prediction.png',5.73,.18,.65,.76,mask=True)
    text(ax,6.52,.55,r'$\widehat M_q$',size=9)
    save(fig,'cgics_pipeline')

def update_audit():
    path=OUT/'method_figure_audit.json'
    audit=json.loads(path.read_text())
    audit['audited_at']='2026-11-17'
    audit['drawing_type']='Original vector schematics with exact archived input/output raster thumbnails'
    audit['style_revision']={'date':'2026-11-17','reference':'User-supplied group_07_report_project2.pdf Figure 1',
        'palette':PALETTE,'font_family':'STIXGeneral','intended_include_scale':1,
        'changes':['Soft panel grouping and coherent reading order','Actual reference/query/mask examples',
                   'Small original feature and prompt icons','Compact serif typography and selective accents'],
        'scientific_changes':[]}
    rows=[]
    for name,layout in LAYOUT_AUDIT.items():
        files=[]
        for ext in ('pdf','svg','png'):
            p=OUT/f'{name}.{ext}';files.append({'path':str(p.relative_to(HERE.parent)),
              'bytes':p.stat().st_size,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()})
        rows.append({'name':name,'files':files,**layout})
    audit['figures']=rows
    audit['example_provenance']='paper/figures/assets/example_provenance.json'
    audit['captions']['insid3_pipeline']='INSID3: visual features, reference correspondence, and seed-guided region aggregation.'
    audit['captions']['cgics_pipeline']='CG-ICS: concept guidance and visual transfer provide complementary prompts to the final SAM 3 prediction.'
    audit['generator']={'path':'paper/build_method_figures.py','sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    path.write_text(json.dumps(audit,indent=2)+'\n')
    (OUT/'method_figure_audit.md').write_text('''# Method-figure audit\n\nThe two original schematics use soft panel grouping and serif labels, following the supplied Group report. All panels, arrows, text and feature/prompt icons are vector objects. The exact reference, query, reference mask, and predictions are from archived COCO episode `coco-f3-q000061-c0bcbff23c1abff2-r000` (case06). Actual CG-ICS candidates and selected concept `red octagon` are recorded in `assets/example_provenance.json`. Feature grids, cluster-color grids and the query-box icon are schematic, not claimed intermediate outputs. No query ground truth is an inference input.\n\nINSID3 keeps original features for query grouping and seed affinity, with positional debiasing before reference correspondence. The reference mask bypasses the image encoder. Reference correspondence guides seed selection and aggregation; the final threshold is retained. CG-ICS separates reference-driven concept generation/scoring/search from reference-query visual transfer and combines the resulting concept/boxes in a final query prediction. Arrows express logical dependencies, not concurrent execution.\n\nAll archive inputs are local and bundled as exact small assets for standalone regeneration. Primary-paper and pinned-source provenance is retained in the JSON audit.\n''')

if __name__=='__main__':
    insid3();cgics();update_audit()
    print(json.dumps(LAYOUT_AUDIT,indent=2))
