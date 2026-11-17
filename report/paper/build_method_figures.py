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
plt.rcParams.update({'font.family':'STIXGeneral','font.size':7.8,
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
    text(ax,x+.13,y+h-.18,label,size=8.5,weight='bold',color=PALETTE['blue'],ha='left')

def module(ax,x,y,w,h,lines,accent=None,size=7.8):
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
    if max(t.get_fontsize() for t in texts) > 8.5:raise RuntimeError(f'{stem}: oversized diagram label')
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
    panel(ax,.04,.04,2.46,2.92,'(a) Visual features')
    panel(ax,2.72,1.64,4.11,1.32,'(b) Reference correspondence')
    panel(ax,2.72,.04,4.11,1.38,'(c) Region selection')
    # Reference mask remains an input to matching, not to the image encoder.
    for x,name,label,mask in [(.17,'reference.jpg',r'Reference $I_r$',False),
                              (.93,'query.jpg',r'Query $I_q$',False),
                              (1.69,'reference_mask.png',r'Mask $M_r$',True)]:
        text(ax,x+.33,2.57,label,size=7.7)
        raster(ax,name,x,1.99,.66,.50,mask=mask)
    module(ax,.22,1.28,2.10,.38,'Frozen DINOv3',accent='blue',size=8.2)
    arrow(ax,[(.50,1.98),(.50,1.68)])
    arrow(ax,[(1.26,1.98),(1.26,1.68)])
    # Two uses of original query features are explicit: grouping and seed affinity.
    arrow(ax,[(1.26,1.26),(1.26,.95)])
    text(ax,1.49,1.08,r'$F_q$',size=8,color=PALETTE['blue'],ha='left')
    module(ax,.22,.32,2.10,.61,[],accent='blue')
    text(ax,1.27,.78,'Query clustering',size=8.1,weight='bold')
    grid(ax,.49,.43,mode='features');arrow(ax,[(.99,.56),(1.48,.56)])
    grid(ax,1.59,.43,mode='clusters')
    text(ax,1.26,.41,r'$\{G_k\}$',size=7.5)
    # Debiased features carry cross-image evidence.
    arrow(ax,[(2.34,1.47),(2.57,1.47),(2.57,2.31),(2.91,2.31)])
    module(ax,2.93,2.09,1.43,.43,['Positional debiasing',r'$F(I-BB^{\mathsf{T}})$'],accent='cyan',size=7.8)
    arrow(ax,[(4.38,2.31),(4.65,2.31)])
    text(ax,4.51,2.48,r'$\widetilde F$',size=7.8,color=PALETTE['blue'])
    module(ax,4.67,2.09,1.94,.43,['Reference matching','Prototype + two-way matches'],accent='cyan',size=7.8)
    arrow(ax,[(2.02,1.98),(2.02,1.84),(5.34,1.84),(5.34,2.07)],color=PALETTE['blue'])
    text(ax,3.79,1.84,r'Reference mask $M_r$',size=7.5,color=PALETTE['blue'],
         bbox=dict(facecolor='white',edgecolor='none',pad=.8))
    # The correspondence evidence includes candidates and debiased cluster features.
    arrow(ax,[(6.15,2.07),(6.15,1.53),(2.61,1.53),(2.61,.84),(2.91,.84)])
    arrow(ax,[(5.43,1.53),(5.43,1.08)])
    text(ax,4.62,1.53,'Candidates + debiased features',size=7.1,color=PALETTE['blue'],
         bbox=dict(facecolor='white',edgecolor='none',pad=.7))
    arrow(ax,[(2.34,.62),(2.91,.62)])
    module(ax,2.93,.49,1.08,.46,['Seed cluster $G_*$', 'Prototype match'],accent='gold',size=7.6)
    arrow(ax,[(4.03,.73),(4.23,.73)])
    module(ax,4.25,.42,1.61,.64,[],accent='gold')
    text(ax,5.055,.92,'Aggregate regions',size=8.0,weight='bold')
    text(ax,5.055,.74,r'$s_k=a_k b_k w_k$',size=8)
    text(ax,5.055,.55,'cross · seed · coverage',size=7.2,color='#4C555A')
    # Original-space seed affinity bypasses positional debiasing.
    arrow(ax,[(.60,1.26),(.60,1.07),(.12,1.07),(.12,.16),(5.05,.16),(5.05,.40)],color=PALETTE['blue'])
    text(ax,3.56,.16,r'Original $F_q$: seed affinity',size=7.4,color=PALETTE['blue'],
         bbox=dict(facecolor='white',edgecolor='none',pad=.7))
    arrow(ax,[(5.88,.73),(6.14,.73)])
    text(ax,6.00,.48,r'$s_k>\theta$',size=7.3)
    raster(ax,'insid3_prediction.png',6.19,.31,.47,.63,mask=True)
    text(ax,6.42,1.12,r'$\widehat M_q$',size=8.1)
    text(ax,6.42,.17,'Upsample',size=7.2)
    save(fig,'insid3_pipeline')

def cgics():
    fig,ax=canvas()
    panel(ax,.04,.04,1.25,2.92,'(a) Inputs')
    panel(ax,1.50,1.58,3.56,1.38,'(b) Concept guidance',color='gold')
    panel(ax,1.50,.04,3.56,1.34,'(c) Visual transfer',color='cyan')
    panel(ax,5.28,.04,1.55,2.92,'(d) Prediction')
    text(ax,.66,2.60,'Reference views',size=7.7)
    raster(ax,'reference.jpg',.16,1.98,.49,.53)
    raster(ax,'reference.jpg',.68,1.98,.49,.53,overlay=True)
    text(ax,.41,1.86,r'$I_r$',size=7.7)
    text(ax,.93,1.86,r'$I_r$ + $M_r$',size=7.7)
    text(ax,.66,1.52,r'Query $I_q$',size=8)
    raster(ax,'query.jpg',.28,.30,.77,1.10)
    # Only reference views enter concept generation; query pixels enter scoring explicitly.
    arrow(ax,[(1.18,2.27),(1.67,2.27)])
    arrow(ax,[(1.18,2.12),(1.36,2.12),(1.36,.74),(1.67,.74)])
    arrow(ax,[(1.07,.93),(1.36,.93)],head=False)
    module(ax,1.69,2.04,.81,.48,['MLLM','Concepts'],accent='coral',size=8)
    arrow(ax,[(2.52,2.28),(2.70,2.28)])
    # Actual initial concepts and scores from the archived example, shown in selection order.
    for y,lab,score,chosen in [(2.46,'stop sign','0.763',False),
                              (2.24,'red octagon','0.904',True),
                              (2.02,'traffic sign','0.719',False)]:
        ax.add_patch(FancyBboxPatch((2.71,y-.095),1.11,.19,
              boxstyle='round,pad=0,rounding_size=.025',lw=.45,
              ec=PALETTE['coral'] if chosen else '#BBC0C2',
              fc=tint(PALETTE['coral'],.27) if chosen else 'white',zorder=4))
        text(ax,3.265,y,lab,size=7.6,weight='bold' if chosen else 'normal')
    arrow(ax,[(3.84,2.28),(4.03,2.28)])
    module(ax,4.05,2.00,.84,.56,['SAM 3 scoring','Ref. IoU ×','query presence'],accent='coral',size=7.5)
    # The search loop is a controller over candidate generation and SAM scoring.
    arrow(ax,[(4.47,1.98),(4.47,1.85),(2.09,1.85),(2.09,2.02)],dashed=True,color=PALETTE['coral'])
    text(ax,3.32,1.70,'Search: expand · prune · rerank',size=7.4,color='#7E5048')
    # Reference components are converted to boxes before stitched-image SAM inference.
    raster(ax,'reference.jpg',1.68,.46,.66,.50,overlay=True)
    raster(ax,'query.jpg',2.35,.46,.34,.50)
    text(ax,2.19,.27,r'Stitch $[I_r\mid I_q]$',size=7.7)
    text(ax,2.21,1.06,r'$M_r \rightarrow$ reference boxes',size=7.3,color=PALETTE['blue'])
    arrow(ax,[(2.73,.73),(2.96,.73)])
    module(ax,2.98,.48,.83,.48,['SAM 3','Spatial prompts'],accent='cyan',size=7.6)
    arrow(ax,[(3.83,.73),(4.03,.73)])
    # Query boxes are an illustrative icon, never derived from the final prediction.
    ax.add_patch(Rectangle((4.08,.50),.73,.45,fc='white',ec='#9DA7AB',lw=.6,zorder=4))
    ax.add_patch(Rectangle((4.25,.57),.31,.27,fc=tint(PALETTE['cyan'],.30),ec=PALETTE['blue'],lw=.75,zorder=5))
    text(ax,4.45,.28,r'Query boxes $V_q$',size=7.6)
    text(ax,4.45,1.06,'Query mask → box',size=7.1,color=PALETTE['blue'])
    # Distinct concept and visual prompts merge at the final query segmentation call.
    arrow(ax,[(4.91,2.28),(5.45,2.28)],color=PALETTE['coral'])
    module(ax,5.47,2.09,1.17,.37,'red octagon',accent='coral',size=8)
    text(ax,6.055,1.92,r'Selected concept $c^*$',size=7.5)
    arrow(ax,[(6.055,1.78),(6.055,1.63)],color=PALETTE['coral'])
    arrow(ax,[(4.84,.73),(5.16,.73),(5.16,1.39),(5.47,1.39)],color=PALETTE['blue'])
    module(ax,5.49,1.16,1.14,.46,['SAM 3',r'Query $I_q$ + $c^*$ + $V_q$'],accent='olive',size=7.8)
    arrow(ax,[(6.055,1.14),(6.055,.94)])
    raster(ax,'cgics_prediction.png',5.77,.19,.55,.72,mask=True)
    text(ax,6.48,.57,r'$\widehat M_q$',size=8.3)
    save(fig,'cgics_pipeline')

def update_audit():
    path=OUT/'method_figure_audit.json'
    audit=json.loads(path.read_text())
    audit['audited_at']='2026-11-17'
    audit['drawing_type']='Original vector schematics with exact archived input/output raster thumbnails'
    audit['style_revision']={'date':'2026-11-17','reference':'User-supplied group_07_report_project2(2).pdf Figure 1',
        'palette':PALETTE,'font_family':'STIXGeneral','intended_include_scale':1,
        'changes':['Soft panel grouping and coherent reading order','Actual reference/query/mask examples',
                   'Small original feature and prompt icons','Compact serif typography and selective accents'],
        'scientific_changes':['Made the original query-feature seed-affinity dependency explicit; no algorithm or result changed','Separated SAM 3 scoring from the concept-search controller']}
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
