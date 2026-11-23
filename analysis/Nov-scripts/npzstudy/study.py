"""Analysis of compact records. Every stage writes its tables before proceeding."""
from pathlib import Path
from collections import Counter
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from reduce import csv_rows, M0

FIT_MASS=float(np.float32(.135))  # Public source stores a float steering default; historical analysis M0 remains unchanged.
COLORS={'g4':'#2166ac','cc3':'#d6604d'}
ENERGY=[(5,10),(10,20),(20,30),(30,50),(50,125)]
PHOTON_ENERGY=[(10,20),(20,30),(30,40),(40,60),(60,80),(80,100),(100,125)]
OUTCOME=['neither strong best link','one strong best link','shared best object','distinct non-photon object(s)','distinct clean photons']

def table(out,name,rows):csv_rows(out/'tables'/f'{name}.csv',rows)

def finish(out,name,fig,title,caption):
    import textwrap
    for ax in fig.axes:
        populated=any(np.isfinite(np.asarray(line.get_ydata(),dtype=float)).any() for line in ax.lines if len(line.get_ydata())>2) or any(len(c.get_paths()) for c in ax.collections) or bool(ax.patches)
        if not populated and not ax.get_legend_handles_labels()[0]:
            ax.text(.5,.5,'No usable entries / insufficient support',ha='center',va='center',transform=ax.transAxes,fontsize=9,color='.4')
    fig.suptitle(title,fontsize=14)
    fig.text(.02,.01,textwrap.fill(caption,145),fontsize=8,va='bottom')
    fig.tight_layout(rect=(0,.12,1,.94))
    for ext in ('png','pdf'):
        path=out/'figures'/f'{name}.{ext}';tmp=path.with_suffix('.tmp')
        fig.savefig(tmp,format=ext,dpi=170);tmp.replace(path)
    plt.close(fig)

def stats(x,event,weights=None,bootstrap=0):
    x=np.asarray(x,dtype=float);event=np.asarray(event)
    valid=np.isfinite(x)
    if weights is not None:valid&=np.isfinite(weights)&(weights>0)
    x=x[valid];event=event[valid];w=np.ones(len(x)) if weights is None else np.asarray(weights)[valid]
    result=dict(N_total=len(valid),N_valid=len(x),N_events=0,mean=np.nan,se_event=np.nan,p16=np.nan,median=np.nan,p84=np.nan,s68=np.nan,median_boot_se=np.nan,s68_boot_se=np.nan)
    if not len(x):return result
    _,inv=np.unique(event,return_inverse=True);ne=int(inv.max()+1)
    avg=np.average(x,weights=w);res=np.bincount(inv,weights=w*(x-avg))
    result.update(N_events=ne,mean=avg,se_event=np.sqrt(ne/(ne-1)*np.dot(res,res))/w.sum() if ne>1 else np.nan)
    order=np.argsort(x);vs=x[order];ww=w[order];cum=np.cumsum(ww)
    quant=vs[np.searchsorted(cum,np.array([.16,.5,.84])*cum[-1])]
    result.update(p16=quant[0],median=quant[1],p84=quant[2],s68=(quant[2]-quant[0])/2)
    if bootstrap and ne>1:
        rng=np.random.default_rng(20261106);est=[];iv=inv[order]
        for _ in range(bootstrap):
            mult=rng.multinomial(ne,np.full(ne,1/ne));cw=np.cumsum(ww*mult[iv])
            if cw[-1]>0:
                q=vs[np.searchsorted(cw,np.array([.16,.5,.84])*cw[-1])];est.append([q[1],(q[2]-q[0])/2])
        if len(est)>1:result['median_boot_se'],result['s68_boot_se']=np.std(est,axis=0,ddof=1)
    return result

def fraction(x,events):
    r=stats(np.asarray(x,float),events)
    n=len(x);k=int(np.sum(x));ph=k/n if n else np.nan
    # Wilson interval is an explicitly labelled object-independent diagnostic, not event inference.
    z=1.959963984540054;den=1+z*z/n if n else np.nan
    mid=(ph+z*z/(2*n))/den if n else np.nan
    half=z*np.sqrt(ph*(1-ph)/n+z*z/(4*n*n))/den if n else np.nan
    return dict(N=n,passed=k,N_events=r['N_events'],fraction=ph,se_event=r['se_event'],wilson95_lo=mid-half,wilson95_hi=mid+half)

def photon_regions(p):
    t=p['theta'];d=np.minimum(t,180-t);crack=(p['phi']+158.01)%45<=9.01
    return {'all':np.ones(len(p),bool),'barrel':(t>=40)&(t<=140)&~crack,
      'barrel_phi_crack':(t>=40)&(t<=140)&crack,
      'forward':(t>=15)&(t<30),'backward':(t>150)&(t<=165),
      'endcap':(d>=15)&(d<30),'endcap_wide':(d>=12)&(d<33),'endcap_tight':(d>=17)&(d<27),
      'transition':(d>=33)&(d<40),'extreme_forward':d<12}

def pair_regions(p):
    a,b=p['A_theta'],p['B_theta'];d=np.c_[np.minimum(a,180-a),np.minimum(b,180-b)]
    same=(a<90)==(b<90)
    result={'all':np.ones(len(p),bool),'barrel':p['region']==0,'barrel_phi_crack':p['region']==1,
      'forward':p['region']==2,'backward':p['region']==3,'endcap':np.isin(p['region'],[2,3]),
      'endcap_wide':same&((d>=12)&(d<33)).all(1),'endcap_tight':same&((d>=17)&(d<27)).all(1),
      'transition':p['region']==5,'endcap_rest':p['region']==4,'mixed_other':p['region']==6,
      'gap_cross':p['gap_cross']==1}
    return result

def ebin(p,lo,hi):return (p['E']>=lo)&(p['E']<hi)
def win(x,w=.04):return np.isfinite(x)&(abs(x-M0)<=w)

def angular(data,out):
    rows=[];responses=[];phirows=[]
    fig,axes=plt.subplots(3,1,figsize=(11,9),sharex=True)
    for tag,d in data.items():
        p=d['photons'];energycuts=[('all_energy',p['E']>=0),('Ege10',p['E']>=10),('10to20',ebin(p,10,20)),('20to50',ebin(p,20,50)),('50to125',ebin(p,50,125))]
        for label,energy in energycuts:
          for parent,pm in [('all_parents',np.ones(len(p),bool)),('no_direct_e_parent',p['electron_parent']==0)]:
            for lo in range(180):
                mask=energy&pm&(p['theta']>=lo)&(p['theta']<lo+1);q=p[mask]
                known=(q['fast']==0)|(q['fast']==1)
                row=dict(sample=tag,energy=label,parent=parent,theta_lo=lo,theta_hi=lo+1,**fraction(q['owned_clean']==1,q['loop']))
                row.update(flag_known=int(known.sum()) if tag=='cc3' else '',flag_unknown=int((~known).sum()) if tag=='cc3' else '',flagged=int((q['fast']==1).sum()) if tag=='cc3' else '',flag_fraction=float(np.mean(q['fast'][known]==1)) if tag=='cc3' and known.any() else np.nan)
                rows.append(row)
                if label=='Ege10' and parent=='all_parents':
                    found=q[q['owned_clean']==1]
                    for name,v in [('energy_response',found['owned_E']/found['E']),('direction_mrad',found['owned_direction_mrad']),('theta_residual_deg',found['owned_theta']-found['theta'])]:
                        responses.append(dict(sample=tag,theta_lo=lo,theta_hi=lo+1,quantity=name,**stats(v,found['loop'])))
        rr=[r for r in rows if r['sample']==tag and r['energy']=='Ege10' and r['parent']=='all_parents']
        x=np.arange(180)+.5
        axes[0].step(x,[r['N'] for r in rr],where='mid',label=tag.upper(),color=COLORS[tag])
        axes[1].plot(x,[r['fraction'] for r in rr],color=COLORS[tag])
        if tag=='cc3':axes[2].plot(x,[r['flag_fraction'] for r in rr],color=COLORS[tag])
        for region,geo in photon_regions(p).items():
            if region not in ('barrel','endcap','transition'):continue
            for lo in range(-180,180,5):
                q=p[geo&(p['E']>=10)&(p['phi']>=lo)&(p['phi']<lo+5)]
                phirows.append(dict(sample=tag,region=region,phi_lo=lo,phi_hi=lo+5,**fraction(q['owned_clean']==1,q['loop'])))
    for ax in axes:
        for t in (12,15,30,33,40,140,147,150,165,168):ax.axvline(t,color='.8',ls=':',lw=.6)
    axes[0].legend();axes[0].set_ylabel('Generated photons / degree')
    axes[1].set_ylabel('Clean owned recovery');axes[2].set_ylabel('CC3 flag fraction');axes[2].set_xlabel('True photon θ [degrees]')
    table(out,'01_angular_coverage',rows);table(out,'01_angular_response',responses);table(out,'01_phi_coverage',phirows)
    finish(out,'01_angular_map',fig,'Where are photons generated, recovered and fast simulated?',
      'Generator-status 1 photons, true E ≥10 GeV, all parentages; no phi mask. Recovery uses V6 owned photon PFO, cw≥0.5. Flag fraction uses known CC3 flags only. Curves descriptive; event SE and counts in CSV. Truth occupancy is not detector acceptance.')

def photons(data,out,boot):
    rows=[];profiles=[]
    for tag,d in data.items():
        p=d['photons']
        for region,geo in photon_regions(p).items():
          for lo,hi in PHOTON_ENERGY:
            base=geo&ebin(p,lo,hi)&(p['electron_parent']==0)
            groups={'inclusive':base,'quiet20mrad':base&(p['nearest_photon_mrad']>=20)&(p['nearest_charged_mrad']>=20),
                    'photon_near20mrad':base&(p['nearest_photon_mrad']<20),'pi0_daughter':base&(p['pi0_parent']==1)}
            if tag=='cc3':groups.update(flagged=base&(p['fast']==1),unflagged=base&(p['fast']==0))
            for population,mask in groups.items():
                q=p[mask];clean=q['owned_clean']==1;f=q[clean]
                obs={'energy_response':f['owned_E']/f['E'],'direction_mrad':f['owned_direction_mrad']}
                obs.update({k:f[k] for k in ('width_ip','width_centered','centroid_offset','R50','R90','core_fraction','tail_fraction','depth_mean','depth_spread','all_width_centered','all_centroid_offset')})
                for name,v in obs.items():
                    nboot=boot if region in ('barrel','endcap') and population=='inclusive' and name=='energy_response' else 0
                    rows.append(dict(sample=tag,region=region,population=population,E_lo=lo,E_hi=hi,truth=len(q),recovered=int(clean.sum()),quantity=name,**stats(v,f['loop'],bootstrap=nboot)))
                if population=='inclusive':
                    valid=(f['n_hits']>=2)&(f['raw_hit_E']>0)&np.isfinite(f['ring_E_00']);ff=f[valid]
                    for j in range(21):
                        val=ff[f'ring_E_{j:02d}'];norm=ff['raw_hit_E']
                        profiles.append(dict(sample=tag,region=region,E_lo=lo,E_hi=hi,r_lo=j*2.5,r_hi=(j+1)*2.5 if j<20 else np.inf,
                          clusters=len(ff),pooled_E_fraction=val.sum()/norm.sum() if len(ff) else np.nan,**stats(val/norm,ff['loop'])))
    table(out,'02_photon_response',rows);table(out,'02_radial_profiles',profiles)
    fig,ax=plt.subplots(2,3,figsize=(13,7))
    for row,region in enumerate(('barrel','endcap')):
      for col,quantity in enumerate(('energy_response','width_centered','centroid_offset')):
        a=ax[row,col]
        for tag,color in COLORS.items():
          for population,style in [('inclusive','-'),('quiet20mrad','--')]:
            rr=[r for r in rows if r['sample']==tag and r['region']==region and r['quantity']==quantity and r['population']==population and r['N_valid']>=5]
            a.errorbar([np.sqrt(r['E_lo']*r['E_hi']) for r in rr],[r['mean'] for r in rr],yerr=[r['se_event'] for r in rr],color=color,ls=style,marker='o',ms=3,label=tag.upper()+' '+population)
        a.set_xscale('log');a.set_xlabel('True photon energy [GeV]');a.set_title(region);a.set_ylabel({'energy_response':'Mean reconstructed / true energy','width_centered':'Centered transverse RMS [mm]','centroid_offset':'Transverse centroid offset [mm]'}[quantity])
    ax[0,0].legend(fontsize=6)
    finish(out,'02_single_photon_baseline',fig,'Do regional differences persist without nearby particles?',
      'Status 1 photons, E≥10 GeV, no direct electron parent, owned clean photon PFO. Barrel excludes historical phi cracks. Quiet: no generator photon or charged generator particle within 20 mrad. First-cluster moments; error bars are event-cluster SE, not cross-sample difference errors.')

def stages(data,out):
    rows=[];outcomes=[];reco=[];flags=[]
    for tag,d in data.items():
        p=d['pairs']
        for region,geo in pair_regions(p).items():
          for lo,hi in ENERGY:
            q=p[geo&ebin(p,lo,hi)];t=q['T']==1;te=q['TE']==1;w=te&win(q['mass']);f=w&(q['found']==1)
            for stage,den,num in [('resolved',np.ones(len(q),bool),t),('energy',t,te),('window40',te,w),('finder_after40',w,f),('final40',np.ones(len(q),bool),f)]:
                rows.append(dict(sample=tag,region=region,E_lo=lo,E_hi=hi,stage=stage,**fraction(num[den],q['loop'][den])))
            for code,label in enumerate(OUTCOME):
                outcomes.append(dict(sample=tag,region=region,E_lo=lo,E_hi=hi,outcome=label,**fraction(q['outcome']==code,q['loop'])))
            outcomes.append(dict(sample=tag,region=region,E_lo=lo,E_hi=hi,outcome='alternative distinct clean among failed',**fraction(q['alternate_distinct_clean'][~t]==1,q['loop'][~t])))
            if tag=='cc3':
                for stage,mask in [('truth',np.ones(len(q),bool)),('TE',te),('window40',w),('found40',f)]:
                    a,b=q['A_fast'][mask],q['B_fast'][mask];known=np.isin(a,[0,1])&np.isin(b,[0,1]);nfast=(a==1).astype(int)+(b==1)
                    flags.append(dict(region=region,E_lo=lo,E_hi=hi,stage=stage,N=len(a),both_known=int(known.sum()),unknown=int((~known).sum()),both=int((known&(nfast==2)).sum()),one=int((known&(nfast==1)).sum()),neither=int((known&(nfast==0)).sum())))
        r=d['reco_pairs']
        for code,label in enumerate(('barrel','barrel_phi_crack','forward','backward','endcap_rest','transition','mixed_other')):
          for lo,hi in ENERGY:
            q=r[(r['region']==code)&ebin(r,lo,hi)]
            for stage,sel in [('clean',np.ones(len(q),bool)),('energy',q['energy_pass']==1),('window40',(q['energy_pass']==1)&win(q['mass'])),('found40',(q['energy_pass']==1)&win(q['mass'])&(q['found']==1))]:
                a=q[sel];unique=len(set(zip(a['loop'],a['cid'],a['idx'])))
                reco.append(dict(sample=tag,region=label,E_lo=lo,E_hi=hi,stage=stage,combinations=len(a),unique_parents=unique,events=len(np.unique(a['loop']))))
    table(out,'03_stages',rows);table(out,'03_failure_outcomes',outcomes);table(out,'03_reco_first_yields',reco);table(out,'03_cc3_flag_composition',flags)
    fig,ax=plt.subplots(1,3,figsize=(13,4))
    for a,stage in zip(ax,('resolved','window40','final40')):
        for tag,c in COLORS.items():
          for reg,ls in [('barrel','-'),('endcap','--')]:
            rr=[r for r in rows if r['sample']==tag and r['region']==reg and r['stage']==stage]
            a.errorbar([np.sqrt(r['E_lo']*r['E_hi']) for r in rr],[r['fraction'] for r in rr],yerr=[r['se_event'] for r in rr],color=c,ls=ls,marker='o',label=tag.upper()+' '+reg)
        a.set_xscale('log');a.set_xlabel('True π⁰ energy [GeV]');a.set_ylabel('Fraction');a.set_title(stage)
    ax[0].legend(fontsize=7)
    finish(out,'03_pair_stages',fig,'At which stage does the π⁰ efficiency change?',
      'Truth-first two generator photon daughters; best cw×PFO energy, cw≥0.5, distinct photon PFOs. Resolved and final40 use all truth pairs; window40 uses resolved pairs with both reco E≥0.5 GeV. Final40 additionally requires exact stored candidate pair. Event-cluster SE.')

PAIR_CAP='Truth-first clean distinct photon PFOs, both reco E≥0.5 GeV; no mass or finder selection. Barrel: both 40–140°, historical phi cracks removed. Endcap: same side, both folded θ in [15,30)°. Densities normalized to all finite entries; displayed tails not renormalized. Histograms descriptive.'

def histograms(data,out):
    for name,field,label,edges in [('04_opening_error',None,'Reco − true opening angle [mrad]',np.linspace(-5,8,53)),('05_mass_before_window','mass','Unfitted mass [MeV]',np.linspace(0,300,61))]:
        fig,axes=plt.subplots(2,3,figsize=(14,8));rows=[]
        for row,region in enumerate(('barrel','endcap')):
          for col,(lo,hi) in enumerate(ENERGY[2:]):
            ax=axes[row,col]
            for tag,d in data.items():
                p=d['pairs'];q=p[pair_regions(p)[region]&ebin(p,lo,hi)&(p['TE']==1)]
                v=1000*q[field] if field else q['alpha_reco_mrad']-q['alpha_true_mrad'];v=v[np.isfinite(v)]
                counts=np.histogram(v,edges)[0];under=int((v<edges[0]).sum());over=int((v>edges[-1]).sum())
                if len(v):ax.stairs(counts/(len(v)*np.diff(edges)),edges,color=COLORS[tag],ls='--' if tag=='cc3' else '-',label=f'{tag.upper()} N={len(v)}; tails={under}/{over}')
                for l,h,n in zip(edges[:-1],edges[1:],counts):rows.append(dict(sample=tag,region=region,E_lo=lo,E_hi=hi,bin_lo=l,bin_hi=h,count=n,N_finite=len(v),N_invalid=len(q)-len(v),underflow=under,overflow=over))
            if field:
                for x in (1000*(M0-.04),1000*(M0+.04)):ax.axvline(x,c='.5',ls=':')
            ax.set_title(f'{region}, {lo}–{hi} GeV');ax.set_xlabel(label);ax.set_ylabel('Pair fraction / '+('MeV' if field else 'mrad'))
            if ax.get_legend_handles_labels()[0]:ax.legend(fontsize=7)
        table(out,name+'_hist',rows);finish(out,name,fig,'Where do reconstructed pairs move before selection?',PAIR_CAP)

def common_weights(samples,keys,min_count=5):
    counts={t:Counter(keys[t]) for t in samples}
    support={k:min(n,counts['cc3'].get(k,0)) for k,n in counts['g4'].items() if min(n,counts['cc3'].get(k,0))>=min_count}
    return {t:np.array([support.get(k,0)/counts[t][k] for k in keys[t]]) for t in samples}

def pair_keys(p,finder=False):
    if finder:
        vars=[(p['mass']*1000,np.arange(30,241,5)),(p['A_reco_E']+p['B_reco_E'],[10,20,30,40,50,70,100]),(p['B_reco_E']/(p['A_reco_E']+p['B_reco_E']),[.1,.2,.3,.4,.5,.6,.7,.8,.9])]
    else:
        vars=[(p['E'],[10,15,20,25,30,35,40,45,50,60,80,100]),(p['alpha_true_mrad'],[4,6,8,10,15,25,50]),(p['B_true_E']/(p['A_true_E']+p['B_true_E']),[.1,.2,.3,.4])]
    vars.extend([(np.minimum(p['A_theta'],180-p['A_theta']),[12,15,20,25,30,33,40,60,80]),(p['A_theta']>90,[.5])])
    return list(zip(*(np.digitize(x,b) for x,b in vars)))

def controlled_mass(data,out):
    rows=[]
    fields=('mass','mass_massless','mass_true_angle','mass_true_energy')
    for region in ('barrel','forward','backward','endcap','endcap_wide','endcap_tight','transition'):
      for lo,hi in ENERGY[2:]:
        ss={}
        for tag,d in data.items():
            p=d['pairs'];m=pair_regions(p)[region]&ebin(p,lo,hi)&(p['TE']==1)
            m&=np.logical_and.reduce([np.isfinite(p[f]) for f in fields]);ss[tag]=p[m]
        weights=common_weights(ss,{t:pair_keys(q) for t,q in ss.items()})
        for tag,q in ss.items():
            keep=weights[tag]>0
            for field in fields:
              for w in (.04,.1):
                rows.append(dict(sample=tag,region=region,E_lo=lo,E_hi=hi,mass_type=field,window=w,N_input=len(q),N_support=int(keep.sum()),weight_sum=weights[tag].sum(),**stats(win(q[field][keep],w),q['loop'][keep],weights[tag][keep])))
    table(out,'05_matched_mass_acceptance',rows)

def controlled_photons(data,out):
    rows=[]
    for reg in ('barrel','forward','backward','endcap','transition'):
      for lo,hi in PHOTON_ENERGY:
       for activity in ('inclusive','quiet20mrad'):
        ss={}
        for t,d in data.items():
            p=d['photons'];m=photon_regions(p)[reg]&ebin(p,lo,hi)&(p['electron_parent']==0)&(p['owned_clean']==1)
            if activity=='quiet20mrad':m&=(p['nearest_photon_mrad']>=20)&(p['nearest_charged_mrad']>=20)
            ss[t]=p[m]
        for field in ('energy_response','width_ip','width_centered','centroid_offset','direction_mrad'):
            gg={};vv={};kk={}
            for t,q in ss.items():
                v=q['owned_E']/q['E'] if field=='energy_response' else q['owned_direction_mrad'] if field=='direction_mrad' else q[field]
                ok=np.isfinite(v);gg[t]=q[ok];vv[t]=v[ok];p=q[ok]
                kk[t]=list(zip(np.digitize(p['E'],np.arange(lo,hi+1,max(2,(hi-lo)/4))),np.digitize(p['theta'],np.arange(0,181,5)),np.digitize(p['nearest_photon_mrad'],[6,10,20,50]),np.digitize(p['nearest_charged_mrad'],[10,20,50])))
            weights=common_weights(gg,kk)
            for t,q in gg.items():
                keep=weights[t]>0
                rows.append(dict(sample=t,region=reg,E_lo=lo,E_hi=hi,activity=activity,quantity=field,N_input=len(ss[t]),N_support=int(keep.sum()),**stats(vv[t][keep],q['loop'][keep],weights[t][keep])))
    table(out,'02_matched_photons',rows)

def direction_and_mass(data,out,boot):
    response=[];sub=[];matched=[];joint=[]
    for region in ('barrel','forward','backward','endcap','endcap_tight','endcap_wide','transition','gap_cross'):
      for lo,hi in ENERGY[2:]:
        samples={tag:d['pairs'][pair_regions(d['pairs'])[region]&ebin(d['pairs'],lo,hi)&(d['pairs']['TE']==1)] for tag,d in data.items()}
        for quality in ('inclusive','cw07','no_conversion_indicator','one_cluster_each','no_shared_hits'):
            groups={}
            for tag,q in samples.items():
                select=np.ones(len(q),bool)
                if quality=='cw07':select=(q['A_cw']>=.7)&(q['B_cw']>=.7)
                if quality=='no_conversion_indicator':select=(q['A_conversion_daughters']==0)&(q['B_conversion_daughters']==0)
                if quality=='one_cluster_each':select=(q['A_n_clusters']==1)&(q['B_n_clusters']==1)
                if quality=='no_shared_hits':select=q['shared_hits']==0
                groups[tag]=q[select]
            for sep_lo,sep_hi in [(0,np.inf),(0,6),(6,10),(10,20),(20,np.inf)]:
                g={t:q[(q['alpha_true_mrad']>=sep_lo)&(q['alpha_true_mrad']<sep_hi)] for t,q in groups.items()}
                for tag,q in g.items():
                    values={'opening_error':q['alpha_reco_mrad']-q['alpha_true_mrad']}
                    for side in ('A','B'):
                        values[side+'_energy_response']=q[side+'_reco_E']/q[side+'_true_E']
                        for v in ('out_mrad','perp_mrad','centroid_out','centroid_perp','side_asym','rotated_asym','theta_asym','phi_asym','width_centered','centroid_offset','all_centroid_out'):
                            values[side+'_'+v]=q[side+'_'+v]
                    for name,v in values.items():
                        response.append(dict(sample=tag,region=region,E_lo=lo,E_hi=hi,quality=quality,sep_lo=sep_lo,sep_hi=sep_hi,quantity=name,**stats(v,q['loop'])))
                # Observable-specific common support avoids matching against missing measurements.
                for quantity in ('opening_error','A_out_mrad','B_out_mrad','A_centroid_out','B_centroid_out','A_side_asym','B_side_asym','mass'):
                    gg={};vv={}
                    for t,q in g.items():
                        v=q['alpha_reco_mrad']-q['alpha_true_mrad'] if quantity=='opening_error' else q[quantity]
                        ok=np.isfinite(v);gg[t]=q[ok];vv[t]=v[ok]
                    weights=common_weights(gg,{t:pair_keys(q) for t,q in gg.items()})
                    for t,q in gg.items():
                        keep=weights[t]>0
                        matched.append(dict(sample=t,region=region,E_lo=lo,E_hi=hi,quality=quality,sep_lo=sep_lo,sep_hi=sep_hi,quantity=quantity,N_input=len(g[t]),N_support=int(keep.sum()),weight_sum=weights[t].sum(),**stats(vv[t][keep],q['loop'][keep],weights[t][keep])))
        for tag,q in samples.items():
            ef=np.sqrt(q['A_reco_E']*q['B_reco_E']/(q['A_true_E']*q['B_true_E']))
            af=np.sin(q['alpha_reco_mrad']/2000)/np.sin(q['alpha_true_mrad']/2000)
            joint.append(dict(sample=tag,region=region,E_lo=lo,E_hi=hi,N=len(q),both_above=int(((ef>1)&(af>1)).sum()),both_below=int(((ef<1)&(af<1)).sum()),opposed=int(((ef-1)*(af-1)<0).sum()),factorization_max_abs=float(np.nanmax(abs(q['mass_massless']-q['truth_mass']*ef*af))) if len(q) else np.nan))
            for group,fields in [('energy_angle',('mass','mass_massless','mass_true_angle','mass_true_energy','truth_mass')),('centroid',('mass','mass_centroid'))]:
                valid=np.logical_and.reduce([np.isfinite(q[k]) for k in fields]);a=q[valid]
                for field in fields:
                  for window in (.04,.1):
                    v=a[field];nboot=boot if region in ('barrel','endcap') and window==.04 and group=='energy_angle' else 0
                    sub.append(dict(sample=tag,region=region,E_lo=lo,E_hi=hi,fixed_population=group,N_input=len(q),N_common=len(a),mass_type=field,window=window,lower=int((v<M0-window).sum()),upper=int((v>M0+window).sum()),**fraction(win(v,window),a['loop']),median_mass=stats(v,a['loop'],bootstrap=nboot)['median']))
    table(out,'04_direction_response',response);table(out,'04_matched_direction',matched);table(out,'05_fixed_population_substitutions',sub);table(out,'05_joint_energy_angle',joint)
    fig,axes=plt.subplots(1,3,figsize=(13,4))
    for ax,v,label in zip(axes,('A_out_mrad','B_out_mrad','opening_error'),('Higher-true-energy photon outward error [mrad]','Lower-true-energy photon outward error [mrad]','Opening-angle error [mrad]')):
        for tag,color in COLORS.items():
            rr=[r for r in matched if r['sample']==tag and r['region']=='endcap' and r['E_lo']==30 and r['quality']=='inclusive' and r['quantity']==v and not (r['sep_lo']==0 and np.isinf(r['sep_hi']))]
            ax.errorbar(np.arange(len(rr)),[r['mean'] for r in rr],yerr=[r['se_event'] for r in rr],color=color,marker='o',label=tag.upper())
        ax.set_xticks(range(4),['0–6','6–10','10–20','≥20']);ax.set_xlabel('True pair separation [mrad]');ax.set_ylabel(label);ax.axhline(0,c='.6',lw=.5)
    axes[0].legend()
    finish(out,'06_direction_mechanism',fig,'Does excess opening come from one daughter or both?',
      'Endcap interior, true π⁰ energy 30–50 GeV; TE before mass/finder. Common binned support in true energy, sharing, separation, folded θ and detector side; ≥5 per sample/cell. Fixed-support event SE. Empty points mean insufficient support. Positive is away from partner.')

def legacy_pairs(path):
    """Read old pi0-tables files without changing their truth-first definitions."""
    from reduce import rows_array
    with np.load(path,allow_pickle=False) as z:d={k:z[k] for k in z.files}
    candidates={(int(ev),min(int(a),int(b)),max(int(a),int(b))) for ev,a,b,n in zip(d['cand_loop_evt'],d['cand_d0_pfo_idx'],d['cand_d1_pfo_idx'],d['cand_n_dau']) if n==2 and a>=0 and b>=0 and a!=b}
    def col(side,k):return d[f'mc_ph{side}_b0_{k}']
    pa=np.column_stack([col(0,'p'+k) for k in 'xyz']);pb=np.column_stack([col(1,'p'+k) for k in 'xyz'])
    ar=np.arctan2(np.linalg.norm(np.cross(pa,pb),axis=1),np.sum(pa*pb,axis=1));at=d['mc_true_open_angle']
    ea,eb=col(0,'E'),col(1,'E');m2=(ea+eb)**2-np.sum((pa+pb)**2,axis=1);mass=np.where(m2>=-1e-8,np.sqrt(np.maximum(m2,0)),np.nan)
    clean=[(col(j,'pfo_idx')>=0)&(col(j,'cw')>=.5-1e-6)&(col(j,'pdg')==22) for j in (0,1)]
    t=clean[0]&clean[1]&(col(0,'pfo_idx')!=col(1,'pfo_idx'));rows=[]
    from reduce import region
    for i in np.flatnonzero((d['mc_ph0_gen']==1)&(d['mc_ph1_gen']==1)):
        ev=int(d['mc_loop_evt'][i]);a=int(col(0,'pfo_idx')[i]);b=int(col(1,'pfo_idx')[i])
        th=np.array([d[f'mc_ph{j}_theta'][i] for j in (0,1)]);ph=np.array([d[f'mc_ph{j}_phi'][i] for j in (0,1)])
        fold=np.minimum(th,180-th);rg=region(th,ph)
        r=dict(loop=ev,cid=-1,idx=int(d['mc_pi0_idx'][i]),E=d['mc_pi0_E'][i],region=rg,gap_cross=int(((fold>=33)&(fold<40)).any() and rg!=5),T=int(t[i]),TE=int(t[i] and min(ea[i],eb[i])>=.5),mass=mass[i],alpha_true_mrad=at[i]*1000,alpha_reco_mrad=ar[i]*1000,found=int(t[i] and (ev,min(a,b),max(a,b)) in candidates))
        for j,side,pvec in [(0,'A',pa),(1,'B',pb)]:
            r.update({side+'_reco_E':col(j,'E')[i],side+'_true_E':d[f'mc_ph{j}_E'][i],side+'_theta':th[j],side+'_reco_idx':col(j,'pfo_idx')[i],side+'_true_idx':d[f'mc_ph{j}_idx'][i],side+'_reco_theta':np.arctan2(np.linalg.norm(pvec[i,:2]),pvec[i,2]),side+'_reco_phi':np.arctan2(pvec[i,1],pvec[i,0])})
        rows.append(r)
    return rows_array(rows)

def closure(data,out,legacy_dir):
    checks=[]
    for tag,d in data.items():
        p=d['pairs'];ph=d['photons']
        for title,values,tolerance in [('massless_vs_fourvector_GeV',p['mass_massless'][p['TE']==1]-p['mass'][p['TE']==1],1e-6),('raw_hit_vs_moment_width_mm',ph['raw_width_closure'],1e-5),('width_variance_identity_mm2',ph['width_identity_error'],1e-5)]:
            v=values[np.isfinite(values)];maxerr=float(np.max(abs(v))) if len(v) else np.nan
            checks.append(dict(sample=tag,check=title,N=len(v),max_abs=maxerr,tolerance=tolerance,status='PASS' if len(v) and maxerr<=tolerance else 'REVIEW' if len(v) else 'UNAVAILABLE'))
        # Exhaustive outcome partition and a separately recomputed nested-stage product.
        if not np.isin(p['outcome'],np.arange(5)).all() or not np.array_equal(p['outcome']==4,p['T']==1):raise AssertionError('Failure partition does not close')
        n=len(p);nt=int(p['T'].sum());ne=int(p['TE'].sum());nw=int(((p['TE']==1)&win(p['mass'])).sum());nf=int(((p['TE']==1)&win(p['mass'])&(p['found']==1)).sum())
        product=(nt/n)*(ne/nt)*(nw/ne)*(nf/nw) if min(n,nt,ne,nw)>0 else np.nan
        checks.append(dict(sample=tag,check='nested_efficiency_product',N=n,max_abs=abs(product-nf/n),tolerance=1e-12,status='PASS' if np.isfinite(product) and abs(product-nf/n)<1e-12 else 'UNAVAILABLE'))
        path=legacy_dir/f'pi0-tables-{tag}-full.npz' if legacy_dir else None
        if path and path.exists():
            old=legacy_pairs(path);lookup={(int(r['loop']),int(r['idx'])):r for r in old}
            # Event loops alone are not proof of same source. Require exact truth energies,
            # daughter IDs, angles and parent IDs over every compared row.
            matched=[];bad=0
            for r in p:
                q=lookup.get((int(r['loop']),int(r['idx'])))
                if q is None:bad+=1;continue
                truth_ok=(r['A_true_idx']==q['A_true_idx'] and r['B_true_idx']==q['B_true_idx'] and np.isclose(r['E'],q['E'],rtol=1e-8,atol=1e-9) and np.isclose(r['alpha_true_mrad'],q['alpha_true_mrad'],rtol=1e-7,atol=1e-7))
                if not truth_ok:bad+=1;continue
                matched.append((r,q))
            checks.append(dict(sample=tag,check='legacy_truth_row_alignment',N=len(p),max_abs=bad,tolerance=0,status='PASS' if bad==0 else 'FAIL'))
            if bad==0:
                for key in ('T','TE','found','A_reco_idx','B_reco_idx'):
                    mismatch=sum(a[key]!=b[key] for a,b in matched)
                    checks.append(dict(sample=tag,check='legacy_'+key,N=len(matched),max_abs=mismatch,tolerance=0,status='PASS' if mismatch==0 else 'FAIL'))
        else:checks.append(dict(sample=tag,check='historical_pi0_closure',N=0,max_abs=np.nan,tolerance=0,status='NOT_RUN_no_legacy_file'))
    table(out,'00_closure',checks)
    (out/'validation'/'analysis_checks.json').write_text(json.dumps(checks,indent=2))
    if any(r['status']=='FAIL' for r in checks):raise AssertionError('Historical closure failed: inspect 00_closure.csv')

def fit_mass(x):
    a,t,f,b,u,g=np.moveaxis(np.asarray(x),-1,0)
    h=np.sin((t-u)/2)**2+np.sin(t)*np.sin(u)*np.sin((f-g)/2)**2
    return 2*np.sqrt(np.maximum(a*b,0)*np.maximum(h,0))

def fit_diagnostic(q,out,tag,label):
    """Public-default model diagnostic only; never claimed to be installed production fit."""
    try:
        from scipy.optimize import minimize
        from scipy.stats import chi2
    except ImportError:return [dict(sample=tag,version=label,status='SKIPPED_scipy_not_installed')]
    x=np.column_stack([q[s+'_'+k] for s in ('A','B') for k in ('reco_E','reco_theta','reco_phi')])
    err=np.where(np.array([1,0,0,1,0,0],bool),.16*np.sqrt(x[:,[0,0,0,3,3,3]]),.001/np.sqrt(x[:,[0,0,0,3,3,3]]))
    step=np.eye(6)*1e-3
    gradient=(fit_mass(x[:,None,:]+err[:,None,:]*step)-fit_mass(x[:,None,:]-err[:,None,:]*step))/.002
    sigma=np.linalg.norm(gradient,axis=1);pull=(fit_mass(x)-FIT_MASS)/sigma;cut=chi2.ppf(.999,1)
    rows=[];rng=np.random.default_rng(20261106)
    # Stratify on mass shell, region and observed found flag; export sampling rates.
    shell=np.where(abs(q['mass']-M0)<=.04,0,np.where(abs(q['mass']-M0)<=.1,1,2))
    for rg in np.unique(q['region']):
      for sh in (0,1):
       for found in (0,1):
        ids=np.flatnonzero((q['region']==rg)&(shell==sh)&(q['found']==found))
        selected=rng.choice(ids,min(20,len(ids)),replace=False)
        for i in selected:
            def constraint(z):return (fit_mass(x[i]+err[i]*z)-FIT_MASS)/FIT_MASS
            bounds=[(-x[i,j]/err[i,j]+1e-7,None) if j in (0,3) else (None,None) for j in range(6)]
            opt=minimize(lambda z:z@z,np.zeros(6),jac=lambda z:2*z,bounds=bounds,constraints={'type':'eq','fun':constraint},method='SLSQP',options={'ftol':1e-10,'maxiter':150})
            success=bool(opt.success and abs(constraint(opt.x))<1e-6)
            rows.append(dict(sample=tag,version=label,status='public_defaults_unverified_production',fit_nominal_mass=FIT_MASS,analysis_nominal_mass=M0,loop=int(q['loop'][i]),parent_idx=int(q['idx'][i]),region=int(rg),shell=int(sh),stratum_N=len(ids),stratum_sampled=len(selected),mass=q['mass'][i],stored_found=found,linear_chi2=pull[i]**2,linear_pass=int(pull[i]**2<cut),nonlinear_chi2=opt.fun,converged=success,nonlinear_pass=int(opt.fun<cut) if success else -1,constraint_residual=constraint(opt.x)))
    return rows

def finder(data,out,legacy_dir):
    rows=[];matched=[];fits=[];alignment=[]
    datasets={('extracted',t):d['pairs'] for t,d in data.items()}
    if legacy_dir:
        for tag in COLORS:
            base=legacy_dir/f'pi0-tables-{tag}-full.npz';wide=legacy_dir/f'pi0-tables-{tag}-full-dmgg0.1.npz'
            if base.exists() and wide.exists():
                a,b=legacy_pairs(base),legacy_pairs(wide)
                def key(q):
                    k=np.empty(len(q),dtype=[('loop','i8'),('idx','i8')]);k['loop']=q['loop'];k['idx']=q['idx'];return k
                _,ia,ib=np.intersect1d(key(a),key(b),return_indices=True)
                same=np.ones(len(ia),bool)
                for field in ('E','A_true_E','B_true_E','alpha_true_mrad','A_reco_idx','B_reco_idx','A_reco_E','B_reco_E','alpha_reco_mrad','T','TE'):
                    same&=np.isclose(a[field][ia],b[field][ib],rtol=1e-7,atol=1e-8,equal_nan=True)
                bad=int((~same).sum())
                alignment.append(dict(sample=tag,baseline=len(a),wide=len(b),shared=len(ia),upstream_mismatches=bad,unchanged_pairs=int(same.sum()),status='PASS' if not bad and len(ia)==len(a)==len(b) else 'USE_UNCHANGED_SUBSET'))
                datasets[('legacy_baseline',tag)]=a;datasets[('legacy_wide',tag)]=b
                datasets[('legacy_baseline_unchanged',tag)]=a[ia[same]]
                datasets[('legacy_wide_unchanged',tag)]=b[ib[same]]
    table(out,'06_baseline_wide_alignment',alignment)
    for (version,tag),p in datasets.items():
        for region in ('barrel','endcap','forward','backward','transition'):
          for lo,hi in ENERGY[2:]:
            q=p[pair_regions(p)[region]&ebin(p,lo,hi)&(p['TE']==1)]
            for shell,mask in [('inside40',win(q['mass'])),('new_40to100',win(q['mass'],.1)&~win(q['mass'])),('inside100',win(q['mass'],.1))]:
                rows.append(dict(version=version,sample=tag,region=region,E_lo=lo,E_hi=hi,shell=shell,**fraction(q['found'][mask]==1,q['loop'][mask])))
        fitq=p[(p['TE']==1)&(p['E']>=20)&(p['E']<125)&win(p['mass'],.1)]
        if not version.endswith('_unchanged'):fits.extend(fit_diagnostic(fitq,out,tag,version))
    for version in sorted({v for v,t in datasets}):
      if not all((version,t) in datasets for t in COLORS):continue
      for reg in ('barrel','endcap'):
       for lo,hi in ENERGY[2:]:
        ss={t:datasets[(version,t)] for t in COLORS}
        ss={t:p[pair_regions(p)[reg]&ebin(p,lo,hi)&(p['TE']==1)&win(p['mass'],.1)] for t,p in ss.items()}
        weights=common_weights(ss,{t:pair_keys(p,True) for t,p in ss.items()})
        for t,q in ss.items():
            use=weights[t]>0
            matched.append(dict(version=version,sample=t,region=reg,E_lo=lo,E_hi=hi,N_input=len(q),N_support=int(use.sum()),**stats(q['found'][use],q['loop'][use],weights[t][use])))
    table(out,'06_finder_retention',rows);table(out,'06_finder_matched',matched)
    usable_fits=[r for r in fits if 'linear_chi2' in r]
    table(out,'06_fit_diagnostic',usable_fits)
    # Optional scipy skip records have a different schema: store JSON, not ambiguous CSV.
    (out/'validation'/'06_fit_diagnostic.json').write_text(json.dumps(fits,indent=2,default=lambda x:x.item() if isinstance(x,np.generic) else str(x)))
    fig,axes=plt.subplots(1,2,figsize=(10,4))
    version='legacy_wide_unchanged' if ('legacy_wide_unchanged','g4') in datasets and ('legacy_wide_unchanged','cc3') in datasets else 'extracted'
    for ax,reg in zip(axes,('barrel','endcap')):
        for tag,c in COLORS.items():
            rr=[r for r in rows if r['version']==version and r['sample']==tag and r['region']==reg and r['shell']=='new_40to100']
            ax.errorbar([np.sqrt(r['E_lo']*r['E_hi']) for r in rr],[r['fraction'] for r in rr],yerr=[r['se_event'] for r in rr],label=tag.upper(),color=c,marker='o')
        ax.set_xscale('log');ax.set_xlabel('True π⁰ energy [GeV]');ax.set_ylabel('Stored found fraction, 40–100 MeV shell');ax.set_title(reg)
    axes[0].legend()
    finish(out,'07_finder_shell',fig,'Does the finder retain pairs in the wider mass shell?',f'Input: {version}. TE pairs; 40 < |unfitted mass − nominal mass| ≤100 MeV. Event SE. If extracted input was produced with a 40 MeV preselection, this is NOT a widened-finder test. Production fit configuration and rejected-candidate reasons are not stored.')


def benchmark_impact(data,out):
    rows=[];combined=[];cache={}
    for tag,d in data.items():
        p=d['pairs'];base=(p['E']>=5)&(p['E']<125);n=int(base.sum())
        for rg in range(7):
          for lo,hi in ENERGY:
            q=p[base&(p['region']==rg)&ebin(p,lo,hi)]
            passed=(q['TE']==1)&win(q['mass'])&(q['found']==1)
            row=dict(sample=tag,region_code=rg,E_lo=lo,E_hi=hi,total_truth_5to125=n,cell_truth=len(q),cell_found=int(passed.sum()),cell_weight=len(q)/n if n else np.nan,cell_efficiency=passed.mean() if len(q) else np.nan,contribution_to_total_efficiency=passed.sum()/n if n else np.nan)
            rows.append(row);cache[(tag,rg,lo)]=row
    for rg in range(7):
      for lo,hi in ENERGY:
        a=cache['g4',rg,lo];b=cache['cc3',rg,lo]
        support=a['cell_truth']>0 and b['cell_truth']>0
        combined.append(dict(region_code=rg,E_lo=lo,E_hi=hi,g4_truth=a['cell_truth'],cc3_truth=b['cell_truth'],common_support=int(support),difference_efficiency_contribution=b['contribution_to_total_efficiency']-a['contribution_to_total_efficiency'],response_component=.5*(a['cell_weight']+b['cell_weight'])*(b['cell_efficiency']-a['cell_efficiency']) if support else np.nan,composition_component=.5*(a['cell_efficiency']+b['cell_efficiency'])*(b['cell_weight']-a['cell_weight']) if support else np.nan))
    table(out,'07_benchmark_impact',rows);table(out,'07_response_vs_composition',combined)

def supporting_plots(out):
    import csv
    def read(name):
        with (out/'tables'/f'{name}.csv').open() as f:return list(csv.DictReader(f))
    rows=read('01_angular_response')
    fig,ax=plt.subplots(1,2,figsize=(11,4))
    for a,quantity,label in zip(ax,('energy_response','direction_mrad'),('Mean reco / true energy','Mean direction error [mrad]')):
        for t,c in COLORS.items():
            rr=[r for r in rows if r['sample']==t and r['quantity']==quantity and int(r['N_valid'])>=5]
            a.errorbar([float(r['theta_lo'])+.5 for r in rr],[float(r['mean']) for r in rr],yerr=[float(r['se_event']) for r in rr],color=c,lw=.7,label=t.upper())
        a.set_xlabel('True photon θ [degrees]');a.set_ylabel(label)
    ax[0].legend()
    finish(out,'S01_response_vs_theta',fig,'Where do photon energy and direction responses change?', 'Generator status 1, true energy ≥10 GeV, all parents, clean owned photon PFO. No phi mask. Native spectrum: these curves are not energy-matched. Event-cluster SE; bins with fewer than five valid photons omitted.')
    rows=read('02_radial_profiles');fig,ax=plt.subplots(1,2,figsize=(11,4))
    for a,reg in zip(ax,('barrel','endcap')):
      for t,c in COLORS.items():
        rr=[r for r in rows if r['sample']==t and r['region']==reg and float(r['E_lo'])==20 and float(r['r_hi'])<np.inf]
        for key,ls in [('mean','-'),('pooled_E_fraction','--')]:
            a.plot([float(r['r_lo'])+1.25 for r in rr],[float(r[key]) for r in rr],color=c,ls=ls,label=t.upper()+' '+('equal photon' if key=='mean' else 'pooled energy'))
        a.set_title(reg);a.set_xlabel('Distance from IP-to-centroid axis [mm]');a.set_ylabel('Energy fraction / 2.5 mm ring')
    ax[0].legend(fontsize=7)
    finish(out,'S02_radial_profile',fig,'Does the shower-profile difference depend on weighting?', 'True photon energy 20–30 GeV; status 1, no direct electron parent, clean owned photon PFO, first cluster with ≥2 hits and positive total raw energy. Equal-photon and pooled raw-hit-energy definitions. Overflow ≥50 mm retained in CSV. Descriptive curves.')
    rows=read('05_fixed_population_substitutions');fig,ax=plt.subplots(1,2,figsize=(11,4));fields=['mass','mass_true_angle','mass_true_energy']
    for a,reg in zip(ax,('barrel','endcap')):
      for t,c in COLORS.items():
        rr=[next(r for r in rows if r['sample']==t and r['region']==reg and float(r['E_lo'])==30 and r['fixed_population']=='energy_angle' and float(r['window'])==.04 and r['mass_type']==f) for f in fields]
        a.errorbar(range(3),[float(r['fraction']) for r in rr],yerr=[float(r['se_event']) for r in rr],color=c,marker='o',label=t.upper())
      a.set_xticks(range(3),['Reconstructed','Use true angle','Use true energies']);a.set_title(reg);a.set_ylabel('±40 MeV acceptance')
    ax[0].legend()
    finish(out,'S03_mass_substitutions',fig,'Do energy and angle differences reinforce or compensate?', 'True π⁰ energy 30–50 GeV; TE population before mass/finder; all substitutions use the same finite pair population. Event SE. These are diagnostic mass calculations, not additive causal fractions or demonstrated detector corrections.')
    rows=read('03_stages');fig,ax=plt.subplots(1,2,figsize=(11,4))
    regions=['endcap_tight','endcap','endcap_wide','forward','backward']
    for a,stage in zip(ax,('resolved','window40')):
      for t,c in COLORS.items():
        rr=[next(r for r in rows if r['sample']==t and r['region']==reg and r['stage']==stage and float(r['E_lo'])==30) for reg in regions]
        a.errorbar(range(len(regions)),[float(r['fraction']) for r in rr],yerr=[float(r['se_event']) for r in rr],color=c,marker='o',label=t.upper())
      a.set_xticks(range(5),['17–27°','15–30°','12–33°','Forward','Backward']);a.set_title(stage);a.set_ylabel('Fraction')
    ax[0].legend()
    finish(out,'S04_boundary_robustness',fig,'Does the endcap effect survive angular-boundary changes?', 'True π⁰ energy 30–50 GeV. Both daughters on the same detector side. First three selections use folded θ; forward/backward use the 15–30° core. Resolved denominator: truth pairs. Window denominator: TE. Event SE; overlapping selections are correlated.')

def run(out,bootstrap=200,legacy_dir=None):
    data={}
    for t in COLORS:
        data[t]={}
        for kind in ('photons','pairs','reco_pairs'):
            with np.load(out/'compact'/f'{t}_{kind}.npz',allow_pickle=False) as f:data[t][kind]=f['rows']
    stages_list=[('00 validation',lambda:closure(data,out,legacy_dir)),('01 angular map',lambda:angular(data,out)),('02 single photons',lambda:photons(data,out,bootstrap)),('02b matched photons',lambda:controlled_photons(data,out)),('03 recovery',lambda:stages(data,out)),('04 histograms',lambda:histograms(data,out)),('05 directions and mass',lambda:direction_and_mass(data,out,bootstrap)),('05b controlled mass',lambda:controlled_mass(data,out)),('06 finder',lambda:finder(data,out,legacy_dir)),('07 benchmark impact',lambda:benchmark_impact(data,out)),('08 supporting plots',lambda:supporting_plots(out))]
    for name,action in stages_list:
        print('Starting '+name,flush=True);action()
        (out/'status.json').write_text(json.dumps(dict(stage=name,status='saved')))
        from run_study import pack_share
        pack_share(out)
        print('Saved '+name,flush=True)
