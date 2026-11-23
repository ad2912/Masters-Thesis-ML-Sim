#!/usr/bin/env python3
"""November 11 photon cluster study — standalone revision. (from chatgpt)

Run: python3 -u nov11_cluster_centroid.py
Requires Python 3.9+, NumPy, Matplotlib. No companion scripts needed.
Input: November 6 truth/reco/clusters NPZ triplets and their manifests.
Default output: AFS plots/nov11clustercentroidstudy.
Upload share_here.zip after the run. See study_notes.txt there for definitions.
Optional path overrides: --input DIRECTORY --outdir DIRECTORY

The historical November 6 reducer functions below are preserved verbatim.
Original reducer SHA256:
626da8b0de5d4291315f862aac92cc33c94a7962bd182aa233595f56dc08a1c1
"""

# Historical reducer: preserved functions and constants.
import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import json
from pathlib import Path

import numpy as np

M0 = .1349766
FLOOR = .5 - 1e-6
GROUPS = ('truth', 'reco', 'clusters')
REGIONS = ('barrel', 'barrel crack', 'forward core', 'backward core',
           'same endcap outside core', 'same transition gap', 'mixed / other')
DEFAULT_INPUT = '/data/dust/user/alimuham/thesis/reco/npz-data-summary/full'

def table(data, name, fields):
    return {f: data[f'{name}_{f}'] for f in fields.split()}

def keys(t, pre=''):
    return zip(t['loop'], t[pre+'cid'], t[pre+'idx'])

def index(t):
    result = {k:i for i,k in enumerate(keys(t))}
    if len(result) != len(t['loop']):
        raise ValueError('Duplicate object keys')
    return result

def xyz(t, fields='px py pz'):
    return np.column_stack([t[k] for k in fields.split()])

def norm(v):
    return np.linalg.norm(v, axis=-1)

def unit(v):
    with np.errstate(invalid='ignore', divide='ignore'):
        return v / np.expand_dims(norm(v),-1)

def opening(a,b):
    if not (norm(a)>0 and norm(b)>0):
        return np.nan
    return np.arctan2(norm(np.cross(a,b)), np.dot(a,b))

def mass(e,p,q):
    m2 = float(e)**2 - np.dot(p+q,p+q)
    return np.sqrt(max(0.,m2)) if m2 >= -1e-8 else np.nan

def region(th,ph):
    a,b=th
    if 40<=a<=140 and 40<=b<=140:
        return int(np.any((ph+158.01)%45<=9.01))
    d=np.minimum(th,180-th)
    if (a<90)==(b<90):
        if np.all((d>=15)&(d<30)):
            return 2 if a<90 else 3
        if np.all(d<33): return 4
        if np.all((d>=33)&(d<40)): return 5
    return 6

def rows_array(rows):
    if not rows:
        return None
    integer={'loop','run','event','cid','idx','region','R','T','TE','found','outcome','gap_cross','energy_pass','candidate_count','same_first_cluster','shared_hits','electron_parent','pi0_parent','alternate_distinct_clean','owned_clean','best_clean','owned_pdg','best_pdg'}
    dtype=[(k,'i8' if k in integer or k.endswith(('_cid','_idx','_fast','_gen')) or k=='fast' else 'f8') for k in rows[0]]
    return np.array([tuple(r.values()) for r in rows],dtype=dtype)

def open_part(folder, part, group, manifest, verify, stack):
    label=manifest['sample']
    filename=f'{label}_{group}_{part["part"]:05d}.npz'
    rec=next(r for r in part['files'] if r['file']==filename)
    path=folder/filename
    if path.stat().st_size!=rec['bytes']:
        raise ValueError(f'File size differs from manifest: {path}')
    if verify:
        digest=hashlib.sha256()
        with path.open('rb') as f:
            for block in iter(lambda:f.read(1<<20),b''): digest.update(block)
        if digest.hexdigest()!=rec['sha256']:
            raise ValueError(f'Hash mismatch: {path}')
    obj=stack.enter_context(np.load(path,allow_pickle=False))
    if obj['meta_sample'].item()!=label or obj['meta_schema_version'].item()!=1:
        raise ValueError(f'Wrong sample or schema: {path}')
    for name,n in rec['rows'].items():
        for token in manifest['schemas'][name].split():
            field=token.split(':')[0]
            if f'{name}_{field}' in obj.files and len(obj[f'{name}_{field}'])!=n:
                raise ValueError(f'Column length differs in {name}_{field}: {path}')
            if f'{name}_{field}' not in obj.files:
                raise ValueError(f'Missing field {name}_{field}: {path}')
        if len(obj[f'{name}_loop'])!=n:
            raise ValueError(f'Row count mismatch in {name}: {path}')
    return obj

def project_pair(u,v,ra,rb):
    mid,axis=unit(u+v),unit(u-v)
    normal=np.cross(mid,axis)
    def proj(x): return np.arctan2(np.dot(x,axis),np.dot(x,mid))
    return ((proj(ra)-proj(u))*1000,(proj(v)-proj(rb))*1000,
            np.arcsin(np.clip(np.dot(ra,normal),-1,1))*1000,
            np.arcsin(np.clip(np.dot(rb,normal),-1,1))*1000)

class Clusters:
    def __init__(self,data,reco,wanted,with_hits):
        self.c = table(data,'cluster','loop cid idx n_hits hit_E_sum mean_x mean_y mean_z mean_xx mean_yy mean_zz mean_xy mean_xz mean_yz')
        ci=index(self.c)
        rel=table(reco,'pfo_cluster','loop pfo_cid pfo_idx order cluster_cid cluster_idx')
        self.first={}; self.ncluster=Counter(); self.all_rows=defaultdict(list)
        for pk,ck,order in zip(keys(rel,'pfo_'),keys(rel,'cluster_'),rel['order']):
            self.ncluster[pk]+=1
            self.all_rows[pk].append(ci[ck])
            row=ci[ck]
            if order==0:
                if pk in self.first: raise ValueError('Duplicate first cluster')
                self.first[pk]=row
        self.center=xyz(self.c,'mean_x mean_y mean_z')
        m=self.c
        self.second=np.stack([m[k] for k in ('mean_xx','mean_xy','mean_xz','mean_xy','mean_yy','mean_yz','mean_xz','mean_yz','mean_zz')],axis=1).reshape(-1,3,3)
        self.hits={}
        if not with_hits or not wanted:
            return
        self.load_hits(data, {row for pk in wanted for row in self.all_rows[pk]})

    def load_hits(self,data,wanted_rows):
        # Filter membership using full keys. Numeric structured keys avoid millions of Python tuples.
        dtype=np.dtype([(n,'i8') for n in ('loop','cid','idx')])
        def pack(t,prefix=''):
            out=np.empty(len(t['loop']),dtype=dtype)
            for k in dtype.names: out[k]=t[k if k=='loop' else prefix+k]
            return out
        membership=table(data,'cluster_hit','loop cluster_cid cluster_idx hit_cid hit_idx')
        selected=np.array(sorted(wanted_rows),dtype=int)
        if not len(selected): return
        ck=pack(self.c)[selected]
        order=np.argsort(ck)
        ck=ck[order];selected=selected[order]
        rel=pack(membership,'cluster_')
        i=np.searchsorted(ck,rel).clip(0,len(ck)-1)
        keep=ck[i]==rel
        ci=selected[i[keep]]
        hit_key=pack(membership,'hit_')[keep]
        hit=table(data,'hit','loop cid idx E x y z')
        hk=pack(hit)
        order=np.argsort(hk)
        sh=hk[order]
        if len(sh)>1 and np.any(sh[1:]==sh[:-1]):raise ValueError('Duplicate hit identity')
        j=np.searchsorted(sh,hit_key)
        if (j>=len(sh)).any() or not np.all(sh[j]==hit_key):
            raise ValueError('Dangling selected cluster-hit reference')
        ids=order[j]
        order=np.argsort(ci,kind='stable')
        ci,ids=ci[order],ids[order]
        # Check selected membership against the stored cluster hit count.
        unique,start,count=np.unique(ci,return_index=True,return_counts=True)
        if not np.array_equal(count,self.c['n_hits'][unique]):
            raise ValueError('Cluster hit counts disagree')
        for row,a,n in zip(unique,start,count):
            ix=ids[a:a+n]
            self.hits[int(row)]=(ix,hit['E'][ix],np.column_stack([hit[k][ix] for k in ('x','y','z')]))

    def measure(self,pk,u,vertex,partner):
        row=self.first.get(pk,-1)
        values=dict(cluster=row,cluster_cid=-1,cluster_idx=-1,n_clusters=self.ncluster[pk],n_hits=0)
        names=('width_ip width_truth width_centered centroid_offset depth_mean depth_spread '
               'centroid_out centroid_perp centroid_x centroid_y centroid_z side_asym rotated_asym '
               'theta_asym phi_asym positive_hit_E negative_hits raw_hit_E R50 R90 core_fraction '
               'tail_fraction all_width_centered all_centroid_offset all_centroid_out all_R90 '
               'all_unique_hits raw_width_closure width_identity_error').split()
        values.update({k:np.nan for k in names})
        for j in range(21): values[f'ring_E_{j:02d}']=np.nan
        if row<0: return values
        values['cluster_cid']=int(self.c['cid'][row]); values['cluster_idx']=int(self.c['idx'][row])
        c,second=self.center[row],self.second[row]
        values['n_hits']=int(self.c['n_hits'][row])
        cov=second-np.outer(c,c); disp=c-vertex
        if self.c['hit_E_sum'][row]>0 and np.isfinite(c).all() and np.isfinite(u).all():
            values['raw_hit_E']=self.c['hit_E_sum'][row]
            if values['n_hits']>=2 and norm(c)>0:
                axis=unit(c); values['width_ip']=np.sqrt(max(0.,np.trace(second)-axis@second@axis))
            vcenter=np.trace(cov)-u@cov@u
            offset2=np.dot(disp,disp)-(disp@u)**2
            if vcenter>=-1e-6:
                values['width_centered']=np.sqrt(max(0.,vcenter))
            values['centroid_offset']=np.sqrt(max(0.,offset2))
            centered=second-np.outer(c,vertex)-np.outer(vertex,c)+np.outer(vertex,vertex)
            vt=np.trace(centered)-u@centered@u
            values['width_truth']=np.sqrt(max(0.,vt)) if vt>=-1e-6 else np.nan
            values['width_identity_error']=vt-vcenter-offset2
            values['depth_mean']=disp@u
            values['depth_spread']=np.sqrt(max(0.,u@cov@u))
            direction=unit(disp)
            values.update(zip(('centroid_x','centroid_y','centroid_z'),direction))
            if partner is not None and norm(np.cross(u,partner))>1e-12:
                outward=unit(u*np.dot(u,partner)-partner); perp=np.cross(u,outward)
                values['centroid_out']=1000*np.arctan2(disp@outward,disp@u)
                values['centroid_perp']=1000*np.arctan2(disp@perp,disp@u)
        if row not in self.hits: return values
        ids,w,x=self.hits[row]
        values['negative_hits']=int((w<0).sum())
        valid=np.isfinite(w)&np.isfinite(x).all(1)
        # Historical radial profile uses signed raw energies, exactly as V6.
        if valid.all() and w.sum()>0 and norm(c)>0 and len(w)>=2:
            axis=unit(c); d=np.sqrt(np.maximum(0,np.sum(x*x,axis=1)-(x@axis)**2))
            values['raw_width_closure']=np.sqrt(max(0,np.dot(w,d*d)/w.sum()))-values['width_ip']
            hist=np.histogram(d,bins=np.r_[np.arange(0,52.5,2.5),np.inf],weights=w)[0]
            values.update({f'ring_E_{j:02d}':v for j,v in enumerate(hist)})
            values['core_fraction']=w[d<9].sum()/w.sum()
            values['tail_fraction']=w[d>=25].sum()/w.sum()
            if np.all(w>=0):
                order=np.argsort(d); cum=np.cumsum(w[order])/w.sum()
                for q,name in ((.5,'R50'),(.9,'R90')):
                    values[name]=d[order[min(np.searchsorted(cum,q),len(d)-1)]]
        ok=valid&(w>0); total=w[ok].sum(); values['positive_hit_E']=total
        if total>0 and np.isfinite(u).all():
            dx=x[ok]-vertex
            theta_axis=np.array([u[2]*np.cos(np.arctan2(u[1],u[0])),u[2]*np.sin(np.arctan2(u[1],u[0])),-np.hypot(u[0],u[1])])
            phi_axis=np.array([-np.sin(np.arctan2(u[1],u[0])),np.cos(np.arctan2(u[1],u[0])),0.])
            axes={'theta_asym':theta_axis,'phi_asym':phi_axis}
            if partner is not None and norm(np.cross(u,partner))>1e-12:
                outward=unit(u*np.dot(u,partner)-partner)
                axes.update(side_asym=outward,rotated_asym=np.cross(u,outward))
            for name,axis in axes.items(): values[name]=np.dot(w[ok],np.sign(dx@axis))/total
        # All-associated-cluster control deduplicates hits by full event-local hit identity.
        bundles=[self.hits[r] for r in self.all_rows[pk] if r in self.hits]
        if bundles:
            ids=np.concatenate([b[0] for b in bundles]); ww=np.concatenate([b[1] for b in bundles]); xx=np.concatenate([b[2] for b in bundles])
            _,ix=np.unique(ids,return_index=True); ww,xx=ww[ix],xx[ix]
            values['all_unique_hits']=len(ix)
            good=(ww>0)&np.isfinite(ww)&np.isfinite(xx).all(1); ww,xx=ww[good],xx[good]
            if len(ww) and np.isfinite(u).all():
                cc=np.average(xx,axis=0,weights=ww); dd=xx-cc
                radial=np.sqrt(np.maximum(0,np.sum(dd*dd,axis=1)-(dd@u)**2))
                values['all_width_centered']=np.sqrt(np.average(radial**2,weights=ww))
                dd=cc-vertex; values['all_centroid_offset']=norm(dd-u*(dd@u))
                if partner is not None and norm(np.cross(u,partner))>1e-12:
                    outward=unit(u*np.dot(u,partner)-partner)
                    values['all_centroid_out']=1000*np.arctan2(dd@outward,dd@u)
                order=np.argsort(radial);cum=np.cumsum(ww[order])/ww.sum()
                values['all_R90']=radial[order[min(np.searchsorted(cum,.9),len(radial)-1)]]
        return values

def analyse_part(tr,rr,cl,sample,counters,with_hits):
    m=table(tr,'mc','loop cid idx pdg gen fast E charge px py pz vx vy vz ex ey ez')
    p=table(rr,'pfo','loop cid idx pdg E px py pz charge')
    mi,pi=index(m),index(p)
    mk,pk=list(keys(m)),list(keys(p))
    mp,pp,vertex=xyz(m),xyz(p),xyz(m,'vx vy vz')
    u=unit(mp)
    theta=np.degrees(np.arctan2(norm(mp[:,:2]),mp[:,2]))
    theta[norm(mp)==0]=np.nan
    phi=np.degrees(np.arctan2(mp[:,1],mp[:,0]))
    ancestry=table(tr,'ancestry','loop child_cid child_idx parent_cid parent_idx')
    parents=defaultdict(list); daughters=defaultdict(list)
    for child,parent in zip(keys(ancestry,'child_'),keys(ancestry,'parent_')):
        a,b=mi[child],mi[parent]
        parents[a].append(b); daughters[b].append(a)
    links=table(rr,'links','loop pfo_cid pfo_idx mc_cid mc_idx cw tw raw')
    by_p=defaultdict(list); by_m=defaultdict(list); seen=set()
    for a,b,cw,tw,raw in zip(keys(links,'pfo_'),keys(links,'mc_'),links['cw'],links['tw'],links['raw']):
        i,j=pi[a],mi[b]
        if not np.isfinite([cw,tw,raw]).all():raise ValueError('Nonfinite truth link weight')
        packed=int(round(raw))
        if not (np.isclose(cw,(packed//10000)/1000) and np.isclose(tw,(packed%10000)/1000)):
            raise ValueError('Stored link weight disagrees with packed value')
        if (i,j) in seen:
            counters['duplicate_links']+=1
            continue
        seen.add((i,j)); by_m[j].append((i,cw));by_p[i].append((j,cw,tw))
    owner=defaultdict(list); reco_parent=defaultdict(set); reco_groups=defaultdict(list)
    for i,ls in by_p.items():
        if p['pdg'][i]!=22: continue
        # Single-photon V6 kept the first link on a cluster-weight tie.
        owned_j,owned_cw,_=max(ls,key=lambda x:x[1])
        if owned_cw>=.5:owner[owned_j].append((i,owned_cw))
        # Historical reco-first pion ranking also breaks ties by track weight.
        j,cw,_=min(ls,key=lambda x:(-x[1],-x[2],mk[x[0]][2],mk[x[0]][1]))
        if cw<FLOOR:continue
        if m['pdg'][j]==22:
            parent=next((a for a in parents[j] if m['pdg'][a]==111),None)
            if parent is not None:
                reco_parent[parent].add(j); reco_groups[parent].append((i,j,cw))
    candidates=table(rr,'candidate','loop cid idx goodness n_daughters E px py pz')
    ci=index(candidates)
    cd=table(rr,'candidate_daughter','loop candidate_cid candidate_idx pfo_cid pfo_idx')
    cands=defaultdict(list)
    for ck,pf in zip(keys(cd,'candidate_'),keys(cd,'pfo_')):
        cands[ci[ck]].append(pi[pf])
    found=defaultdict(list)
    for j,n in enumerate(candidates['n_daughters']):
        ids=cands[j]
        if len(ids)!=n:raise ValueError('Candidate daughter count mismatch')
        if n==2 and ids[0]!=ids[1]:found[tuple(sorted(ids))].append(j)
    photons=np.flatnonzero((m['pdg']==22)&(m['gen']==1))
    best={}
    for j in photons:
        ls=by_m[j]
        best[j]=min(ls,key=lambda x:(-x[1]*p['E'][x[0]],-x[1],pk[x[0]][2],pk[x[0]][1])) if ls else (-1,np.nan)
    pair_defs=[]
    for parent in np.flatnonzero(m['pdg']==111):
        ds=daughters[parent]
        if len(ds)==2 and all(m['pdg'][j]==22 and m['gen'][j]==1 for j in ds):
            a,b=sorted(ds,key=lambda j:(-m['E'][j],mk[j][2],mk[j][1]))
            pair_defs.append((parent,a,b))
    def clean(i,w):
        return i>=0 and w>=FLOOR and p['pdg'][i]==22
    needed={pk[i] for ls in owner.values() for i,w in ls}
    for parent,a,b in pair_defs:
        i,wa=best[a];j,wb=best[b]
        if i!=j and clean(i,wa) and clean(j,wb) and min(p['E'][i],p['E'][j])>=.5:
            needed.update((pk[i],pk[j]))
    clusters=Clusters(cl,rr,needed,with_hits)
    event=table(tr,'events','loop run event')
    event_key=dict(zip(event['loop'],zip(event['run'],event['event'])))
    # Neighbour activity is based on all generator photons and charged generator particles,
    # not on reconstructed neighbours; missing neighbours are +infinity, not zero.
    neighbours={}
    for evloop in np.unique(m['loop'][photons]):
        aa=photons[m['loop'][photons]==evloop]
        charged=np.flatnonzero((m['loop']==evloop)&(m['gen']==1)&(np.abs(m['charge'])>0))
        for a in aa:
            other=aa[aa!=a]
            angles=1000*np.arctan2(norm(np.cross(u[a],u[other])),u[other]@u[a])
            cha=1000*np.arctan2(norm(np.cross(u[a],u[charged])),u[charged]@u[a])
            neighbours[a]=dict(nearest_photon_mrad=float(np.min(angles)) if len(angles) else np.inf,
                nearest_charged_mrad=float(np.min(cha)) if len(cha) else np.inf,
                photon_E_cone20=float(m['E'][other][angles<20].sum()),
                charged_E_cone20=float(m['E'][charged][cha<20].sum()))
    photon_rows=[];pair_rows=[];reco_rows=[]
    for a in photons:
        i,w=best[a]; own,ow=max(owner[a],key=lambda x:p['E'][x[0]]) if owner[a] else (-1,np.nan)
        ps=parents[a]; ds=daughters[a]
        conversion=any(m['pdg'][d]==11 for d in ds) and any(m['pdg'][d]==-11 for d in ds)
        r=dict(loop=mk[a][0],cid=mk[a][1],idx=mk[a][2],run=event_key[mk[a][0]][0],event=event_key[mk[a][0]][1],
               E=m['E'][a],theta=theta[a],phi=phi[a],fast=m['fast'][a],
               electron_parent=int(any(abs(m['pdg'][j])==11 for j in ps)),
               pi0_parent=int(any(m['pdg'][j]==111 for j in ps)),conversion_daughters=int(conversion),
               end_r=np.hypot(m['ex'][a],m['ey'][a]),end_z=m['ez'][a],
               n_links=len(by_m[a]),n_links_cw05=sum(cw>=FLOOR for _,cw in by_m[a]))
        r.update(neighbours[a])
        for name,k,cw in [('best',i,w),('owned',own,ow)]:
            r.update({name+'_clean':int(clean(k,cw)),name+'_cw':cw,name+'_pdg':int(p['pdg'][k]) if k>=0 else 0,
                      name+'_cid':int(pk[k][1]) if k>=0 else -1,name+'_idx':int(pk[k][2]) if k>=0 else -1,
                      name+'_theta':np.degrees(np.arctan2(norm(pp[k,:2]),pp[k,2])) if k>=0 else np.nan,name+'_phi':np.degrees(np.arctan2(pp[k,1],pp[k,0])) if k>=0 else np.nan,
                      name+'_E':p['E'][k] if k>=0 else np.nan,name+'_direction_mrad':1000*opening(mp[a],pp[k]) if k>=0 else np.nan})
        shape=clusters.measure(pk[own] if own>=0 else None,u[a],vertex[a],None)
        r.update(shape)
        photon_rows.append(r)
    for parent,a,b in pair_defs:
        i,wa=best[a];j,wb=best[b]
        t=clean(i,wa) and clean(j,wb) and i!=j
        te=t and min(p['E'][i],p['E'][j])>=.5
        at=opening(mp[a],mp[b]); ar=opening(pp[i],pp[j]) if t else np.nan
        measured=mass(p['E'][i]+p['E'][j],pp[i],pp[j]) if t else np.nan
        cids=found.get(tuple(sorted((i,j))),[]) if t else []
        fit_values=candidates['goodness'][cids]
        valid=fit_values[np.isfinite(fit_values)]
        loop,cid,idx=mk[parent]
        fold=np.minimum(theta[[a,b]],180-theta[[a,b]])
        gap=(fold>=33)&(fold<40)
        r=dict(loop=loop,cid=cid,idx=idx,gap_cross=int(gap.any() and region(theta[[a,b]],phi[[a,b]])!=5),run=event_key[loop][0],event=event_key[loop][1],
               E=m['E'][parent],region=region(theta[[a,b]],phi[[a,b]]),
               R=int(len(reco_parent[parent])>=2),T=int(t),TE=int(te),found=int(bool(cids)),
               candidate_count=len(cids),goodness=np.max(valid) if len(valid) else np.nan,
               mass=measured,alpha_true_mrad=at*1000,alpha_reco_mrad=ar*1000,
               mass_true_angle=2*np.sqrt(p['E'][i]*p['E'][j])*np.sin(at/2) if te else np.nan,
               mass_true_energy=2*np.sqrt(m['E'][a]*m['E'][b])*np.sin(ar/2) if te else np.nan,
               mass_massless=2*np.sqrt(p['E'][i]*p['E'][j])*np.sin(ar/2) if te else np.nan)
        has_a=i>=0 and wa>=FLOOR; has_b=j>=0 and wb>=FLOOR
        r['outcome']=(0 if not has_a and not has_b else 1 if has_a != has_b else 2 if i==j else 3 if not t else 4)
        r['alternate_distinct_clean']=int(any(ia!=ib and clean(ia,ca) and clean(ib,cb) for ia,ca in by_m[a] for ib,cb in by_m[b]))
        r['truth_mass']=2*np.sqrt(m['E'][a]*m['E'][b])*np.sin(at/2)
        r['true_energy_sum']=m['E'][a]+m['E'][b]
        dirs=project_pair(u[a],u[b],unit(pp[i]),unit(pp[j])) if te else (np.nan,)*4
        for side,k,f,w,other,out,perp in [('A',a,i,wa,b,dirs[0],dirs[2]),('B',b,j,wb,a,dirs[1],dirs[3])]:
            values=dict(true_cid=mk[k][1],true_idx=mk[k][2],true_E=m['E'][k],theta=theta[k],phi=phi[k],fast=m['fast'][k],
                        reco_cid=pk[f][1] if f>=0 else -1,reco_idx=pk[f][2] if f>=0 else -1,
                        reco_charge=p['charge'][f] if f>=0 else np.nan,reco_E=p['E'][f] if f>=0 else np.nan,reco_theta=np.arctan2(norm(pp[f,:2]),pp[f,2]) if f>=0 else np.nan,reco_phi=np.arctan2(pp[f,1],pp[f,0]) if f>=0 else np.nan,cw=w,clean=int(clean(f,w)),out_mrad=out,perp_mrad=perp,
                        conversion_daughters=int(any(m['pdg'][d]==11 for d in daughters[k]) and any(m['pdg'][d]==-11 for d in daughters[k])),
                        end_r=np.hypot(m['ex'][k],m['ey'][k]),end_z=m['ez'][k])
            shape=clusters.measure(pk[f] if te else None,u[k],vertex[k],u[other])
            values.update(neighbours[k]); values['n_links_cw05']=sum(cw>=FLOOR for _,cw in by_m[k])
            values.update(shape)
            r.update({side+'_'+key:value for key,value in values.items()})
        ca,cb=r['A_cluster'],r['B_cluster']
        r['same_first_cluster']=int(ca>=0 and ca==cb)
        r['shared_hits']=(len(np.intersect1d(clusters.hits[ca][0],clusters.hits[cb][0])) if ca in clusters.hits and cb in clusters.hits else -1)
        ca_dir=np.array([r['A_centroid_'+v] for v in ('x','y','z')]); cb_dir=np.array([r['B_centroid_'+v] for v in ('x','y','z')])
        ac=opening(ca_dir,cb_dir)
        r['alpha_centroid_mrad']=1000*ac
        r['mass_centroid']=2*np.sqrt(p['E'][i]*p['E'][j])*np.sin(ac/2) if te else np.nan
        pair_rows.append(r)
    from itertools import combinations
    for parent,ls in reco_groups.items():
        for (i,a,wa),(j,b,wb) in combinations(ls,2):
            if a==b: continue
            loop,cid,idx=mk[parent]
            reco_rows.append(dict(loop=loop,run=event_key[loop][0],event=event_key[loop][1],cid=cid,idx=idx,
                E=m['E'][parent],region=region(theta[[a,b]],phi[[a,b]]),
                A_true_idx=mk[a][2],B_true_idx=mk[b][2],A_reco_idx=pk[i][2],B_reco_idx=pk[j][2],
                A_reco_cid=pk[i][1],B_reco_cid=pk[j][1],
                A_gen=m['gen'][a],B_gen=m['gen'][b],
                energy_pass=int(min(p['E'][i],p['E'][j])>=.5),
                mass=mass(p['E'][i]+p['E'][j],pp[i],pp[j]),found=int(tuple(sorted((i,j))) in found)))
    counters['events']+=len(event['loop'])
    counters['generator_photons']+=len(photons)
    counters['generator_photons_Ege10']+=int(np.sum(m['E'][photons]>=10))
    counters['generator_photons_Ege10_no_e_parent']+=sum(r['E']>=10 and not r['electron_parent'] for r in photon_rows)
    counters['truth_pi0_pairs']+=len(pair_rows)
    counters['reco_first_all_parents']+=sum(len(v)>=2 for v in reco_parent.values())
    counters['reco_first_generator_parents']+=sum(r['R'] for r in pair_rows)
    counters['generator_resolved']+=sum(r['T'] for r in pair_rows)
    counters['generator_resolved_E']+=sum(r['TE'] for r in pair_rows)
    counters['generator_found']+=sum(r['found'] for r in pair_rows)
    counters['invalid_resolved_mass']+=sum(r['TE'] and not np.isfinite(r['mass']) for r in pair_rows)
    counters['found_outside_40']+=sum(r['found'] and abs(r['mass']-M0)>.04 for r in pair_rows)
    return rows_array(photon_rows),rows_array(pair_rows),rows_array(reco_rows)


# Current plotting and ownership audit.
from contextlib import ExitStack
import csv
import hashlib
import json
from pathlib import Path
import shutil
import time
from zipfile import ZipFile, ZIP_DEFLATED

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

EDGES = np.array([10, 20, 30, 40, 60, 80, 100, 125])
QUANTITIES = ("width_ip", "width_centered", "centroid_offset")
REGIONS = ("Full detector", "Barrel", "Endcap")
POPULATIONS = ("All selected photons", "π⁰ daughter photons")
DEFAULT_OUT = "/afs/desy.de/user/a/alimuham/thesis-ml-sim/plots/nov11clustercentroidstudy"


def digest(path):
    checksum = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            checksum.update(block)
    return checksum.hexdigest()


def save_json(path, value):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False, default=lambda x: x.item() if isinstance(x, np.generic) else str(x)))
    temporary.replace(path)


def write_csv(path, rows):
    if not rows:
        path.write_text("")
        return
    if rows:
        with path.open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=rows[0])
            writer.writeheader()
            writer.writerows(rows)


def regions(photon):
    theta, phi = photon["theta"], photon["phi"]
    fold = min(theta, 180 - theta)
    return (True, 40 <= theta <= 140 and (phi + 158.01) % 45 > 9.01,
            15 <= fold < 30)


def association_graph(truth, reco):
    mc = table(truth, "mc", "loop cid idx pdg gen fast sim_status vx vy vz ex ey ez")
    lookup = index(mc)
    children = defaultdict(list)
    relations = table(truth, "ancestry", "loop child_cid child_idx parent_cid parent_idx")
    for child, parent in zip(keys(relations, "child_"), keys(relations, "parent_")):
        children[lookup[parent]].append(lookup[child])
    pfo = table(reco, "pfo", "loop cid idx pdg E")
    pfo_lookup = index(pfo)
    links = table(reco, "links", "loop pfo_cid pfo_idx mc_cid mc_idx cw")
    by_mc = defaultdict(list)
    seen_links = set()
    for pk, mk, cw in zip(keys(links, "pfo_"), keys(links, "mc_"), links["cw"]):
        if (pk, mk) in seen_links:
            continue
        seen_links.add((pk, mk))
        by_mc[lookup[mk]].append((pfo_lookup[pk], float(cw)))
    return mc, lookup, children, pfo, by_mc


def trace(photon, graph):
    mc, lookup, children, pfo, links = graph
    origin = lookup[tuple(int(photon[k]) for k in ("loop", "cid", "idx"))]
    visited = {origin}
    descendants = []
    pending = list(children[origin])
    while pending:
        node = pending.pop()
        if node in visited:
            continue
        visited.add(node)
        descendants.append(node)
        pending.extend(children[node])
    direct = links[origin]
    related = [link for node in descendants for link in links[node]]
    strong = {i for i, cw in related if cw >= .5}
    return {
        "original_sim_status": int(mc["sim_status"][origin]),
        "has_direct_link": bool(direct),
        "has_direct_strong_link": any(cw >= .5 for _, cw in direct),
        "has_direct_strong_photon_link": any(cw >= .5 and pfo["pdg"][i] == 22 for i, cw in direct),
        "has_descendant": bool(descendants),
        "has_fast_descendant": any(mc["fast"][i] == 1 for i in descendants),
        "has_unknown_descendant_flag": any(mc["fast"][i] not in (0, 1) for i in descendants),
        "has_any_descendant_link": bool(related),
        "has_strong_descendant_link": bool(strong),
        "has_strong_descendant_photon_link": any(pfo["pdg"][i] == 22 for i in strong),
        "has_strong_descendant_electron_link": any(abs(pfo["pdg"][i]) == 11 for i in strong),
        "descendant_count": len(descendants),
        "descendant_records": [
            {"cid": int(mc["cid"][node]), "idx": int(mc["idx"][node]),
             "pdg": int(mc["pdg"][node]), "fast": int(mc["fast"][node]),
             "links": [{"pfo_cid": int(pfo["cid"][i]), "pfo_idx": int(pfo["idx"][i]),
                        "pdg": int(pfo["pdg"][i]), "E": float(pfo["E"][i]), "cw": cw}
                       for i, cw in links[node]]}
            for node in descendants[:40]],
        "descendant_records_truncated": len(descendants) > 40,
        "descendant_pdgs": sorted({int(mc["pdg"][i]) for i in descendants}),
        "descendant_pfo_pdgs": sorted({int(pfo["pdg"][i]) for i in strong}),
        "direct_links": [{"pfo_cid": int(pfo["cid"][i]), "pfo_idx": int(pfo["idx"][i]),
                          "pdg": int(pfo["pdg"][i]), "E": float(pfo["E"][i]), "cw": cw}
                         for i, cw in direct],
    }


def failure_reason(photon, evidence):
    if photon["owned_clean"] == 1:
        return "owned photon"
    if not evidence["has_direct_link"]:
        return "no direct truth link"
    if not evidence["has_direct_strong_link"]:
        return "direct links below 0.5"
    if not evidence["has_direct_strong_photon_link"]:
        return "strong links only to non-photon PFOs"
    return "strong photon link but another truth particle owns PFO"


def add_moments(bucket, values, events):
    valid = np.isfinite(values)
    values, events = values[valid], events[valid]
    if not len(values):
        return
    _, indices = np.unique(events, return_inverse=True)
    sums = np.bincount(indices, weights=values)
    counts = np.bincount(indices)
    bucket += np.array([len(values), len(sums), values.sum(), sums @ sums,
                        sums @ counts, counts @ counts], dtype=float)


def process(truth, reco, clusters, sample):
    counters = Counter()
    photons, pairs, _ = analyse_part(truth, reco, clusters, sample, counters, False)
    graph = association_graph(truth, reco)
    pair_daughters = set()
    if pairs is not None:
        for side in ("A", "B"):
            pair_daughters.update(zip(pairs["loop"], pairs[side + "_true_cid"], pairs[side + "_true_idx"]))
    bins = defaultdict(list)
    audit = defaultdict(Counter)
    examples = []
    example_counts = Counter()
    flags = Counter()
    checks = Counter()
    if photons is None:
        return dict(bins=[], examples=[], counters=dict(counters), flags={}, checks={})
    for photon in photons:
        flags[str(int(photon["fast"]))] += 1
        if not (10 <= photon["E"] < 125 and photon["electron_parent"] == 0):
            continue
        if not np.isfinite([photon["theta"], photon["phi"]]).all():
            checks["invalid_truth_angles"] += 1
            continue
        evidence = trace(photon, graph)
        reason = failure_reason(photon, evidence)
        daughter = tuple(photon[k] for k in ("loop", "cid", "idx")) in pair_daughters
        bin_index = int(np.searchsorted(EDGES, photon["E"], side="right") - 1)
        groups = ["G4"] if sample == "g4" else ["CC3 hybrid", {1: "CC3 fast", 0: "CC3 unflagged"}.get(int(photon["fast"]), "CC3 unknown")]
        for region, accepted in zip(REGIONS, regions(photon)):
            if not accepted:
                continue
            for population in POPULATIONS[:1 + int(daughter)]:
                for group in groups:
                    key = (region, population, group, bin_index)
                    bins[key].append(photon)
                    count = audit[key]
                    count["truth"] += 1
                    count["owned"] += int(photon["owned_clean"] == 1)
                    count["best_clean"] += int(photon["best_clean"] == 1)
                    count["first_cluster"] += int(photon["owned_clean"] == 1 and photon["cluster_idx"] >= 0)
                    count["direct_conversion_indicator"] += int(photon["conversion_daughters"] == 1)
                    count["zero_endpoint"] += int(photon["end_r"] == 0 and photon["end_z"] == 0)
                    for name, value in evidence.items():
                        if name.startswith("has_"):
                            count[name] += int(value)
                    count["failure_" + reason] += 1
                    if photon["owned_clean"] != 1:
                        count["failed_with_descendant_link"] += int(evidence["has_strong_descendant_link"])
                        count["failed_with_fast_descendant"] += int(evidence["has_fast_descendant"])
            example_key = (region, int(photon["fast"]), reason,
                           daughter and region == "Endcap" and bin_index == 1)
            if example_counts[example_key] < 1:
                example_counts[example_key] += 1
                detail = {k: float(photon[k]) for k in ("E", "theta", "phi", "end_r", "end_z")}
                detail.update({k: int(photon[k]) for k in ("loop", "run", "event", "cid", "idx", "fast", "owned_clean", "best_clean")})
                # JSON null records missing numeric values explicitly.
                detail = {k: v if not isinstance(v, float) or np.isfinite(v) else None for k, v in detail.items()}
                examples.append(dict(sample=sample, region=region, pi0_daughter=daughter, reason=reason, photon=detail, evidence=evidence))
        if photon["owned_clean"] == 1:
            identity = photon["width_identity_error"]
            if np.isfinite(identity):
                checks["identity_checked"] += 1
                checks["identity_failed"] += int(abs(identity) > 1e-5)
    output = []
    for key, entries in bins.items():
        rows = np.array(entries, dtype=photons.dtype)
        clean = rows["owned_clean"] == 1
        state = np.zeros(6)
        add_moments(state, clean.astype(float), rows["loop"])
        moments = {"owned_fraction": state.tolist()}
        for quantity in QUANTITIES:
            state = np.zeros(6)
            add_moments(state, rows[quantity][clean], rows["loop"][clean])
            moments[quantity] = state.tolist()
        state = np.zeros(6)
        add_moments(state, (rows["width_ip"] - rows["width_centered"])[clean], rows["loop"][clean])
        moments["width_ip_minus_centered"] = state.tolist()
        output.append(dict(key=list(key), counts=dict(audit[key]), moments=moments))
    return dict(bins=output, examples=examples, counters=dict(counters), flags=dict(flags), checks=dict(checks))


def summarize(parts):
    counts = defaultdict(Counter)
    moments = defaultdict(lambda: np.zeros(6))
    examples = []
    for part in parts:
        examples.extend(part["examples"])
        for row in part["bins"]:
            key = tuple(row["key"])
            counts[key].update(row["counts"])
            for quantity, state in row["moments"].items():
                moments[key + (quantity,)] += state
    names = sorted({name for count in counts.values() for name in count})
    audit_rows, response_rows = [], []
    for key, count in sorted(counts.items()):
        region, population, sample, index = key
        labels = dict(region=region, population=population, sample=sample,
                      energy_lo=int(EDGES[index]), energy_hi=int(EDGES[index + 1]))
        if sum(v for k, v in count.items() if k.startswith("failure_")) != count["truth"]:
            raise ValueError("Ownership failure categories do not sum to truth")
        audit_rows.append(dict(**labels, **{name: count[name] for name in names}))
        for quantity in (*QUANTITIES, "width_ip_minus_centered", "owned_fraction"):
            n, ne, total, sum2, sum_n, n2 = moments[key + (quantity,)]
            mean = total / n if n else np.nan
            variance = max(0., sum2 - 2 * mean * sum_n + mean * mean * n2)
            se = np.sqrt(ne / (ne - 1) * variance) / n if ne > 1 else np.nan
            response_rows.append(dict(**labels, quantity=quantity, n_truth=count["truth"],
                                      n_owned=count["owned"], n_valid=int(n), n_events=int(ne),
                                      mean=mean, se_event=se))
    return audit_rows, response_rows, examples


def comparisons(rows):
    """Descriptive native-sample differences; SE assumes independent G4/CC3."""
    lookup = {(r['region'], r['population'], r['energy_lo'], r['quantity'], r['sample']): r for r in rows}
    result = []
    for row in rows:
        if row['sample'] not in ('CC3 hybrid', 'CC3 fast') or row['quantity'] not in ('width_ip', 'centroid_offset'):
            continue
        ref = lookup.get((row['region'], row['population'], row['energy_lo'], row['quantity'], 'G4'))
        if ref is None:
            continue
        result.append(dict(region=row['region'], population=row['population'], sample=row['sample'],
            energy_lo=row['energy_lo'], energy_hi=row['energy_hi'], quantity=row['quantity'],
            difference_mm=row['mean']-ref['mean'],
            relative_difference_percent=100*(row['mean']/ref['mean']-1) if ref['mean'] else np.nan,
            difference_se_independent_samples_mm=np.hypot(row['se_event'], ref['se_event']),
            n_valid_cc3=row['n_valid'], n_valid_g4=ref['n_valid'],
            interpretation='Native recovered samples; eligibility and survival may differ; cross-sample covariance unknown'))
    return result


def plots(rows, output):
    styles = [('G4', 'Geant4', '#2166ac', 'o', '-'),
              ('CC3 hybrid', 'CC3 inclusive', '#d6604d', 's', '-'),
              ('CC3 fast', 'CC3 flagged', '#d6604d', '^', '--')]
    for quantity, title, ylabel, filename in (
        ('width_ip', 'Reconstructed photon cluster width', 'Mean transverse RMS [mm]', 'cluster_width.png'),
        ('centroid_offset', 'Reconstructed photon centroid error', 'Mean distance to true photon trajectory [mm]', 'centroid_error.png'),
        ('owned_fraction', 'Truth photons with a directly owned photon PFO', 'Selected truth photons reconstructed [%]', 'photon_reconstruction_survival.png'),
    ):
        survival = quantity == 'owned_fraction'
        selected_styles = styles + ([('CC3 unflagged', 'CC3 unflagged', '#777777', 'x', ':')] if survival else [])
        fig, axes = plt.subplots(3, 2, figsize=(11, 9), sharex=True, sharey=True)
        fig.subplots_adjust(left=.13, right=.98, top=.83, bottom=.16, hspace=.28, wspace=.12)
        any_points = False
        for i, reg in enumerate(REGIONS):
            for j, population in enumerate(POPULATIONS):
                ax = axes[i, j]
                for sample, label, color, marker, linestyle in selected_styles:
                    subset = [r for r in rows if r['region'] == reg and r['population'] == population
                              and r['sample'] == sample and r['quantity'] == quantity]
                    # Explicit NaNs preserve gaps instead of connecting across unavailable bins.
                    by_energy = {r['energy_lo']: r for r in subset}
                    x, y, err = [], [], []
                    for lo, hi in zip(EDGES[:-1], EDGES[1:]):
                        r = by_energy.get(lo)
                        valid = r is not None and r['n_valid'] >= 20 and r['n_events'] >= 2 and np.isfinite(r['mean'])
                        x.append(np.sqrt(lo*hi))
                        scale = 100 if survival else 1
                        y.append(scale*r['mean'] if valid else np.nan)
                        err.append(scale*r['se_event'] if valid else np.nan)
                        any_points |= valid
                    ax.errorbar(x, y, yerr=err, color=color, marker=marker, linestyle=linestyle,
                                ms=4, capsize=2, label=label)
                ax.set_xscale('log')
                ax.set_xlim(10, 125)
                ax.set_xticks([10,20,40,80,125])
                ax.set_xticklabels(['10','20','40','80','125'])
                ax.grid(axis='y', alpha=.2)
                if survival:
                    ax.set_ylim(-3, 103)
                if i == 0:
                    ax.set_title(population, fontsize=11)
                if j == 0:
                    ax.set_ylabel({'Full detector':'Full acceptance','Barrel':'Barrel interior','Endcap':'Endcap interior'}[reg])
        fig.suptitle(title, fontsize=16, y=.98)
        fig.legend(*axes[0,0].get_legend_handles_labels(), loc='upper center',
                   bbox_to_anchor=(.5,.945), ncol=2 if survival else 3, frameon=False)
        fig.text(.025,.50,ylabel,rotation=90,va='center',fontsize=11)
        fig.text(.55,.10,'True photon energy [GeV]',ha='center',fontsize=12)
        caption = ('Truth denominator includes reconstruction failures.' if survival else
                   'First cluster of directly owned photon PFOs; comparison is conditional on reconstruction.')
        fig.text(.55,.055,caption,ha='center',fontsize=9)
        fig.text(.55,.03,'Event-level standard errors; ≥20 photons and ≥2 events per point. π⁰ daughters are a subset of all photons.',ha='center',fontsize=9)
        if not any_points:
            fig.text(.55,.49,'No bins meet the plotting threshold',ha='center',bbox={'facecolor':'white','edgecolor':'gray'})
        fig.savefig(output / filename,dpi=170)
        plt.close(fig)


STUDY_NOTES = "Study revision: one standalone script, November 11 photon widths and centroids.\n\nPRIMARY POPULATIONS\nAll selected photons: PDG22, generator status1, true E in [10,125) GeV,\nno direct e-/e+ parent. No isolation cut. This restores the historical V6\npopulation and supersedes the unapproved quiet-20-mrad primary selection.\nPi0 daughters: subset of the above with a PDG111 parent having exactly two\ndirect daughters, both generator-status-1 photons. Parent status unrestricted.\nIndividual daughter energy is plotted. Sibling reconstruction is not required.\nThe two columns overlap; they are not independent or exclusive samples.\nNo mass window, finder cut, or minimum reconstructed energy selects the shapes.\nBins: 10,20,30,40,60,80,100,125 GeV, upper edges exclusive.\n\nGEOMETRY\nFull acceptance: finite theta/phi; no additional angular cut.\nBarrel: 40 <= theta <= 140 deg, excluding phi in [C-0.51,C+8.50] deg,\nC=-157.5+45k, with wrapping. Endcap: 15 <= min(theta,180-theta) < 30 deg.\n\nMATCHING AND OBSERVABLES\nPreserved historical November 6 reducer functions: each PDG22 PFO is owned\nby the largest cluster-weight MC link, cw>=0.5, first link on a tie.\nEach truth photon uses its highest-energy owned photon PFO and first cluster.\nwidth_ip: energy-weighted transverse RMS about origin-to-reco-centroid axis.\nwidth_centered: RMS about a true-direction-parallel line through that centroid.\ncentroid_offset: unsigned distance |(C_reco-V_true) x u_true| in mm.\nwidth_truth^2 = width_centered^2 + centroid_offset^2 is checked numerically.\nStored raw signed hit moments are preserved. No hit filtering was added.\nAssociated hits are not verified to be exclusively ECAL hits. The numerical\nidentity checks geometry/algebra, not independent correctness of stored moments.\n\nPLOTS AND COMPARISON\ncluster_width.png and centroid_error.png: G4, CC3 inclusive (all flags), and\nCC3 flagged. The flagged subset is part of inclusive CC3, not independent.\nphoton_reconstruction_survival.png: owned/truth, also showing unflagged CC3.\nUnflagged shape values remain in photon_response.csv for audit, without a\npotentially misleading grey shape curve. Flag=-1 or other unknown values are\nkept separate in tables and included in CC3 inclusive.\nThese are native populations with identical stated cuts, not matched ensembles.\nFlag eligibility and reconstruction survival can differ. Event-level SEs give\nuncertainty on the photon-weighted mean, allowing within-event correlations.\nZero survival may have zero plug-in SE; it does not establish exactly zero\npopulation probability. Tables retain all bins, including bins hidden in plots.\nwidth_centroid_differences.csv gives absolute and relative conditional mean\ndifferences. Its combined SE assumes independent samples. Do not treat it as\na significance claim if common events induce cross-sample covariance.\nA 1-2 mm difference on a 20 mm RMS scale is 5-10%; its relevance to direction\nor pion reconstruction requires additional evidence. RMS is not a containment\nradius. A broader shower need not have a displaced centroid. Unsigned centroid\ndistance cannot establish an outward angular bias; signed pair measurements\nand the reconstruction direction definition are needed for that connection.\n\nUNFLAGGED AUDIT\nownership_audit.csv records truth denominators, owned PFOs, first clusters,\nmutually exclusive ownership failure categories, and descendant-link evidence.\nThe failure categories sum to truth; other diagnostic columns may overlap.\nA strong descendant link means an individual cw>=0.5 link, not energy closure.\nOriginal-photon fast=0 does not establish pure Geant4 for the whole shower.\nassociation_examples.json records deterministic first examples per region,\nflag and failure category per chunk, with event/MC IDs and descendant links.\nIt is not a random sample. Descendant tracing is limited to stored ancestry.\nThis run diagnoses the reported unflagged zero-survival anomaly; it does not\nassume its cause. A direct e+/e- daughter indicator alone is not a detector\nreach or full conversion classification.\n\nSCOPE AND REPRODUCIBILITY\nThe existing pair-direction study remains a secondary result and is not\nredefined or regenerated here. There are no SimCalorimeterHit contributions\nin these NPZs; simulation-level widths require a separate SIM-level export.\nNo new extraction is launched by this script.\nrun.json contains input provenance, count closure, checks and completion status.\nOnly status=complete denotes the full successful dataset. Checkpoints allow\nreruns to reuse unchanged chunks; changed provenance triggers recomputation.\nshare_here.zip contains only this revision's outputs and its standalone source.\nFull-data physics validation must await the NAF run; local tests are synthetic.\n"

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path(DEFAULT_INPUT))
    parser.add_argument("--outdir", type=Path, default=Path(DEFAULT_OUT))
    parser.add_argument("--manifest-dir", type=Path)
    parser.add_argument("--max-chunks", type=int, default=0, help="0 processes all chunks")
    parser.add_argument("--verify-hashes", action="store_true")
    args = parser.parse_args()
    args.input = args.input.expanduser()
    if args.max_chunks < 0:
        parser.error("--max-chunks must be nonnegative")
    output = args.outdir.expanduser()
    output.mkdir(parents=True, exist_ok=True)
    checkpoints = output / "checkpoints_standalone"
    checkpoints.mkdir(exist_ok=True)
    source = Path(__file__).resolve()
    signature = {source.name: digest(source)}
    completed = []
    report = dict(status="running", signature=signature, samples={}, limitations=[
        "No SimCalorimeterHit or contribution tables in this NPZ schema.",
        "Native conditional populations; no G4/CC3 matching or difference significance.",
        "Strong descendant links diagnose attribution; they do not establish shower energy completeness.",
        "Raw signed first-cluster hit moments are inherited from November 6. ECAL-only membership is unverified.",
    ])
    started = time.time()
    try:
        for sample in ("g4", "cc3"):
            folder = args.input / sample
            choices = [folder / "manifest.json", folder / f"{sample}_manifest.json"]
            if args.manifest_dir:
                choices.insert(0, args.manifest_dir / f"{sample}_manifest.json")
            manifest_path = next((p for p in choices if p.is_file()), None)
            if manifest_path is None:
                raise FileNotFoundError(f"Missing {sample} manifest in {choices}")
            manifest = json.loads(manifest_path.read_text())
            if manifest["sample"] != sample or manifest["schema_version"] != 1 or not manifest["complete"]:
                raise ValueError(f"Unexpected manifest {manifest_path}")
            selected = manifest["chunks"][:args.max_chunks or None]
            expected_loop = 0
            totals = Counter()
            flag_counts, checks = Counter(), Counter()
            for part in selected:
                if part["first_event_loop"] != expected_loop:
                    raise ValueError("Noncontiguous manifest")
                stamp = dict(source=signature, manifest=digest(manifest_path),
                             input=str(folder.resolve()), verify_hashes=args.verify_hashes,
                             files={r["file"]: [(folder / r["file"]).stat().st_size,
                                               (folder / r["file"]).stat().st_mtime_ns]
                                    for r in part["files"]})
                checkpoint = checkpoints / f'{sample}_{part["part"]:05d}.json'
                for record in part["files"]:
                    if (folder / record["file"]).stat().st_size != record["bytes"]:
                        raise ValueError(f'Input size mismatch {record["file"]}')
                cached = json.loads(checkpoint.read_text()) if checkpoint.exists() else None
                if cached is not None and cached["stamp"] == stamp:
                    if args.verify_hashes:
                        for record in part["files"]:
                            if digest(folder / record["file"]) != record["sha256"]:
                                raise ValueError("Cached input checksum mismatch")
                    result = cached["result"]
                else:
                    with ExitStack() as stack:
                        tr, rr, cl = [open_part(folder, part, g, manifest, args.verify_hashes, stack)
                                      for g in GROUPS]
                        if not np.array_equal(tr["events_loop"], np.arange(expected_loop, part["last_event_loop"] + 1)):
                            raise ValueError("Unexpected event loops")
                        for data in (rr, cl):
                            for field in ("loop", "run", "event"):
                                if not np.array_equal(tr["events_" + field], data["events_" + field]):
                                    raise ValueError("Triplet event identity mismatch")
                        for data in (tr, rr, cl):
                            for field in data.files:
                                if field.endswith("_loop") and not np.isin(data[field], tr["events_loop"]).all():
                                    raise ValueError(f"Orphan event in {field}")
                        result = process(tr, rr, cl, sample)
                    save_json(checkpoint, dict(stamp=stamp, result=result))
                completed.append(result)
                totals.update(result["counters"])
                flag_counts.update(result["flags"])
                checks.update(result["checks"])
                expected_loop = part["last_event_loop"] + 1
                print(f'{sample}: chunk {part["part"] + 1}, {expected_loop} events, {(time.time()-started)/60:.1f} min', flush=True)
            full = len(selected) == len(manifest["chunks"])
            closure = {}
            for name, reference in (("events", "events_processed"), ("generator_photons", "generator_photons"),
                                     ("generator_photons_Ege10_no_e_parent", "generator_photons_Ege10_no_e_parent")):
                expected = manifest["counters"][reference]
                closure[name] = dict(actual=totals[name], expected=expected,
                                     status=("PASS" if totals[name] == expected else "FAIL") if full else "PILOT")
            report["samples"][sample] = dict(closure=closure, fast_flags=flag_counts, checks=checks,
                                             source=manifest.get("source"), manifest_sha256=digest(manifest_path))
            if checks["identity_failed"] or any(r["status"] == "FAIL" for r in closure.values()):
                raise ValueError("Closure failed; inspect run.json")
        report["status"] = "complete" if args.max_chunks == 0 else "pilot_complete"
    except BaseException as error:
        report["status"] = "failed"
        report["error"] = repr(error)
        raise
    finally:
        try:
            save_json(output / "run.json", report)
            if completed:
                audit, responses, examples = summarize(completed)
                write_csv(output / "ownership_audit.csv", audit)
                write_csv(output / "photon_response.csv", responses)
                save_json(output / "association_examples.json", examples)
                plots(responses, output)
                write_csv(output / "width_centroid_differences.csv", comparisons(responses))
            (output / "study_notes.txt").write_text(STUDY_NOTES, encoding="utf-8")
            if source != (output / source.name).resolve():
                shutil.copy2(source, output / source.name)
            archive = output / "share_here.zip"
            temporary = archive.with_suffix(".tmp")
            with ZipFile(temporary, "w", ZIP_DEFLATED) as package:
                names = ["run.json", "study_notes.txt", source.name]
                if completed:
                    names += ["ownership_audit.csv", "photon_response.csv", "association_examples.json",
                              "width_centroid_differences.csv", "cluster_width.png", "centroid_error.png",
                              "photon_reconstruction_survival.png"]
                for name in names:
                    package.write(output / name, name)
            temporary.replace(archive)
            print(f"Status: {report['status']}. Share {archive}", flush=True)
        except Exception as output_error:
            report["status"] = "failed"
            report["output_error"] = repr(output_error)
            save_json(output / "run.json", report)
            raise


if __name__ == "__main__":
    main()
