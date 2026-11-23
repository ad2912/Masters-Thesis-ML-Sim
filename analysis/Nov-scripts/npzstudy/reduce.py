#!/usr/bin/env python3
"""Reduce the schema-v1 DUST extraction to photon/pair records and analysis plots."""
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

def atomic_npz(path, **arrays):
    tmp=path.with_suffix('.tmp')
    with tmp.open('wb') as stream: np.savez_compressed(stream,**arrays)
    tmp.replace(path)

def csv_rows(path, rows):
    if rows:
        with path.open('w', newline='') as stream:
            w=csv.DictWriter(stream,fieldnames=list(rows[0]))
            w.writeheader();w.writerows(rows)

def mean_se(values, event, weight=None):
    ok=np.isfinite(values)
    v=values[ok]; ev=event[ok]
    if not len(v): return np.nan,np.nan,0
    w=np.ones(len(v)) if weight is None else weight[ok]
    avg=np.average(v,weights=w)
    se=norm(np.bincount(ev,weights=w*(v-avg)))/w.sum()
    return avg,se,len(v)
