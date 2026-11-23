"""End-to-end synthetic fixture with full IDs, an unmatched photon and shared hits."""
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import numpy as np

def fixture(root,schemas,groups,tag):
    folder=root/tag;folder.mkdir()
    m0=.1349766
    a=2*np.arcsin(m0/40)
    u=np.array([np.sin(.35),0,np.cos(.35)])
    v=np.array([np.sin(.35+a),0,np.cos(.35+a)])
    centroid=2400*u
    hit=centroid+np.array([0.,2.,0.])
    mc=[
        dict(cid=9,idx=0,pdg=111,gen=2,E=40,px=20*(u[0]+v[0]),pz=20*(u[2]+v[2])),
        dict(cid=9,idx=1,pdg=22,gen=1,E=20,px=20*u[0],pz=20*u[2],fast=int(tag=='cc3')),
        dict(cid=9,idx=2,pdg=22,gen=1,E=20,px=20*v[0],pz=20*v[2]),
        dict(cid=9,idx=3,pdg=22,gen=1,E=12,px=12)]
    P=[]
    for j,d in enumerate((u,v)):
        P.append(dict(cid=31,idx=j,pdg=22,E=20,px=20*d[0],pz=20*d[2]))
    C=[]
    for j in range(2):
        values=dict(zip(('mean_x','mean_y','mean_z'),hit))
        values.update(dict(zip(('mean_xx','mean_yy','mean_zz','mean_xy','mean_xz','mean_yz'),
                               [hit[0]**2,hit[1]**2,hit[2]**2,hit[0]*hit[1],hit[0]*hit[2],hit[1]*hit[2]])))
        C.append(dict(cid=77,idx=j,n_hits=1,hit_E_sum=1,**values))
    data={
        'events':[dict(run=13,event=42,n_mc=4,n_pfo=2,n_candidate=1)],
        'mc':mc,
        'ancestry':[dict(child_cid=9,child_idx=j,parent_cid=9,parent_idx=0) for j in (1,2)],
        'pfo':P,
        'links':[dict(pfo_cid=31,pfo_idx=j,mc_cid=9,mc_idx=j+1,cw=.8,raw=8000000) for j in range(2)],
        'pfo_cluster':[dict(pfo_cid=31,pfo_idx=j,cluster_cid=77,cluster_idx=j) for j in range(2)],
        'candidate':[dict(cid=41,idx=0,n_daughters=2,E=40,goodness=.9)],
        'candidate_daughter':[dict(candidate_cid=41,candidate_idx=0,pfo_cid=31,pfo_idx=j,order=j) for j in range(2)],
        'cluster':C,
        'hit':[dict(cid=18,idx=7,E=1,x=hit[0],y=hit[1],z=hit[2])],
        'cluster_hit':[dict(cluster_cid=77,cluster_idx=j,hit_cid=18,hit_idx=7) for j in range(2)]}
    files=[]
    for group,names in groups.items():
        arrays={'meta_sample':np.asarray(tag),'meta_schema_version':np.asarray(1)}
        for name in names:
            for field in schemas[name].split():
                k,t=field.split(':')
                arrays[f'{name}_{k}']=np.asarray([r.get(k,0) for r in data[name]],dtype='i8' if t=='i' else 'f8')
        path=folder/f'{tag}_{group}_00000.npz';np.savez_compressed(path,**arrays)
        files.append(dict(file=path.name,bytes=path.stat().st_size,sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                          rows={n:len(data[n]) for n in names}))
    manifest=dict(sample=tag,schema_version=1,complete=True,schemas=schemas,groups=groups,
                  chunks=[dict(part=0,first_event_loop=0,last_event_loop=0,files=files)],
                  counters=dict(events_processed=1,generator_photons=3,generator_photons_Ege10=3,
                                generator_photons_Ege10_no_e_parent=3),
                  source='synthetic',script_sha256='synthetic')
    (folder/'manifest.json').write_text(json.dumps(manifest))
