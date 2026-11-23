#!/usr/bin/env python3
"""Chunked CC3/G4 study. ROOT/podio are not needed; input files remain unchanged."""
import argparse
from collections import Counter
from contextlib import ExitStack
import hashlib
import json
from pathlib import Path
import shutil
import sys
import time
import zipfile
import numpy as np
from reduce import GROUPS, DEFAULT_INPUT, open_part, analyse_part, atomic_npz

HERE=Path(__file__).resolve().parent

def write_json(path,value):
    tmp=path.with_suffix('.tmp'); tmp.write_text(json.dumps(value,indent=2,default=lambda x:x.item() if isinstance(x,np.generic) else str(x)));tmp.replace(path)

def fingerprint(path): return hashlib.sha256(path.read_bytes()).hexdigest()

def pack_share(out):
    target=out/'share_first.zip';tmp=target.with_suffix('.tmp')
    with zipfile.ZipFile(tmp,'w',zipfile.ZIP_DEFLATED) as z:
        for directory in ('tables','figures','validation'):
            for p in sorted((out/directory).rglob('*')):
                if p.is_file() and p.suffix!='.pdf': z.write(p,p.relative_to(out))
        for name in ('run.json','status.json','output_manifest.json'):
            if (out/name).exists(): z.write(out/name,name)
    tmp.replace(target)

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--input',type=Path,default=Path(DEFAULT_INPUT))
    ap.add_argument('--outdir',type=Path,default=Path('/data/dust/user/alimuham/thesis/plots/npz_study_Nov6'))
    ap.add_argument('--manifest-dir',type=Path)
    ap.add_argument('--max-chunks',type=int,default=0)
    ap.add_argument('--resume',action='store_true')
    ap.add_argument('--verify-hashes',action='store_true')
    ap.add_argument('--moments-only',action='store_true')
    ap.add_argument('--bootstrap',type=int,default=200)
    ap.add_argument('--legacy-dir',type=Path,help='Optional directory with pi0-tables-{g4,cc3}-full.npz and full-dmgg0.1.npz')
    ap.add_argument('--plots-only',action='store_true')
    args=ap.parse_args()
    if args.max_chunks<0 or args.bootstrap<0:ap.error('Counts must be nonnegative')
    out=args.outdir.expanduser().resolve()
    if out.exists() and any(out.iterdir()) and not (args.resume or args.plots_only):ap.error('Output exists. Use --resume or a different --outdir.')
    for name in ('compact','checkpoints','figures','tables','validation','source'): (out/name).mkdir(parents=True,exist_ok=True)
    config={k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()}
    script_hash={p.name:fingerprint(p) for p in HERE.glob('*.py')}
    signature=dict(input=str(args.input.resolve()),moments_only=args.moments_only,scripts=script_hash)
    previous=out/'run.json'
    if previous.exists():
        old=json.loads(previous.read_text())
        if not args.plots_only and old['signature']!=signature:raise ValueError('Resume requires identical input path, hit mode and source code. Use a new output directory.')
    else:write_json(previous,dict(config=config,signature=signature,numpy=np.__version__,python=sys.version))
    for path in HERE.glob('*.py'):shutil.copy2(path,out/'source'/path.name)
    start=time.time()
    try:
        if not args.plots_only:
            for tag in ('g4','cc3'):
                folder=args.input/tag
                candidates=[args.manifest_dir/f'{tag}_manifest.json'] if args.manifest_dir else [folder/'manifest.json',folder/f'{tag}_manifest.json']
                path=next((p for p in candidates if p.is_file()),None)
                if path is None:raise FileNotFoundError(f'No manifest: {candidates}')
                manifest=json.loads(path.read_text()); mh=fingerprint(path)
                if manifest['sample']!=tag or manifest['schema_version']!=1 or not manifest['complete']:raise ValueError(f'Invalid manifest: {path}')
                write_json(out/'validation'/f'{tag}_manifest.json',manifest)
                counts=Counter(); parts=manifest['chunks'][:args.max_chunks or None];next_loop=0
                if not parts:raise ValueError('Empty chunk list')
                for part in parts:
                    if part['first_event_loop']!=next_loop:raise ValueError('Noncontiguous manifest chunks')
                    stem=out/'checkpoints'/f'{tag}_{part["part"]:05d}'
                    checkpoint=stem.with_suffix('.json')
                    paths={kind:Path(str(stem)+'_'+kind+'.npz') for kind in ('photons','pairs','reco_pairs')}
                    loaded=False
                    if args.resume and checkpoint.exists() and all(p.exists() for p in paths.values()):
                        ck=json.loads(checkpoint.read_text())
                        if ck['manifest_sha256']!=mh:raise ValueError('Manifest changed since checkpoint')
                        if any(fingerprint(paths[k])!=v for k,v in ck['outputs_sha256'].items()):raise ValueError('Checkpoint damaged')
                        counts.update(ck['counts']);loaded=True
                    if not loaded:
                        counter=Counter()
                        with ExitStack() as stack:
                            tr,rr,cl=[open_part(folder,part,g,manifest,args.verify_hashes,stack) for g in GROUPS]
                            ev=tr['events_loop']
                            if not np.array_equal(ev,np.arange(next_loop,part['last_event_loop']+1)):raise ValueError('Event loops do not match manifest')
                            for d in (rr,cl):
                                if any(not np.array_equal(tr['events_'+k],d['events_'+k]) for k in ('loop','run','event')):raise ValueError('Triplet event IDs disagree')
                            # Every object and relation must belong to an event in this chunk.
                            for data in (tr,rr,cl):
                                for field in data.files:
                                    if field.endswith('_loop') and not np.isin(data[field],ev).all():raise ValueError(f'Orphan event in {field}')
                            arrays=analyse_part(tr,rr,cl,tag,counter,not args.moments_only)
                        for kind,array in zip(paths,arrays):
                            atomic_npz(paths[kind],rows=array if array is not None else np.empty(0,dtype=[('loop','i8')]))
                        write_json(checkpoint,dict(manifest_sha256=mh,counts=dict(counter),outputs_sha256={k:fingerprint(v) for k,v in paths.items()}))
                        counts.update(counter)
                    next_loop=part['last_event_loop']+1
                    print(f'{tag}: chunk {part["part"]+1}/{len(parts)} {"reused" if loaded else "saved"}; {next_loop} events; {(time.time()-start)/60:.1f} min',flush=True)
                    write_json(out/'status.json',dict(stage='reduction',sample=tag,chunk=part['part'],counts=dict(counts)))
                complete=len(parts)==len(manifest['chunks'])
                checks=[]
                for k in ('events','generator_photons','generator_photons_Ege10','generator_photons_Ege10_no_e_parent'):
                    expected=manifest['counters']['events_processed' if k=='events' else k]
                    checks.append(dict(counter=k,actual=counts[k],expected=expected,status=('PASS' if counts[k]==expected else 'FAIL') if complete else 'PILOT'))
                write_json(out/'validation'/f'{tag}_closure.json',dict(counts=dict(counts),checks=checks,complete=complete))
                if any(c['status']=='FAIL' for c in checks):raise ValueError('Manifest count closure failed')
                # Compact event shards remain available independently. One merged file per purpose.
                for kind in ('photons','pairs','reco_pairs'):
                    arrays=[]
                    for part in parts:
                        with np.load(out/'checkpoints'/f'{tag}_{part["part"]:05d}_{kind}.npz',allow_pickle=False) as z:
                            if len(z['rows']):arrays.append(z['rows'])
                    if not arrays:raise ValueError(f'No {kind} rows in {tag}; use a larger pilot')
                    atomic_npz(out/'compact'/f'{tag}_{kind}.npz',rows=np.concatenate(arrays));del arrays
        from study import run
        run(out,args.bootstrap,args.legacy_dir)
        write_json(out/'status.json',dict(stage='complete',elapsed_minutes=(time.time()-start)/60))
    except BaseException as error:
        write_json(out/'status.json',dict(stage='interrupted_or_failed',error=repr(error),elapsed_minutes=(time.time()-start)/60))
        pack_share(out)
        raise
    finally:
        records=[dict(path=str(p.relative_to(out)),bytes=p.stat().st_size) for p in sorted(out.rglob('*')) if p.is_file() and p.name not in ('output_manifest.json','share_first.zip')]
        write_json(out/'output_manifest.json',dict(files=records,total_bytes=sum(r['bytes'] for r in records)))
    pack_share(out)
    print(f'Finished. Plots: {out}/figures\nShare first: {out}/share_first.zip',flush=True)

if __name__=='__main__':main()
