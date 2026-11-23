#!/usr/bin/env python3
"""Software/analytic tests, not validation of the unavailable DUST dataset."""
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from collections import Counter
import numpy as np
from fixture import fixture
from reduce import analyse_part,opening,project_pair,unit,mass,region,Clusters
from study import fit_mass,stats,fraction,common_weights
HERE=Path(__file__).resolve().parent

class PhysicsTests(unittest.TestCase):
    def test_stable_opening_and_mass(self):
        rng=np.random.default_rng(7)
        for _ in range(100):
            theta,phi=rng.uniform(.15,2.9),rng.uniform(-3,3)
            theta2,phi2=theta+rng.normal(0,.005),phi+rng.normal(0,.005)
            u=np.array([np.sin(theta)*np.cos(phi),np.sin(theta)*np.sin(phi),np.cos(theta)])
            v=np.array([np.sin(theta2)*np.cos(phi2),np.sin(theta2)*np.sin(phi2),np.cos(theta2)])
            ea,eb=rng.uniform(5,100,2);a=opening(u,v)
            expected=2*np.sqrt(ea*eb)*np.sin(a/2)
            self.assertAlmostEqual(expected,fit_mass(np.array([ea,theta,phi,eb,theta2,phi2])),places=10)
            self.assertAlmostEqual(expected,mass(ea+eb,ea*u,eb*v),places=7)
    def test_direction_sign_and_closure(self):
        u=np.array([np.sin(.3),0,np.cos(.3)]);v=np.array([np.sin(.31),0,np.cos(.31)])
        a=np.array([np.sin(.299),0,np.cos(.299)]);b=np.array([np.sin(.312),0,np.cos(.312)])
        oa,ob,pa,pb=project_pair(u,v,a,b)
        self.assertAlmostEqual(oa,1,places=10);self.assertAlmostEqual(ob,2,places=10)
        self.assertAlmostEqual(oa+ob,1000*(opening(a,b)-opening(u,v)),places=10)
        self.assertEqual(pa,0);self.assertEqual(pb,0)
    def test_region_boundaries(self):
        self.assertEqual(region(np.array([20.,21.]),np.zeros(2)),2)
        self.assertEqual(region(np.array([159.,160.]),np.zeros(2)),3)
        self.assertEqual(region(np.array([35.,36.]),np.zeros(2)),5)
        self.assertEqual(region(np.array([35.,50.]),np.zeros(2)),6)
        self.assertEqual(region(np.array([90.,90.]),np.array([-157.5,-157.5])),1)
    def test_event_uncertainty_and_support(self):
        self.assertTrue(np.isnan(stats(np.array([1,0]),np.array([0,0]))['se_event']))
        self.assertAlmostEqual(stats(np.array([0,0,1,1]),np.array([0,0,1,1]))['se_event'],.5)
        self.assertGreater(fraction(np.zeros(10),np.arange(10))['wilson95_hi'],0)
        w=common_weights({'g4':None,'cc3':None},{'g4':['a']*10+['b']*10,'cc3':['a']*5+['c']*10})
        self.assertEqual(w['g4'].sum(),5);self.assertEqual(w['cc3'].sum(),5)

class ReaderTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.root=Path(self.tmp.name)
        m=json.loads((HERE/'schema.json').read_text());fixture(self.root,m['schemas'],m['groups'],'cc3')
        self.arrays=[]
        for kind in ('truth','reco','clusters'):
            with np.load(self.root/'cc3'/f'cc3_{kind}_00000.npz',allow_pickle=False) as z:self.arrays.append({k:z[k].copy() for k in z.files})
    def tearDown(self):self.tmp.cleanup()
    def run_arrays(self):return analyse_part(*self.arrays,'cc3',Counter(),True)
    def test_shapes_and_ids(self):
        ph,p,r=self.run_arrays()
        self.assertEqual(len(ph),3);self.assertEqual(ph['owned_clean'].sum(),2)
        self.assertEqual(p['outcome'][0],4);self.assertEqual(len(r),1)
        self.assertEqual(p['shared_hits'][0],1)
        self.assertAlmostEqual(p['A_width_truth'][0],2,places=7)
        self.assertAlmostEqual(p['A_width_centered'][0],0,places=7)
        self.assertAlmostEqual(p['A_centroid_offset'][0],2,places=7)
        self.assertAlmostEqual(p['mass'][0],.1349766,places=9)
        self.assertTrue(np.isnan(ph['width_ip'][-1]));self.assertEqual(ph['owned_idx'][-1],-1)
    def test_unresolved_categories(self):
        rr=self.arrays[1]
        rr['links_cw'][:]=.1;rr['links_raw'][:]=1000000
        self.assertEqual(self.run_arrays()[1]['outcome'][0],0)
        rr['links_cw'][0]=.8;rr['links_raw'][0]=8000000
        self.assertEqual(self.run_arrays()[1]['outcome'][0],1)
        rr['links_cw'][:]=.8;rr['links_raw'][:]=8000000;rr['links_pfo_idx'][:]=0
        self.assertEqual(self.run_arrays()[1]['outcome'][0],2)
        rr['links_pfo_idx'][:]=[0,1];rr['pfo_pdg'][1]=211
        self.assertEqual(self.run_arrays()[1]['outcome'][0],3)
    def test_bad_link_rejected(self):
        self.arrays[1]['links_raw'][0]=1
        with self.assertRaisesRegex(ValueError,'packed'):self.run_arrays()
    def test_dangling_reference_rejected(self):
        self.arrays[1]['links_mc_idx'][0]=1234
        with self.assertRaises(KeyError):self.run_arrays()

if __name__=='__main__':unittest.main(verbosity=2)
