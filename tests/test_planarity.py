"""Independent geometric cases for periodic and swept spring constraints."""
import unittest
import numpy as np
from lss.engineering.planarity import PlanarConstraints


class PlanarityTests(unittest.TestCase):
    def make(self, points, edges):
        return PlanarConstraints(np.array(points,float),np.array(edges,int).T,[10,10],.05)

    def test_proper_crossing(self):
        x=np.array([[-1,-1],[1,1],[-1,1],[1,-1]],float)
        p=self.make(x,[[0,1],[2,3]])
        self.assertEqual(len(p.violating_pairs(x,enforce_clearance=False)),1)

    def test_shared_joint_and_overlap(self):
        x=np.array([[0,0],[2,0],[0,2]],float)
        p=self.make(x,[[0,1],[0,2]])
        self.assertEqual(len(p.violating_pairs(x)),0)
        y=x.copy();y[2]=[1,0]
        self.assertEqual(len(p.violating_pairs(y)),1)
        y[2]=[-1,0]
        self.assertEqual(len(p.violating_pairs(y)),0)

    def test_collinear_nonincident_and_t_junction(self):
        x=np.array([[0,0],[2,0],[3,0],[4,0]],float)
        p=self.make(x,[[0,1],[2,3]])
        self.assertEqual(len(p.violating_pairs(x)),0)
        y=x.copy();y[2]=[1,0]
        self.assertEqual(len(p.violating_pairs(y)),1)
        y[3]=[1,1]
        self.assertEqual(len(p.violating_pairs(y)),1)

    def test_periodic_crossing(self):
        x=np.array([[4.5,0],[-4.5,0],[-4.8,-1],[-4.8,1]],float)
        p=self.make(x,[[0,1],[2,3]])
        self.assertEqual(len(p.violating_pairs(x,enforce_clearance=False)),1)

    def test_periodic_shared_joint(self):
        x=np.array([[4.5,0],[-4.5,0],[4.5,1]],float)
        p=self.make(x,[[0,1],[0,2]])
        self.assertEqual(len(p.violating_pairs(x)),0)

    def test_continuous_move_no_endpoint_crossing(self):
        x=np.array([[-1,0],[1,0],[-.5,1],[.5,1]],float)
        p=self.make(x,[[0,1],[2,3]])
        y=x.copy();y[2:,1]=-1
        self.assertEqual(len(p.violating_pairs(y)),0)
        self.assertEqual(len(p.swept_pairs(x,y)),1)
        self.assertTrue(p.blocked_nodes(x,y).all())

    def test_clearance_preserves_original_narrow_gap(self):
        x=np.array([[-1,0],[1,0],[-1,.01],[1,.01]],float)
        p=self.make(x,[[0,1],[2,3]])
        self.assertEqual(len(p.violating_pairs(x)),0)
        y=x.copy();y[2:,1]=.001
        self.assertEqual(len(p.violating_pairs(y)),1)
        self.assertEqual(len(p.violating_pairs(y,enforce_clearance=False)),0)

    def test_clear_motion_and_duplicate_orientations(self):
        x=np.array([[-1,0],[1,0],[-1,2],[1,2]],float)
        p=self.make(x,[[0,1],[1,0],[2,3]])
        self.assertEqual(len(p.edges),2)
        self.assertFalse(p.blocked_nodes(x,x+np.array([.1,.1])).any())

    def test_bond_cannot_collapse_and_reverse(self):
        x=np.array([[-1,0],[1,0],[-1,2],[1,2]],float)
        p=self.make(x,[[0,1],[2,3]])
        y=x.copy();y[[0,1]]=y[[1,0]]
        self.assertTrue(p.blocked_nodes(x,y)[:2].all())

    def test_shared_joint_cannot_sweep_through_overlap(self):
        x=np.array([[0,0],[2,0],[1,1]],float)
        p=self.make(x,[[0,1],[0,2]])
        y=x.copy();y[2,1]=-1
        self.assertEqual(len(p.violating_pairs(y)),0)
        self.assertEqual(len(p.swept_pairs(x,y)),1)

    def test_periodic_swept_contact(self):
        x=np.array([[4.5,0],[-4.5,0],[-4.8,1],[-4.8,2]],float)
        p=self.make(x,[[0,1],[2,3]])
        y=x.copy();y[2:,1]-=3
        self.assertEqual(len(p.violating_pairs(x)),0)
        self.assertEqual(len(p.violating_pairs(y)),0)
        self.assertEqual(len(p.swept_pairs(x,y)),1)

    def test_swept_detection_against_sampled_independent_orientations(self):
        rng=np.random.default_rng(8452)
        for _ in range(100):
            x=rng.uniform(-2,2,(4,2));y=rng.uniform(-2,2,(4,2))
            p=self.make(x,[[0,1],[2,3]])
            if len(p.violating_pairs(x)) or len(p.violating_pairs(y)):
                continue
            crossed=False
            for t in np.linspace(0,1,101):
                z=x+t*(y-x)
                def orient(a,b,c):
                    v,w=b-a,c-a
                    return v[0]*w[1]-v[1]*w[0]
                crossed |= (orient(z[0],z[1],z[2])*orient(z[0],z[1],z[3])<0 and
                            orient(z[2],z[3],z[0])*orient(z[2],z[3],z[1])<0)
            if crossed:
                self.assertTrue(p.blocked_nodes(x,y).any())


if __name__=='__main__':unittest.main()
