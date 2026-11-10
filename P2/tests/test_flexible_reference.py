"""Independent reference equations have hand-derived force fixtures."""
import importlib.util
import unittest
import numpy as np


class FlexibleReferenceTests(unittest.TestCase):
    def reference(self):
        self.assertIsNotNone(importlib.util.find_spec('P2.validate_flexible_dynamics'),
                             'Independent dynamic cable reference required')
        from P2.validate_flexible_dynamics import reference_forces
        return reference_forces

    def test_two_equal_stretched_segments_have_balanced_internal_force(self):
        law=self.reference()
        force,end,tension=law(np.array([[1.1,0.,0.]]),np.zeros((1,3)),np.array([2.2,0.,0.]),
            np.zeros(3),np.zeros(3),length=2.,EA=10.,BA=.5,density=.01,diameter=.003,Cd=1.2,rho=1.2,g=9.8)
        np.testing.assert_allclose(tension,[1.,1.],atol=1e-12)
        np.testing.assert_allclose(end,[-1.,0.,0.],atol=1e-12)
        np.testing.assert_allclose(force,[[0.,0.,.196]],atol=1e-12)

    def test_endpoint_axial_velocity_changes_tension_without_compression(self):
        law=self.reference()
        kw=dict(length=2.,EA=10.,BA=.5,density=.01,diameter=.003,Cd=1.2,rho=1.2,g=9.8)
        force,end,tension=law(np.array([[1.1,0.,0.]]),np.zeros((1,3)),np.array([2.2,0.,0.]),
            np.array([.2,0.,0.]),np.zeros(3),**kw)
        np.testing.assert_allclose(tension,[1.,1.1],atol=1e-12)
        np.testing.assert_allclose(force,[[.1,0.,.196]],atol=1e-12)
        _,end,tension=law(np.array([[.9,0.,0.]]),np.zeros((1,3)),np.array([1.8,0.,0.]),
            np.array([-.2,0.,0.]),np.zeros(3),**kw)
        np.testing.assert_array_equal(tension,[0.,0.]);np.testing.assert_array_equal(end,[0.,0.,0.])


if __name__=='__main__':unittest.main()
