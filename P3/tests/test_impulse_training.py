import unittest
import numpy as np
from P3.tests.test_aligned_residual import ROOT
import json


class ImpulseTrainingTests(unittest.TestCase):
    def test_balanced_basis_separates_collective_force_from_differential_motors(self):
        from P3.impulse_train import fit_features,export_weights
        features=np.arange(56,dtype=float).reshape(2,28)/10
        shifted=features.copy();shifted[:,13:17]+=10
        x,indices=fit_features(features,balanced=True)
        y,_=fit_features(shifted,balanced=True)
        np.testing.assert_allclose(x,y,atol=1e-12)
        compact=np.arange(3*len(indices),dtype=float).reshape(3,-1)/10
        weights=export_weights(compact,indices,balanced=True)
        np.testing.assert_allclose(weights[:,13:17].sum(1),0,atol=1e-12)
        np.testing.assert_allclose(features@weights.T,x@compact.T,atol=1e-12)

    def test_own_increment_target_uses_current_gyro_only_as_supervision(self):
        from P3.impulse_train import runtime_target
        nominal=json.loads((ROOT/'px4_offboard/acceleration_dual/assets/config.json').read_text())['nominal']
        data=dict(previous_observed_gyro=np.zeros((2,3)),previous_filtered_force=np.tile([0.,0.,-9.8],(2,1)),
            filtered_force=np.tile([0.,0.,-9.8],(2,1)),previous_rpm=np.full((2,4),530.),rpm=np.full((2,4),530.),
            observed=np.array([[.01,0.,0.,0.,0.,-9.8],[.02,0.,0.,0.,0.,-9.8]]),
            sensor_x=np.zeros((2,7)),dt=np.full(2,.01),reference_alpha=np.full((2,3),999.))
        features,target=runtime_target(data,np.zeros((6,7)),nominal)
        np.testing.assert_allclose(target,[[1.,0.,0.],[2.,0.,0.]],atol=1e-12)
        data['observed'][:,:3]*=2
        other,new_target=runtime_target(data,np.zeros((6,7)),nominal)
        np.testing.assert_array_equal(features,other)
        np.testing.assert_allclose(new_target,2*target,atol=1e-12)

