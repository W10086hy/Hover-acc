"""Offline prediction must not consume future measured outputs."""
import importlib.util,unittest
import torch
from P2.hover_bptt.environment import plant_config
from P2.hover_bptt.model import ExactWingOffBodyDragModel
from P3.mismatch_dynamics import DynamicsResidual
from P2.hover_bptt.onboard import NominalVehicle

class MismatchPredictionTests(unittest.TestCase):
    def test_given_input_forecast_cannot_read_future_targets(self):
        name='P3.mismatch_forecast'
        self.assertIsNotNone(importlib.util.find_spec(name),'Mismatch forecasting module missing')
        m=__import__(name,fromlist=['predict_window']);cfg=plant_config();model=ExactWingOffBodyDragModel(cfg,device='cpu',arms=torch.tensor([[0.,0.,.05]]),lengths=torch.tensor([4.9]))
        state=model.initial(1);obs=torch.zeros(1,35);obs[:,0]=1.;obs[:,10:14]=state[:,13:17]/model.max_speed
        obs[:,7:10]=model.loads(state,torch.zeros(1,3))[2]/20
        packet=torch.cat((state[:,6:10],state[:,10:13],obs[:,7:10]*20,state[:,13:17],-state[:,2:3]),-1)
        batch={'initial_state':state,'initial_packet':packet,'obs':obs,'hidden':torch.zeros(1,128),'context':torch.zeros(1,14),'commands':torch.full((1,2,4),.5),'wind':torch.zeros(1,2,3),'future_states':torch.zeros(1,2,19),'future_packets':torch.zeros(1,2,15)}
        net=DynamicsResidual({'width':32});first=m.predict_window(net,model,cfg,batch,learned=True)
        batch['future_states'].fill_(999.);batch['future_packets'].fill_(999.)
        second=m.predict_window(net,model,cfg,batch,learned=True)
        self.assertTrue(torch.equal(first['states'],second['states']));self.assertTrue(torch.equal(first['packets'],second['packets']))
        batch['commands'][:,1]=.7;third=m.predict_window(net,model,cfg,batch,learned=True)
        self.assertTrue((third['states'][:,1,13:17]>first['states'][:,1,13:17]).all())

if __name__=='__main__':unittest.main()
