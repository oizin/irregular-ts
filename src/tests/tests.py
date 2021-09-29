import unittest
import torch
from ..models.output import GaussianOutputNN
from ..models.odenet import FF1
from torchctrnn import ODERNNCell
from ..models.base import BaseModel

class TestOutputNN(unittest.TestCase):

    def test_gaussian_output_loss(self):
        hidden_dim = 10
        batch_size = 5
        model = GaussianOutputNN(hidden_dim)
        z = torch.randn(batch_size,hidden_dim)
        output = model(z)
        self.assertEqual(output.size(), (batch_size,2))
        m,s = output[:,0],output[:,1]
        y = torch.rand(batch_size)
        msk = torch.ones(batch_size).bool()
        self.assertEqual(y[msk].size(), m[msk].size())
        loss = model.loss_fn(m[msk],s[msk],y[msk])
        self.assertEqual(loss.size(), ())
        
class TestODENet(unittest.TestCase):

    def test_ff1_output(self):
        hidden_dim = 10
        feature_dim = 8
        batch_size = 5
        model = FF1(hidden_dim,feature_dim)
        z = torch.randn(batch_size,hidden_dim)
        x = torch.randn(batch_size,feature_dim)
        t = torch.zeros(1)
        output = model(x,t,z)
        self.assertEqual(output.size(), (batch_size,hidden_dim))

class TestFullModel(unittest.TestCase):
        
    def test_full_model_output_loss(self):
        batch_size = 5
        hidden_dim = 6
        seq_len = 10
        feature_dim = 8
        ode_feature_dim = 0
        odenet = FF1(hidden_dim,ode_feature_dim)
        odernn = ODERNNCell(odenet,feature_dim)
        gaussianNN = GaussianOutputNN(hidden_dim)
        model = BaseModel(odernn,gaussianNN)
        x = torch.randn(batch_size,seq_len,feature_dim)
        dt = torch.tensor([0.,1.]).expand(batch_size,seq_len,2)
        output = model(dt,x)
        self.assertEqual(output.size(), (batch_size,seq_len,2))
        y = torch.rand(batch_size,seq_len)
        msk = torch.ones(batch_size,seq_len).bool()
        m,s = output[:,:,0],output[:,:,1]
        self.assertEqual(m[msk].size(), y[msk].size())
        loss = model.loss_fn(m[msk],s[msk],y[msk])
        self.assertEqual(loss.size(), ())

if __name__ == '__main__':
    unittest.main()