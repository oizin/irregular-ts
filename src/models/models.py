from base import BaseModel
from base import GaussianOutputNN
from torchctrnn import *

class latentJumpNN(BaseModel):

    def __init__(self,LatentJumpODECell,GaussianOutputNN):
        BaseModel.__init__(self,LatentJumpODECell,GaussianOutputNN)

        