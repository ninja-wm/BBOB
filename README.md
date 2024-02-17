
**A implement of BBOB functions based on Pytorch.**

# Installation
```powershell
cd BBOB
./install.sh
```

# About BBOB
For an introduction to BBOB, please refer to the official introduction document **bbobdocfunctions.pdf**

# Example
## BBOB Functions
```python

from tqdm import tqdm
import numpy as np
import BBOB.bbobfunctions as BBOB
from BBOB.utils import getFitness,genOffset,setOffset,getOffset
import torch
from imports import *
from GLHF.problem import Problem
#define your task
class taskProblem(Problem):
    def __init__(self,fun=None,repaire=True,dim=None):
        super().__init__()
        self.fun=fun
        self.useRepaire=repaire
        self.dim=dim

    def repaire(self,x):
        xlbmask=torch.zeros_like(x,device=DEVICE)
        xlbmask[x<self.fun['xlb']]=1
        normalmask=1-xlbmask
        xlbmask=xlbmask*self.fun['xlb']
        x=normalmask*x+xlbmask

        xubmask=torch.zeros_like(x,device=DEVICE)
        xubmask[x>self.fun['xub']]=1
        normalmask=1-xubmask
        xubmask=xubmask*self.fun['xub']
        x=normalmask*x+xubmask
        return x
    
    def calfitness(self,x):
        if self.useRepaire:
            x1=self.repaire(x)
        else:
            x1=x
        b,n,d=x.shape
        x1=x1.view((-1,d))
        r=getFitness(x1,self.fun)   #b,n,1
        r=torch.unsqueeze(r,-1)
        r=r.view((b,n,1))
        return r
    
    
    def genRandomPop(self,batchShape):
        lb=self.fun['xlb'] 
        ub=self.fun['xub']
        return torch.rand(batchShape,device=DEVICE)*(ub-lb)+lb

    def reoffset(self):
        genOffset(self.dim,self.fun)
        
        
    def setOffset(self,offset):
        for key in offset.keys():
            self.fun[key]=offset[key]
    

    def getfunname(self):
        return self.fun['fid']
    
    def setfun(self,fun):
        self.fun=fun
        
#
def test(fid,x):
    f=BBOB.FUNCTIONS[fid]
    f['xlb']=5 #lower bound
    f['xub']=5 #upper bound
    genOffset(x.shape[-1],f)
    if not fid in [5,24]:
        f['xopt']=torch.zeros((x.shape[-1],)).to(DEVICE)
    task=taskProblem(fun=f,repaire=True,dim=x.shape[-1])
    value=task.calfitness(x)
    return value

if __name__=='__main__':
    dim=10
    batch=2
    popsize=3
    x=torch.randn((batch,popsize,dim),device=DEVICE) 
    vals=test(2,x)
    print(vals)
```

## CEC Functions

```python
from tqdm import tqdm
import numpy as np
from BBOB.cecfunctions import FUNCTIONS as F
from BBOB.utils import getFitness,genOffset
import torch
from GLHF.problem import Problem

#define your task

DEVICE=torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

class taskProblem(Problem):
    def __init__(self,fun=None,repaire=True,dim=None):
        super().__init__()
        self.fun=fun
        self.useRepaire=repaire
        self.dim=dim

    def repaire(self,x):
        xlbmask=torch.zeros_like(x,device=DEVICE)
        xlbmask[x<self.fun['xlb']]=1
        normalmask=1-xlbmask
        xlbmask=xlbmask*self.fun['xlb']
        x=normalmask*x+xlbmask

        xubmask=torch.zeros_like(x,device=DEVICE)
        xubmask[x>self.fun['xub']]=1
        normalmask=1-xubmask
        xubmask=xubmask*self.fun['xub']
        x=normalmask*x+xubmask
        return x
    
    def calfitness(self,x):
        if self.useRepaire:
            x1=self.repaire(x)
        else:
            x1=x
        b,n,d=x.shape
        # x1=x1.view((-1,d))
        r=getFitness(x1,self.fun)   #b,n,1
        r=torch.unsqueeze(r,-1)
        r=r.view((b,n,1))
        return r
    
    def genRandomPop(self,batchShape):
        lb=self.fun['xlb'] 
        ub=self.fun['xub']
        return torch.rand(batchShape,device=DEVICE)*(ub-lb)+lb

    def reoffset(self):
        genOffset(self.dim,self.fun)
        
        
    def setOffset(self,offset):
        for key in offset.keys():
            self.fun[key]=offset[key]
    

    def getfunname(self):
        return self.fun['fid']
    
    def setfun(self,fun):
        self.fun=fun
        
#
def test(fid,x):
    f=F['cecf%d'%fid]
    problem=taskProblem(fun=f,dim=dim,repaire=True)  
    genOffset(x.shape[-1],f)
    task=taskProblem(fun=f,repaire=True,dim=x.shape[-1])
    value=task.calfitness(x)
    return value


if __name__=='__main__':
    dim=10
    batch=2
    popsize=3
    x=torch.randn((batch,popsize,dim),device=DEVICE) 
    vals=test(2,x)
    print(vals)
```
