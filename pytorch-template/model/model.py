import torch.nn as nn
import torch.nn.functional as F
from base import BaseModel
from trainer import c1Linear, c2Linear
from torch.nn import Linear

class sModel(BaseModel):
    def __init__(self, inputt, hidden=8, num_classes=5):
        super().__init__()
        self.fc1 = nn.Linear(inputt, hidden)
        self.fc2 = nn.Linear(hidden, num_classes)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)

class Model(BaseModel):
    def __init__(self, inputt=10, hidden=30, num_classes=3):
        super().__init__()
        self.relu = nn.ReLU()
        
        self.fc1 = c1Linear(inputt, hidden, bias=True)
        self.fc2 = c1Linear(hidden, num_classes, bias=True)
        
    def forward(self, x):
        x2=self.relu(self.fc1(x))
        out=self.fc2(x2)
        return F.log_softmax(out, dim=1) 

class Model2(BaseModel):
    def __init__(self, inputt=10, hidden=30, num_classes=3):
        super().__init__()
        self.relu = nn.ReLU()
        
        self.fc1 = c2Linear(inputt, hidden, bias=True)
        self.fc2 = c2Linear(hidden, num_classes, bias=False)
        
    def forward(self, x):
        x2=self.relu(self.fc1(x))
        out=self.fc2(x2)
        return F.log_softmax(out, dim=1) 

