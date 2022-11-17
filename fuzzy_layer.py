#%%
import torch
import torch.nn as nn
import numpy as np
#%%
size_in = 4
size_out = 3
x = torch.FloatTensor([0, 1, 0, 0])
initial_scales = [[1,1,1,1],[2,2,2,2],[3,3,3,3]]
initial_centers = [[0,-1,0,0],[4,4,4,4],[5,5,5,5]]

diags = []
for s,c in zip(initial_scales, initial_centers):
    diags.append(np.insert(np.diag(s), size_in, c, axis = 1))
a = torch.FloatTensor(diags)

const_row = np.zeros(size_in+1)
const_row[size_in] = 1
const_row = np.array([const_row]*size_out)
const_row = np.reshape(const_row, (size_out, 1, size_in+1))
c_r = torch.FloatTensor(const_row)
c_one = torch.FloatTensor([1])

ta = torch.cat([a,c_r],1)
tx = torch.transpose(torch.reshape(torch.cat([x,c_one]), (1,size_in+1)),0,1)
mul = torch.matmul(ta, tx)
exponents = torch.norm(mul[:,:size_in],p=2,dim=1)
final = torch.exp(-exponents)
final.flatten()
#%%
class FuzzyLayer(torch.nn.Module):

    def __init__(self,
                 size_in, size_out, 
                 initial_centers=None,
                 initial_scales=None, ):
        """
        mu_j(x,a,c) = exp(-|| a . x ||^2)
        """
        super().__init__()

        self.size_in, self.size_out = size_in, size_out

        diags = []
        for s,c in zip(initial_scales, initial_centers):
            diags.append(np.insert(np.diag(s), size_in, c, axis = 1))
        a = torch.FloatTensor(diags)

        const_row = np.zeros(size_in+1)
        const_row[size_in] = 1
        const_row = np.array([const_row]*size_out)
        const_row = np.reshape(const_row, (size_out, 1, size_in+1))
        self.c_r = torch.FloatTensor(const_row)
        self.c_one = torch.FloatTensor([1])
        self.A = nn.Parameter(a) 

    def forward(self, x):
        ta = torch.cat([self.A, self.c_r],1)
        tx = torch.transpose(torch.reshape(torch.cat([x,self.c_one]), (1,self.size_in+1)),0,1)
        mul = torch.matmul(ta, tx)
        exponents = torch.norm(mul[:,:self.size_in], p=2, dim=1)
        memberships = torch.exp(-exponents)
        return memberships.flatten()

    
# %%
model = FuzzyLayer(4,3,initial_centers, initial_scales)
criterion = torch.nn.MSELoss(reduction='sum')
optimizer = torch.optim.SGD(model.parameters(), lr=1e-6)
for t in range(10):
    
    y_pred = model(x)

    loss = criterion(y_pred, torch.FloatTensor([0.8, 0.0, 0.0]))
    print(t, loss.item())

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
