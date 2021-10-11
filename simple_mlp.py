import torch
import torch.nn as nn
import torch.nn.functional as F
from dataloader import synthetic_example
import matplotlib.pyplot as plt


device = 'cuda'
class NN(nn.Module):
    def __init__(self):
        super(NN,self).__init__()
        self.fc1 = nn.Linear(2,8)
        self.fc2 = nn.Linear(8,2)
    
    def forward(self,x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

net = NN().to(device)
dataset = synthetic_example()
train_loader = torch.utils.data.DataLoader(dataset, batch_size=4)
optimizer = torch.optim.Adam(net.parameters(),lr=1e-3,weight_decay=1e-9)
criterion = F.mse_loss

for e in range(5):
    total_loss = 0
    for batch_in,batch_out in train_loader:
        prediction = net(batch_in.float().to(device))
        loss = criterion(prediction,batch_out.float().to(device))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss +=  loss
    total_loss /= len(train_loader)
    print("Epoch : %d, Loss: %.3f"%(e,total_loss))

net.eval()
traj_x = []
traj_y = []
x = 1.
y = 2.
for i in range(50):
    traj_x.append(x)
    traj_y.append(y)
    with torch.no_grad():
        input = torch.FloatTensor([x,y]).unsqueeze(0)
        prediction = net(input.to(device))
        dx = prediction[0][0].cpu().item()
        dy = prediction[0][1].cpu().item()
        x += dx
        y += dy
plt.figure()
plt.ylim(0,4)
plt.xlim(-2,2)
plt.scatter(traj_x,traj_y)
plt.show()