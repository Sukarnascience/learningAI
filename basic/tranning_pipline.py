import torch
import torch.nn as nn

x = torch.tensor([[1],[2],[3],[4]],dtype=torch.float32)
y = torch.tensor([[2],[4],[6],[8]],dtype=torch.float32)
x_test = torch.tensor([5],dtype=torch.float32)
no_samples,no_feature = x.shape

input_size = no_feature
output_size = no_feature

# make models
# use defalt : model = nn.Linear(input_size,output_size)
class NeuralNetwork(nn.Module):
    def __init__(self,input_D,output_D):
        super(NeuralNetwork,self).__init__()
        self.lin = nn.Linear(input_D,output_D)
    def forward(self,x):
        return self.lin(x)

model = NeuralNetwork(input_size,output_size)

print(f"Prediction before tranning: f(5) = {model(x_test).item()}")

# tranning 
learning_rate = 0.01
no_iteration = 20
loss = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(),lr=learning_rate)

for epoch in range(no_iteration):
    y_pred = model(x)
    ls = loss(y,y_pred)
    ls.backward()
    optimizer.step()
    optimizer.zero_grad()
    [w,b] = model.parameters()
    print(f"epoch :{epoch+1} ,weights :{w[0][0].item()} ,loss :{ls}")

print(f"Prediction after tranning: f(5) = {model(x_test).item()}")
