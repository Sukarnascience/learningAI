'''
    Author : Sukarna Jana
    Date   : 28/04/2021
    Topic  : Linear Regression

    Steps:
        1. Design model (input,output,forward pass)
        2. Construct loss and optimizer
        3. Training loop:
            - forward pass: compute prediction and loss
            - backward pass: gradients
            - update weights
'''

import torch
import torch.nn as nn 
import numpy as np 
import matplotlib.pyplot as plt 
# we need data to work with so we will take it from sklearn
from sklearn import datasets
# we need to generate a regression dataset

# Prepare data
x_data,y_data = datasets.make_regression(n_samples=200,n_features=1,noise=40,random_state=1)
# it will be in numpy So, lets convert into tensor
# x_data is a double data type so lets convert to float
x = torch.from_numpy(x_data.astype(np.float32))
y = torch.from_numpy(y_data.astype(np.float32))
# we need to reshape our 'y' becasue it's has only 1Row we need to make it into Col. Vector each value in 1Row
y = y.view(y.shape[0],1)

n_samples,n_features = x.shape

# lets create model
class linearRegression(nn.Module):
    def __init__(self,inputSize,outputSize):
        super(linearRegression,self).__init__()
        self.lin = nn.Linear(inputSize,outputSize)
    def forward(self,x):
        return self.lin(x)

input_size = n_features
output_size = 1  # we need only 1 value for each sample
model = linearRegression(input_size,output_size)

# loss and optimizer
# we will use biltin loss function from pytorch : we need MSE(Mean Square Error) for linear regression
criterion = nn.MSELoss()
learning_rate = 0.01
optimizer = torch.optim.SGD(model.parameters(),lr=learning_rate) # SGD(stachastic gradient descent)

# tranning loop
num_epochs = 2000
for epoch in range(num_epochs):
    # forward pass
    y_predicted = model(x)
    # loss
    loss = criterion(y_predicted,y)
    # backward pass
    loss.backward()
    # update 
    optimizer.step()
    # empty 
    optimizer.zero_grad()

    # lets print some info to see what is going on
    print(f'epoch: {epoch+1}, loss: {loss.item():.4f}') # :.4f means we want to see only 4 decimal value

# now lets plot in graph and see
# we will convert into numpy but before that we need to detach out tensor for 
# preventing this operation form being track in our computation graphs 
predicted = model(x).detach().numpy()

plt.plot(x_data,y_data,'bo')
plt.plot(x_data,predicted,'red')
plt.savefig('Linear_Regression')
plt.show()
