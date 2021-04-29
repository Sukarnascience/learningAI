'''
    Author : Sukarna Jana
    Date   : 29/04/2021
    Topic  : Logistic Regression

    Steps:
        1. Design model (input,output,forward pass)
        2. Construct loss and optimizer
        3. Training loop:
            - forward pass: compute prediction and loss
            - backward pass: gradients
            - update weights
'''

import torch
# for neural networking we use torch.nn
import torch.nn as nn 
# for some data transformation we will use np
import numpy as np
# load some binary classification data set
from sklearn import datasets
# for scaling our feature
from sklearn.preprocessing import StandardScaler
# for spelliting our training and testing data
from sklearn.model_selection import train_test_split

# lets prepare our data
# lets load brest cancer dataset (its our binary classification)
bc = datasets.load_breast_cancer()
x = bc.data
y = bc.target
n_samples,n_features = x.shape

# let's split our data into training data & testing data
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=1234)

# let's scale our features
# by using StandardScaler() fun. which will make our features to have 0 mean & unit
# veriene ( this is always recommended to do & when we want to deal with a logistic regression )
sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test)
# let's convert into tensor
x_train = torch.from_numpy(x_train.astype(np.float32))
x_test = torch.from_numpy(x_test.astype(np.float32))
y_train = torch.from_numpy(y_train.astype(np.float32))
y_test = torch.from_numpy(y_test.astype(np.float32))
# need to reshape our y
y_train = y_train.view(y_train.shape[0],1)
y_test = y_test.view(y_test.shape[0],1)

# let's create our model
# (our model is a linear combination with logistic fun. where we put sigmoid at the end)
# f = wx + b , sigmoid at the end
class logisticRegression(nn.Module):
    def __init__(self,n_input_features):
        super(logisticRegression,self).__init__()
        self.lin = nn.Linear(n_input_features,1)
    def forward(self,x):
        y_predicted = torch.sigmoid(self.lin(x))
        return y_predicted

model = logisticRegression(n_features)

# loss and optimizer
# for checking loss this time we will use BCE(Binary Cross Entropy)
# for optimizer we will use the same SGD(Stochastic Gradient Descent)
learning_rate = 0.01
criterion = nn.BCELoss()
optimizer = torch.optim.SGD(model.parameters(),lr=learning_rate)

# let's train
num_epochs = 5000
for epoch in range(num_epochs):
    # forward pass
    y_predicted = model(x_train)
    # loss
    loss = criterion(y_predicted,y_train)
    # backward pass
    loss.backward()
    # update
    optimizer.step()
    optimizer.zero_grad()
    # lets print some data to see what is going on...
    print(f'epoch :{epoch}, loss :{loss.item():4f}')

# let's eveluate our model
# So, the evaluation should not be the part of our computational graph
# where we want to prevent the track of the history...
with torch.no_grad():
# here we want to see the accuracy
    y_predicted = model(x_test)
# due to our sigmoid function it will return a value b/w : 0 & 1
    y_predicted_cls = y_predicted.round()
# accuracy
    acc = y_predicted_cls.eq(y_test).sum()/float(y_test.shape[0])
    print(f'accuracy :{acc:.4f}')
