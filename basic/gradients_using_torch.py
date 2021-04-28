import torch

# for linear regression the formula will be for forward
# f = w * X

# tranning data
x = torch.tensor([1,2,3,4],dtype=torch.float32)
# eg: f = 2 * x So,since our formula is 2X so ... 
y = torch.tensor([2,4,6,8],dtype=torch.float32)
w = torch.tensor(0.0,requires_grad=True,dtype=torch.float32)

# model prediction
def forward(x):
    return w*x

# loss = mean
def loss(y,y_predicted):
    return ((y_predicted-y)**2).mean()

# gradient will be now not be calculated manually 

print(f"Prediction before tranning: f(5) = {forward(5)}")

# tranning 
learning_rate = 0.01
no_iteration = 20

for epoch in range(no_iteration):
    # prediction = forword pass
    y_pred = forward(x)
    # loss
    ls = loss(y,y_pred)
    # gradients 
    # dj/dw
    ls.backward()
    # update weights
    with torch.no_grad():
        w -= learning_rate*w.grad
    # zero gradient to privent accumilation in weights
    w.grad.zero_()

    print(f"epoch :{epoch+1} ,weights :{w} ,loss :{ls}")

print(f"Prediction after tranning: f(5) = {forward(5)}")