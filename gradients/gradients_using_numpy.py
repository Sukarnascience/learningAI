import numpy as np 

# for linear regression the formula will be for forward
# f = w * X

# tranning data
x = np.array([1,2,3,4],dtype=np.float32)
# eg: f = 2 * x So,since our formula is 2X so ... 
y = np.array([2,4,6,8],dtype=np.float32)
w = 0.0

# model prediction
def forward(x):
    return w*x

# loss = mean
def loss(y,y_predicted):
    return ((y_predicted-y)**2).mean()

# gradient manual
# mean = 1/n*(w*x-y)**2
# dj/dw = 1/n*(2*x(w*x-y))
def gradient(x,y,y_predicted):
    return np.dot(2*x,y_predicted-y).mean()

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
    dw = gradient(x,y,y_pred)
    # update weights
    w -= learning_rate*dw

    print(f"epoch :{epoch+1} ,weights :{w} ,loss :{ls}")

print(f"Prediction after tranning: f(5) = {forward(5)}")
