import numpy as np
import matplotlib.pyplot as plt

#              x y b l    b = bias, l=label
xorFunction = [[1,1,1,0],
               [1,0,1,1],
               [0,1,1,1],
               [0,0,1,0]]

xorFunction1 =[[1,1,1,0],
               [1,0,1,1],
               [0,1,1,1],
               [0,0,1,1]]
xorFunction2 =[[1,1,1,1],
               [1,0,1,1],
               [0,1,1,1],
               [0,0,1,0]]

for i in range(len(xorFunction1)):
    itemForPlot = xorFunction[i]
    plt.plot(itemForPlot[0],itemForPlot[1], 'ro',ms=10)

plt.plot(xorFunction[0][0],xorFunction[0][1],'bo',ms=10)
plt.plot(xorFunction[3][0],xorFunction[3][1],'bo',ms=10)
plt.ylim((-1,2))
plt.xlim((-1,2))
plt.show()


xorWeights = [0,0,0]


#dot product:
def netValue (weights, inputs):
    netVal = weights[0] * inputs[0] + weights[1] * inputs[1] + weights[2] * inputs[2]
    return netVal


#Sigmoid Activation:
def sigmoid(x):
    return 1.0/(1.0 + np.exp(-x))

def sigmoidUpdate(weights, inputs, learningRate):
    w1 = weights[0]
    w2 = weights[1]
    w3 = weights[2]
    dataLabel = inputs[3]
    netVal = netValue(weights, inputs[0:3])
    actVal = sigmoid(netVal)
    w1 = w1 + learningRate*inputs[0]*(dataLabel-actVal)*(actVal)*(1-actVal)
    w2 = w2 + learningRate*inputs[1]*(dataLabel-actVal)*(actVal)*(1-actVal)
    w3 = w3 + learningRate*inputs[2]*(dataLabel-actVal)*(actVal)*(1-actVal)
    newWeights = [w1, w2, w3]
    return newWeights


xorWeights = [0,0,0]
learningRate = 0.09
epoch = 5000

for i in range(epoch):
    for j in range(4):
        inputs = xorFunction1[j]
        xorWeights = sigmoidUpdate(xorWeights, inputs, learningRate)

forTesting1  = xorWeights

xorWeights = [0,0,0]
for i in range(epoch):
    for j in range(4):
        inputs = xorFunction2[j]
        xorWeights = sigmoidUpdate(xorWeights, inputs, learningRate)

forTesting2  = xorWeights

#hidden layer
Input1=[[x,-(forTesting1[0] / forTesting1[1])*x  -(forTesting1[2] / forTesting1[1]),1,1 ]for x in list(np.linspace(-1, 2, 1000))]
Input2=[[x,-(forTesting2[0] / forTesting2[1])*x  -(forTesting2[2] / forTesting2[1]),1,0 ]for x in list(np.linspace(-1, 2, 1000))]
Input1.extend(Input2)
weights3 = [0,0,0]
for i in range(epoch):
    for j in range(4):
        inputs = Input1[j]
        weights3 = sigmoidUpdate(weights3, inputs, learningRate)
        
forTesting  = weights3

for i in range(len(xorFunction)):
    itemForPlot = xorFunction[i]
    plt.plot(itemForPlot[0],itemForPlot[1], 'ro',ms=10)

plt.plot(xorFunction[0][0],xorFunction[0][1],'bo',ms=10)
plt.plot(xorFunction[3][0],xorFunction[3][1],'bo',ms=10)
plt.ylim((-1,2))
plt.xlim((-1,2))

x = np.linspace(-1, 2, 1000)
plt.plot(x,-(forTesting1[0] / forTesting1[1])*x  -(forTesting1[2] / forTesting1[1]))
plt.plot(x,-(forTesting2[0] / forTesting2[1])*x  -(forTesting2[2] / forTesting2[1]))
plt.plot(x,-(forTesting[0] / forTesting[1])*x  -(forTesting[2] / forTesting[1]))
plt.title('xorFunction')
plt.show()
