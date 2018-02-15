import numpy as np
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
import pdb

#in the case the std are assume to be identical across classes
#the decision boundary is a linear function, a straight line.
#if we calculated the std for each class, the decision boundary
#is closed to the real boundry of each classes.
x1 = np.random.normal(0, 1, 200)
y1 = np.random.normal(0, 2, 200)

x2 = np.random.normal(4, 0.1, 200)
y2 = np.random.normal(5, 0.2, 200)

plt.plot(x1,y1,'bo',ms=10)
plt.plot(x2,y2,'ro',ms=10)
plt.ylim((-7,7))
plt.xlim((-7,7))
plt.title('Generic Gaussian Data')
plt.xlabel('x')
plt.ylabel('y')
plt.legend(['Class 1','Class 2'], numpoints=1,loc = 'upper left')
plt.show()


probClass1 = 1/2
probClass2 = 1/2

muX_given1 = sum(x1)/float(200)
muY_given1 = sum(y1)/float(200)

muX_given2 = sum(x2)/float(200)
muY_given2 = sum(y2)/float(200)
#std for different classes
sigX1 = np.sqrt(sum((x1-muX_given1)**2)/float(200))
sigY1 = np.sqrt(sum((y1-muY_given1)**2)/float(200))
sigX2 = np.sqrt(sum((x2-muX_given2)**2)/float(200))
sigY2 = np.sqrt(sum((y2-muY_given2)**2)/float(200))


g1 = np.arange(-6,7,0.1)
g2 = np.arange(-6,7,0.1)
gg1,gg2 = np.meshgrid(g1,g2)

testGridG = np.array((gg1.ravel(), gg2.ravel())).T

plt.plot(testGridG[:,0],testGridG[:,1],'go',ms=10)
plt.plot(x1,y1,'bo',ms=10)
plt.plot(x2,y2,'ro',ms=10)
plt.ylim((-7,7))
plt.xlim((-7,7))
plt.title('Generic Gaussian Data')
plt.xlabel('x')
plt.ylabel('y')
plt.legend(['Test Points','Class 1','Class 2'], numpoints=1,loc = 'upper left')
plt.show()



class1Points = np.empty(shape=[0, 2])
class2Points = np.empty(shape=[0, 2])

for dataPoint in testGridG:
    prob1 = probClass1*mlab.normpdf(dataPoint[0], muX_given1, sigX1)*mlab.normpdf(dataPoint[1], muY_given1, sigY1)
    prob2 = probClass2* mlab.normpdf(dataPoint[0], muX_given2, sigX2)*mlab.normpdf(dataPoint[1], muY_given2, sigY2)
    #pdb.set_trace()
    #add bias to solve runtime error
    logOdds = np.log(prob1+1) - np.log(prob2+1)
    if (logOdds >= 0):
        class1Points = np.append(class1Points, [dataPoint], axis=0)
    else:
        class2Points = np.append(class2Points, [dataPoint], axis=0)


plt.plot(class1Points[:,0],class1Points[:,1],'bo',ms=10)
plt.plot(class2Points[:,0],class2Points[:,1],'ro',ms=10)
plt.ylim((-7,7))
plt.xlim((-7,7))
plt.title('Generic Gaussian Data')
plt.xlabel('x')
plt.ylabel('y')
plt.legend(['Class 1','Class 2'], numpoints=1,loc = 'upper left')
plt.show()

