import numpy as np
from scipy.optimize import minimize
from scipy.io import loadmat
from numpy.linalg import det, inv
from math import sqrt, pi
import scipy.io
import matplotlib.pyplot as plt
import pickle
import sys

def ldaLearn(X,y):
    # Inputs
    # X - a N x d matrix with each row corresponding to a training example
    # y - a N x 1 column vector indicating the labels for each training example
    #
    # Outputs
    # means - A d x k matrix containing learnt means for each of the k classes
    # covmat - A single d x d learnt covariance matrix 
    
    # IMPLEMENT THIS METHOD 
    
    #Extracting the data sets
    dataList = np.array(X)
    dataLabels = np.array(y)
    #Obtaining number of rows and columns
    Xrow=dataList.shape[0]
    Xcolumn=dataList.shape[1]
    Yrow=dataLabels.shape[0]
    Ycolumn=dataLabels.shape[1]
    #Converting array to Tuple
    labelList = y.reshape(y.size)
    #Obtaining the unique range of values for labellist
    labelRange = np.unique(labelList)
    #Initialising and computingthe mean and covariance
    means=np.zeros((Xcolumn,labelRange.size))
    for l in range(labelRange.size):
        means[:,l] = np.mean(dataList[labelList == labelRange[l]],axis=0)
    covmat = np.cov(X , rowvar = 0)
    return means,covmat

def qdaLearn(X,y):
    # Inputs
    # X - a N x d matrix with each row corresponding to a training example
    # y - a N x 1 column vector indicating the labels for each training example
    #
    # Outputs
    # means - A d x k matrix containing learnt means for each of the k classes
    # covmats - A list of k d x d learnt covariance matrices for each of the k classes
    
    # IMPLEMENT THIS METHOD

    #Extracting the data sets
    dataList = np.array(X)
    dataLabels = np.array(y)
    #Obtaining number of rows and columns
    Xrow=dataList.shape[0]
    Xcolumn=dataList.shape[1]
    Yrow=dataLabels.shape[0]
    Ycolumn=dataLabels.shape[1]
    #Converting array to Tuple
    labelList = y.reshape(y.size)
    #Obtaining the unique range of values for labellist
    labelRange = np.unique(labelList)
    #Initialising and computingthe mean and covariance
    means=np.zeros((Xcolumn,labelRange.size))
    covmats=[np.zeros((Xcolumn,Xcolumn))]*labelRange.size
    ##mean matrix is 2*5 one row represents x mean,other y mean 
    for i in range(labelRange.size):
        means[:,i]=np.mean(dataList[labelList==labelList[i]],axis=0)
        covmats[i]=np.cov(dataList[labelList==labelList[i]],rowvar=0)
    return means,covmats

def ldaTest(means,covmat,Xtest,ytest):
    # Inputs
    # means, covmat - parameters of the LDA model
    # Xtest - a N x d matrix with each row corresponding to a test example
    # ytest - a N x 1 column vector indicating the labels for each test example
    # Outputs
    # acc - A scalar accuracy value
    # ypred - N x 1 column vector indicating the predicted labels

    # IMPLEMENT THIS METHOD

    inverse_covmat = np.linalg.inv(covmat)
    covmat_determinant = np.linalg.det(covmat)
    yPredA= np.zeros((Xtest.shape[0],means.shape[1]))
    for i in range(means.shape[1]):
        yPredA[:,i] = np.exp(-0.5*np.sum((Xtest - means[:,i])* 
        np.dot(inverse_covmat, (Xtest - means[:,i]).T).T,1))/(np.sqrt(np.pi*2)*(np.power(covmat_determinant,2)))
    #Getting the index of the class with the highest probability
    label = np.argmax(yPredA,1)
    #Index start from 0,class index start from 1.So to balance the index adding 1 to all the index
    label = label + 1
    ytest = ytest.reshape(ytest.size)
    
    acc = 100*np.mean(label == ytest) 
    return acc,yPredA

def qdaTest(means,covmats,Xtest,ytest):
    # Inputs
    # means, covmats - parameters of the QDA model
    # Xtest - a N x d matrix with each row corresponding to a test example
    # ytest - a N x 1 column vector indicating the labels for each test example
    # Outputs
    # acc - A scalar accuracy value
    # ypred - N x 1 column vector indicating the predicted labels

    # IMPLEMENT THIS METHOD
    
    ypred= np.zeros((Xtest.shape[0],means.shape[1]))
    for i in range(means.shape[1]):
        inverse_covmat = np.linalg.inv(covmat)
        covmat_determinant = np.linalg.det(covmat)
        ypred[:,i] = np.exp(-0.5*np.sum((Xtest - means[:,i])* 
        np.dot(inverse_covmat, (Xtest - means[:,i]).T).T,1))/(np.sqrt(np.pi*2)*(np.power(covmat_determinant,2)))
    #Getting the index of the class with the highest probability
    label = np.argmax(ypred,1)
    #Index start from 0,class index start from 1.So to balance the index adding 1 to all the index
    label = label + 1
    ytest = ytest.reshape(ytest.size)
    
    acc = 100*np.mean(label == ytest) 
    return acc,ypred

def learnOLERegression(X,y):
    # Inputs:                                                         
    # X = N x d 
    # y = N x 1                                                               
    # Output: 
    # w = d x 1 
	
    # IMPLEMENT THIS METHOD 
    X_transpose = np.transpose(X)

    transpose_product = np.dot(X_transpose, X)

    inverse_Product = inv(transpose_product)

    XtransposeY = np.dot(X_transpose, y)

    w = np.dot(inverse_Product,XtransposeY )        
                             
    return w                                                  

def learnRidgeRegression(X,y,lambd):
    # Inputs:
    # X = N x d                                                               
    # y = N x 1 
    # lambd = ridge parameter (scalar)
    # Output:                                                                  
    # w = d x 1                                                                

    # IMPLEMENT THIS METHOD                                                   
    return w

def testOLERegression(w,Xtest,ytest):
    # Inputs:
    # w = d x 1
    # Xtest = N x d
    # ytest = X x 1
    # Output:
    # mse
    
    # IMPLEMENT THIS METHOD
    Xtestrows = Xtest.shape[0]
    diff = np.subtract(ytest,np.dot(Xtest,w))
                
    # Output:
    mse = np.dot(np.transpose(diff), diff)
    mse = np.divide(np.sqrt(mse), Xtestrows)
    
    return mse

def regressionObjVal(w, X, y, lambd):

    # compute squared error (scalar) and gradient of squared error with respect
    # to w (vector) for the given data X and y and the regularization parameter
    # lambda                                                                  

    # IMPLEMENT THIS METHOD      
    error = ""
    error_grad = ""                                       
    return error, error_grad

def mapNonLinear(x,p):
    # Inputs:                                                                  
    # x - a single column vector (N x 1)                                       
    # p - integer (>= 0)                                                       
    # Outputs:                                                                 
    # Xp - (N x (p+1)) 
	
    # IMPLEMENT THIS METHOD
    Xp = ""
    return Xp

# Main script

# Problem 1 - Gaussian Discrimination using LDA and QDA
# load the sample data                                                                 
if sys.version_info.major == 2:
    X,y,Xtest,ytest = pickle.load(open('basecode/sample.picklee','rb'))
else:
    X,y,Xtest,ytest = pickle.load(open('basecode/sample.pickle','rb'),encoding = 'latin1')

# LDA
means,covmat = ldaLearn(X,y)
ldaacc,ldares = ldaTest(means,covmat,Xtest,ytest)
print('LDA Accuracy = '+str(ldaacc))
# QDA
means,covmats = qdaLearn(X,y)
qdaacc,qdares = qdaTest(means,covmats,Xtest,ytest)
print('QDA Accuracy = '+str(qdaacc))

# plotting boundaries
x1 = np.linspace(-5,20,100)
x2 = np.linspace(-5,20,100)
xx1,xx2 = np.meshgrid(x1,x2)
xx = np.zeros((x1.shape[0]*x2.shape[0],2))
xx[:,0] = xx1.ravel()
xx[:,1] = xx2.ravel()

fig = plt.figure(figsize=[12,6])
plt.subplot(1, 2, 1)

zacc,zldares = ldaTest(means,covmat,xx,np.zeros((xx.shape[0],1)))
plt.contourf(x1,x2,zldares.reshape((x1.shape[0],x2.shape[0])),alpha=0.3)
plt.scatter(Xtest[:,0],Xtest[:,1],c=ytest.ravel())
plt.title('LDA')

plt.subplot(1, 2, 2)

zacc,zqdares = qdaTest(means,covmats,xx,np.zeros((xx.shape[0],1)))
plt.contourf(x1,x2,zqdares.reshape((x1.shape[0],x2.shape[0])),alpha=0.3)
plt.scatter(Xtest[:,0],Xtest[:,1],c=ytest.ravel())
plt.title('QDA')

plt.show()
# Problem 2 - Linear Regression
if sys.version_info.major == 2:
    X,y,Xtest,ytest = pickle.load(open('basecode/diabetes.pickle','rb'))
else:
    X,y,Xtest,ytest = pickle.load(open('basecode/diabetes.pickle','rb'),encoding = 'latin1')

# add intercept
X_i = np.concatenate((np.ones((X.shape[0],1)), X), axis=1)
Xtest_i = np.concatenate((np.ones((Xtest.shape[0],1)), Xtest), axis=1)

w = learnOLERegression(X,y)
mle = testOLERegression(w,Xtest,ytest)

w_i = learnOLERegression(X_i,y)
mle_i = testOLERegression(w_i,Xtest_i,ytest)

print('MSE without intercept '+str(mle))
print('MSE with intercept '+str(mle_i))

# Problem 3 - Ridge Regression
k = 101
lambdas = np.linspace(0, 1, num=k)
i = 0
mses3_train = np.zeros((k,1))
mses3 = np.zeros((k,1))
for lambd in lambdas:
    w_l = learnRidgeRegression(X_i,y,lambd)
    mses3_train[i] = testOLERegression(w_l,X_i,y)
    mses3[i] = testOLERegression(w_l,Xtest_i,ytest)
    i = i + 1
fig = plt.figure(figsize=[12,6])
plt.subplot(1, 2, 1)
plt.plot(lambdas,mses3_train)
plt.title('MSE for Train Data')
plt.subplot(1, 2, 2)
plt.plot(lambdas,mses3)
plt.title('MSE for Test Data')

plt.show()
# Problem 4 - Gradient Descent for Ridge Regression
k = 101
lambdas = np.linspace(0, 1, num=k)
i = 0
mses4_train = np.zeros((k,1))
mses4 = np.zeros((k,1))
opts = {'maxiter' : 20}    # Preferred value.                                                
w_init = np.ones((X_i.shape[1],1))
for lambd in lambdas:
    args = (X_i, y, lambd)
    w_l = minimize(regressionObjVal, w_init, jac=True, args=args,method='CG', options=opts)
    w_l = np.transpose(np.array(w_l.x))
    w_l = np.reshape(w_l,[len(w_l),1])
    mses4_train[i] = testOLERegression(w_l,X_i,y)
    mses4[i] = testOLERegression(w_l,Xtest_i,ytest)
    i = i + 1
fig = plt.figure(figsize=[12,6])
plt.subplot(1, 2, 1)
plt.plot(lambdas,mses4_train)
plt.plot(lambdas,mses3_train)
plt.title('MSE for Train Data')
plt.legend(['Using scipy.minimize','Direct minimization'])

plt.subplot(1, 2, 2)
plt.plot(lambdas,mses4)
plt.plot(lambdas,mses3)
plt.title('MSE for Test Data')
plt.legend(['Using scipy.minimize','Direct minimization'])
plt.show()


# Problem 5 - Non Linear Regression
pmax = 7
lambda_opt = 0 # REPLACE THIS WITH lambda_opt estimated from Problem 3
mses5_train = np.zeros((pmax,2))
mses5 = np.zeros((pmax,2))
for p in range(pmax):
    Xd = mapNonLinear(X[:,2],p)
    Xdtest = mapNonLinear(Xtest[:,2],p)
    w_d1 = learnRidgeRegression(Xd,y,0)
    mses5_train[p,0] = testOLERegression(w_d1,Xd,y)
    mses5[p,0] = testOLERegression(w_d1,Xdtest,ytest)
    w_d2 = learnRidgeRegression(Xd,y,lambda_opt)
    mses5_train[p,1] = testOLERegression(w_d2,Xd,y)
    mses5[p,1] = testOLERegression(w_d2,Xdtest,ytest)

fig = plt.figure(figsize=[12,6])
plt.subplot(1, 2, 1)
plt.plot(range(pmax),mses5_train)
plt.title('MSE for Train Data')
plt.legend(('No Regularization','Regularization'))
plt.subplot(1, 2, 2)
plt.plot(range(pmax),mses5)
plt.title('MSE for Test Data')
plt.legend(('No Regularization','Regularization'))
plt.show()
