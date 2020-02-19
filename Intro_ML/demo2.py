import numpy as np
import matplotlib
import matplotlib.pyplot as plt


from sklearn import datasets, linear_model


# Load the diabetes dataset
diabetes = datasets.load_diabetes() ## Loads specific data set
X = diabetes.data
y = diabetes.target

######
print("type X is")
print(type(X))
## numpy.ndarray

##Predictor variables within matrix X
## age, sex, body mass index, average blood pressure, and six blood serum measurements

##AGE
print("AGE",str(X[0:5,0]))
#[ 0.03807591 -0.00188202  0.08529891 -0.08906294  0.00538306]

## SEX
print("SEX",str(X[0:5,1]))
#[ 0.05068012 -0.04464164  0.05068012 -0.04464164 -0.04464164]


nsamp, natt = X.shape
print("num samples="+str(nsamp)+" num attributes="+str(natt))

#####
#$$ y \approx \beta_{0,k} + \beta_{1,k}x_k$$
##Using Simple Linear Regression for Each Feature Individually

ym = np.mean(y)
losses = np.zeros(natt)
beta0 = np.zeros(natt)
beta1 = np.zeros(natt)
for k in range(natt):
    xm = np.mean(X[:,k])
    sxy = np.mean((X[:,k]-xm)*(y-ym))
    sxx = np.mean((X[:,k]-xm)**2)
    beta1[k] = sxy/sxx
    beta0[k] = ym - beta1[k]*xm
    errs = y - beta1[k]*X[:,k] - beta0[k]
    losses[k] = np.sum(errs**2)

    print(str(k)+" loss="+"{:.2e}".format(losses[k])+" beta0="+str(beta0[k])+" beta1="+str(beta1[k]))




#######


syy = np.mean((y-ym)**2)
rsqr = 1 - np.min(losses)/(nsamp*syy)
print("Rsqr="+"{:.3}".format(rsqr))

###########

# Find the index of the single variable with the best squared loss
imax = np.argmin(losses)

# Regression line over the range of x values
xmin = np.min(X[:,imax])
xmax = np.max(X[:,imax])
ymin = beta0[imax] + beta1[imax]*xmin
ymax = beta0[imax] + beta1[imax]*xmax
plt.plot([xmin,xmax], [ymin,ymax], 'r-', linewidth=3)

# Scatter plot of points
plt.scatter(X[:,imax],y)
plt.grid()


###


# Remove the means
ym = np.mean(y)
y1 = y-ym  # a column vecotor each minus mean
Xm = np.mean(X,axis=0) # averaging over column, resulting a row vector of dimension natt
X1 = X - Xm[None,:] # minus the same mean in each column

# Compute the correlations per features
Sxx = np.mean(X1**2,axis=0) #a row vector with each element indicating the variance of one attribute
Sxy = np.mean(X1*y1[:,None],axis=0) #a row vector with each element indicating the covarance on one attribute to the targer

# Compute the coefficients and losses per feature
beta1 = Sxy/Sxx # element wise division, resulting a row vector containing  beta1 for each attribute
beta0 = ym - beta1*Xm # element wise multiplication, resulting a row vector containing beta0 for each attribute
errs = ((X*beta1) + beta0) - y[:,None] # results in a matrix where every column contains the residuals for predictor k
losses = np.sum(errs**2,axis=0) # sums up the squared size of the residuals across each column


##


for k in range(natt):
    print(str(k)+" loss="+"{:.2e}".format(losses[k])+" beta0="+str(beta0[k])+" beta1="+str(beta1[k]))




###3 Multiple Variable Linear Model

ones = np.ones((nsamp,1))
X_orig = X
X = np.hstack((ones,X_orig))
X.shape

Xt = np.transpose(X)
beta=np.dot(np.linalg.inv(np.dot(Xt,X)),np.dot(Xt,y))
print(beta)

Xmat=np.matrix(X)
ymat=np.matrix(y)
ymat=np.transpose(ymat)
Xmatt=np.transpose(Xmat)
beta=np.linalg.inv(Xmatt*Xmat)*Xmatt*ymat
print(beta)


out = np.linalg.lstsq(X,y)
beta = out[0]
print(beta)


### LOWEST SQUARE

import scipy as sp


out = sp.sparse.linalg.lsqr(X,y)
beta = out[0]
print(beta)

import time
t = time.time()
beta=np.linalg.inv(Xmatt*Xmat)*Xmatt*ymat
elapsed = time.time() - t
print("direct matrix computation time: " + str(elapsed) + " seconds")

yp = Xmat*beta
errs = np.array(ymat - yp)
lossm = np.sum(errs**2)

print("multiple variable loss="+"{:.2e}".format(lossm))

yp = Xmat*beta
lossm = np.linalg.norm(np.array(ymat - yp))**2

print("multiple variable loss="+"{:.2e}".format(lossm))

rsqr = 1 - lossm/(nsamp*syy)
rsqr


####sci-kit learn

X = X[:,1:11]
regr = linear_model.LinearRegression()
regr.fit(X,y)

regr.intercept_

regr.coef_

y_pred = regr.predict(X)
lossm = np.linalg.norm(y_pred - y)**2

print("multiple variable loss="+"{:.2e}".format(lossm))
