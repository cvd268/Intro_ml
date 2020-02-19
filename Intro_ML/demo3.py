import numpy as np


### AXIS Parameters

X = np.array([[0,1],[2,3],[4,5]])
print(X)

print(np.mean(X))
print(np.sum(X))

print(np.sum(X,axis=0))

print(np.sum(X,axis=1))


X = np.arange(24).reshape(2,3,4)  # shape = (2,3,4)
print(X)

Y1 = np.sum(X,axis=0)             # shape = (3,4)
Y2 = np.sum(X,axis=1)             # shape = (2,4)
print('Y1 = ')
print(Y1)
print('Y2 = ')
print(Y2)

print("Broadcasting")

a = np.array([[1,2,3],[4,5,6]])
b = 2
print(b*a)
print(a*b)

# Generate some random data
n = 100
p = 5
X = np.random.rand(n,p)

Xm = np.zeros(p)      # Mean for each feature
X_demean = np.zeros((n,p))  # Transformed features with the means removed
for j in range(p):
    Xm[j] = np.mean(X[:,j])
    for i in range(n):
        X_demean[i,j] = X[i,j] - Xm[j]
print(X_demean[0:7,:]) # print the first several rows of the result to compare to later

# Compute the mean per column using the axis command
Xm = np.mean(X,axis=0)  # This is a p-dim matrix
print(Xm)

print(Xm[None,:])

# Subtract the mean
X_demean = X - Xm[None,:]
print(X_demean[0:7,:])

print("Example 2:")

Xstd = np.std(X,axis=0)
Z = (X-Xm[None,:])/Xstd[None,:]

print("Example 3:")
# Some random data
nx = 100
ny = 10
x = np.random.rand(nx)
y = np.random.rand(ny)

# Compute the outer product in o

Z = x[:,None]*y[None,:]

X = np.random.rand(4,3)
# Y = ...


X = np.random.rand(5,3)
d = np.random.rand(5)
# Y = ...

print("Matrix operations with numpy ")

X = np.array([[1,2],[3,4]])
z = np.array([[2],[3]])
Y = np.array([[-1,-1],[-1,-1]])
print(str(X)+'\n\n'+str(z)+'\n\n'+str(Y))

print(str(X*z))
print()
print(X*Y)

print(np.dot(X,z))
print()
print(np.dot(np.transpose(z),z));print()
print(np.dot(X,Y))
print()

print(X@z)
print()
print(np.transpose(z)@z)
print()
print(X@Y)

print(z@np.transpose(z))
