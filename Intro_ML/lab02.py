import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
#matplotlib inline


def fit_mult_linear(X,y):
    """
    Given matrix of predictors X and target vector y fit for a multiple linear regression model under the squared loss.
    """
    # TODO complete the following code
    nsamp,natt=X.shape

    ones=np.ones((nsamp,1))
    X_orig=X
    X=np.hstack((ones,X_orig))
    X.shape
    Xt=np.transpose(X)

    beta = np.dot(np.linalg.inv(np.dot(Xt,X)),np.dot(Xt,y))
    return beta

def main():
    names =[
        't',                                  # Time (secs)
        'q1', 'q2', 'q3',                     # Joint angle   (rads)
        'dq1', 'dq2', 'dq3',                  # Joint velocity (rads/sec)
        'I1', 'I2', 'I3',                     # Motor current (amp)
        'eps21', 'eps22', 'eps31', 'eps32',   # Strain gauge measurements ($\mu$m /m )
        'ddq1', 'ddq2', 'ddq3'                # Joint accelerations (rad/sec^2)
    ]
    # TODO
    #exp1.csv
    #exp2.csv
    #train = pd.read_csv("exp1.csv",header=None,delim_whitespace=True,names=names,na_values='?')
    train = pd.read_csv("exp1.csv",names=names)
    test = pd.read_csv("exp2.csv",names=names)

    ## print the first six lines
     #print(dftrain[0:5,0])
    #print(type(train))
    trainLst=train.values.tolist()
    testLst=test.values.tolist()
    print("trainLst")
    print(trainLst[0:5][0:5])
    print("testLst")
    print(testLst[0:5][0:5])
    print()

    xx=trainLst[0:5][0:5]
    yy=testLst[0:5][0:5]

    if(xx==yy):
        print("yes same")
        ## no they are not the same

    # TODO
    ytrain = train['I2'].values
    t = train['t'].values
    plt.plot(t,ytrain,"x")
    plt.grid(True)
    #plt.show()


    ######

    # TODO

    # 't',                                  # Time (secs)
    # 'q1', 'q2', 'q3',                     # Joint angle   (rads)
    # 'dq1', 'dq2', 'dq3',                  # Joint velocity (rads/sec)
    # 'I1', 'I2', 'I3',                     # Motor current (amp)
    # 'eps21', 'eps22', 'eps31', 'eps32',   # Strain gauge measurements ($\mu$m /m )
    # 'ddq1', 'ddq2', 'ddq3'                # Joint accelerations (rad/sec^2)
    # ]

    #3,6,11,12,13,14,16
    #['q2','dq2','eps21', 'eps22', 'eps31', 'eps32','ddq2']
    Xtrain = train

    Xtrain=Xtrain.drop(['t'],axis=1) #t
    #print(Xtrain.head(6))
    Xtrain=Xtrain.drop(['q1'],axis=1) # q1
    Xtrain=Xtrain.drop(['q3'],axis=1) #q3
    #print(Xtrain.head(6))
    Xtrain=Xtrain.drop(['dq1'],axis=1) #dq1
    #print(Xtrain.head(6))
    Xtrain=Xtrain.drop(['dq3'],axis=1) #dq3
    Xtrain=Xtrain.drop(['I1'],axis=1)
    Xtrain=Xtrain.drop(['I2'],axis=1)
    Xtrain=Xtrain.drop(['I3'],axis=1)
    Xtrain=Xtrain.drop(['ddq1'],axis=1)
    Xtrain=Xtrain.drop(['ddq3'],axis=1)

    print(Xtrain.head(6))


    print("Made it here")

    print("Measure the Fit on an Indepedent Dataset")
    test = pd.read_csv("exp2.csv",names=names)
    exp2Lst=testLst

    y = test['I3'].values
    tt = test['t'].values
    plt.plot(t,ytrain,"o")
    plt.grid(True)

    plt.show()









main()
