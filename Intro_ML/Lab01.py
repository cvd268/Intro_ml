#lab01
#Carlos Valle-Diaz

import pandas as pd
import numpy as np
import math

import matplotlib
import matplotlib.pyplot as plt




def fit_linear(x,y):
    ## Parameter types are arrays

    """
    Given vectors of data points (x,y), performs a fit for the linear model:
       y = beta0 + beta1*x,
    The function returns beta0, beta1, and loss, where loss is the sum-of-squares loss of.
    """

    mean_x=np.mean(x)
    mean_y=np.mean(y)
    syx = np.mean((y-mean_y)*(x-mean_x))
    sxx = np.mean((x-mean_x)**2)
    syy = np.mean((x-mean_y)**2)



    # TODO complete the following code
    beta1 = syx/sxx
    beta0 = mean_y-beta1*mean_x
    loss =np.sum((y-beta0-beta1*x)**2)

    #print("beta0=",beta0,"beta1=",beta1,"Loss=",loss)
    return beta0, beta1, loss


def fit_approx(x,y):
    """
    Given vectors of data points (x,y), performs an *approximate* fit for the linear model:
       y = beta0 + beta1*x,
    under the sum-of-squares loss. The min_loss returned is the lost of the best beta0 and beta1 you found.
    """
    b0=np.arange(-50,50,1)
    b1=np.arange(-10,10,.1)

    min_loss=np.sum((y-b0-b1*2)**2)
    # min_loss = ...

    return b0, b1, min_loss

def fit_approx_l1(x,y):
    """
    Given vectors of data points (x,y), performs an *approximate* fit for the linear model:
       y = beta0 + beta1*x,
    under the least absolute deviations loss.
    """
    # TODO complete the following code

    b0=np.arange(-50,50,1)
    b1=np.arange(-10,10,.1)

    min_loss = math.abs(np.sum((y-b0-b1*2)))

    return b0, b1, min_loss



def fit_approx_max(x,y):
    """
    Given vectors of data points (x,y), performs an *approximate* fit for the linear model:
       y = beta0 + beta1*x,
    under the max loss.
    """
    # TODO complete the following code

    b0=np.arange(-50,50,1)
    b1=np.arange(-10,10,.1)

    min_loss = math.max(np.sum((y-b0-b1*2)))

    return b0, b1, min_loss



def main():
    colnames =[
        'CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM',
        'AGE',  'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT', 'PRICE'
    ]

    # TODO:  Complete the code
    df = pd.read_csv('housing.data',names=colnames)

    df = pd.read_csv('housing.data',header=None,delim_whitespace=True,names=colnames,na_values='?')


    print(df.head(6))


    #What is the shape of the data? How many attributes are there? How many samples?

    #The number of samples is 500, there is a total of 500 entries
    # The total number of attributes to the data is 14 as there is 14 unique coloums of data

    y = df['PRICE'].values
    x= df['RM'].values

    #print("y=",y)

    mean_price=np.mean(y)

    print("Mean price of houses=",mean_price)

    #####

    plt.plot(x,y,'x')
    plt.xlabel('Rooms')
    plt.ylabel('Price')

    plt.scatter(x,y,color='orange')
    plt.grid(True)


    b0,b1,loss=fit_linear(x,y)
    print("beta0=",b0,"beta1=",b1,"Loss=",loss)
    #print(fit_linear(x,y))
    ypred = b1*x + b0
    plt.plot(x,ypred,'-',linewidth=3)


    ######### Part 2: compare

    # yhat=beta0+beta1*x
    # loss = np.sum((y-yhat)**2)

    for i in df:
        xx=df[i].values

        b00,b11,loss1=fit_linear(xx,y)
        #print(fit_linear(x,y))
        yhat=b00+b11*xx
        loser=np.sum((y-yhat)**2)
        print(i,loser)

    #Lstat is the best to determine price

    ## Part 3:Compare different loss function

#     Compare the output of fit_approx and fit_linear with y being PRICE and x being the LSAT variable from our dataset. You should do so:
#
# quantitatively, by printing the loss that each approach achieves
# and visually, by plotting the different fit lines obtained.
    py=df['PRICE'].values
    pl=df["LSTAT"].values

    shape = df.shape
    print('num samples=', shape[0] ,', num attributes=', shape[1])
    #
    # by,by1,ploss=fit_approx(x,py)
    # bl,bl1,lloss=fit_approx(x,pl)
    #
    # print("Price Loss",ploss,"Price LSTAT",pl)
    #
    # ypredp = by1*x + by
    # plt.plot(x,ypredp,'- -',linewidth=3)
    #
    # ypredl = bl1*x + bl
    # plt.plot(x,ypredl,'o',linewidth=3)





    plt.show()
    # Kept trying to change the colors and for some reason it never changed hope I got the graph right



main()
