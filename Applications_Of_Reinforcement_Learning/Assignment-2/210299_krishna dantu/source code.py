#collection of all functions and methods used to write the notebook for ml assignment in summer project 2022
#raw code with little to no formatting
#krishna dantu
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
url="https://raw.githubusercontent.com/lumidenoir/Project2022/assignment/Applications_Of_Reinforcement_Learning/Assignment-2/210299_krishna%20dantu/CarPrice_Assignment.csv"
my_data=pd.read_csv(url, skipinitialspace=True, usecols=['carlength','carwidth','carheight','enginesize','stroke','horsepower'])
price_data=pd.read_csv(url, skipinitialspace=True, usecols=['price'])
query=[[190.9,70.3,54.9,183,3.64,123]]
alpha=0.001
cycle=5000
theta=np.zeros((1,my_data.shape[1]+1))

def normalise(matrix):
    factor1=matrix.mean()
    factor2=matrix.max()-matrix.min()
    matrix_nor = (matrix-factor1)/factor2
    return matrix_nor,factor1,factor2

def stdnormalise(matrix):
    factor1=matrix.mean()
    factor2=matrix.std()
    matrix_std = (matrix-factor1)/factor2
    return matrix_std,factor1,factor2

def split(matrix,ratio):
    m=matrix.shape[0]
    n=int(m*ratio)
    traindata=matrix.iloc[0:n].values
    testdata=matrix.iloc[n:m].values
    return traindata,testdata

def concatenate(matrix):
    matrix=np.concatenate((np.ones((matrix.shape[0],1)),matrix), axis=1)
    return matrix

def costfunction(X,y,theta):
    sqr_error=((X.dot(theta.T))-y)**2
    sumofsquares=np.sum(sqr_error)
    return (1/2*X.shape[0])*sumofsquares

def gradientdescent(X,y,theta,cycle,alpha):
    cost = np.zeros(cycle)
    for i in range(cycle):
        error=((X.dot(theta.T))-y)
        multiplyerror = (error.T@X)
        theta = theta-(alpha/X.shape[1])*multiplyerror
        cost[i] = costfunction(X, y, theta)
    
    return theta,cost

my_data_nor,fac1_nor,fac2_nor=normalise(my_data)
price_data_nor,fac3_nor,fac4_nor=normalise(price_data)
my_data_std,fac1_std,fac2_std=stdnormalise(my_data)
price_data_std,fac3_std,fac4_std=stdnormalise(price_data)

train_data_nor,test_data_nor=split(my_data_nor,0.8)
train_price_nor,test_price_nor=split(price_data_nor,0.8)
train_data_std,test_data_std=split(my_data_std,0.8)
train_price_std,test_price_std=split(price_data_std,0.8)

train_data_nor=concatenate(train_data_nor)
test_data_nor=concatenate(test_data_nor)
train_data_std=concatenate(train_data_std)
test_data_std=concatenate(test_data_std)



theta_nor, cost_nor=gradientdescent(train_data_nor,train_price_nor,theta,cycle,alpha)
theta_std, cost_std=gradientdescent(train_data_std,train_price_std,theta,cycle,alpha)

plt.plot(range(cycle),cost_nor, label="normal")
plt.xlabel("Iterations")
plt.ylabel("Cost")
plt.grid()
plt.title("Cost vs Iterations")
plt.show()

plt.plot(range(cycle),cost_std, label="std normal")
plt.xlabel("Iterations")
plt.ylabel("Cost")
plt.grid()
plt.title("Cost vs Iterations")
plt.show()

def predict(X,theta):
    y_predict=X.dot(theta.T)
    return y_predict

train_predict_nor=predict(train_data_nor,theta_nor)
test_predict_nor=predict(test_data_nor,theta_nor)
train_predict_std=predict(train_data_std,theta_std)
test_predict_std=predict(test_data_std,theta_std)

def r2score(y_pred, y):
    rss = np.sum((y_pred - y) ** 2)
    tss = np.sum((y-y.mean()) ** 2)
    r2 = 1 - (rss / tss)
    return r2

print("r2 score for train values (nor)",r2score(train_predict_nor,train_price_nor))
print("r2 score for test values (nor)",r2score(test_predict_nor,test_price_nor))
print("r2 score for train values (std)",r2score(train_predict_std,train_price_std))
print("r2 score for test values (std)",r2score(test_predict_std,test_price_std))


query_nor=(query-np.array(fac1_nor))/np.array(fac2_nor)
query_nor=concatenate(query_nor)
query_predict_nor=predict(query_nor,theta_nor)
predict_convert_nor=query_predict_nor[0,0]*fac4_nor+fac3_nor
print("price prediction for query (nor)",float(predict_convert_nor))

query_std=(query-np.array(fac1_std))/np.array(fac2_std)
query_std=concatenate(query_std)
query_predict_std=predict(query_std,theta_std)
predict_convert_std=query_predict_std[0,0]*fac4_std+fac3_std
print("price prediction for query (std)",float(predict_convert_std))

from sklearn.linear_model import LinearRegression
model=LinearRegression()
model.fit(train_data_nor,train_price_nor)
y_prediction=model.predict(query_nor)
y_predict=model.predict(train_data_nor)
y1_predict=model.predict(test_data_nor)
print(r2score(y_predict,train_price_nor),"\n",r2score(y1_predict,test_price_nor))
print(y_prediction[0,0]*fac4_nor+fac3_nor)