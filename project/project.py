# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import scale


data_path = r'E:\My Documents\NYU\Courses\FRE 9733 Machine Learning\Project' 
data = pd.read_csv(data_path+'\data_project.csv', delimiter=r"\s+", index_col = ['timestamp'])
data.index = pd.to_datetime(data.index/1000000,unit='s')

#
#asks = []
#bids = []
#for i in range(1,6):
#    asks.append('ask'+str(i))
#    asks.append('ask'+str(i)+'_vol')
#    bids.append('bid'+str(i))
#    bids.append('bid'+str(i)+'_vol')
#
#
#data2 =pd.read_csv(data_path+'\data_project2.csv',sep = " ",names=range(12), index_col = [0])
#data2.index = pd.to_datetime(data2.index/1000000,unit='s')
#
#data2_asks = data2[data2.iloc[:,0]=='asks']
#data2_bids = data2[data2.iloc[:,0]=='bids']
#
#data2_asks = data2_asks.drop(columns=[1])
#data2_asks.columns=asks
#
#data2_bids = data2_bids.drop(columns=[1])
#data2_bids.columns=bids
#
#data2 = pd.concat([data2_asks, data2_bids],axis=1)

data['mid'] = (data['bid'] + data['ask'])/2
data['spread'] = data['bid'] - data['ask']
data['vol_diff']=data['bidsize'] - data['asksize']

data['delta_time']=data.index.to_series().diff(1).apply(pd.Timedelta.total_seconds)

data['d_ask'] = (data['ask'].diff(1)/data['delta_time']/10000).apply(np.tanh)
data['d_bid'] = (data['bid'].diff(1)/data['delta_time']/10000).apply(np.tanh)
data['d_asksize'] = (data['asksize'].diff(1)/data['delta_time']/1000000).apply(np.tanh)
data['d_bidsize'] = (data['bidsize'].diff(1)/data['delta_time']/1000000).apply(np.tanh)

data.drop(columns=['delta_time'],inplace=True)

data_1s = pd.DataFrame()

data_1s['mid'] = data.mid.resample('1s').last().fillna(method='ffill')
data_1s.drop(data_1s.head(19).index, inplace = True)

X = pd.DataFrame(index = data_1s.index, columns=range(60))

for i in data_1s.index:
    temp = data[:i].tail(5)
    var = np.array((i - temp.index).total_seconds()).reshape(1,-1)
    temp = temp.values.reshape(1,-1)
    var = np.concatenate((var,temp),axis=1)
    X.loc[i] = var
        
X['y'] = data_1s.shift(-10)
X['price_now'] = data_1s

X = X.dropna()
Y = X['y']
price_now = X['price_now']
X.drop(columns=['y','price_now'],inplace=True)
y = np.array(Y).ravel()


train_x = X.iloc[:4500,:]
test_x = X.iloc[4500:,:]
train_y = y[:4500]
test_y = y[4500:]
price_now_train = price_now[:4500]
price_now_test = price_now[4500:]


import graphviz
from sklearn import tree
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
from sklearn import model_selection

kfold = model_selection.TimeSeriesSplit(n_splits=3)


param_grid = {
              "n_estimators": [100, 150],
              "max_features": [20, 25, 30]
              }

model = RandomForestRegressor()

grid_search = GridSearchCV(model, param_grid=param_grid, cv=kfold,return_train_score=False)

grid_search.fit(train_x, train_y)

result = pd.DataFrame(grid_search.cv_results_)

print(grid_search.best_params_)

model = grid_search.best_estimator_

#clf = tree.DecisionTreeRegressor(max_depth=10)
#
#clf.fit(X, y)

#dot_data = tree.export_graphviz(clf, out_file=None, 
#                         feature_names=X.columns,
#                         filled=True, rounded=True,  
#                         special_characters=True)  
#graph = graphviz.Source(dot_data)
#graph


y_pre = model.predict(test_x)
condition1 = np.array(price_now_test<y_pre)
condition2 = (y_pre < test_y)
success = (condition1 == condition2)

print(success.sum()/len(success))

plt.figure()
plt.plot(test_y)
plt.plot(y_pre)
plt.show()

ret = np.array((test_y - price_now_test)/price_now_test)
long_or_short = 2*condition1 - 1

ret = ret * long_or_short
ret_cum = np.cumprod((ret+1)) - 1


