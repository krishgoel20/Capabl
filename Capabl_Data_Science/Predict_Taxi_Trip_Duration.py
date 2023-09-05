#!/usr/bin/env python
# coding: utf-8

# # INTRODUCTION
# It is a model that predicts the total ride duration of taxi trips in New York City. The primary dataset is released by the NYC Taxi and Limousine Commission, which include pickup time, geo-coordinates, number of passengers, and several other variables.

# In[10]:


import numpy as np
import pandas as pd


# In[11]:


import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib','inline')


# In[12]:


sns.set_style('whitegrid')


# In[13]:


df_train = pd.read_csv(r"C:\Users\Vipin Goel\Desktop\Dataset\train.csv",nrows=200000,parse_dates=["pickup_datetime"])


# # Data Exploration

# In[14]:


df_train.info()


# In[15]:


df_train.describe()


# In[16]:


df_train.dtypes


# #  NAN or Missing Values:

# In[17]:


df_train['trip_duration'].isnull().sum()


# In[18]:


df_train.drop_duplicates(inplace=True)
df_train.shape


# In[19]:


df_train['passenger_count'].value_counts().reset_index()


# In[20]:


df_train[df_train['passenger_count']==0].count()


# In[21]:


df_train = df_train[df_train['passenger_count']!=0]


# In[22]:


df_train["pickup_datetime"][0]


# In[23]:


df_train['pickup_datetime'] = pd.to_datetime(df_train['pickup_datetime'])
df_train['dropoff_datetime'] = pd.to_datetime(df_train['dropoff_datetime'])
df_train.head()


# # Data Preprocessing

# In[24]:


plt.figure(figsize=(10, 5))
sns.distplot(np.log10(df_train['trip_duration']),color="red").set(title='Distribution Plot with Log Transformation for Trip Duration')


# In[25]:


for col in df_train.describe().columns:
  fig = plt.figure(figsize=(9, 6))
  ax = fig.gca()
  df_train.boxplot(column=col,ax=ax)
  ax.set_ylabel(col)
plt.title("Box Plot for Trip Duration")
plt.show()


# In[26]:


plt.figure(figsize=(10,5))
sns.distplot(df_train['trip_duration'],color="b").set(title='Distribution Plot for Trip Duration')


# In[27]:


def calculate_trip_duration(pickup,dropoff):
    return (dropoff-pickup).total_seconds()


# In[28]:


df_train['calculate_trip_duration'] = df_train.apply(lambda x: calculate_trip_duration(x['pickup_datetime'],x['dropoff_datetime']),axis=1)


# In[29]:


(df_train['calculate_trip_duration']==df_train['trip_duration']).value_counts()


# Here, we see that there the trip duration is consistent with the calculated trip duration. so, this large value are purely an outlier.

# In[30]:


df_train.drop(['calculate_trip_duration'],axis=1,inplace=True)


# In[31]:


plt.figure(figsize=[10,5])
labels = ['less then 1min','within 10 mins','within 30 mins','within hour','within day','within two days','more then two day']
df_train.groupby(pd.cut(df_train['trip_duration'],bins=[0,60,600,1800,3600,86400,86400*2,10000000],labels=labels))['trip_duration'].count().plot(kind='bar',fontsize=10)
plt.title("Bar Plot for Trip Duration")
plt.ylabel("Trip Counts")
plt.ylabel("Trip Duration")
plt.xticks(rotation=45)


# In[32]:


numeric_features = df_train.describe().columns
numeric_features


# In[33]:


for col in numeric_features:
    fig = plt.figure(figsize=(9,6))
    ax = fig.gca()
    feature = df_train[col]
    feature.hist(bins=50,ax=ax)
    ax.axvline(feature.mean(),color='magenta',linestyle='dashed',linewidth=2)
    ax.axvline(feature.median(),color='cyan',linestyle='dashed',linewidth=2)    
    ax.set_title(col)
plt.show()


# In[34]:


df_train.describe().columns


# In[35]:


for col in numeric_features[1:-1]:
    fig = plt.figure(figsize=(9,6))
    ax = fig.gca()
    feature = df_train[col]
    label = df_train['Trip_Duration']
    correlation = feature.corr(label)
    plt.scatter(x=feature,y=label)
    plt.xlabel(col)
    plt.ylabel('Trip Duration')
    ax.set_title('Trip Duration v/s ' + col + '- correlation: ' + str(correlation))
    z = np.polyfit(df_train[col],df_train['trip_duration'],1)
    y_hat = np.poly1d(z)(df_train[col])

    plt.plot(df_train[col],y_hat,"r--",lw=1)

plt.show()


# # Data Visualization

# In[36]:


city_long_border = [-74.03,-73.75]
city_lat_border = [40.63,40.85]

df_train.plot(kind='scatter',x='dropoff_longitude',y='dropoff_latitude',color='Red',s=0.2,alpha=.6)
plt.title('Dropoffs')

plt.ylim(city_lat_border)
plt.xlim(city_long_border)


# In[37]:


city_long_border = [-74.03,-73.75]
city_lat_border = [40.63,40.85]

df_train.plot(kind='scatter',x='pickup_longitude',y='pickup_latitude',color='green',s=0.2,alpha=.6)
plt.title('Pickups')

plt.ylim(city_lat_border)
plt.xlim(city_long_border)


# In[38]:


def select_within_boundingbox(df,BB):
    return ((df_train["pickup_longitude"] >= BB[0]) & (df_train["pickup_longitude"] <= BB[1]) & (df_train["pickup_latitude"] >= BB[2]) & (df_train["pickup_latitude"] <= BB[3]) & (df_train["dropoff_longitude"] >= BB[0]) & (df_train["dropoff_longitude"] <= BB[1]) & (df_train["dropoff_latitude"] >= BB[2]) & (df_train["dropoff_latitude"] <= BB[3]))
BB = (-74.3, -73.0, 40.6, 41.7)


# In[39]:


get_ipython().system('pip install folium')


# In[40]:


import folium


# In[41]:


nyc = folium.Map(location=[40.730610,-73.935242],zoom_start=12,control_scale=True)
nyc


# In[42]:


for i in df_train.index[:100]:
  folium.Marker(location=[df_train['pickup_latitude'][i],df_train['pickup_longitude'][i]],icon=folium.Icon(color="blue")).add_to(nyc)
nyc


# In[43]:


for i in df_train.index[:100]:
  folium.Marker(location=[df_train['dropoff_latitude'][i],df_train['dropoff_longitude'][i]],icon=folium.Icon(color="red",icon="info-sign")).add_to(nyc)
nyc


# # Data Modelling

# In[44]:


df_train['pickup_day'] = df_train['pickup_datetime'].dt.day_name()
df_train['dropoff_day'] = df_train['dropoff_datetime'].dt.day_name()
df_train.head()


# In[45]:


figure,ax = plt.subplots(nrows=2,ncols=1,figsize=(10,10))
sns.countplot(x='pickup_day',data=df_train,ax=ax[0])
ax[0].set_title('Number of Pickups done on each day of the week')
sns.countplot(x='dropoff_day',data=df_train,ax=ax[1])
ax[1].set_title('Number of Dropoffs done on each day of the week')

plt.tight_layout()


# Thus we see most trips were taken on Friday and Monday being the least. The distribution of trip duration with the days of the week is something to look into as well.

# In[46]:


bins = np.array([0,1800,3600,5400,7200,90000])
df_train['duration_time'] = pd.cut(df_train.trip_duration,bins,labels=["<5","5-10","10-15","15-20",">20"])


# In[47]:


import datetime
def timezone(x):
    if x>=datetime.time(4,0,1) and x<=datetime.time(10,0,0):
        return 'morning'
    elif x>=datetime.time(10,0,1) and x<=datetime.time(16,0,0):
        return 'midday'
    elif x>=datetime.time(16,0,1) and x<=datetime.time(22,0,0):
        return 'evening'
    elif x>=datetime.time(22,0,1) or x<=datetime.time(4,0,0):
        return 'late night'

df_train['pickup_timezone'] = df_train['pickup_datetime'].apply(lambda x :timezone(datetime.datetime.strptime(str(x),"%Y-%m-%d %H:%M:%S").time()))
df_train['dropoff_timezone'] = df_train['dropoff_datetime'].apply(lambda x :timezone(datetime.datetime.strptime(str(x),"%Y-%m-%d %H:%M:%S").time()))


# In[48]:


figure,ax = plt.subplots(nrows=1,ncols=2,figsize=(15,10))
sns.countplot(x='pickup_timezone',data=df_train,ax=ax[0])
ax[0].set_title('The Distribution of Number of Pickups on each part of the day')
sns.countplot(x='dropoff_timezone',data=df_train,ax=ax[1])
ax[1].set_title('The Distribution of Number of Dropoffs on each part of the day')
plt.show()


# In[49]:


def calc_distance(df):
    pickup = (df['pickup_latitude'],df['pickup_longitude'])
    drop = (df['dropoff_latitude'],df['dropoff_longitude'])
    return haversine(pickup,drop)


# In[50]:


from math import radians,sin,cos,sqrt,atan2

def haversine(coord1,coord2):
    # Radius of the Earth in km
    R = 6371.0

    lat1,lon1 = radians(coord1[0]),radians(coord1[1])
    lat2,lon2 = radians(coord2[0]),radians(coord2[1])

    dlat = lat2-lat1
    dlon = lon2-lon1

    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2 * atan2(sqrt(a),sqrt(1-a))

    distance = R * c
    return distance


# In[51]:


df_train["distance"] = df_train.apply(lambda x: calc_distance(x),axis=1)


# # Outlier detection using IQR Method

# In[52]:


plt.figure(figsize=(15,8))
plt.title("Box Plot of Distance")
ax = sns.boxplot(data=df_train['distance'],orient="v")


# In[53]:


percentile_q1 = np.percentile(df_train['distance'],25)
print(percentile_q1)
percentile_q2 = np.percentile(df_train['distance'],50)
print(percentile_q2)
percentile_q3 = np.percentile(df_train['distance'],75)
print(percentile_q3)


# In[54]:


iqr = percentile_q3 - percentile_q1
lower_limit_outlier = percentile_q1 - 1.5*iqr
upper_limit_outlier = percentile_q3 + 1.5*iqr

print("Lower limit for outlier :",lower_limit_outlier)
print("Upper limit for outlier :",upper_limit_outlier)


# In[55]:


df_train = df_train[df_train['distance']>lower_limit_outlier]
df_train = df_train[df_train['distance']<upper_limit_outlier]


# In[56]:


df_train.shape


# In[57]:


plt.figure(figsize=(15,8))
plt.title("Box Plot of Trip Duration")
ax = sns.boxplot(data=df_train['trip_duration'],orient="v")


# In[58]:


import numpy as np


# In[59]:


percentile_q1_trip_duration = np.percentile(df_train['trip_duration'],25)
print(percentile_q1_trip_duration)
percentile_q2_trip_duration = np.percentile(df_train['trip_duration'],50)
print(percentile_q2_trip_duration)
percentile_q3_trip_duration = np.percentile(df_train['trip_duration'],75)
print(percentile_q3_trip_duration)


# In[60]:


iqr = percentile_q3_trip_duration - percentile_q1_trip_duration
lower_limit_outlier_trip_duration = percentile_q1_trip_duration - 1.5*iqr
upper_limit_outlier_trip_duration = percentile_q3_trip_duration + 1.5*iqr

print("Lower limit for outlier :",lower_limit_outlier_trip_duration)
print("Upper limit for outlier :",upper_limit_outlier_trip_duration)


# In[61]:


df_train = df_train[df_train['trip_duration']>0]
df_train = df_train[df_train['trip_duration']<upper_limit_outlier_trip_duration]


# In[62]:


df_train.shape


# In[63]:


plt.figure(figsize=(15,8))
plt.title("Box Plot of Passenger Count")
ax = sns.boxplot(data=df_train['passenger_count'])


# In[64]:


percentile_q1_passenger_count = np.percentile(df_train['passenger_count'],25)
print(percentile_q1_passenger_count)
percentile_q2_passenger_count = np.percentile(df_train['passenger_count'],50)
print(percentile_q2_passenger_count)
percentile_q3_passenger_count = np.percentile(df_train['passenger_count'],75)
print(percentile_q3_passenger_count)


# In[65]:


iqr = percentile_q3_passenger_count - percentile_q1_passenger_count
lower_limit_outlier_passenger_count = percentile_q1_passenger_count - 1.5*iqr
upper_limit_outlier_passenger_count = percentile_q3_passenger_count + 1.5*iqr

print("Lower limit for outlier :",lower_limit_outlier_passenger_count)
print("Upper limit for outlier :",upper_limit_outlier_passenger_count)


# In[66]:


df_train = df_train[df_train['passenger_count']>0]
df_train = df_train[df_train['passenger_count']<upper_limit_outlier_passenger_count]


# In[67]:


df_train.shape


# In[68]:


df_train["pickup_datetime"] = pd.to_datetime(df_train["pickup_datetime"],format="%Y-%m-%d %H:%M:%S")
df_train['Day'] = df_train['pickup_datetime'].dt.day_name()


# In[69]:


df_train["year"] = df_train["pickup_datetime"].apply(lambda x: x.year)
df_train["month"] = df_train["pickup_datetime"].apply(lambda x: x.month)
df_train["day_num"] = df_train["pickup_datetime"].apply(lambda x: x.day)
df_train["hour"] = df_train["pickup_datetime"].apply(lambda x: x.hour)
df_train["minute"] = df_train["pickup_datetime"].apply(lambda x: x.minute)


# In[70]:


df_train.head()


# In[71]:


df_train['trip_duration_hour'] = df_train['trip_duration']/3600
df_train['log_distance'] = np.log(df_train.distance)
df_train['log_trip_duration'] = np.log(df_train.trip_duration_hour)


# In[72]:


sns.countplot(x='vendor_id',data=df_train)


# # Store and fwd flag

# This flag indicates whether the trip record was held in vehicle memory before seding to the vendor because the vehicle did not have a connection to the server - Y store and forward; N=not a store and forward trip.

# In[73]:


sns.catplot(x="store_and_fwd_flag",y="log_trip_duration",kind="strip",data=df_train)


# # Bivariate Analysis

# It is used to find out if there is a relationship between teo sets of values. It usually involves the variables X and Y.
# 

# In[74]:


fig = plt.figure(figsize=(15,10))
sns.scatterplot(x='distance',y='trip_duration',data=df_train)


# In[75]:


fig = plt.figure(figsize=(15,10))
sns.scatterplot(x='log_distance',y='log_trip_duration',data=df_train)


# In[76]:


sample = ['distance','trip_duration_hour']
for i in sample:
  plt.figure(figsize=(15,10))
  sns.distplot(df_train[i],color="g")


# In[77]:


df_train = pd.get_dummies(df_train,columns=["store_and_fwd_flag","Day"],prefix=["store_and_fwd_flag",'Day'])
#Feature for the Machine learning models
features = ['vendor_id','passenger_count','pickup_longitude','pickup_latitude','dropoff_longitude','dropoff_latitude','distance','month','hour','minute','store_and_fwd_flag_N','store_and_fwd_flag_Y','Day_Friday','Day_Monday','Day_Saturday','Day_Sunday','Day_Thursday','Day_Tuesday','Day_Wednesday']
newdata = ['vendor_id','passenger_count','pickup_longitude','pickup_latitude','dropoff_longitude','dropoff_latitude','distance','month','hour','minute','store_and_fwd_flag_N','store_and_fwd_flag_Y','Day_Friday','Day_Monday','Day_Saturday','Day_Sunday','Day_Thursday','Day_Tuesday','Day_Wednesday','trip_duration_hour']
trip_data = df_train[newdata]
df_train.shape


# In[79]:


from scipy.stats import zscore
#Train test split
X = df_train[features].apply(zscore)[:100000]
y = df_train['trip_duration_hour'][:100000]
# Importing train_test_split
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=42)
print(X_train.shape,y_train.shape)
print(X_test.shape,y_test.shape)


# # Correlation Analysis

# It is a method of statistical evaluation used to study the strength of a relationship between two or more, numerically measured, continuous variables. This analysis is useful to check if there are possible connections between variables. 

# Heatmap:

# In[81]:


plt.figure(figsize=(30,20))

sns.heatmap(trip_data.corr(),cmap='RdYlGn',annot=True,vmin=-1,vmax=1,square=True)
plt.title("Correlation Heatmap",fontsize=16)
plt.show()


# In[82]:


from matplotlib import legend
# Function for evaluation metric for regression
def EvaluationMetric(Xt,yt,yp,disp="on"):
  ''' Take the different set of parameter and prints evaluation metrics '''
  MSE = round(mean_squared_error(y_true=yt,y_pred=yp),4)
  RMSE = (np.sqrt(MSE))
  R2 = (r2_score(y_true=yt,y_pred=yp))
  Adjusted_R2 = (1-(1-r2_score(yt,yp))*((Xt.shape[0]-1)/(Xt.shape[0]-Xt.shape[1]-1)))
  if disp=="on":
    print("MSE: ",MSE,"RMSE: ",RMSE)
    print("R2: ",R2,"Adjusted R2: ",Adjusted_R2)

  #Plotting Actual and Predicted Values
  plt.figure(figsize=(18,6))
  plt.plot((yp)[:100]) 
  plt.plot((np.array(yt)[:100]))
  plt.legend(["Predicted","Actual"])
  plt.title('Actual and Predicted Time Duration')



  return (MSE,RMSE,R2,Adjusted_R2) 


# # Linear Regression

# Linear Regression Analysis is used to predict the value of a variable based on the value of another variable. The variable you want to predict is called the dependant variable. The variable used to predict the other variable value is called the independent variable.

# In[84]:


from sklearn.linear_model import LinearRegression

reg = LinearRegression().fit(X_train,y_train)
reg.score(X_train,y_train)


# In[87]:


from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
y_pred_train = reg.predict(X_train)
y_pred_test = reg.predict(X_test)
#Evaluation metrics for Train set
EvaluationMetric(X_train,y_train,y_pred_train)


# In[88]:


EvaluationMetric(X_test,y_test,y_pred_test)


# In[90]:


plt.figure(figsize=(18,6))

importance = reg.coef_
importance = np.sort(importance)
feature = features
indices = np.argsort(importance)
indices = indices[:10:-1]
#plotting the features and their score in ascending order
sns.set_style("darkgrid")
plt.bar(range(len(indices)),importance[indices])
plt.xticks(range(len(indices)),[feature[i] for i in indices])
plt.show()


# In[91]:


residuals = y_pred_test-y_test

plt.figure(figsize=(10,6),dpi=120,facecolor='w',edgecolor='b')
f = range(0,len(y_test))
k = [0 for i in range(0,len(y_test))]
plt.scatter(f,residuals,label='residuals')
plt.plot(f,k,color='red',label='Regression Line')
plt.xlabel('Fitted Points')
plt.ylabel('Residuals')
plt.title('Residual Plot')
plt.legend()


# # Decision Tree

# In[93]:


max_depth = [4,6,8,10]

# Minimum number of samples required to split a node
min_samples_split = [10,20,30]

# Minimum number of samples required at each leaf node
min_samples_leaf = [10,16,20]

# HYperparameter Grid
param_dt = {'max_depth' : max_depth,'min_samples_split' : min_samples_split,'min_samples_leaf' : min_samples_leaf}


# In[97]:


from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import GridSearchCV

dt_model = DecisionTreeRegressor()

# Grid search
dt_grid = GridSearchCV(estimator=dt_model,param_grid=param_dt,cv=5,verbose=2,scoring='r2')

dt_grid.fit(X_train,y_train)


# In[98]:


dt_grid.best_score_


# In[99]:



dt_grid.best_estimator_


# In[100]:


dt_optimal_model = dt_grid.best_estimator_
y_pred_dt_test = dt_optimal_model.predict(X_test)
y_pred_dt_train = dt_optimal_model.predict(X_train)
y_pred_dt_test = dt_optimal_model.predict(X_test)
y_pred_dt_train = dt_optimal_model.predict(X_train)

EvaluationMetric(X_train,y_train,y_pred_dt_train)


# In[101]:


EvaluationMetric(X_test,y_test,y_pred_dt_test)


# In[102]:


X_train.columns


# In[103]:


dt_optimal_model.feature_importances_


# In[104]:


importances = dt_optimal_model.feature_importances_

importance_dict = {'Feature' : list(X_train.columns),'Feature Importance' : importances}

importance_df = pd.DataFrame(importance_dict)
importance_df.sort_values(by=['Feature Importance'],ascending=False,inplace=True)
importance_df


# In[105]:


plt.figure(figsize=(15,13))
plt.title('Features Importance')
sns.barplot(x='Feature',y="Feature Importance",data=importance_df[:10])


# In[109]:


get_ipython().system('pip install graphviz')


# In[111]:


from sklearn.tree import DecisionTreeRegressor,export_graphviz
from sklearn import tree
from IPython.display import SVG
from graphviz import Source
from IPython.display import display


# # Lasso Regression

# In[115]:


from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import Lasso

### Cross validation
lasso = Lasso()
parameters = {'alpha' : [1e-15,1e-13,1e-10,1e-8,1e-5,1e-4,1e-3,1e-2,1e-1,1,5,10,20,30,40,45,50,55,60,100]}
lasso_regressor = GridSearchCV(lasso,parameters,scoring='r2',cv=5)
lasso_regressor.fit(X_train,y_train)


# In[116]:


lasso_regressor.score(X_train,y_train)


# In[117]:


print("The best fit alpha value is found out to be: ",lasso_regressor.best_params_)
print("\nUsing ",lasso_regressor.best_params_," the negative mean squared error is: ",lasso_regressor.best_score_)


# In[120]:


from sklearn.linear_model import Ridge


ridge_regressor = Ridge(alpha=1.0)  
ridge_regressor.fit(X_train,y_train)
y_pred_ridge_test = ridge_regressor.predict(X_test)
y_pred_ridge_train = ridge_regressor.predict(X_train)


# In[121]:


y_pred_ridge_test = ridge_regressor.predict(X_test)
y_pred_ridge_train = ridge_regressor.predict(X_train)


# In[122]:



residuals = y_pred_ridge_test - y_test

plt.figure(figsize=(10,6),dpi=120,facecolor='w',edgecolor='b')
f = range(0,len(y_test))
k = [0 for i in range(0,len(y_test))]
plt.scatter(f,residuals,label='Residuals')
plt.plot(f,k,color='red',label='Regression Line')
plt.xlabel('Fitted Points')
plt.ylabel('Residuals')
plt.title('Residual Plot')
plt.legend()


# In[123]:


EvaluationMetric(X_train,y_train,y_pred_ridge_train)


# In[124]:


EvaluationMetric(X_test,y_test,y_pred_ridge_test)


# In[125]:


sns.distplot(y_test-y_pred_ridge_test).set_title("Error Distribution b/w Actual and Predicted Values")
plt.show()


# # XGBoost

# It is an optimized distributed gradient boosting library designed to be higly efficient, flexible and portable. It implements machine learning algorithms under the Gradient Boosting framework.

# In[126]:


# Number of trees
n_estimators = [50,100,120]

# Maximum depth of trees
max_depth = [5,7,9]
min_samples_split = [40,50]
# learning_rate = [0.1,0.3,0.5]

# Hyperparameter Grid
param_xgb = {'n_estimators' : n_estimators,'max_depth' : max_depth,'min_samples_split' : min_samples_split}


# In[127]:


import xgboost as xgb
xgb_model = xgb.XGBRegressor()

# Grid search
xgb_grid = GridSearchCV(estimator=xgb_model,param_grid=param_xgb,cv=3,verbose=2,scoring="r2")

xgb_grid.fit(X_train,y_train)


# In[129]:


xgb_grid.best_score_


# In[130]:


xgb_grid.best_params_


# In[131]:


xgb_optimal_model = xgb_grid.best_estimator_


# In[132]:


y_pred_xgb_test = xgb_optimal_model.predict(X_test)
y_pred_xgb_train = xgb_optimal_model.predict(X_train)


# In[133]:


EvaluationMetric(X_train,y_train,y_pred_xgb_train)


# In[134]:


EvaluationMetric(X_test,y_test,y_pred_xgb_test)


# In[135]:


X_train.columns


# In[136]:


xgb_optimal_model.feature_importances_


# In[137]:


importances = xgb_optimal_model.feature_importances_

importance_dict = {'Feature' : list(X_train.columns),'Feature Importance' : importances}

importance_df = pd.DataFrame(importance_dict)


# In[138]:


importance_df.sort_values(by=['Feature Importance'],ascending=False,inplace=True)
importance_df


# In[139]:


plt.figure(figsize=(17,9))
plt.title('Features Importance')
sns.barplot(x='Feature',y="Feature Importance",data=importance_df[:10])


# # GradientBoosting

# In[140]:


# Number of trees
n_estimators = [100,120]

# Maximum depth of trees
max_depth = [5,8,10]

# Minimum number of samples required to split a node
min_samples_split = [50,80]

# Minimum number of samples required at each leaf node
min_samples_leaf = [40,50]


# HYperparameter Grid
param_gb = {'n_estimators' : n_estimators,'max_depth' : max_depth,'min_samples_split' : min_samples_split,'min_samples_leaf' : min_samples_leaf}


# In[141]:


# Create an instance of the  GradientBoostingRegressor
from sklearn.ensemble import GradientBoostingRegressor
gb_model = GradientBoostingRegressor()

# Grid search
gb_grid = GridSearchCV(estimator=gb_model,param_grid=param_gb,cv=3,verbose=2,scoring='r2')

gb_grid.fit(X_train,y_train)


# In[143]:


gb_grid.best_params_


# In[144]:


gb_grid.best_estimator_


# In[145]:


gb_optimal_model = gb_grid.best_estimator_


# In[146]:


y_preds_gb = gb_optimal_model.predict(X_test)
y_pred_gb_train = gb_optimal_model.predict(X_train)


# In[147]:



EvaluationMetric(X_train,y_train,y_pred_gb_train)


# In[148]:


EvaluationMetric(X_test,y_test,y_preds_gb)


# In[149]:


xgb_optimal_model.feature_importances_


# In[150]:


importances = gb_optimal_model.feature_importances_

importance_dict = {'Feature' : list(X_train.columns),'Feature Importance' : importances}

importance_df = pd.DataFrame(importance_dict)


# In[151]:


importance_df.sort_values(by=['Feature Importance'],ascending=False,inplace=True)
importance_df


# In[152]:


plt.figure(figsize=(17,9))
plt.title('Top 5 Features')
sns.barplot(x='Feature',y="Feature Importance",data=importance_df[:10])


# # Light GBM

# In[154]:


get_ipython().system('pip install lightgbm')
from lightgbm import LGBMRegressor


# In[155]:


n_estimator = [5,10,20] # No. of trees
max_depth = [5,7,9] # max depth of tree
min_samples_split = [40,50]
params = {"n_estimator" : n_estimator,"max_depth" : max_depth,"min_samples_split" : min_samples_split}
lgb = LGBMRegressor()
gs_lgb = GridSearchCV(lgb,params,cv=3,verbose=2,scoring='r2')
gs_lgb.fit(X_train,y_train)
print(gs_lgb.best_score_)
print(gs_lgb.best_params_)


# In[156]:


gs_lgb.best_estimator_


# In[157]:


gs_lgb_opt_model = gs_lgb.best_estimator_


# In[158]:


y_preds_lgb = gs_lgb_opt_model.predict(X_test)
y_pred_lgb_train = gs_lgb_opt_model.predict(X_train)


# In[159]:


EvaluationMetric(X_train,y_train,y_pred_lgb_train)


# In[160]:


EvaluationMetric(X_test,y_test,y_preds_lgb)


# In[161]:


importances = gs_lgb_opt_model.feature_importances_

importance_dict = {'Feature' : list(X_train.columns),'Feature Importance' : importances}

importance_df = pd.DataFrame(importance_dict)


# In[162]:


importance_df.sort_values(by=['Feature Importance'],ascending=False,inplace=True)
importance_df


# In[163]:


plt.figure(figsize=(17,9))
plt.title('Top 5 Features')
sns.barplot(x='Feature',y="Feature Importance",data=importance_df[:10])


# In[165]:


get_ipython().system('pip install prettytable')


# In[166]:


from prettytable import PrettyTable


# In[167]:


from prettytable import PrettyTable
train = PrettyTable(['SL. NO.',"MODEL_NAME","Train MSE","Train RMSE",'Train R^2','Train Adjusted R^2'])
train.add_row(['1','Linear Regression','0.0055473742826857575', '0.07448069738318619', '0.4975386610902135','0.49741929668112017'])
train.add_row(['2','Lasso Regression','0.005544777800761673','0.07446326477372364','0.4977738411443343','0.4976545326044711'])
train.add_row(['3','Ridge Regression','0.0055447795071197434','0.07446327623144004','0.49777368658851806','0.4976543780119387'])
train.add_row(['4','DecisionTree Regressor','0.003933693284929908','0.0627191620234989','0.6437001193563989','0.6436154769741506'])
train.add_row(['5','XGBRegressor','0.001963513255289278','0.044311547651704496','0.8221519859766687','0.8221097365109717'])
train.add_row(['6','GradientBoosting','0.0022675373651197994','0.04761866614175369','0.7946145685424261','0.7945657773046455'])
train.add_row(['7','LightGBM','0.0032980298940813377','0.057428476334318135','0.7012762377478683','0.7012052731131748'])
print(train)


# In[168]:


from prettytable import PrettyTable
test = PrettyTable(['SL. NO.',"MODEL_NAME","Test MSE","Test RMSE",'Test R^2','Test Adjusted R^2'])
test.add_row(['1','Linear Regression','0.005472765706091576','0.07397814343501448','0.4957919161505717','0.495312438993758'])
test.add_row(['2','Lasso Regression','0.0054708980396948005','0.07396551926198315','0.49596398499944283','0.49548467147166453'])
test.add_row(['3','Ridge Regression','0.005470874962507452','0.07396536326218815','0.49596611110990296','0.4954867996039515'])
test.add_row(['4','DecisionTree Regressor','0.004235598377226902','0.0650814749158845','0.609772634819681','0.6094015477356758'])
test.add_row(['5','XGBRegressor','0.0031445622329224813','0.056076396397436966','0.7102902292633624','0.7100147294813806'])
test.add_row(['6','GradientBoosting','0.003121489155427573','0.05587028866425851','0.7124159610810551',' 0.7121424827657667'])
test.add_row(['7','LightGBM','0.003418832582409957',' 0.058470784007142895','0.6850215927460219','0.6847220637301148'])
print(test)


# We conclude that Gradient Boosting perform well.

# # Conclusion 

# ### Objective:

# - We observe that both models show somewhat similar learning rate but with visible differences in error rates.
# - Gradient boosting performed very well out of all the models
# - XGBoost training curve on the other hand starts quite low and further improves with the increase in the training size and it too plateau towards the end.
# - Validation curve seems to show similar trend in both the models i.e. starts very high but improves with the training size with some differences in error rate i.e. XGBoost curve learning is quite fast and more accurate as compared to the RF one.
# 
# - Both the models seems to suffer from high variance since the training curve error is very less in both the models.
# 
# - The large gap at the end also indicates that the model suffers from quite a low bias i.e. overfitting the training data.
# 
# - Also, both the model's still has potential to decrease and converge towards the training curve by the end.

# ### End Notes:

# We discussed a variety of machine learning development cycle topics in this project. We found that the exploration of the data and variable analysis is a crucial part of the process and should be carried out for a complete knowledge of the data. While exploring, we also cleansed the data because some outliers needed to be dealt with before feature engineering. Additionally, we used feature engineering to filter out and collect only the best features—those that are more important and accounted for the majority of the variance in the dataset. Finally, we trained the models using the featureset that produced the best results.

# ### Improvement:

# - Add more training instances to improve validation curve in the XGBoost model.
# - Increase the regularization for the learning algorithm. This should decrease the variance and increase the bias towards the validation curve.
# - Reduce the numbers of features in the training data that we currently use. The algorithm will still fit the training data very well, but due to the decreased number of features, it will build less complex models. This should increase the bias and decrease the variance.