#!/usr/bin/env python
# coding: utf-8

# # EDA and XGBoost regression for each store

# ## Imports

# In[1]:


conda install -c anaconda py-xgboost


# In[2]:


import numpy as np 
import pandas as pd 
import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

import seaborn as sns
import matplotlib.pyplot as plt
pd.options.display.float_format = '{:.2f}'.format


# ## Reading the data

# In[3]:


oil_df = pd.read_csv('oil.csv')
train_df = pd.read_csv('train.csv', index_col=0)
test_df = pd.read_csv('test.csv')
sample_subm = pd.read_csv('sample_submission.csv')
holi_df = pd.read_csv('holidays_events.csv')
store_df =  pd.read_csv('stores.csv')
trans_df = pd.read_csv('transactions.csv')


# ## Analyze

# In[4]:


# training data 형태 및 행, 열, 데이터 유형 등의 개수 확인
train_df[train_df.sales>0]


# In[5]:


# unique stores 
np.sort(train_df.store_nbr.unique())


# In[6]:


# unique family of items
train_df.family.unique()


# In[7]:


train_df.info()


# In[8]:


train_df.describe()


# In[9]:


# 각 상점의 sales 양
train_df.groupby(by='store_nbr')['sales'].sum()


# In[10]:


#매출량 시각화 (특이치 확인에 용이)
import plotly.express as px
fig = px.scatter(train_df[train_df.store_nbr==1], x="date", y="sales")
fig.show()


# In[11]:


#month, year, day 등으로 date 쪼개서 저장

train_df['month'] = pd.to_datetime(train_df['date']).dt.month
train_df['day'] = pd.to_datetime(train_df['date']).dt.day
train_df['day_name'] = pd.to_datetime(train_df['date']).dt.day_name()
train_df['year'] = pd.to_datetime(train_df['date']).dt.year
train_df.head(1)


# In[12]:


# 월별 매출 확인

table = pd.pivot_table(train_df, values ='sales', index =['store_nbr'],
                         columns =['month'], aggfunc = np.sum)
fig, ax = plt.subplots(figsize=(15,12))         
sns.heatmap(table, annot=False, linewidths=.5, ax=ax, cmap="YlGnBu")
plt.show()


# 시간이 지남에 따른 점진적인 매출 증가없고 거의 모든 상점 지난 달과 같은 사업 진행하는 것으로 확인

# In[13]:


# 다른 제품군이 상품판매에 어떤 영향을 미치는지 확인 

table1 = pd.pivot_table(train_df, values ='sales', index =['family'], aggfunc = np.sum)
fig, ax = plt.subplots(figsize=(10,10))         
sns.heatmap(table1, annot=True, linewidths=.5, ax=ax, cmap="YlGnBu")
plt.show()


# In[14]:


# 각 상품의 percentage of sales 

total_sum = table1.sales.sum()
table1/total_sum


# In[15]:


# promotion이 sales에 미치는 영향확인 (상점 1, 2)

import plotly.express as px
fig = px.scatter(train_df[train_df.store_nbr==1], x="onpromotion", y="sales")
fig.show()


# In[16]:


import plotly.express as px
fig = px.scatter(train_df[train_df.store_nbr==2], x="onpromotion", y="sales")
fig.show()


# 판매와 판촉 사이 간극 일주일

# In[17]:


# total sales with and without promotion

import plotly.express as px
df = px.data.tips()
fig = px.histogram(train_df[train_df.onpromotion<200], x="onpromotion", nbins=20)
fig.show()


# In[18]:


# 요일별 sales 확인

table3 = pd.pivot_table(train_df, values ='sales', index =['day_name'], aggfunc = np.sum)
fig, ax = plt.subplots(figsize=(10,10))         
sns.heatmap(table3, annot=True, linewidths=.5, ax=ax, cmap="YlGnBu")
plt.show()


# 주말 매출이 항상 높음

# In[19]:


# 연도에 따른 sales 확인
table_year = pd.pivot_table(train_df, values ='sales', index =['store_nbr'],
                         columns =['year'], aggfunc = np.sum)

fig, ax = plt.subplots(figsize=(15,12))         
sns.heatmap(table_year, annot=False, linewidths=.5, ax=ax, cmap="YlGnBu")
plt.show()


# year과 sales 간 약간의 연관성이 있어 보임

# ## model building / feature engineering을 위한 train데이터 조작

# In[20]:


# making the groups for family

family_map       = {'AUTOMOTIVE': 'rest',
                   'BABY CARE': 'rest',
                   'BEAUTY': 'rest',
                   'BOOKS': 'rest',
                   'CELEBRATION': 'rest',
                   'GROCERY II': 'rest',
                   'HARDWARE': 'rest',
                   'HOME AND KITCHEN I': 'rest',
                   'HOME AND KITCHEN II': 'rest',
                   'HOME APPLIANCES': 'rest',
                   'LADIESWEAR': 'rest',
                   'LAWN AND GARDEN': 'rest',
                   'LINGERIE': 'rest',
                   'MAGAZINES': 'rest',
                   'PET SUPPLIES': 'rest',
                   'PLAYERS AND ELECTRONICS': 'rest',
                   'SCHOOL AND OFFICE SUPPLIES': 'rest',
                   'SEAFOOD': 'rest',
                   'DELI': 'first_sec',
                    'EGGS': 'first_sec',
                    'FROZEN FOODS': 'first_sec',
                    'HOME CARE': 'first_sec',
                    'LIQUOR,WINE,BEER': 'first_sec',
                    'PREPARED FOODS': 'first_sec',
                    'PERSONAL CARE': 'first_sec',
                    'BREAD/BAKERY': 'third',
                    'MEATS': 'third',
                    'POULTRY': 'third',
                    'CLEANING':'fourth',
                    'DAIRY':'fourth',
                    'PRODUCE':'seventh',
                    'BEVERAGES':'fifth',
                    'GROCERY I': 'sixth'
                   }

train_df['new_family'] = train_df['family'].map(family_map)
train_df.head(2)


# In[21]:


# ouliers 해결을 위한 그래프

import plotly.express as px
fig = px.scatter(train_df[train_df.store_nbr==4], x="date", y="sales")
fig.show()


# In[22]:


# handling the ouliers for each store

for i in range(1,len(train_df.store_nbr.unique())+1):
    val = train_df[train_df.store_nbr == i].sales.quantile(0.99)
    train_df = train_df.drop(train_df[(train_df.store_nbr==i) & (train_df.sales > val)].index)
    
# outlier 제거 시 전체 행의 약 1%가 손상됨
fig = px.scatter(train_df[train_df.store_nbr==4], x="date", y="sales")
fig.show()


# ## analysis with store metadata and holiday data

# In[23]:


store_df.shape


# In[24]:


# store data train 데이터에 병합

train_df = pd.merge(train_df, store_df, on='store_nbr', how='left') 
train_df.head(3)


# In[25]:


#holiday data

holi_df.head(2)


# In[26]:


holi_df.locale.unique()


# In[27]:


# unique local names (store meta data와 결합시 train데이터와 병합하는데 도움)
holi_df.locale_name.unique()


# In[28]:


# unique city data
store_df.city.unique()


# In[29]:


store_df.state.unique()


# In[30]:


holi_df.type.unique()


# In[31]:


# 'type'이 중복되므로 이름 변경
holi_df.rename(columns={'type': 'day_nature'},
          inplace=True, errors='raise')

# only 1 national name Ecuador
holi_df[holi_df.locale=='National'].head(3)


# In[32]:


# holidays 종류에 따라 새로운 data frame 생성

holiday_loc = holi_df[holi_df.locale == 'Local']
holiday_reg = holi_df[holi_df.locale == 'Regional']
holiday_nat = holi_df[holi_df.locale == 'National']


holiday_loc.rename(columns={'locale_name': 'city'},
          inplace=True, errors='raise')
holiday_reg.rename(columns={'locale_name': 'state'},
          inplace=True, errors='raise')


# In[33]:


holiday_loc


# In[34]:


holiday_reg


# In[35]:


train_df = pd.merge(train_df, holiday_loc, on=['date', 'city'], how='left') 
train_df = train_df[~((train_df.day_nature == 'Holiday') & (train_df.transferred == False))]
train_df.drop(['day_nature', 'locale', 'description','transferred'], axis=1, inplace=True)

train_df = pd.merge(train_df, holiday_reg, on=['date', 'state'], how='left') 
train_df = train_df[~((train_df.day_nature == 'Holiday') & (train_df.transferred == False))]
train_df.drop(['day_nature', 'locale', 'description','transferred'], axis=1, inplace=True)

train_df = pd.merge(train_df, holiday_nat, on=['date'], how='left') 
train_df = train_df[~((train_df.day_nature == 'Holiday') & (train_df.transferred == False))]
train_df.drop(['day_nature', 'locale', 'description','transferred'], axis=1, inplace=True)


# In[36]:


train_df


# In[37]:


train_df.drop(['date', 'family', 'month', 'day','city','state','type', 'cluster', 'locale_name', 'year'],axis=1, inplace=True)


# In[38]:


train_df = pd.get_dummies(train_df, columns = ['day_name','new_family'])
train_df.reset_index(inplace=True)
train_df.drop(['index'],axis=1, inplace=True)
train_df.head(2)


# In[39]:


from sklearn.linear_model import LinearRegression
from xgboost import XGBRegressor
from sklearn import preprocessing


for i in range(1,len(train_df.store_nbr.unique())+1):  
    temp_df = train_df[train_df.store_nbr == i]
    sale_out = temp_df[['sales']]
    globals()['max_%s' % i] = temp_df['onpromotion'].max()
    temp_df['onpromotion'] = (temp_df['onpromotion']/globals()['max_%s' % i])
    temp_df.onpromotion = np.where(temp_df.onpromotion<0, 0, temp_df.onpromotion)
    temp_df.drop(['sales','store_nbr'],axis=1, inplace=True)
    globals()['model_%s' % i] = XGBRegressor(verbosity=0)
    globals()['model_%s' % i].fit(temp_df, sale_out)


# In[40]:


test_df['day_name'] = pd.to_datetime(test_df['date']).dt.day_name()
test_df['new_family'] = test_df['family'].map(family_map)
test_df.drop(['date','family'],axis=1, inplace=True)
test_df = pd.get_dummies(test_df, columns = ['day_name','new_family'])
test_df.head(2)


# In[41]:


backup_df_1 = pd.DataFrame()

for i in range(1,len(train_df.store_nbr.unique())+1):
    temp_df = test_df[test_df.store_nbr == i]
    temp_df['onpromotion'] = (temp_df['onpromotion']/globals()['max_%s' % i])
    temp_df.onpromotion = np.where(temp_df.onpromotion<0, 0, temp_df.onpromotion)
    save_id = temp_df[['id']].reset_index()
    temp_df.drop(['id','store_nbr'],axis=1, inplace=True)
    submit = globals()['model_%s' % i].predict(temp_df)
    save_id['sales'] = submit
    df11 = pd.DataFrame(submit, columns = ['sales'])
    backup_df = pd.concat([save_id[['id']], df11], axis = 1, ignore_index = True)
    backup_df_1 = backup_df_1.append(backup_df, ignore_index=True)

backup_df_1.rename(columns={0 : "id", 1 : "sales"}, inplace=True, errors='raise')
test_df = pd.merge(test_df, backup_df_1, on='id', how='left') 


# In[42]:


backup_df_1.head(4)


# In[43]:


sample_df = test_df[['id', 'sales']]
sample_df.sales = np.where(sample_df.sales<0, 0, sample_df.sales)
sample_df.head(3)


# In[44]:


sample_df.to_csv('submission.csv' , index = False)


# ## result
# 
# ![image.png](attachment:image.png)

# ## XGBoost
# 
# https://webnautes.tistory.com/1643
# 
# Gradient Boosting 을 사용해서 모델링한다.
# 
# 경사 하강법을 사용하여 모델의 손실이 줄어들게 되는(모델 손실의 그레디언트가 낮아지도록) 트리를 모델에 추가하는 그리드 방식 이용
# 
# *결정트리앙상블
# 
