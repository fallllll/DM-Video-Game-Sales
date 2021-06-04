import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

data = pd.read_csv('vgsales.csv')
print(data)

popular_game = data[['Name', 'Global_Sales']]
popular_game.set_index('Name', inplace=True)
print(popular_game.index)

p20_game = data[:20][['Name', 'Global_Sales']]
p20_game.set_index('Name', inplace=True)
p20_game.plot.bar()

popular_genre = data[['Genre', 'Global_Sales']]
popular_genre.set_index('Genre', inplace=True)
print(popular_genre.index)

p20_genre = data[:20][['Genre', 'Global_Sales']]
p20_genre.set_index('Genre', inplace=True)
p20_genre.plot.bar()

popular_platform = data[['Platform', 'Global_Sales']]
popular_platform.set_index('Platform', inplace=True)
print(popular_platform.index)

p20_platform = data[:20][['Platform', 'Global_Sales']]
p20_platform.set_index('Platform', inplace=True)
p20_platform.plot.bar()


popular_publisher = data[['Publisher', 'Global_Sales']]
popular_publisher.set_index('Publisher', inplace=True)
print(popular_publisher.index)

p20_publisher = data[:20][['Publisher', 'Global_Sales']]
p20_publisher.set_index('Publisher', inplace=True)
p20_publisher.plot.bar()


#分析数据得到每年电子游戏的历史销售额数据
yearset = set()
for y in data['Year']:
    yearset.add(y)
yearlist = list(yearset)
while np.isnan(yearlist[0]):
    yearlist.remove(yearlist[0])
print(yearlist)
print(len(yearlist))

sales = []
for year in yearlist:
    year_index = data['Year'] == year
    year_salenum = data['Global_Sales'][year_index].sum()
    sales.append(year_salenum)
print(sales)
print(len(sales))


#统计Platform、Publisher、Genre的种类
platformset = set()
for p in data['Platform']:
    platformset.add(p)
platformlist = list(platformset)
print(platformlist)
print(len(platformlist))

publisherset = set()
for pub in data['Publisher']:
    publisherset.add(pub)
publisherlist = list(publisherset)
print(publisherlist)
print(len(publisherlist))

genreset = set()
for g in data['Genre']:
    genreset.add(g)
genrelist = list(genreset)
print(genrelist)
print(len(genrelist))


#选择排名前20的platform、publisher和12种Genre作为自变量

genre_data = data[['Genre','Global_Sales']]
genre_data.set_index('Genre',inplace=True)
genre_data_20 = genre_data.groupby('Genre').sum()
genre_data_20.sort_values(by='Global_Sales', ascending=False, inplace=True)
genre_data_20 = genre_data_20[:20]
#print(genre_data_20)

platform_data = data[['Platform','Global_Sales']]
platform_data.set_index('Platform',inplace=True)
platform_data_20 = platform_data.groupby('Platform').sum()
platform_data_20.sort_values(by='Global_Sales', ascending=False, inplace=True)
platform_data_20 = platform_data_20[:20]
#print(platform_data_20)

publisher_data = data[['Publisher','Global_Sales']]
publisher_data.set_index('Publisher',inplace=True)
publisher_data_20 = publisher_data.groupby('Publisher').sum()
publisher_data_20.sort_values(by='Global_Sales', ascending=False, inplace=True)
publisher_data_20 = publisher_data_20[:20]
#print(publisher_data_20)

score = 0
Inputvec = {}
for item in genre_data_20.index:
        Inputvec[item] = score
        score += 1
for item in platform_data_20.index:
        Inputvec[item] = score
        score += 1
for item in publisher_data_20.index:
        Inputvec[item] = score
        score += 1
print(Inputvec)

ys = 0
yearvec = {}
for y in yearlist:
    yearvec[y] = ys
    ys +=1
print(yearvec)

# 生成输入向量
index = ['Genre', 'Platform', 'Publisher']
Input = np.zeros((len(yearlist), 52))

# genre
data1 = data[['Year', 'Genre', 'Global_Sales']]
# data1.set_index(['Year','Genre'],inplace=True)
for ye in yearlist:
    for gen in genre_data_20.index:
        df = data1[(data1['Year'] == ye) & (data1['Genre'] == gen)]
        # print(df)
        # print(df['Global_Sales'].sum())
        numsum = df['Global_Sales'].sum()
        a = yearvec[ye]
        b = Inputvec[gen]
        Input[a][b] = numsum

# platform
data2 = data[['Year', 'Platform', 'Global_Sales']]
for ye in yearlist:
    for pla in platform_data_20.index:
        df2 = data2[(data2['Year'] == ye) & (data2['Platform'] == pla)]
        # print(df)
        # print(df['Global_Sales'].sum())
        numsum = df2['Global_Sales'].sum()
        a = yearvec[ye]
        b = Inputvec[pla]
        Input[a][b] = numsum

# publisher
data3 = data[['Year', 'Publisher', 'Global_Sales']]
for ye in yearlist:
    for pub in publisher_data_20.index:
        df3 = data3[(data3['Year'] == ye) & (data3['Publisher'] == pub)]
        # print(df)
        # print(df['Global_Sales'].sum())
        numsum = df3['Global_Sales'].sum()
        a = yearvec[ye]
        b = Inputvec[pub]
        Input[a][b] = numsum

print(Input)


#线性回归预测
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold
x = Input
y = np.array(sales)

from scipy.stats import pearsonr
x = x  / np.sum(x, 0)
print(x)
print(x.shape)
print(y.shape)
kf = KFold(n_splits=5, shuffle=True, random_state=123)
model = LinearRegression()

y_pred = np.zeros(y.shape)
for train, test in kf.split(x, y):
    model.fit(x[train], y[train])
    y_pred[test] = model.predict(x[test])

rmse = np.sqrt(np.sum((y_pred - y) ** 2) / y.shape[0])
print(rmse)
r = pearsonr(y, y_pred)[0]
print(r)
plt.scatter(y, y_pred)
plt.xlabel('real sales')
plt.ylabel('predicted sales')
plt.show()


x = data.groupby(['Year']).count()
gamenum = x['Name']
plt.figure(figsize=(12,8))
gamenum.plot.bar()
plt.show()


x = data.groupby(['Year']).count()
gamenum = x['Global_Sales']
plt.figure(figsize=(12,8))
gamenum.plot.bar()
plt.show()


x = data.groupby(['Platform']).count()
gamenum = x['Global_Sales']
plt.figure(figsize=(12,8))
gamenum.plot.bar()
plt.show()


x = data.groupby(['Publisher']).count()
gamenum = x[:30]['Global_Sales']
plt.figure(figsize=(12,8))
gamenum.plot.bar()
plt.show()




