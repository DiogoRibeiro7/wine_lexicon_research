import numpy as np
import pandas as pd 

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import KFold, cross_val_score, GridSearchCV, train_test_split

from sklearn.preprocessing import LabelEncoder, RobustScaler

from sklearn.linear_model import Lasso, Ridge
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor

import warnings
warnings.filterwarnings('ignore')


def load_data():
  red = pd.read_csv("https://github.com/ASapayev/wine_CSV/raw/main/Red.csv")
  red['Type'] = 'red'
  sparkling = pd.read_csv("https://github.com/ASapayev/wine_CSV/raw/main/Sparkling.csv")
  sparkling['Type'] = 'sparkling'
  white = pd.read_csv("https://github.com/ASapayev/wine_CSV/raw/main/White.csv")
  white['Type'] = 'white'
  rose = pd.read_csv("https://github.com/ASapayev/wine_CSV/raw/main/Rose.csv")
  rose['Type'] = 'rose'
  return pd.concat([red, sparkling, white, rose], ignore_index=True)

wines = load_data()

wines.info()

wines.Year.value_counts().sort_index()

wines.Year = wines.Year.replace('N.V.', 2024).astype("int")
wines['Name'] = wines['Name'].str.split().str[:-1].str.join(' ')

wines


def print_summarize_dataset(dataset):
    print('Dataset shape: ', dataset.shape, '\n\n')
    print('Dataset info:', dataset.info(), '\n\n')
    print('Data summarize:\n', dataset.describe(), '\n\n')
    print('Total countries', dataset.Country.nunique(), '\n\n')
    print(dataset.Country.value_counts())
print_summarize_dataset(wines)


country = wines.Country.value_counts()[:12]

plt.figure(figsize=(12,5))
sns.barplot( y=country.values, x=country.index, palette=sns.color_palette("Set2", 12))
plt.xticks(rotation=60)
plt.title('Countries with the largest export volume')
plt.xlabel("Country")
plt.ylabel("Volume")
plt.show()


wines.corr()

plt.figure()
sns.heatmap(wines.corr(), cmap=sns.color_palette("RdPu", 10))
plt.show()

plt.figure(figsize=(12, 5))
sns.countplot(data=wines, x='Rating', color='purple')
plt.title("Rating Count distribuition ", fontsize=20)
plt.xlabel("Rating", fontsize=15) 
plt.ylabel("Count", fontsize=15)
plt.show()

plt.figure(figsize=(10,15))

plt.subplot(3,1,1)
graph = sns.distplot(wines['NumberOfRatings'], color='olive')
graph.set_title("Number Of Ratings distribuition", fontsize=20) 
graph.set_xlabel("Number Of Ratings", fontsize=15)
graph.set_ylabel("Frequency", fontsize=15) 

plt.subplot(3,1,2)
graph1 = sns.distplot(np.log(wines['NumberOfRatings']), color='olive')
graph1.set_title("Number Of Ratings Log distribuition", fontsize=20) 
graph1.set_xlabel("Number Of Ratings", fontsize=15) 
graph1.set_ylabel("Frequency", fontsize=15)
graph1.set_xticklabels(np.exp(graph1.get_xticks()).astype(int))

plt.subplot(3,1,3)
graph = sns.distplot(wines[wines['NumberOfRatings']<1000]['NumberOfRatings'], color='olive')
graph.set_title("Number Of Ratings <1000 distribuition", fontsize=20)
graph.set_xlabel("Number Of Ratings", fontsize=15) 
graph.set_ylabel("Frequency", fontsize=15) 

plt.subplots_adjust(hspace = 0.3,top = 0.9)
plt.show()


plt.figure(figsize=(13,5))

graph = sns.regplot(x=np.log(wines['Price']), y='Rating', 
                    data=wines, fit_reg=False, color='green')
graph.set_title("Rating x Price Distribuition", fontsize=20)
graph.set_xlabel("Price(EUR)", fontsize= 15)
graph.set_ylabel("Rating", fontsize= 15)
graph.set_xticklabels(np.exp(graph.get_xticks()).astype(int))

plt.show()


varieties = pd.read_csv("https://github.com/ASapayev/wine_CSV/raw/main/Varieties.csv")
varieties


wines['Variety'] = np.nan
for index in wines.index:
    for variety in varieties['Variety']:    
        if variety in wines.loc[index, 'Name']:
            wines.loc[index, 'Variety'] = variety
            break

print('Now we have variety for', wines.Variety.notna().sum(),'wines,',
      '%s%%' % int(wines.Variety.notna().sum()/len(wines)*100), 'of all')


# replace NaN's
wines.Variety = wines.Variety.fillna('unknown')
wines_enc = wines.copy().drop(columns = ['Name'])
#One-hot encoder for winestyle
wines_enc = pd.get_dummies(wines_enc, columns = ['Type'])
wines_enc


categorical_cols = [col for col in wines_enc.columns if wines_enc[col].dtype == "object"]
# Apply label encoder
label_encoder = LabelEncoder()
for col in categorical_cols:
    wines_enc[col] = label_encoder.fit_transform(wines_enc[col])
wines_enc.head()

lgbm = lightgbm.fit(X_train, y_train)
res_low_NumberOfRatings = lgbm.predict(X_low_NumberOfRatings_test)
res_high_NumberOfRatings = lgbm.predict(X_high_NumberOfRatings_test)
res_random_NumberOfRatings = lgbm.predict(X_random_test)

print('MAE of predictions with low NumberOfRatings:   ', mean_absolute_error(y_low_NumberOfRatings_test, res_low_NumberOfRatings))
print('MAE of predictions with high NumberOfRatings:  ', mean_absolute_error(y_high_NumberOfRatings_test, res_high_NumberOfRatings))
print('MAE of predictions with middle NumberOfRatings:', mean_absolute_error(y_random_test, res_random_NumberOfRatings))


print('MAE of predictions with low NumberOfRatings:   ', mean_absolute_error(y_low_NumberOfRatings_test, res_low_NumberOfRatings))
print('MAE of predictions with high NumberOfRatings:  ', mean_absolute_error(y_high_NumberOfRatings_test, res_high_NumberOfRatings))
print('MAE of predictions with middle NumberOfRatings:', mean_absolute_error(y_random_test, res_random_NumberOfRatings))

