import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import missingno as msno
import warnings
warnings.filterwarnings('ignore')

#Data loading
rain = pd.read_csv("weatherAUS.csv")
pd.set_option('display.expand_frame_repr', False)

print(rain.head())
print(rain.shape)
print(rain.info())
print(rain.describe())
print(rain.columns)

#EDA
fig = plt.figure(figsize = (8,5))
rain.RainTomorrow.value_counts(normalize = True).plot(kind='bar', alpha = 0.9, rot=0)
plt.title('RainTomorrow Distribution')
plt.show()

sns.countplot(x='RainToday', hue='RainTomorrow',data=rain)
plt.show()

rain['RainToday'].replace({'No': 0, 'Yes': 1},inplace = True)
rain['RainTomorrow'].replace({'No': 0, 'Yes': 1},inplace = True)

#Removing samples with Nan label
rain.dropna(how='all', subset=['RainTomorrow'], inplace=True)

num_features=[col for col in rain.columns if rain[col].dtypes !='O']
cat_features=[col for col in rain.columns if col not in num_features]

for k in ['MinTemp', 'MaxTemp', 'Rainfall', 'WindGustSpeed', 'WindSpeed9am', 'WindSpeed3pm', 'Humidity9am', 'Humidity3pm', 'Pressure9am', 'Sunshine', 'Cloud9am', 'Cloud3pm', 'Evaporation']:
    c=sns.FacetGrid(data=rain,col="RainTomorrow",height=6)
    c.map(sns.histplot, k, bins=25)
    plt.show()

# NaN values visualization
msno.matrix(rain)
plt.show()
msno.bar(rain.sample(100))
plt.show()
nanFeatures=[col for col in rain.columns if rain[col].isnull().any()]
print("features contain NaN values:{}".format(nanFeatures))
print(rain[nanFeatures].isnull().mean())

missing = pd.DataFrame(rain.isnull().sum(),columns = ['no.of missing values'])

missing['% missing_values']= (missing/len(rain)).round(2)*100
print(missing)

cols = ['Evaporation', 'Sunshine', 'Cloud9am', 'Cloud3pm']
plt.style.use('seaborn-dark')
fig, ax = plt.subplots(4, 2, figsize=(12, 8), constrained_layout=True)
for i, num_var in enumerate(cols):
    sns.kdeplot(data=rain, x=num_var, ax=ax[i][0],
                fill=True, alpha=0.6, linewidth=1.5)
    ax[i][0].set_ylabel(num_var)
    ax[i][0].set_xlabel(None)
    sns.histplot(data=rain, x=num_var, ax=ax[i][1])
    ax[i][1].set_ylabel(None)
    ax[i][1].set_xlabel(None)
fig.suptitle('Features con molti valori NaN', size=16);
plt.show()
print(rain.shape)

#NaN value management
rain.dropna(how='all', subset=['RainToday'], inplace=True)
rain['Evaporation']=rain['Evaporation'].fillna(rain['Evaporation'].mean())
rain['MinTemp']=rain['MinTemp'].fillna(rain['MinTemp'].median())
rain['MaxTemp']=rain['MaxTemp'].fillna(rain['MaxTemp'].median())
rain['Rainfall']=rain['Rainfall'].fillna(rain['Rainfall'].median())
rain['Sunshine']=rain['Sunshine'].fillna(rain['Sunshine'].median())
rain['WindGustSpeed']=rain['WindGustSpeed'].fillna(rain['WindGustSpeed'].median())
rain['WindSpeed9am']=rain['WindSpeed9am'].fillna(rain['WindSpeed9am'].median())
rain['WindSpeed3pm']=rain['WindSpeed3pm'].fillna(rain['WindSpeed3pm'].median())
rain['Humidity9am']=rain['Humidity9am'].fillna(rain['Humidity9am'].median())
rain['Humidity3pm']=rain['Humidity3pm'].fillna(rain['Humidity3pm'].median())
rain['Pressure9am']=rain['Pressure9am'].fillna(rain['Pressure9am'].median())
rain['Pressure3pm']=rain['Pressure3pm'].fillna(rain['Pressure3pm'].median())
rain['Cloud9am']=rain['Cloud9am'].fillna(rain['Cloud9am'].median())
rain['Cloud3pm']=rain['Cloud3pm'].fillna(rain['Cloud3pm'].median())
rain['Temp9am']=rain['Temp9am'].fillna(rain['Temp9am'].median())
rain['Temp3pm']=rain['Temp3pm'].fillna(rain['Temp3pm'].median())
rain['WindDir9am'] = rain['WindDir9am'].fillna(rain['WindDir9am'].mode()[0])
rain['WindGustDir'] = rain['WindGustDir'].fillna(rain['WindGustDir'].mode()[0])
rain['WindDir3pm'] = rain['WindDir3pm'].fillna(rain['WindDir3pm'].mode()[0])

msno.matrix(rain)
plt.show()

#correlation
plt.figure(figsize=(20,10))
sns.heatmap(rain.corr(), annot = True)
plt.show()

# multicolinearity
def correlation(df, threshold):
    col_corr = set()  # names of correlated columns
    corr_matrix = df.corr()
    for i in range(len(corr_matrix.columns)):
        for j in range(i):
            if abs(corr_matrix.iloc[i, j]) > threshold:
                colname = corr_matrix.columns[i]
                col_corr.add(colname)
    return col_corr
print(correlation(rain[num_features], 0.8))
plt.close()

rain.drop(['Temp3pm', 'Pressure3pm', 'Temp9am'], axis=1, inplace=True)
print(rain.shape)

#Outliers
plt.close()
plt.figure(figsize=(20,10))
plt.style.use('seaborn-dark')
sns.boxenplot(data = rain )
plt.xticks(rotation=90)
plt.show()
print(rain.shape)

plt.style.use('seaborn')
fig, axis = plt.subplots(13, 2, figsize=(12, 24))
for i, num_var in enumerate(['MinTemp', 'MaxTemp', 'Rainfall', 'WindGustSpeed', 'WindSpeed9am', 'WindSpeed3pm', 'Humidity9am', 'Humidity3pm', 'Pressure9am', 'Sunshine', 'Cloud9am', 'Cloud3pm', 'Evaporation']):
    sns.boxplot(y=num_var, data=rain, ax=axis[i][0], color='skyblue')
    sns.kdeplot(x=num_var, data=rain, ax=axis[i][1], color='skyblue',
                fill=True, alpha=0.6, linewidth=1.5)
    axis[i][0].set_ylabel(f"{num_var}", fontsize=12)
    axis[i][0].set_xlabel(None)
    axis[i][1].set_xlabel(None)
    axis[i][1].set_ylabel(None)
fig.suptitle('Numeric Features', fontsize=16, y=1)
plt.tight_layout()
plt.show()

fig, axes = plt.subplots(4,1, figsize=(10, 10), sharey=False)
fig.suptitle('Distribution of some continuous variables')

sns.boxplot(x= 'Rainfall', data = rain, palette = 'Set2', ax = axes[0])
axes[0].set_title("")
sns.boxplot(x= 'Evaporation', data = rain, palette = 'Set2', ax = axes[1])
axes[1].set_title("")
sns.boxplot(x= 'WindSpeed9am', data = rain, palette = 'Set2', ax = axes[2])
axes[2].set_title("")
sns.boxplot(x= 'WindSpeed3pm', data = rain, palette = 'Set2', ax = axes[3])
axes[3].set_title("")
plt.tight_layout()
plt.show()

fig, axes = plt.subplots(4,1, figsize=(10, 10), sharey=False)
fig.suptitle('Distribution of some continuous variables')


sns.histplot(x= 'Rainfall', data = rain, palette = 'Set2', ax = axes[0], bins = 10)
axes[0].set_title("")
sns.histplot(x= 'Evaporation', data = rain, palette = 'Set2', ax = axes[1], bins = 10)
axes[1].set_title("")
sns.histplot(x= 'WindSpeed9am', data = rain, palette = 'Set2', ax = axes[2], bins = 10)
axes[2].set_title("")
sns.histplot(x= 'WindSpeed3pm', data = rain, palette = 'Set2', ax = axes[3], bins = 10)
axes[3].set_title("")
plt.tight_layout()
plt.show()

# outliers removal
threshold = 0.0005
for col in ['MinTemp', 'MaxTemp', 'Rainfall', 'WindGustSpeed', 'WindSpeed9am', 'WindSpeed3pm', 'Humidity9am', 'Humidity3pm', 'Pressure9am', 'Sunshine', 'Cloud9am', 'Cloud3pm', 'Evaporation']:

    lower_threshold = rain[col].quantile(threshold)
    upper_threshold = rain[col].quantile(1 - threshold)

    rain = rain[(rain[col] >= lower_threshold) & (rain[col] <= upper_threshold)]

plt.figure(figsize=(20,10))
plt.style.use('seaborn-dark')
sns.boxenplot(data = rain )
plt.xticks(rotation=90)
plt.show()
print(rain.shape)

sns.pairplot(data=rain, vars=('MaxTemp','MinTemp','Pressure9am','Pressure3pm', 'Temp9am', 'Temp3pm', 'Evaporation'), hue='RainTomorrow' )
plt.show()

# date management
rain['Date'] = pd.to_datetime(rain['Date'])
rain['Day'] = rain['Date'].dt.day
rain['Month'] = rain['Date'].dt.month
rain['Year'] = rain['Date'].dt.year
rain.drop("Date",axis=1,inplace=True)
print(rain.head())

sns.countplot(x='Month',hue='RainTomorrow',data=rain, palette='Set2')
plt.tight_layout()
plt.show()
sns.countplot(x='Year',hue='RainTomorrow',data=rain, palette='Set2')
plt.tight_layout()
plt.show()

#windDir management
fig, ax =plt.subplots(3,1)
plt.figure(figsize=(10,10))
sns.countplot(data=rain,x='WindDir9am',ax=ax[0])
sns.countplot(data=rain,x='WindDir3pm',ax=ax[1])
sns.countplot(data=rain,x='WindGustDir',ax=ax[2])
fig.tight_layout()
plt.show()

for i in ['WindDir9am','WindDir3pm','WindGustDir','Location']:
    c = sns.FacetGrid(data=rain, col="RainTomorrow", height=6)
    c.map(sns.histplot, i, bins=25)
    plt.show()
print(rain.shape)

#correlation with label
print(rain.corr().abs()['RainTomorrow'].sort_values(ascending = False))

rain.drop("Day",axis=1,inplace=True)
rain.drop("Year",axis=1,inplace=True)
rain.drop("WindDir9am",axis=1,inplace=True)

#one-hot-encode
rain = pd.get_dummies(rain, drop_first = True).reset_index(drop = True)
print(rain.head())

rain.to_csv('C:/Users/Tommaso/Desktop/uni/Australia1.csv',index=False)
