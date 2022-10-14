import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# import libraries for plotting
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')


def chargement_data():
#   from google.colab import drive
#   drive.mount('/content/drive')


  df = pd.read_csv('weatherAUS.csv')

  #Remplacement des valeurs manquantes par la moyenne pour les variables quantitatives
  df['MinTemp']=df['MinTemp'].fillna(df['MinTemp'].mean())
  df['MaxTemp']=df['MinTemp'].fillna(df['MaxTemp'].mean())
  df['Rainfall']=df['Rainfall'].fillna(df['Rainfall'].mean())
  df['Evaporation']=df['Evaporation'].fillna(df['Evaporation'].mean())
  df['Sunshine']=df['Sunshine'].fillna(df['Sunshine'].mean())
  df['WindGustSpeed']=df['WindGustSpeed'].fillna(df['WindGustSpeed'].mean())
  df['WindSpeed9am']=df['WindSpeed9am'].fillna(df['WindSpeed9am'].mean())
  df['WindSpeed3pm']=df['WindSpeed3pm'].fillna(df['WindSpeed3pm'].mean())
  df['Humidity9am']=df['Humidity9am'].fillna(df['Humidity9am'].mean())
  df['Humidity3pm']=df['Humidity3pm'].fillna(df['Humidity3pm'].mean())
  df['Pressure9am']=df['Pressure9am'].fillna(df['Pressure9am'].mean())
  df['Pressure3pm']=df['Pressure3pm'].fillna(df['Pressure3pm'].mean())
  df['Cloud9am']=df['Cloud9am'].fillna(df['Cloud9am'].mean())
  df['Cloud3pm']=df['Cloud3pm'].fillna(df['Cloud3pm'].mean())
  df['Temp9am']=df['Temp9am'].fillna(df['Temp9am'].mean())
  df['Temp3pm']=df['Temp3pm'].fillna(df['Temp3pm'].mean())

  #Remplacement des valeurs manquantes par lle mode pour les variables qualitativesàsq
  df['RainToday']=df['RainToday'].fillna(df['RainToday'].mode()[0])
  df['RainTomorrow']=df['RainTomorrow'].fillna(df['RainTomorrow'].mode()[0])
  df['WindDir9am'] = df['WindDir9am'].fillna(df['WindDir9am'].mode()[0])
  df['WindGustDir'] = df['WindGustDir'].fillna(df['WindGustDir'].mode()[0])
  df['WindDir3pm'] = df['WindDir3pm'].fillna(df['WindDir3pm'].mode()[0])



  return df

def chargement_data_time():
#   from google.colab import drive
#   drive.mount('/content/drive')


  df = pd.read_csv('weatherAUS.csv', header=0, parse_dates=[0], index_col=0, squeeze=True) 

  #Remplacement des valeurs manquantes par la moyenne pour les variables quantitatives
  df['MinTemp']=df['MinTemp'].fillna(df['MinTemp'].mean())
  df['MaxTemp']=df['MinTemp'].fillna(df['MaxTemp'].mean())
  df['Rainfall']=df['Rainfall'].fillna(df['Rainfall'].mean())
  df['Evaporation']=df['Evaporation'].fillna(df['Evaporation'].mean())
  df['Sunshine']=df['Sunshine'].fillna(df['Sunshine'].mean())
  df['WindGustSpeed']=df['WindGustSpeed'].fillna(df['WindGustSpeed'].mean())
  df['WindSpeed9am']=df['WindSpeed9am'].fillna(df['WindSpeed9am'].mean())
  df['WindSpeed3pm']=df['WindSpeed3pm'].fillna(df['WindSpeed3pm'].mean())
  df['Humidity9am']=df['Humidity9am'].fillna(df['Humidity9am'].mean())
  df['Humidity3pm']=df['Humidity3pm'].fillna(df['Humidity3pm'].mean())
  df['Pressure9am']=df['Pressure9am'].fillna(df['Pressure9am'].mean())
  df['Pressure3pm']=df['Pressure3pm'].fillna(df['Pressure3pm'].mean())
  df['Cloud9am']=df['Cloud9am'].fillna(df['Cloud9am'].mean())
  df['Cloud3pm']=df['Cloud3pm'].fillna(df['Cloud3pm'].mean())
  df['Temp9am']=df['Temp9am'].fillna(df['Temp9am'].mean())
  df['Temp3pm']=df['Temp3pm'].fillna(df['Temp3pm'].mean())

  #Remplacement des valeurs manquantes par lle mode pour les variables qualitativesàsq
  df['RainToday']=df['RainToday'].fillna(df['RainToday'].mode()[0])
  df['RainTomorrow']=df['RainTomorrow'].fillna(df['RainTomorrow'].mode()[0])
  df['WindDir9am'] = df['WindDir9am'].fillna(df['WindDir9am'].mode()[0])
  df['WindGustDir'] = df['WindGustDir'].fillna(df['WindGustDir'].mode()[0])
  df['WindDir3pm'] = df['WindDir3pm'].fillna(df['WindDir3pm'].mode()[0])



  return df

def chargement_data():
#   from google.colab import drive
#   drive.mount('/content/drive')


  #df = pd.read_csv('/content/drive/MyDrive/Projet Meteo/weatherAUS.csv')
  df = pd.read_csv('weatherAUS.csv')

  #Remplacement des valeurs manquantes par la moyenne pour les variables quantitatives
  df['MinTemp']=df['MinTemp'].fillna(df['MinTemp'].mean())
  df['MaxTemp']=df['MinTemp'].fillna(df['MaxTemp'].mean())
  df['Rainfall']=df['Rainfall'].fillna(df['Rainfall'].mean())
  df['Evaporation']=df['Evaporation'].fillna(df['Evaporation'].mean())
  df['Sunshine']=df['Sunshine'].fillna(df['Sunshine'].mean())
  df['WindGustSpeed']=df['WindGustSpeed'].fillna(df['WindGustSpeed'].mean())
  df['WindSpeed9am']=df['WindSpeed9am'].fillna(df['WindSpeed9am'].mean())
  df['WindSpeed3pm']=df['WindSpeed3pm'].fillna(df['WindSpeed3pm'].mean())
  df['Humidity9am']=df['Humidity9am'].fillna(df['Humidity9am'].mean())
  df['Humidity3pm']=df['Humidity3pm'].fillna(df['Humidity3pm'].mean())
  df['Pressure9am']=df['Pressure9am'].fillna(df['Pressure9am'].mean())
  df['Pressure3pm']=df['Pressure3pm'].fillna(df['Pressure3pm'].mean())
  df['Cloud9am']=df['Cloud9am'].fillna(df['Cloud9am'].mean())
  df['Cloud3pm']=df['Cloud3pm'].fillna(df['Cloud3pm'].mean())
  df['Temp9am']=df['Temp9am'].fillna(df['Temp9am'].mean())
  df['Temp3pm']=df['Temp3pm'].fillna(df['Temp3pm'].mean())

  #Remplacement des valeurs manquantes par lle mode pour les variables qualitativesàsq
  df['RainToday']=df['RainToday'].fillna(df['RainToday'].mode()[0])
  df['RainTomorrow']=df['RainTomorrow'].fillna(df['RainTomorrow'].mode()[0])
  df['WindDir9am'] = df['WindDir9am'].fillna(df['WindDir9am'].mode()[0])
  df['WindGustDir'] = df['WindGustDir'].fillna(df['WindGustDir'].mode()[0])
  df['WindDir3pm'] = df['WindDir3pm'].fillna(df['WindDir3pm'].mode()[0])



  return df

def chargement_ville(choix_ville,df):
  df_prepare = df[df['Location'] == choix_ville]
  return df_prepare

def separation_colonnes(df_prepare):
  cat_columns = df_prepare.select_dtypes(include=['object']).columns
  num_columns = df_prepare.select_dtypes(include=['float64', 'int64']).columns
  return cat_columns,num_columns

def separation_colonnes_date(df_prepare):
  cat_columns = df_prepare.select_dtypes(include=['object']).columns
  num_columns = df_prepare.select_dtypes(include=['float64', 'int64']).columns
  cat_columns.drop('Date')
  return date_columns,cat_columns,num_columns

def encodage(df_prepare,cat_columns):
  
  cat_columns = cat_columns.drop(['RainToday','RainTomorrow'])

  #Encodages des variables catégorielles sauf pour RainToday, RainTomorrow
  from sklearn.preprocessing import LabelEncoder
  le = LabelEncoder()
  for i in cat_columns:
      df_prepare[i] = le.fit_transform(df_prepare[i])
  df_prepare.head()

  # Remplace valeur de Rain pour 0 quand non, 1 quand oui
  label_dict = {'No': 0,'Yes':1}
  df_prepare['RainTomorrow'] = df_prepare['RainTomorrow'].map(label_dict)


  df_prepare['RainToday'] = df_prepare['RainToday'].map(label_dict)

  df_prepare.head()
  return df_prepare