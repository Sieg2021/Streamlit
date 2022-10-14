import streamlit as st # web development
import numpy as np # np mean, np random 
import pandas as pd # read csv, df manipulation
import time # to simulate a real time data, time loop 
import plotly.express as px # interactive charts 
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from fonction_utile import *

# read csv from a github repo
df = pd.read_csv("weatherAUS.csv")


st.set_page_config(
    page_title = 'Prévision météo',
    page_icon = '☁️',
    layout = 'wide'
)

# dashboard title

st.title("Prévision météo")

st.write(df)

dummy_regressor_model = joblib.load('mon_dummy_regr.joblib')
st.write(dummy_regressor_model)

df = chargement_data()
df_prepare = chargement_ville('Darwin',df)
df_prepare['Date'] = pd.to_datetime(df_prepare['Date'])
cat_columns, num_columns = separation_colonnes(df_prepare)
df_prepare = encodage(df_prepare,cat_columns)
num_columns = num_columns.drop('Temp9am')

#Séparation entre feature et cible
X = df_prepare.drop(columns=["Temp9am"], axis=1)
y = df_prepare["Temp9am"]


#séparation du jeu de donnée en train et test, on choisi 0.1 car cela représente 1 an de donnée ( 10 ans de données *0.1 = 1 an)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, shuffle = False)

y_predict = dummy_regressor_model.predict(X_test)

test_results = pd.DataFrame(data={'Test Predictions':y_predict, 'Réalité':y_test})
test_result= test_results.reset_index(drop=True)
st.write(test_result)

import matplotlib.pyplot as plt
plt.rcParams["figure.figsize"] = (20,15)

plt.plot(test_results['Test Predictions'][20:300],color='red',label ='Predictions')
plt.plot(test_results['Réalité'][20:300],label='Réalité')
plt.legend()
plt.show()
st.pyplot(plt)

train_df = pd.DataFrame()
train_df['ds'] = pd.to_datetime(df_prepare['Date'])
train_df['y']=df_prepare['Temp9am']
train_df.head(2)
train_df2 = train_df[train_df['ds']<'2016-06-24']
train_df2.tail(2)

prophet_model = joblib.load('prophet.joblib')
st.write(prophet_model)

future = prophet_model.make_future_dataframe(periods=365)
future.tail(2)

forecast = prophet_model.predict(future)
forecast.tail()

plt.rcParams["figure.figsize"] = (20,15)
fig, ax = plt.subplots()

ax.plot(train_df['ds'],train_df['y'],label = "réalité")
ax.plot(forecast['ds'], forecast['yhat'],label = "prédiction")
plt.legend()
plt.show()
st.pyplot(plt)


from keras.models import load_model
model_lstm = load_model('model_lstm.h5')
#st.write(model_lstm)

df = chargement_data()
df_prepare = chargement_ville('Darwin',df)

df_prepare.head()

train_df = pd.DataFrame()
train_df['ds'] = pd.to_datetime(df_prepare['Date'])
train_df['y']=df_prepare['Temp9am']
train_df.head(2)
train_df2 = train_df[train_df['ds']<'2016-06-24']
train_df2.tail(2)
df_rnn= train_df
df_rnn.index = pd.to_datetime(df_rnn['ds'])
df_rnn.drop(['ds'],axis=1)
def df_to_X_y(df, window_size=5):
  df_as_np = df.to_numpy()
  X = []
  y = []
  for i in range(len(df_as_np)-window_size):
    row = [[a] for a in df_as_np[i:i+window_size]]
    X.append(row)
    label = df_as_np[i+window_size]
    y.append(label)
  return np.array(X), np.array(y)
WINDOW_SIZE = 5
temp = df_rnn['y']
X1, y1 = df_to_X_y(temp, WINDOW_SIZE)
X1.shape, y1.shape
X_train1, y_train1 = X1[:2900], y1[:2900]
X_val1, y_val1 = X1[2900:3033], y1[2900:3033]
X_test1, y_test1 = X1[3033:], y1[3033:]
X_train1.shape, y_train1.shape, X_val1.shape, y_val1.shape, X_test1.shape, y_test1.shape
score = model_lstm.evaluate(X_test1, y_test1, verbose=0)

print('x_test / loss      : {:5.4f}'.format(score[0]))
print('x_test / mape       : {:5.4f}'.format(score[1]))
print('x_test / mae       : {:5.4f}'.format(score[2]))

train_predictions = model_lstm.predict(X_train1).flatten()
train_results = pd.DataFrame(data={'Predictions':train_predictions, 'Réalité':y_train1})
train_results
import matplotlib.pyplot as plt
plt.rcParams["figure.figsize"] = (20,15)

plt.plot(train_df['ds'][20:300],train_results['Predictions'][20:300],color='yellow',label ='Predictions')
plt.plot(train_df['ds'][20:300],train_results['Réalité'][20:300],label='Réalité')
plt.legend()
st.pyplot(plt)

val_predictions = model_lstm.predict(X_val1).flatten()
val_results = pd.DataFrame(data={'Val Predictions':val_predictions, 'Réalité':y_val1})
val_results
plt.plot(train_df['ds'][2900:3000],val_results['Val Predictions'][:100],label='Predictions')
plt.plot(train_df['ds'][2900:3000],val_results['Réalité'][:100],label='Réalité')
plt.legend()
st.pyplot(plt)

test_predictions = model_lstm.predict(X_test1).flatten()
test_results = pd.DataFrame(data={'Test Predictions':test_predictions, 'Réalité':y_test1})
test_results

plt.plot(train_df['ds'][3033:3188],test_results['Test Predictions'][:155],label='Predictions')
plt.plot(train_df['ds'][3033:3188],test_results['Réalité'][:155],label='Réalité')
plt.legend()
st.pyplot(plt)



# top-level filters 

# job_filter = st.selectbox("Select the Job", pd.unique(df['job']))


# # creating a single-element container.
# placeholder = st.empty()

# # dataframe filter 

# df = df[df['job']==job_filter]

# # near real-time / live feed simulation 

# for seconds in range(200):
# #while True: 
    
#     df['age_new'] = df['age'] * np.random.choice(range(1,5))
#     df['balance_new'] = df['balance'] * np.random.choice(range(1,5))

#     # creating KPIs 
#     avg_age = np.mean(df['age_new']) 

#     count_married = int(df[(df["marital"]=='married')]['marital'].count() + np.random.choice(range(1,30)))
    
#     balance = np.mean(df['balance_new'])

#     with placeholder.container():
#         # create three columns
#         kpi1, kpi2, kpi3 = st.columns(3)

#         # fill in those three columns with respective metrics or KPIs 
#         kpi1.metric(label="Age ⏳", value=round(avg_age), delta= round(avg_age) - 10)
#         kpi2.metric(label="Married Count 💍", value= int(count_married), delta= - 10 + count_married)
#         kpi3.metric(label="A/C Balance ＄", value= f"$ {round(balance,2)} ", delta= - round(balance/count_married) * 100)

#         # create two columns for charts 

#         fig_col1, fig_col2 = st.columns(2)
#         with fig_col1:
#             st.markdown("### First Chart")
#             fig = px.density_heatmap(data_frame=df, y = 'age_new', x = 'marital')
#             st.write(fig)
#         with fig_col2:
#             st.markdown("### Second Chart")
#             fig2 = px.histogram(data_frame = df, x = 'age_new')
#             st.write(fig2)
#         st.markdown("### Detailed Data View")
#         st.dataframe(df)
#         time.sleep(1)
#     #placeholder.empty()


