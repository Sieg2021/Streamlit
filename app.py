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

prophet_model = joblib.load('prophet.joblib')
st.write(prophet_model)
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


