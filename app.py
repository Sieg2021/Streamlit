import streamlit as st # web development
import numpy as np # np mean, np random 
import pandas as pd # read csv, df manipulation
import time # to simulate a real time data, time loop 
import plotly.express as px # interactive charts 
import plotly.graph_objects as go
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import confusion_matrix, roc_auc_score, roc_curve, classification_report, precision_recall_curve
from collections import deque
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import TimeSeriesSplit
from fonction_utile import *
from streamlit_option_menu import option_menu
import extra_streamlit_components as stx
import streamlit.components.v1 as html
from  PIL import Image

from st_aggrid import AgGrid
import io 

# read csv from a github repo
df = pd.read_csv("weatherAUS.csv")


st.set_page_config(
    page_title = 'Prévision météo',
    page_icon = '☁️',
    layout = 'wide'
)

# dashboard title

#st.title("Prévision météo")

with st.sidebar:
    choose = option_menu("Menu", ["À propos du projet", "Jeux de données","Data Viz","Classification", "Regression","Application"],
                         icons=['cloud-hail','pie-chart','bar-chart', 'diagram-2', 'graph-up','app-indicator'],
                         menu_icon="menu-up", default_index=0,
                         styles={
        "container": {"padding": "5!important", "background-color": "#262730"},
        "icon": {"color": "orange", "font-size": "25px"}, 
        "nav-link": {"font-size": "16px", "text-align": "left", "margin":"0px", "--hover-color": "#eee"},
        "nav-link-selected": {"background-color": "#02ab21"},
    }
    )

if choose == "À propos du projet":
    col1, col2 = st.columns( [0.8, 0.2])
    with col1:               # To display the header text using css style
        st.markdown(""" <style> .font {
        font-size:35px ; font-family: 'Cooper Black'; color: #FF9633;} 
        </style> """, unsafe_allow_html=True)
        st.markdown('<p class="font">Présentation du projet</p>', unsafe_allow_html=True)    
        
        st.image('meteo.png', width=600)
        st.write('Ce projet est basé sur un ensemble d’observations météorologiques journalières d’Australie. Ces observations ont été faites sur l’ensemble du territoire Australien, dans 49 villes différentes.')
        st.title('Objectifs')
        st.write('L’objectif de ce projet est de proposer des modèles capables de prédire efficacement les données météorologiques, à savoir s\'il pleuvra le lendemain ou quelle température il fera.')
        st.write('La température est exprimée en degré celsius. Dans notre jeu de données , les valeurs varient entre -7.2 et 40 °C.')
        
        st.title('Membres du projet')
        st.write('Saïd LATTI')
        st.write('Emma ROBERT')


if choose == "Jeux de données":
    col1, col2 = st.columns( [0.8, 0.2])
    with col1: 
        st.markdown(""" <style> .font {
        font-size:35px ; font-family: 'Cooper Black'; color: #FF9633;} 
        </style> """, unsafe_allow_html=True)
        st.markdown('<p class="font">Jeux de Donnée</p>', unsafe_allow_html=True) 
        chosen_id = stx.tab_bar(data=[
        stx.TabBarItemData(id="tab1", title="Présentation", description="Présentation et description des données"),
        stx.TabBarItemData(id="tab2", title="Préprocessing", description="Etapes de préparation des données ")])
        placeholder = st.container()

        if chosen_id == "tab1":
            st.empty()
            from data import page_data              # To display the header text using css style
            page_data()
        if chosen_id == "tab2":
            st.empty()
            from preprocessing import page_preprocessing    # To display the header text using css style
            page_preprocessing()



if choose == "Data Viz":
    st.empty()
    col1, col2, col3 = st.columns( [0.2, 0.5,0.3])
    with col2:               # To display the header text using css style
        st.markdown(""" <style> .font {
        font-size:35px ; font-family: 'Cooper Black'; color: #FF9633;} 
        </style> """, unsafe_allow_html=True)
        st.markdown('<p class="font">Visualisation des données</p>', unsafe_allow_html=True)    
        path="weatherAUS.csv"
        df = pd.read_csv(path)
        
        #with st.spinner(text='In progress'):
        #    time.sleep(25)
        
        st.write("Afin de mieux comprendre les données et leurs impacts, voici quelques visualisations :")
        fig, ax = plt.subplots(figsize=(1,1))

        #graph nb de villes
        st.bar_chart(df['Location'].value_counts())
        
        st.write('\n')

        #graph visualisation évoluation de la temp min max sur 10 ans
        df2= df
        df2['year'] = df2['Date'].apply(lambda date : date.split('-')[0]).astype(int)
        df2['month'] = df2['Date'].apply(lambda date : date.split('-')[1]).astype(int)
        df2['day'] = df2['Date'].apply(lambda date : date.split('-')[2]).astype(int)
        st.write("Visualisation de l'évolution de la température maximale et minimale")
        fig = plt.figure()
        sns.lineplot( x= 'year', y = 'MinTemp', label = 'température min', data = df2.groupby('year').mean())
        sns.lineplot( x= 'year', y = 'MaxTemp', label = 'température max', data = df2.groupby('year').mean())
        plt.title("Variation de températures à l'année")
        plt.xlabel('Année')
        plt.ylabel('Température (°C) ')
        st.pyplot(fig)
        st.write('\n')

        df2= df
        df2['year'] = df2['Date'].apply(lambda date : date.split('-')[0]).astype(int)
        df2['month'] = df2['Date'].apply(lambda date : date.split('-')[1]).astype(int)
        df2['day'] = df2['Date'].apply(lambda date : date.split('-')[2]).astype(int)
        st.write("Visualisation de l'évolution de la température maximale et minimale")
        fig = plt.figure()
        sns.lineplot( x= 'month', y = 'MinTemp', label = 'température min', data = df2.groupby('month').mean())
        sns.lineplot( x= 'month', y = 'MaxTemp', label = 'température max', data = df2.groupby('month').mean())
        plt.title("Variation de températures par mois")
        plt.xlabel('Mois')
        plt.ylabel('Température (°C) ')
        st.pyplot(fig)
        df = pd.read_csv(path)

        #répartion Rain
        st.write('\n')
        st.write("Visualisation de la réparation des variables RainToday et RainTomorrow")
        fig3 =plt.figure(figsize=(20, 8))
        plt.subplot(121)
        plt.title(label='Modalités de la variable RainToday')
        sns.countplot(x="RainToday", data=df)
        plt.subplot(122)
        plt.title(label='Modalités de la variable RainTomorrow')
        sns.countplot(x="RainTomorrow", data=df);
        st.pyplot(fig3)
        st.write('\n')

        #visualisation multivariable
        st.write("Visualisation de la réparation de chacune des variables dans le jeu de données")
        numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
        num_variable = df.select_dtypes(include=numerics)
        fig = plt.figure(figsize=(20, 40))
        plt.subplots_adjust(left=0.12,
                            bottom=0.12,
                            right=0.9,
                            top=0.9,
                            wspace=0.4,
                            hspace=0.4)
        for i, column in enumerate(num_variable, 1):
            plt.subplot(8,2,i)
            sns.histplot(df[column]).set(title=column);
        st.pyplot(fig)

if choose == "Regression":
    st.markdown(""" <style> .font {
        font-size:35px ; font-family: 'Cooper Black'; color: #FF9633;} 
        </style> """, unsafe_allow_html=True)
    st.markdown('<p class="font">Regression</p>', unsafe_allow_html=True)  
    chosen_id = stx.tab_bar(data=[
    stx.TabBarItemData(id="tab1", title="DummyRegressor", description="Base Line"),
    stx.TabBarItemData(id="tab2", title="Arima", description="Algorithme de moyenne mobile intégrée autorégressive "),
    stx.TabBarItemData(id="tab3", title="Prophet", description="Série temporelle par Facebook"),
    stx.TabBarItemData(id="tab4", title="LSTM Model", description="Reseau de neurones récurrent")])
    placeholder = st.container()
    if chosen_id == "tab1":
        st.write("<h3>DummyRegressor</h3>", unsafe_allow_html=True)
        st.write("DummyRegressor est un modèle de régression qui fait des prédictions en utilisant des règles simples.")
        dummy_regressor_model = joblib.load('mon_dummy_regr.joblib')
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
        #st.write(test_result)

        import matplotlib.pyplot as plt_dummy
        plt_dummy.rcParams["figure.figsize"] = (20,15)

        plt_dummy.plot(test_results['Test Predictions'][20:300],color='red',label ='Predictions')
        plt_dummy.plot(test_results['Réalité'][20:300],label='Réalité')
        plt_dummy.legend()
        plt_dummy.show()
        st.pyplot(plt_dummy)
        plt_dummy.figure().clear()

        st.write("Curieusement, les résultats sont assez correct avec un pourcentage d’erreur absolu moyen de 8,1% comme le montre le graphique")

    elif chosen_id == "tab3":
        st.write("<h3>Prophet</h3>", unsafe_allow_html=True)
        st.write("Pour cette approche, nous avons choisi de nous intéresser au modèle dévelloppé par Facebook : Facebook prophet car il est adapté aux séries temporelles et facile à paramétrer.")
        st.write("Voici la courbe représentant la réalité et les données prédites par ce modèle :")
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

        train_df = pd.DataFrame()
        train_df['ds'] = pd.to_datetime(df_prepare['Date'])
        train_df['y']=df_prepare['Temp9am']
        train_df.head(2)
        train_df2 = train_df[train_df['ds']<'2016-06-24']
        train_df2.tail(2)

        prophet_model = joblib.load('prophet.joblib')

        import matplotlib.pyplot as plt_prophet

        future = prophet_model.make_future_dataframe(periods=365)
        future.tail(2)

        forecast = prophet_model.predict(future)
        forecast.tail()

        plt_prophet.rcParams["figure.figsize"] = (20,15)
        fig, ax = plt_prophet.subplots()

        ax.plot(train_df['ds'],train_df['y'],label = "réalité")
        ax.plot(forecast['ds'], forecast['yhat'],label = "prédiction")
        plt_prophet.legend()
        plt_prophet.show()
        st.pyplot(plt_prophet)
        plt_prophet.figure().clear()

        st.write("On obtient de très bons résultats et qui sont assez proches de la réalité")

    elif chosen_id == "tab2":
        st.write("Etant donné que nos données contiennent une forte temporalité, dans cette approche nous allons utliser un modèle ARIMA.")
        st.write("Voici la courbe représentant la réalité et les données prédites par ce modèle :")

        df = chargement_data()
        df_prepare = chargement_ville('Darwin',df)
        df_ts = df_prepare[['Date','Temp9am']].copy()
        df_ts['Date'] = pd.to_datetime(df_ts['Date'])
        df_ts.reset_index(drop=True, inplace=True)
        df_ts.set_index('Date', inplace=True)
        y_train = df_ts[df_ts.index<'2016-06-24'] #train_df[train_df['ds']<'2016-06-24']
        y_test = df_ts[df_ts.index>='2016-06-24']
        
        arima = joblib.load('arima.joblib')
        #st.write(arima)
        from pylab import rcParams
        rcParams['figure.figsize'] = 18, 8
        pred = arima.get_prediction(start=pd.to_datetime('2016-06-24'), dynamic=False)
        pred_ci = pred.conf_int()
        train_results = pd.DataFrame(data={'Predictions':pred.predicted_mean})
        train_results = train_results.merge(y_test, left_index=True, right_index=True)
        ax = plt.plot(y_test,label='Observation')
        plt.plot(pred.predicted_mean, label='Prediction', alpha=.7)
        plt.fill_between(pred_ci.index,
                        pred_ci.iloc[:, 0],
                        pred_ci.iloc[:, 1], color='k', alpha=.2)
        plt.xlabel('Date')
        plt.ylabel('Temperature')
        plt.legend()
        st.pyplot(plt)
        plt.figure().clear()

        st.write("On obtient de très bons résultats et qui sont assez proches de la réalité, malgré un léger décalage de quelques degrés")

    elif chosen_id == "tab4":
        st.write("<h3>LSTM Model</h3>", unsafe_allow_html=True)

        from keras.models import load_model
        model_lstm = load_model('model_lstm.h5')

        st.write("Dans cette approche, le modèle choisi est un réseau de neuronnes adapté aux séries temporelles : le modèle LSTM")
        st.write("Les résultats obtenus sont bons avec le pourcentage d’erreur absolue moyenne  (MAPE) de 4.6%. Un graphe de comparaison entre les valeurs réelles et les valeurs prédites permettent de se rendre compte visuellement de ces performances:")

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
        #X1.shape, y1.shape
        X_train1, y_train1 = X1[:2900], y1[:2900]
        X_val1, y_val1 = X1[2900:3033], y1[2900:3033]
        X_test1, y_test1 = X1[3033:], y1[3033:]
        #X_train1.shape, y_train1.shape, X_val1.shape, y_val1.shape, X_test1.shape, y_test1.shape
        score = model_lstm.evaluate(X_test1, y_test1, verbose=0)

        print('x_test / loss      : {:5.4f}'.format(score[0]))
        print('x_test / mape       : {:5.4f}'.format(score[1]))
        print('x_test / mae       : {:5.4f}'.format(score[2]))

        train_predictions = model_lstm.predict(X_train1).flatten()
        train_results = pd.DataFrame(data={'Predictions':train_predictions, 'Réalité':y_train1})
        #train_results
        import matplotlib.pyplot as plt_lstm_1
        plt_lstm_1.rcParams["figure.figsize"] = (20,15)

        plt_lstm_1.plot(train_df['ds'][20:300],train_results['Predictions'][20:300],color='yellow',label ='Predictions')
        plt_lstm_1.plot(train_df['ds'][20:300],train_results['Réalité'][20:300],label='Réalité')
        plt_lstm_1.legend()
        #st.pyplot(plt_lstm_1)
        plt_lstm_1.figure().clear()

        val_predictions = model_lstm.predict(X_val1).flatten()
        val_results = pd.DataFrame(data={'Val Predictions':val_predictions, 'Réalité':y_val1})
        #val_results

        import matplotlib.pyplot as plt_lstm_2

        plt_lstm_2.plot(train_df['ds'][2900:3000],val_results['Val Predictions'][:100],label='Predictions')
        plt_lstm_2.plot(train_df['ds'][2900:3000],val_results['Réalité'][:100],label='Réalité')
        plt_lstm_2.legend()
        #st.pyplot(plt_lstm_2)
        plt_lstm_2.figure().clear()


        test_predictions = model_lstm.predict(X_test1).flatten()
        test_results = pd.DataFrame(data={'Test Predictions':test_predictions, 'Réalité':y_test1})
        test_results

        import matplotlib.pyplot as plt
        plt.plot(train_df['ds'][3033:3188],test_results['Test Predictions'][:155],label='Predictions')
        plt.plot(train_df['ds'][3033:3188],test_results['Réalité'][:155],label='Réalité')
        plt.legend()
        st.pyplot(plt)
        plt.figure().clear()




    else:
        placeholder = st.empty()



if choose == "Classification":
    st.markdown(""" <style> .font {
        font-size:35px ; font-family: 'Cooper Black'; color: #FF9633;} 
        </style> """, unsafe_allow_html=True)
    st.markdown('<p class="font">Classification</p>', unsafe_allow_html=True)  
    df = chargement_data()
    df_prepare = chargement_ville('Darwin',df)
    cat_columns, num_columns = separation_colonnes(df_prepare)
    df_prepare = encodage(df_prepare,cat_columns)
    df_prepare.head()
    chosen_id = stx.tab_bar(data=[
    stx.TabBarItemData(id="tab1", title="DummyClassifier", description="BaseLine"),
    stx.TabBarItemData(id="tab2", title="Première approche", description="Modèle Classique"),
    stx.TabBarItemData(id="tab3", title="Optimisation", description="Amélioration du modèle"),
    stx.TabBarItemData(id="tab4", title="Approche par ville", description="Approche par ville"),
    stx.TabBarItemData(id="tab5", title="MLPClassifier", description="Perceptron multicouche") ])
    placeholder = st.container()
    if chosen_id == "tab1":
        st.write("<h3>DummyClassifier</h3>", unsafe_allow_html=True)
        
        dummy_clf = joblib.load('dummy_clf.joblib')

        X = df_prepare.drop(columns=["RainTomorrow"], axis=1)
        y = df_prepare["RainTomorrow"]

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, shuffle = False)

        sc = MinMaxScaler()
        X_train[num_columns] = sc.fit_transform(X_train[num_columns])
        X_test[num_columns] = sc.transform(X_test[num_columns])

        y_predict = dummy_clf.predict(X_test)
        y_pred_prob=dummy_clf.predict_proba(X_test)[:,1]

        test_results = pd.DataFrame(data={'Test Predictions':y_predict, 'Réalité':y_test})
        test_result= test_results.reset_index(drop=True)
        from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
        print("*"*10, "Classification Report", "*"*10)
        print("-"*30)
        print("Dummy classifier: \n ", classification_report(y_test, y_predict))
        print("-"*30)

        rap1 = classification_report(y_test, y_predict)
        st.write("Le premier modèle testé est le Dummy Classifier. Voici les résulats :")
        st.write("Rapport de classification")
        st.text(rap1)

        st.write("\n")
        st.write("Courbe ROC du modèle")

        fpr_dumm, tpr_dumm, thresholds  = roc_curve(y_test, y_pred_prob)

        plt.rcParams['font.size'] = 12
        plt.rcParams["figure.figsize"] = (20,15)
        plt.plot(fpr_dumm, tpr_dumm, color = 'blue', marker = 'o', markerfacecolor = 'red', markersize = 1, label = "Dummy Classifier")


        plt.plot([0,1], [0,1], 'k--' )
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.0])
        plt.xlabel('False Positive Rate(1-Specificity)')
        plt.ylabel('True Positive Rate(Sensitivity)')
        plt.legend()
        plt.grid(True)
        plt.show()
        st.pyplot(plt)
        plt.figure().clear()

        st.write("Les résultats de ce modèle, comme on pouvait s’y attendre, sont totalement erronés pour la classe minoritaire et la courbe ROC montre l'absence de pertinence dans la classification proposée.")

    elif chosen_id == "tab2":
        #Séparation entre feature et cible
        X = df_prepare.drop(columns=["RainTomorrow"], axis=1)
        y = df_prepare["RainTomorrow"]

        #séparation du jeu de donnée en train et test, on choisi 0.1 car cela représente 1 an de donnée ( 10 ans de données *0.1 = 1 an)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, shuffle = False)

        #Normalisation
        sc = MinMaxScaler()
        X_train[num_columns] = sc.fit_transform(X_train[num_columns])
        X_test[num_columns] = sc.transform(X_test[num_columns])
       
        dtree = joblib.load('dtree_clf.joblib')

        y_dtree = dtree.predict(X_test)
        y_dtree_prob=dtree.predict_proba(X_test)[:,1]


        lr = joblib.load('lr_clf.joblib')
        y_lr = lr.predict(X_test)
        y_lr_prob=lr.predict_proba(X_test)[:,1]


        rf = joblib.load('rf_clf.joblib')
        y_rf = rf.predict(X_test)
        y_rf_prob=rf.predict_proba(X_test)[:,1]


        svm = joblib.load('svm_clf.joblib')
        y_svm = svm.predict(X_test) 
        y_svm_prob=svm.predict_proba(X_test)[:,1]

        print("*"*10, "Classification Report", "*"*10)

        print("-"*30)
        print("Logistic Regression: ", classification_report(y_test, y_lr))
        print("-"*30)


        print("-"*30)
        print("Decision Tree: ", classification_report(y_test, y_dtree))
        print("-"*30)


        print("-"*30)
        print("Random Forest: ", classification_report(y_test, y_rf))
        print("-"*30)
        print("-"*30)

        print("-"*30)
        print("SVM : ", classification_report(y_test,y_svm))
        print("-"*30)
        print("-"*30)

        rap2 = classification_report(y_test, y_lr)
        st.write("Les modèles testés lors de cette approche sont les modèles classiques(Régression Logistique, Arbre de décision, Fôret aléatoire et SVM).\
                 Voici les résulats :")
        st.write("Rapports de classification")
        st.write('Regression Logistique :')
        st.text(rap2)
        
        st.write("\n")


        rap6= classification_report(y_test, y_dtree)
        st.write('Arbre de Décision :')
        st.text(rap6)
        
        st.write("\n")

        rap7 = classification_report(y_test, y_rf)
        st.write('Forêt aléatoire :')
        st.text(rap7)
        
        st.write("\n")

        rap8 = classification_report(y_test, y_svm)
        st.write('SVM :')
        st.text(rap8)
        
        st.write("\n")
        st.write("Courbe ROC des modèles")

        fpr_lr, tpr_lr, thresholds  = roc_curve(y_test, y_lr_prob)
        fpr_rf, tpr_rf, thresholds  = roc_curve(y_test, y_rf_prob)
        fpr_dtree, tpr_dtree, thresholds  = roc_curve(y_test, y_dtree_prob)
        fpr_svm, tpr_svm, thresholds  = roc_curve(y_test, y_svm_prob)

        plt.rcParams['font.size'] = 12
        plt.plot(fpr_lr, tpr_lr, color = 'blue', marker = 'o', markerfacecolor = 'red', markersize = 1, label = "Regression logistique")
        plt.plot(fpr_rf, tpr_rf, color = 'red', marker = 'o', markerfacecolor = 'red', markersize = 1, label = "Forêt aléatoire")
        plt.plot(fpr_dtree, tpr_dtree, color = 'yellow', marker = 'o', markerfacecolor = 'red', markersize = 1, label = "Arbre de décision")
        plt.plot(fpr_svm, tpr_svm, color = 'green', marker = 'o', markerfacecolor = 'red', markersize = 1, label = "SVM")


        plt.plot([0,1], [0,1], 'k--' )
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.0])
        #plt.title('ROC Curve for rain prediction CLassifier')
        plt.xlabel('False Positive Rate(1-Specificity)')
        plt.ylabel('True Positive Rate(Sensitivity)')
        plt.legend()
        plt.grid(True)
        plt.show()
        st.pyplot(plt)
        plt.figure().clear()

        st.write("On constate que 2 modèles sont assez proches. Afin de les distinguer, on compare la précision de ces modèles : le meilleur modèle est le Random Forest (accuracy = 0.85).")


    elif chosen_id == "tab3":
        st.write("RandomForestClassifier Optimisé") 
        df = chargement_data()
        df_prepare = chargement_ville('Darwin',df)
        df_prepare['Date'] = pd.to_datetime(df_prepare['Date'])
        cat_columns, num_columns = separation_colonnes(df_prepare)
        df_prepare = encodage(df_prepare,cat_columns)
        df_prepare.head()
        df_prepare.sort_values(['Location', 'Date'], ascending = [True, True])
        df1 = pd.DataFrame()
        for ville in pd.unique(df_prepare['Location']):
            df_ville = df_prepare[df_prepare['Location'] == ville]
            df_ville['Day - 1']=df_ville.RainToday.shift(1)
            df_ville['Day - 2']=df_ville.RainToday.shift(2)
            df_ville['Day - 3']=df_ville.RainToday.shift(3)
            df1 = pd.concat([df1, df_ville])
        df_prepare = df1.copy()
        df_prepare['Day - 1']=df_prepare['Day - 1'].fillna(df_prepare['Day - 1'].mode()[0])
        df_prepare['Day - 2']=df_prepare['Day - 2'].fillna(df_prepare['Day - 2'].mode()[0])
        df_prepare['Day - 3']=df_prepare['Day - 3'].fillna(df_prepare['Day - 3'].mode()[0])
        df_prepare['Date'] = pd.to_numeric(pd.to_datetime(df_prepare['Date']))
        sc = MinMaxScaler()
        df_prepare[num_columns] = sc.fit_transform(df_prepare[num_columns])

        tscv = TimeSeriesSplit(n_splits=2)

            #X = df_prepare.to_numpy()
        X = df_prepare.drop(columns=["RainTomorrow"], axis=1).to_numpy()
            #X = df_prepare.drop(columns=["RainTomorrow"], axis=1)
        y = df_prepare["RainTomorrow"].to_numpy()

        for train_index, test_index in tscv.split(X):
            print("TRAIN:", train_index, "TEST:", test_index)
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]


        X_train = pd.DataFrame(X_train)
        y_train = pd.DataFrame(y_train)  

        clf_rf_opti = joblib.load('clf_rf_opti.joblib')

        y_pred = clf_rf_opti.predict(X_test)
        print("-"*30)
        print("Random forest optimisé : \n ", classification_report(y_test, y_pred))
        print("-"*30)

        
        rap3 = classification_report(y_test, y_pred)
        st.write("Dans cette approche, nous avons cherché à améliorer l'approche précédente en prennant en compte les données métérologiques des jours précédents ainsi qu'en cherchant les meilleurs paramètres du modèle .\
                 Voici les résulats :")
        st.write("Rapport de classification :")
        st.text(rap3)
                
        st.write("\n")
        st.write("Courbe ROC :")


        y_rf_opt = clf_rf_opti.predict(X_test)
        y_rf_opt_prob=clf_rf_opti.predict_proba(X_test)[:,1]

        fpr_rf_opt, tpr_rf_opt, thresholds  = roc_curve(y_test, y_rf_opt_prob)


        plt.rcParams['font.size'] = 12
        plt.plot(fpr_rf_opt, tpr_rf_opt, color = 'green', marker = 'o', markerfacecolor = 'red', markersize = 1, label = "Random forest optimisé")


        plt.plot([0,1], [0,1], 'k--' )
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.0])
        plt.xlabel('False Positive Rate(1-Specificity)')
        plt.ylabel('True Positive Rate(Sensitivity)')
        plt.legend()
        plt.grid(True)
        plt.show()
        st.pyplot(plt)
        plt.figure().clear()

        st.write("On observe une amélioration générale des résultats (en comparaison avec le modèle non optimisé) notamment au niveau de la classe 1.")
    
    elif chosen_id == "tab4":
        st.write("RandomForestClassifier")
        st.write("Dans cette approche, nous avons cherché à améliorer l'approche précédente en prennant en compte les données géographiques.\
                 Voici les résulats :")


        df = chargement_data()
        df_prepare = chargement_ville('Darwin',df)
        cat_columns, num_columns = separation_colonnes(df_prepare)
        df_prepare = encodage(df_prepare,cat_columns)
        df_prepare.head()

            #Séparation entre feature et cible
        X = df_prepare.drop(columns=["RainTomorrow"], axis=1)
        y = df_prepare["RainTomorrow"]

            #séparation du jeu de donnée en train et test, on choisi 0.1 car cela représente 1 an de donnée ( 10 ans de données *0.1 = 1 an)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, shuffle = False)

            #Normalisation
        sc = MinMaxScaler()
        X_train[num_columns] = sc.fit_transform(X_train[num_columns])
        X_test[num_columns] = sc.transform(X_test[num_columns])

       
        dtree_ville = joblib.load('dtree_ville_clf.joblib')

        y_dtree_ville = dtree_ville.predict(X_test)
        y_dtree_ville_prob=dtree_ville.predict_proba(X_test)[:,1]

        lr_ville = joblib.load('lr_ville_clf.joblib')

        y_lr_ville = lr_ville.predict(X_test)
        y_lr_ville_prob=lr_ville.predict_proba(X_test)[:,1]

        rf_ville = joblib.load('rf_ville_clf.joblib')

        y_rf_ville = rf_ville.predict(X_test)
        y_rf_ville_prob=rf_ville.predict_proba(X_test)[:,1]

        svm_ville = joblib.load('svm_ville_clf.joblib')

        y_svm_ville = svm_ville.predict(X_test) 
        y_svm_ville_prob=svm_ville.predict_proba(X_test)[:,1]

        rap9 = classification_report(y_test, y_lr_ville)
        st.write("Les modèles testés lors de cette approche sont les modèles classiques(Régression Logistique, Arbre de décision, Fôret aléatoire et SVM).\
                 Voici les résulats :")
        st.write("Rapports de classification")
        st.write('Regression Logistique :')
        st.text(rap9)
        
        st.write("\n")


        rap10 = classification_report(y_test, y_dtree_ville)
        st.write('Arbre de Décision :')
        st.text(rap10)
        
        st.write("\n")

        rap11 = classification_report(y_test, y_rf_ville)
        st.write('Forêt aléatoire :')
        st.text(rap11)
        
        st.write("\n")

        rap12 = classification_report(y_test, y_svm_ville)
        st.write('SVM :')
        st.text(rap12)
        
        st.write("\n")
        st.write("Courbe ROC des modèles")


        fpr_lr_ville, tpr_lr_ville, thresholds  = roc_curve(y_test, y_lr_ville_prob)
        fpr_rf_ville, tpr_rf_ville, thresholds  = roc_curve(y_test, y_rf_ville_prob)
        fpr_dtree_ville, tpr_dtree_ville, thresholds  = roc_curve(y_test, y_dtree_ville_prob)
        fpr_svm_ville, tpr_svm_ville, thresholds  = roc_curve(y_test, y_svm_ville_prob)

        plt.rcParams['font.size'] = 12
        plt.rcParams["figure.figsize"] = (20,15)
        plt.plot(fpr_lr_ville, tpr_lr_ville, color = 'blue', marker = 'o', markerfacecolor = 'red', markersize = 1, label = "Regression logistique par ville")
        plt.plot(fpr_rf_ville, tpr_rf_ville, color = 'red', marker = 'o', markerfacecolor = 'red', markersize = 1, label = "Forêt aléatoire par ville")
        plt.plot(fpr_dtree_ville, tpr_dtree_ville, color = 'yellow', marker = 'o', markerfacecolor = 'red', markersize = 1, label = "Arbre de décision par ville")
        plt.plot(fpr_svm_ville, tpr_svm_ville, color = 'green', marker = 'o', markerfacecolor = 'red', markersize = 1, label = "SVM par ville")


        plt.plot([0,1], [0,1], 'k--' )
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.0])
        #plt.title('ROC Curve for rain prediction CLassifier')
        plt.xlabel('False Positive Rate(1-Specificity)')
        plt.ylabel('True Positive Rate(Sensitivity)')
        plt.legend()
        plt.grid(True)
        plt.show()
        st.pyplot(plt)
        plt.figure().clear()

        st.write("Le résultat montre une amélioration par rapport à l’approche classique (accuracy = 0.84) et un temps d’exécution court. La classe 1 ( pour rappel : classe 1 = il pleuvra le lendemain)  est aussi mieux prédite.")

    
    elif chosen_id == "tab5":
        st.write("MLPClassifier")
        df = chargement_data()
        df_prepare = chargement_ville('Darwin',df)
        cat_columns, num_columns = separation_colonnes(df_prepare)
        df_prepare = encodage(df_prepare,cat_columns)
        df_prepare.head()
        #Séparation entre feature et cible
        X = df_prepare.drop(columns=["RainTomorrow"], axis=1)
        y = df_prepare["RainTomorrow"]


        #Normalisation
        sc = MinMaxScaler()
        X[num_columns] = sc.fit_transform(X[num_columns])
        tscv = TimeSeriesSplit(n_splits=2)

        #X = df_prepare.to_numpy()
        X = df_prepare.drop(columns=["RainTomorrow"], axis=1).to_numpy()
        #X = df_prepare.drop(columns=["RainTomorrow"], axis=1)
        y = df_prepare["RainTomorrow"].to_numpy()

        for train_index, test_index in tscv.split(X):
            print("TRAIN:", train_index, "TEST:", test_index)
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]

        clf_pipe = joblib.load('mlp_clf.joblib')
        y_pred = clf_pipe.predict(X_test)
        y_pred_prob=clf_pipe.predict_proba(X_test)[:,1]


        print("*"*10, "Classification Report", "*"*10)

        print("-"*30)
        print("MLPClassifier: \n", classification_report(y_test, y_pred))
        print("-"*30)

        fpr_mlpc, tpr_mlpc, thresholds  = roc_curve(y_test, y_pred_prob)

        clf_pipe = joblib.load('mlp_clf_opti.joblib')
        y_pred = clf_pipe.predict(X_test)
        y_pred_prob=clf_pipe.predict_proba(X_test)[:,1]


        print("*"*10, "Classification Report", "*"*10)

        print("-"*30)
        print("MLPClassifier: \n", classification_report(y_test, y_pred))
        print("-"*30)

        rap5 = classification_report(y_test, y_pred)
        st.write("Dans cette approche, nous avons cherché à améliorer l'approche précédente en prennant utilisant un réseau de neurones (MLP) .\
                 Voici les résulats :")
        st.write("Rapport de classification :")
        st.text(rap5)

        fpr_mlpc_opti, tpr_mlpc_opti, thresholds  = roc_curve(y_test, y_pred_prob)

        plt.rcParams['font.size'] = 12
        plt.rcParams["figure.figsize"] = (20,15)
        plt.plot(fpr_mlpc, tpr_mlpc, color = 'blue', marker = 'o', markerfacecolor = 'red', markersize = 1, label = "MLPC de base")
        plt.plot(fpr_mlpc_opti, tpr_mlpc_opti, color = 'red', marker = 'o', markerfacecolor = 'red', markersize = 1, label = "MLPC optimisé")


        plt.plot([0,1], [0,1], 'k--' )
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.0])
        #plt.title('ROC Curve for rain prediction CLassifier')
        plt.xlabel('False Positive Rate(1-Specificity)')
        plt.ylabel('True Positive Rate(Sensitivity)')
        plt.legend()
        plt.grid(True)
        plt.show()
        st.pyplot(plt)
        plt.figure().clear()

        fig = go.Figure()
        fig.add_shape(
            type='line', line=dict(dash='dash'),
            x0=0, x1=1, y0=0, y1=1
        )
        fig.add_trace(go.Scatter(x=fpr_mlpc_opti, y=tpr_mlpc_opti, name="MLPC Opti", mode='lines'))
        fig.add_trace(go.Scatter(x=fpr_mlpc, y=tpr_mlpc, name="MLPC de base", mode='lines'))
        fig.update_layout(
            xaxis_title='False Positive Rate',
            yaxis_title='True Positive Rate',
            yaxis=dict(scaleanchor="x", scaleratio=1),
            xaxis=dict(constrain='domain'),
            width=1400, height=800
        )
        # fig.show()
        st.write(fig)
        st.write("On observe que les résultats décrivent un bon modèle avec un bonne précision/recall pour la classe 0 ainsi que de meilleurs résultats que précédemment pour classe 1 au niveau du recall")
    
    else:
        placeholder = st.empty()


if choose == "Application":
    chosen_id = stx.tab_bar(data=[
    stx.TabBarItemData(id="tab1", title="Regression", description="Prédiction de la température"),
    stx.TabBarItemData(id="tab2", title="Classification", description="Pleuvra t-il demain?")])
    placeholder = st.container()
    if chosen_id == "tab1":
        st.write("Regression")
        select_model = st.container()
        with select_model:
            select_model = st.selectbox('Selectionner un modèle',[" Choix du modèle", "ARIMA", "LSTM", "Prophet"])
        
        min= pd.to_datetime(df['Date']).min()+pd.DateOffset(days=375)
        max= pd.to_datetime(df['Date']).max()-pd.DateOffset(days=365)
        choix_periode= st.date_input('Date input', min_value = min, max_value = max, value = min)

        

        if select_model == "LSTM":
            
            from keras.models import load_model
            model_lstm = load_model('model_lstm.h5')
            #st.write(model_lstm)
            df = chargement_data()

            df_prepare = chargement_ville('Darwin',df)

            df_prepare.head()

            train_df = pd.DataFrame()
            train_df['ds'] = pd.to_datetime(df_prepare['Date'])
            train_df['y']=df_prepare['Temp9am']
            #train_df
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
            #X1.shape, y1.shape
            X_train1, y_train1 = X1[:2900], y1[:2900]
            X_val1, y_val1 = X1[2900:3033], y1[2900:3033]
            X_test1, y_test1 = X1[3033:], y1[3033:]
            #X_train1.shape, y_train1.shape, X_val1.shape, y_val1.shape, X_test1.shape, y_test1.shape
            test_predictions = model_lstm.predict(X1).flatten()
            test_results = pd.DataFrame(data={'ds':train_df[5:]['ds'],'Test Predictions':test_predictions, 'Réalité':y1})
            #test_results.shape
            df = test_results
            train_df.reset_index(drop=True)
            
            df['ds'] = train_df['ds']
            df = df.reset_index(drop=True)
            #df['ds'] = train_df['ds'][3033:3188]
            df = df[df['ds']>pd.to_datetime(choix_periode)]
            #df
            #df.shape
            # for i in range(df.shape[0]):
                # df['ds'][i] =train_df['ds'][3033+i]
            header = st.container()
            plot_spot = st.empty()
            # select_param = st.container()
            # with select_param:
            #     param_lst = list(df.columns)
            #     param_lst.remove('ds')
            #     select_param = st.selectbox('Select a Weather Parameter',   param_lst)
            #function to make chart
            def make_chart(df, y_col1,y_col2, ymin, ymax):
                fig = go.Figure(layout_yaxis_range=[ymin, ymax])
                fig.add_trace(go.Scatter(x=df['ds'], y=df[y_col1],  name=y_col1))
                fig.add_trace(go.Scatter(x=df['ds'], y=df[y_col2],  name=y_col2))
                
                fig.update_layout(width=900, height=570, xaxis_title='time',
                yaxis_title='Température')
                st.write(fig)

            #func call
            n = len(df)
        
            ymax = df['Test Predictions'].max()+5
            ymin = df['Test Predictions'].min()-5
            for i in range(0, n-30, 1):
                df_tmp = df.iloc[i:i+30, :]
                with plot_spot:
                    make_chart(df_tmp, 'Test Predictions','Réalité', ymin, ymax)
                time.sleep(0.5)
        if select_model == "ARIMA":
            
            arima = joblib.load("arima.joblib")
            df = chargement_data()

            df_prepare = chargement_ville('Darwin',df)

            df_ts = df_prepare[['Date','Temp9am']].copy()
            df_ts['Date'] = pd.to_datetime(df_ts['Date'])
            df_ts.reset_index(drop=True, inplace=True)
            df_ts.set_index('Date', inplace=True)
            #df['ds'] = train_df['ds'][3033:3188]
            pred = arima.get_prediction(start=pd.to_datetime(choix_periode), dynamic=False)
            #df_ts
            df = df_ts[df_ts.index>=pd.to_datetime(choix_periode)] 
            #pred.predicted_mean
            #df
            test_results = pd.DataFrame(data={'ds':df.index,'Test Predictions':pred.predicted_mean, 'Réalité':df['Temp9am']})
            # for i in range(df.shape[0]):
                # df['ds'][i] =train_df['ds'][3033+i]
            header = st.container()
            plot_spot = st.empty()
            select_param = st.container()
            df = test_results
            #function to make chart
            def make_chart(df, y_col1,y_col2, ymin, ymax):
                fig = go.Figure(layout_yaxis_range=[ymin, ymax])
                fig.add_trace(go.Scatter(x=df['ds'], y=df[y_col1],  name=y_col1))
                fig.add_trace(go.Scatter(x=df['ds'], y=df[y_col2],  name=y_col2))
                
                fig.update_layout(width=900, height=570, xaxis_title='time',
                yaxis_title='Température')
                st.write(fig)

            #func call
            n = len(df)
            
            ymax = df['Test Predictions'].max()+5
            ymin = df['Test Predictions'].min()-5
            for i in range(0, n-30, 1):
                df_tmp = df.iloc[i:i+30, :]
                with plot_spot:
                    make_chart(df_tmp, 'Test Predictions','Réalité', ymin, ymax)
                time.sleep(0.5)
        if select_model == "Prophet":
            
            prophet = joblib.load("prophet.joblib")
            df = chargement_data()

            df_prepare = chargement_ville('Darwin',df)

            train_df = pd.DataFrame()
            train_df['ds'] = pd.to_datetime(df_prepare['Date'])
            train_df['y']=df_prepare['Temp9am']
            train_df.reset_index(drop=True, inplace=True)
            #train_df
            train_df2 = train_df[train_df['ds']<'2016-06-24']
            train_df2.tail(2)            
            future = prophet.make_future_dataframe(periods=365)
            future.tail(2)
            forecast = prophet.predict(future)
            #st.write(forecast.tail(2))
            #forecast 
            
            test_results = pd.DataFrame(data={'ds':train_df['ds'],'Test Predictions':forecast['yhat'], 'Réalité':train_df['y']})
            #test_results
            # for i in range(df.shape[0]):
                # df['ds'][i] =train_df['ds'][3033+i]
            header = st.container()
            plot_spot = st.empty()
            select_param = st.container()
            df = test_results
            df = test_results[test_results['ds']>=pd.to_datetime(choix_periode)] 
            #function to make chart
            def make_chart(df, y_col1,y_col2, ymin, ymax):
                fig = go.Figure(layout_yaxis_range=[ymin, ymax])
                fig.add_trace(go.Scatter(x=df['ds'], y=df[y_col1],  name=y_col1))
                fig.add_trace(go.Scatter(x=df['ds'], y=df[y_col2],  name=y_col2))
                
                fig.update_layout(width=900, height=570, xaxis_title='time',
                yaxis_title='Température')
                st.write(fig)
           

            #func call
            n = len(df)
            #st.write('vouvou n')
            #st.write(n)
            ymax = df['Test Predictions'].max()+5
            ymin = df['Test Predictions'].min()-5
            for i in range(0, n-30, 1):
                df_tmp = df.iloc[i:i+30, :]
                with plot_spot:
                    make_chart(df_tmp, 'Test Predictions','Réalité', ymin, ymax)
                time.sleep(0.5)
    if chosen_id == "tab2":
        st.write("Classification")
        #choix_ville = st.selectbox( label = "choix de la ville", options = df['Location'].unique(), index= 0)
        min= pd.to_datetime(df['Date']).min()+pd.DateOffset(days=375)
        max= pd.to_datetime(df['Date']).max()-pd.DateOffset(days=365)
        choix_periode= st.date_input('Date input', min_value = min, max_value = max, value = min)
        mlp_clf_pipe_opti = joblib.load("mlp_clf_opti.joblib")
        df = chargement_data()
        df_prepare = chargement_ville('Darwin',df)
        cat_columns, num_columns = separation_colonnes(df_prepare)
        test = encodage(df_prepare[pd.to_datetime(df_prepare['Date']) == pd.to_datetime(choix_periode)],cat_columns)
        test = test.drop(columns=["RainTomorrow"], axis=1)
        sortie = mlp_clf_pipe_opti.predict_proba(test)
        sortie2 = mlp_clf_pipe_opti.predict(test)  
        if sortie[:,1]>0.5:
            # st.image('pluie.png', width=600)
            # st.write("Température max:",test["MaxTemp"].values[0])
            # st.write("Température min:",test["MinTemp"].values[0])
            # st.write("Probabilité de pluie:",sortie[:,1])
            # col1, col4, col2, col3 = st.columns(4)

            # with col1:
            #     st.image('pluie.png', width=200)

            # with col4:
            #     st.header("Ville")
            #     st.write(choix_ville)

            # with col2:
            #     st.write("Température max:",test["MaxTemp"].values[0])
            #     st.write("Température min:",test["MinTemp"].values[0])

            # with col3:
            #     proba = sortie[:,1]
            #     st.write(f"Probabilité de pluie: {round(proba[0]*100, 2)}")
            temp_max_percent = 1.2
            temp_min_percent = -0.8
            pluie_percent = 12
            colTommorow, colImage, col3, colToday, col1, col2 = st.columns(6)
            proba = sortie[:,1]
            colTommorow.header("Demain")
            with colImage:
                st.image('pluie.png', width=200)
            col3.metric("Probabilité de pluie", f"{round(proba[0]*100, 2)}%", f"{pluie_percent} %")
            colToday.header("Aujourd'hui")
            col1.metric("Température max", f"{test.MaxTemp.values[0]} °C", f"{temp_max_percent} °C")
            col2.metric("Température min", f"{test.MinTemp.values[0]} °C", f"{temp_min_percent} °C")

        else:
            # st.image('soleil.png', width=600)
            # st.write("Température max:",test["MaxTemp"].values[0])
            # st.write("Température min:",test["MinTemp"].values[0])
            # st.write("Probabilité de pluie:",sortie[:,0])

            temp_max_percent = 1.2
            temp_min_percent = -0.8
            pluie_percent = 12
            colTommorow, colImage, col3, colToday, col1, col2 = st.columns(6)            
            proba = sortie[:,0]
            colTommorow.header("Demain")
            with colImage:
                st.image('sun-icon-vector-isolated-sun-flat-vector-icons-sun-logo-design-inspiration-700-175858735-removebg-preview (1).png', width=200)
            col3.metric("Probabilité de beau temps", f"{round(proba[0]*100, 2)}%", f"{pluie_percent} %")
            colToday.header("Aujourd'hui")
            col1.metric("Température max", f"{test.MaxTemp.values[0]} °C", f"{temp_max_percent} °C")
            col2.metric("Température min", f"{test.MinTemp.values[0]} °C", f"{temp_min_percent} °C")
