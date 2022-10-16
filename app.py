import streamlit as st # web development
import numpy as np # np mean, np random 
import pandas as pd # read csv, df manipulation
import time # to simulate a real time data, time loop 
import plotly.express as px # interactive charts 
import plotly.graph_objects as go
import joblib
from imblearn.over_sampling import RandomOverSampler, SMOTE
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
import cv2
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

st.title("Prévision météo")

with st.sidebar:
    choose = option_menu("App Gallery", ["À propos du projet", "Classifier", "Regressor"],
                         icons=['house', 'camera fill', 'kanban'],
                         menu_icon="app-indicator", default_index=0,
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
        st.markdown('<p class="font">About the Creator</p>', unsafe_allow_html=True)    
    st.write("Sharone Li is a data science practitioner, enthusiast, and blogger. She writes data science articles and tutorials about Python, data visualization, Streamlit, etc. She is also an amateur violinist who loves classical music.\n\nTo read Sharone's data science posts, please visit her Medium blog at: https://medium.com/@insightsbees")    
    # st.image(profile, width=700 )

if choose == "Regressor":
    chosen_id = stx.tab_bar(data=[
    stx.TabBarItemData(id="tab1", title="DummyRegressor", description="Tasks to take care of"),
    stx.TabBarItemData(id="tab2", title="Prophet", description="Tasks taken care of"),
    stx.TabBarItemData(id="tab3", title="LSTM Model", description="Tasks missed out")])
    placeholder = st.container()
    if chosen_id == "tab1":
        st.write("<h3>DummyRegressor</h3>", unsafe_allow_html=True)
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
        st.write(test_result)

        import matplotlib.pyplot as plt_dummy
        plt_dummy.rcParams["figure.figsize"] = (20,15)

        plt_dummy.plot(test_results['Test Predictions'][20:300],color='red',label ='Predictions')
        plt_dummy.plot(test_results['Réalité'][20:300],label='Réalité')
        plt_dummy.legend()
        plt_dummy.show()
        st.pyplot(plt_dummy)
        plt_dummy.figure().clear()

    elif chosen_id == "tab2":
        st.write("<h3>Prophet</h3>", unsafe_allow_html=True)
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

    elif chosen_id == "tab3":
        st.write("<h3>LSTM Model</h3>", unsafe_allow_html=True)

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
        import matplotlib.pyplot as plt_lstm_1
        plt_lstm_1.rcParams["figure.figsize"] = (20,15)

        plt_lstm_1.plot(train_df['ds'][20:300],train_results['Predictions'][20:300],color='yellow',label ='Predictions')
        plt_lstm_1.plot(train_df['ds'][20:300],train_results['Réalité'][20:300],label='Réalité')
        plt_lstm_1.legend()
        st.pyplot(plt_lstm_1)
        plt_lstm_1.figure().clear()

        val_predictions = model_lstm.predict(X_val1).flatten()
        val_results = pd.DataFrame(data={'Val Predictions':val_predictions, 'Réalité':y_val1})
        val_results

        import matplotlib.pyplot as plt_lstm_2

        plt_lstm_2.plot(train_df['ds'][2900:3000],val_results['Val Predictions'][:100],label='Predictions')
        plt_lstm_2.plot(train_df['ds'][2900:3000],val_results['Réalité'][:100],label='Réalité')
        plt_lstm_2.legend()
        st.pyplot(plt_lstm_2)
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

        import time
        import plotly.graph_objects as go
        import pandas as pd
        import numpy as np

        df = test_results
        df['ds'] = train_df['ds'][3033:3188]
        for i in range(df.shape[0]):
            df['ds'][i] =train_df['ds'][3033+i]
        header = st.container()
        select_param = st.container()
        plot_spot = st.empty()
        st.write(df.shape)
        st.write(df.tail(2))
        #select parmeter drop down
        with select_param:
            param_lst = list(df.columns)
            param_lst.remove('ds')
            select_param = st.selectbox('Select a Weather Parameter',   param_lst)
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
        ymax = max(df[select_param])+5
        ymin = min(df[select_param])-5
        for i in range(0, n-30, 1):
            df_tmp = df.iloc[i:i+30, :]
            with plot_spot:
                make_chart(df_tmp, 'Test Predictions','Réalité', ymin, ymax)
            time.sleep(0.5)

    else:
        placeholder = st.empty()

        

if choose == "Classifier":
    st.write("Classifier")
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

    elif chosen_id == "tab2":
        st.write("DecisionTreeClassifier")
        #Séparation entre feature et cible
        X = df_prepare.drop(columns=["RainTomorrow"], axis=1)
        y = df_prepare["RainTomorrow"]

        #séparation du jeu de donnée en train et test, on choisi 0.1 car cela représente 1 an de donnée ( 10 ans de données *0.1 = 1 an)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, shuffle = False)

        #Normalisation
        sc = MinMaxScaler()
        X_train[num_columns] = sc.fit_transform(X_train[num_columns])
        X_test[num_columns] = sc.transform(X_test[num_columns])
        #équilibrage par Oversampling : SMOTE
        smo = SMOTE()
        X_sm, y_sm = smo.fit_resample(X_train, y_train)
        print('Classes échantillon SMOTE :', dict(pd.Series(y_sm).value_counts()))

        dtree = joblib.load('dtree.joblib')

        y_dtree = dtree.predict(X_test)
        y_dtree_prob=dtree.predict_proba(X_test)[:,1]

        st.write("LogisticRegression")

        lr = joblib.load('lr.joblib')
        y_lr = lr.predict(X_test)
        y_lr_prob=lr.predict_proba(X_test)[:,1]

        st.write("RandomForestClassifier")

        rf = joblib.load('rf.joblib')
        y_rf = rf.predict(X_test)
        y_rf_prob=rf.predict_proba(X_test)[:,1]

        st.write("SVC")

        svm = joblib.load('svm.joblib')
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

    elif chosen_id == "tab3":
        st.write("RandomForestClassifier Opti") 
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
    
    elif chosen_id == "tab4":
        st.write("RandomForestClassifier")
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

        smo = SMOTE()
        X_sm, y_sm = smo.fit_resample(X_train, y_train)
        print('Classes échantillon SMOTE :', dict(pd.Series(y_sm).value_counts()))

        dtree_ville = joblib.load('dtree_ville.joblib')

        y_dtree_ville = dtree_ville.predict(X_test)
        y_dtree_ville_prob=dtree_ville.predict_proba(X_test)[:,1]

        lr_ville = joblib.load('lr_ville.joblib')

        y_lr_ville = lr_ville.predict(X_test)
        y_lr_ville_prob=lr_ville.predict_proba(X_test)[:,1]

        rf_ville = joblib.load('rf_ville.joblib')

        y_rf_ville = rf_ville.predict(X_test)
        y_rf_ville_prob=rf_ville.predict_proba(X_test)[:,1]

        svm_ville = joblib.load('svm_ville.joblib')

        y_svm_ville = svm_ville.predict(X_test) 
        y_svm_ville_prob=svm_ville.predict_proba(X_test)[:,1]


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
    
    else:
        placeholder = st.empty()
    









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


