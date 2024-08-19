#Importing necessary libraries
import numpy as np
import pandas as pd
from flask import *
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization
from keras.optimizers import Adam



app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/about')
def about():
    return render_template('about.html')

# @app.route('/load',methods=["GET","POST"])
# def load():
#     global df, dataset
#     if request.method == "POST":
#         data = request.files['data']
#         df = pd.read_csv(data)
#         dataset = df.head(100)
#         msg = 'Data Loaded Successfully'
#         return render_template('load.html', msg=msg)
#     return render_template('load.html')

@app.route('/view')
def view():
    global df, dataset
    df = pd.read_csv('data.csv')
    dataset = df.head(100)
    return render_template('view.html', columns=dataset.columns.values, rows=dataset.values.tolist())

@app.route('/model',methods=['POST','GET'])
def model():

    if request.method=="POST":
        data = pd.read_csv('data.csv')
        data.head()
        x=data.iloc[:,:-1]
        y=data.iloc[:,-1]

        x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.3,stratify=y,random_state=42)

        print('ccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc')
        s=int(request.form['algo'])
        if s==0:
            return render_template('model.html',msg='Please Choose an Algorithm to Train')
        elif s==1:
            print('aaaaaaaaaaaaaaaaaaaaabbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb')
            from sklearn.linear_model import   LogisticRegression
            lr = LogisticRegression()
            lr=lr.fit(x_train,y_train)
            y_pred = lr.predict(x_test)
            acc_lr = accuracy_score(y_test,y_pred)*100
            print('aaaaaaaaaaaaaaaaaaaaaaaaa')
            msg = 'The accuracy obtained by Logistic Regression is ' + str(acc_lr) + str('%')
            return render_template('model.html', msg=msg)
        elif s==2:
            from sklearn.ensemble import RandomForestClassifier
            rf = RandomForestClassifier()
            rf=rf.fit(x_train,y_train)
            y_pred = rf.predict(x_test)
            acc_rf = accuracy_score(y_test,y_pred)*100
            msg = 'The accuracy obtained by Random Forest Classifier is ' + str(acc_rf) + str('%')
            return render_template('model.html', msg=msg)
        elif s==3:
            dt = DecisionTreeClassifier()
            dt=dt.fit(x_train,y_train)
            y_pred = dt.predict(x_test)
            acc_dt = accuracy_score(y_test,y_pred)*100
            msg = 'The accuracy obtained by Decision Tree Classifier is ' + str(acc_dt) + str('%')
            return render_template('model.html', msg=msg)
        elif s==4:
            # Gradient Boosting
            gbc = GradientBoostingClassifier()
            gbc = gbc.fit(x_train, y_train)
            y_pred = gbc.predict(x_test)
            # Evaluation
            acc_gbc = accuracy_score(y_test, y_pred)*100
            msg = 'The accuracy obtained by Gradient Boosting Classifier is ' + str(acc_gbc) + str('%')
            return render_template('model.html', msg=msg)
        elif s==5:
            svc = SVC()
            svc = svc.fit(x_train, y_train)
            y_pred = svc.predict(x_test)
            acc_svc = accuracy_score(y_test, y_pred)*100
            msg = 'The accuracy obtained by Support Vector Classifier is ' + str(acc_svc) + str('%')
            return render_template('model.html', msg=msg)
        elif s==6:
            # Reshape data for LSTM input
            x_train_lstm = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
            x_test_lstm = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

            # build LSTM model
            lstm_model = Sequential()
            lstm_model.add(LSTM(64, input_shape=(x_train_lstm.shape[1], x_train_lstm.shape[2]), activation='relu', return_sequences=True))
            lstm_model.add(Dropout(0.3))
            lstm_model.add(LSTM(64, activation='relu'))
            lstm_model.add(Dropout(0.3))
            lstm_model.add(Dense(1, activation='sigmoid'))

            lstm_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

            # Early Stopping
            early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

            # Fit model with early stopping
            lstm_epochs_hist = lstm_model.fit(x_train_lstm, y_train, validation_data=(x_test_lstm, y_test),
                    epochs=100, batch_size=32, validation_split=0.2, verbose=0, callbacks=[early_stopping])

            # Prediction
            y_pred_lstm = (lstm_model.predict(x_test_lstm) > 0.5).astype("int32")
            acc_lstm = accuracy_score(y_test, y_pred_lstm)*100
            msg = 'The accuracy obtained by LSTM is ' + str(acc_lstm) + str('%')
            return render_template('model.html', msg=msg)
        elif s==7:
            nn_model = Sequential()
            nn_model.add(Dense(256, input_dim=x_train.shape[1], activation='relu'))
            nn_model.add(BatchNormalization())
            nn_model.add(Dropout(0.3))
            nn_model.add(Dense(128, activation='relu'))
            nn_model.add(BatchNormalization())
            nn_model.add(Dropout(0.3))
            nn_model.add(Dense(64, activation='relu'))
            nn_model.add(BatchNormalization())
            nn_model.add(Dropout(0.3))
            nn_model.add(Dense(1, activation='sigmoid'))

            nn_model.compile(optimizer=Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=['accuracy'])

            early_stopping = EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True)

            lr_scheduler = ReduceLROnPlateau(monitor='val_loss', factor=0.3, patience=10, verbose=1)

            nn_epochs_hist = nn_model.fit(x_train, y_train, validation_split=0.2, epochs=200, batch_size=64,
                                        callbacks=[early_stopping, lr_scheduler], verbose=0)

            y_pred_nn = (nn_model.predict(x_test) > 0.5).astype(int).reshape(-1)
            acc_nn = accuracy_score(y_test, y_pred_nn)*100
            msg = 'The accuracy obtained by Neural Network is ' + str(acc_nn) + str('%')
            return render_template('model.html', msg=msg)

    return render_template('model.html')




if __name__ =='__main__':
    app.run(debug=True)
