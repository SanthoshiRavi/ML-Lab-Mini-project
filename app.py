import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score,classification_report
from flask import *
from sklearn.ensemble import VotingClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import LinearSVC

app=Flask(__name__)

@app.route('/')
def index():
    return render_template("index.html")

@app.route('/about')
def about():
    return render_template("about.html")

@app.route('/load',methods=["GET","POST"])
def load():
    global df, dataset
    if request.method == "POST":
        data = request.files['data']
        df = pd.read_csv(data)
        dataset = df.head(100)
        msg = 'Data Loaded Successfully'
        return render_template('load.html', msg=msg)
    return render_template('load.html')

@app.route('/preprocess', methods=['POST', 'GET'])
def preprocess():
    global x, y, x_train, x_test, y_train, y_test,  hvectorizer,df,data
    if request.method == "POST":
        size = int(request.form['split'])
        size = size / 100
        df = pd.read_csv(r'Employee dataset.csv')
        df.drop('employee_id',axis = 1,inplace = True)
        df.isna().sum()
        df['previous_year_rating'].fillna( df['previous_year_rating'].median(), inplace=True)
        df['education'].fillna( df['education'].mode()[0], inplace=True)
        df.isna().sum()
        # using the labelEncoder to convert into same datatype
        le = LabelEncoder()
        for i in df.columns:
            if df[i].dtype == 'object':
                df[i] = le.fit_transform(df[i])
        print(df)
        x = df.drop('is_promoted',axis = 1)
        y = df['is_promoted']
        x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.3,random_state=70)
        df.head()
        # describes info about train and test set
        print("Number transactions X_train dataset: ", x_train.shape)
        print("Number transactions y_train dataset: ", y_train.shape)
        print("Number transactions X_test dataset: ", x_test.shape)
        print("Number transactions y_test dataset: ", y_test.shape)    
        print(x_train,x_test)
        return render_template('preprocess.html', msg='Data Preprocessed and It Splits Successfully')
    return render_template('preprocess.html')

@app.route('/model', methods=["POST","GET"])
def model():
    if request.method=="POST":
        global model
        s=int(request.form['algo'])
        if s==0:
            return render_template('model.html',msg="Choose an algorithm")
        elif s==1:
            gnb=GaussianNB()
            gnb.fit(x_train[:500],y_train[:500])
            y_pred=gnb.predict(x_test[:500])
            ac_gnb=accuracy_score(y_pred,y_test[:500])*100
            precision_gnb = precision_score(y_test[:500], y_pred, average='weighted')*100
            recall_gnb = recall_score(y_test[:500], y_pred, average='weighted')*100
            f1_gnb = f1_score(y_test[:500], y_pred, average='weighted')*100
            msg="The accuracy : "+str(ac_gnb) + str('%')
            msg1="The precision : "+str(precision_gnb) + str('%')
            msg2="The recall : "+str(recall_gnb) + str('%')
            msg3="The f1_score : "+str(f1_gnb) + str('%')
            return render_template("model.html",msg=msg,msg1=msg1,msg2=msg2,msg3=msg3)
        elif s==2:
            svs=LinearSVC()
            svs.fit(x_train[:500],y_train[:500])
            y_pred=svs.predict(x_test[:500])
            ac_svs=accuracy_score(y_pred,y_test[:500])*100
            precision_svs = precision_score(y_test[:500], y_pred, average='weighted')*100
            recall_svs = recall_score(y_test[:500], y_pred, average='weighted')*100
            f1_svs = f1_score(y_test[:500], y_pred, average='weighted')*100
            msg="The accuracy : "+str(ac_svs) + str('%')
            msg1="The precision : "+str(precision_svs) + str('%')
            msg2="The recall : "+str(recall_svs) + str('%')
            msg3="The f1_score : "+str(f1_svs) + str('%')
            return render_template("model.html",msg=msg,msg1=msg1,msg2=msg2,msg3=msg3)
        elif s==3:
            from xgboost import XGBClassifier
            xgb  = XGBClassifier()
            xgb.fit(x_train, y_train)
            y_pred = xgb.predict(x_test)
            ac_xgb = accuracy_score(y_test, y_pred)
            precision_xgb = precision_score(y_test, y_pred, average='weighted') * 100
            recall_xgb = recall_score(y_test, y_pred, average='weighted') * 100
            f1_xgb = f1_score(y_test, y_pred, average='weighted') * 100
            msg = "The accuracy: " + str(ac_xgb) + '%'
            msg1 = "The precision: " + str(precision_xgb) + '%'
            msg2 = "The recall: " + str(recall_xgb) + '%'
            msg3 = "The f1_score: " + str(f1_xgb) + '%'
            return render_template("model.html", msg=msg, msg1=msg1, msg2=msg2, msg3=msg3)
        elif s==4:
            gnb = GaussianNB()
            svs = LinearSVC()
            vot = VotingClassifier(estimators=[
                            ('gnb', gnb),
                            ('svc', svs)
                        ], voting='hard')
            vot.fit(x_train[:500], y_train[:500])
            y_pred = vot.predict(x_test[:500])
            ac_vot = accuracy_score(y_pred, y_test[:500]) * 100
            precision_vot = precision_score(y_test[:500], y_pred, average='weighted') * 100
            recall_vot = recall_score(y_test[:500], y_pred, average='weighted') * 100
            f1_vot = f1_score(y_test[:500], y_pred, average='weighted') * 100
            msg="The accuracy  :"+str(ac_vot) + str('%')
            msg1="The precision :"+str(precision_vot) + str('%')
            msg2="The recall : "+str(recall_vot) + str('%')
            msg3="The f1_score  :"+str(f1_vot) + str('%')
            return render_template("model.html",msg=msg,msg1=msg1,msg2=msg2,msg3=msg3)
    return render_template("model.html")


@app.route('/prediction' , methods=["POST","GET"])
def prediction():
    							
    			
    if request.method=="POST":
        f1=float(request.form['department'])
        f2=float(request.form['region'])
        f5=float(request.form['education'])
        f6=float(request.form['gender'])
        f7=float(request.form['recruitment_channel'])
        f8=float(request.form['no_of_trainings'])
        f9=int(request.form['age'])
        f10=float(request.form['previous_year_rating'])
        f11=float(request.form['length_of_service'])
        f12=float(request.form['KPIs'])
        f13=float(request.form['awards_won?'])
        f14=float(request.form['avg_training_score'])

        lee=[f1,f2,f5,f6,f7,f8,f9,f10,f11,f12,f13,f14]
        print(lee)

        from sklearn.naive_bayes import GaussianNB
        model=GaussianNB()
        model.fit(x_train,y_train)
        result=model.predict([lee])
        print(result)
        if result==0:
            msg="not promoted"
            return render_template('prediction.html', msg=msg)
        else:
            msg="promoted"
            return render_template('prediction.html', msg=msg)
    return render_template("prediction.html") 

if __name__=="__main__":
    app.run(debug=True)