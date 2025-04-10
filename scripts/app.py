import numpy as np
import pandas as pd
from flask import Flask ,render_template,request,flash,session,redirect,url_for,jsonify
from datetime import date
import bcrypt
import plotly.graph_objs as go
import os
import openai

app = Flask(__name__)
app.config['SECRET_KEY'] = 'diabetes'
import db
import pickle


global gluc
global ins
global predict_date
global gluc_list
global ins_list

dataset = pd.read_csv('diabetes.csv')
dataset_X = dataset.iloc[:,[1, 2, 5, 7]].values

from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler(feature_range = (0,1))
dataset_scaled = sc.fit_transform(dataset_X)

logreg_model = pickle.load(open('logreg_model.pkl', 'rb'))
knn_model = pickle.load(open('knn_model.pkl', 'rb'))
svc_model = pickle.load(open('svc_model.pkl', 'rb'))
nb_model = pickle.load(open('nb_model.pkl', 'rb'))
decision_tree = pickle.load(open('decision_tree.pkl', 'rb'))
random_forest= pickle.load(open('random_forest.pkl', 'rb'))

def hash_it(password):
    
    bytes = password.encode('utf-8')
    hash = bcrypt.hashpw(bytes, b'$2b$12$ikFJLhhV.1ziQsq0cT94IO')
    hash=str(hash)
    return hash[2:-1]

@app.route('/')

@app.route('/index')
def index():
  return render_template('index.html')
@app.route('/signup',methods = ['POST', 'GET'])
def signup():
  if request.method == 'POST':
    name = request.form['name']
    email = request.form['email']
    phone = request.form['phone']
    password= request.form['password']
    hashed_password = hash_it(password)
    print(hashed_password)
    user_account = db.db.Users.find_one({"email":email})
    
    if user_account:
        return render_template('login.html',msg="You are already a member, please login using your details")
    else:
      db.db.Users.insert_one({"name":name,"email":email,"phone":phone,"password":hashed_password})
      return render_template('login.html', msg="Registered successfuly..")

         

@app.route('/signin',methods = ['POST', 'GET'])
def signin():
  if request.method == 'POST':
    email = request.form['email']
    password= request.form['password']
    signin_user = db.db.Users.find_one({'email':email })
    if signin_user and signin_user['password']==hash_it(password):
      global id
      session['loggedin'] = True
      session['name']= signin_user['name']
      session['email']=signin_user['email']
      id=session['email']
      msg = 'Welcome'+" "+session['name']+"!!"
      flash(msg)
      return redirect(url_for('predict'))
      return render_template('predict.html', msg = msg)
    else:
        msg = 'Incorrect username / password !'
        return render_template('login.html', msg = msg)
          


            



@app.route('/register')
def register():
  return render_template('register.html')

@app.route("/predict")
def predict():
  print(session.get('email'))
  if  not session.get('email'):
      return render_template('login.html')
  else:
    return render_template('predict.html')
  

@app.route("/prediction",methods=['POST'])
def prediction():
      
    global gluc 
    global ins 
    today = date.today()
    today = str(today)
    signin_user = db.db.Users.find_one({'email':id })
    user = signin_user['email']
    glucose = request.form['Glucose Level']
    insulin = request.form['Insulin']
    age = request.form['Age']
    BMI= request.form['BMI']
    
    gluc=glucose
    ins = insulin
    print(gluc,ins)
    db.db.History.insert_one({"date":today,"user":user,"glucose":glucose,"insulin":insulin,"age":age,"BMI":BMI})

    prediction_results=[]
    float_features = [float(x) for x in request.form.values()]
    final_features = [np.array(float_features)]
    logreg_prediction = logreg_model.predict( sc.transform(final_features) )
    knn_prediction = knn_model.predict( sc.transform(final_features) )
    nb_prediction = nb_model.predict( sc.transform(final_features) )
    svc_prediction = svc_model.predict( sc.transform(final_features) )
    dt_prediction = decision_tree.predict( sc.transform(final_features) )
    rf_prediction = random_forest.predict( sc.transform(final_features) )
    prediction_results.append(logreg_prediction[0])
    prediction_results.append(knn_prediction[0])
    prediction_results.append(nb_prediction[0])
    prediction_results.append(svc_prediction[0])
    prediction_results.append(dt_prediction[0])
    prediction_results.append(rf_prediction[0])

    print(prediction_results)
    if(prediction_results.count(1.0)> prediction_results.count(0.0)):
          prediction=1
    else:
          prediction=0

    print(prediction)

    if prediction == 1:
        pred = "You have Diabetes, please consult a Doctor."
    elif prediction == 0:
        pred = "You don't have Diabetes."
    output = pred

    return render_template("predict.html",msg=output)


@app.route('/result')
def result():
      
      global gluc
      global ins
      global predict_date
      global gluc_list
      global ins_list
      
     

      x = ['glucose']
      y = [int(gluc)]

      data = go.Bar(x=x, y=y,width=0.1)

      layout = go.Layout(
          title='Glucose Level',
          xaxis=dict(title='X Axis'),
          yaxis=dict(title='Y Axis'),
          width=700,
          height=500
      )

      fig = go.Figure(data=data, layout=layout)
      fig.add_hline(y=125)
      chart1= fig.to_html(full_html=False)

      x = ['Insulin']
      y = [int(ins)]

      data = go.Bar(x=x, y=y,width=0.1)

      layout = go.Layout(
          title='Insulin Level',
          xaxis=dict(title='X Axis'),
          yaxis=dict(title='Y Axis'),
          width=700,
          height=500
      )

      fig = go.Figure(data=data, layout=layout)
      fig.add_hline(y=100)
      chart2= fig.to_html(full_html=False)

      

      return render_template('result.html',chart1=chart1,chart2=chart2)


@app.route('/history')
def history():
      
  if not session.get('email'):
        return render_template('login.html')
  else:
    global predict_date
    global gluc_list
    global ins_list
    signin_user = db.db.Users.find_one({'email':id })
    user = signin_user['email']
    user_history = list(db.db.History.find({'user':user}))
    date_history=[]
    glucose_history=[]
    insulin_history=[]
    print(user_history)
    for i in user_history:
      date_history.append(i['date'])
      glucose_history.append(int(i['glucose']))
      insulin_history.append(int(i['insulin']))
    predict_date= date_history 
    gluc_list = glucose_history
    ins_list = insulin_history
    from datetime import datetime

    dates1=predict_date
    values1= gluc_list
    dates2= predict_date
    values2= ins_list


    # Convert dates to datetime objects
    dates1 = [datetime.strptime(date, '%Y-%m-%d') for date in dates1]
    dates2 = [datetime.strptime(date, '%Y-%m-%d') for date in dates2]

    # Create traces for the line chart
    trace1 = go.Scatter(
        x=dates1,
        y=values1,
        mode='lines+markers',  # Show both lines and markers on the chart
        name='Glucose Levels'  # Name the first line
    )
    trace2 = go.Scatter(
        x=dates2,
        y=values2,
        mode='lines+markers',  # Show both lines and markers on the chart
        name='Insulin Levels'  # Name the second line
    )

    # Create the layout for the chart
    layout = go.Layout(
        title='History of Glucose and Insulin Levels',
        xaxis=dict(title='Date'),
        yaxis=dict(title='Value'),
        width=800,
        height=500
    )

    # Create the figure and show the chart
    fig = go.Figure(data=[trace1, trace2], layout=layout)
    chart3= fig.to_html(full_html=False)


  
  
  
  
  return render_template("history.html",records=user_history,chart3=chart3)




@app.route('/login')
def login():
    return render_template('login.html')

@app.route('/diet')
def diet():
    return render_template('diet.html')

@app.route('/chatbot')
def chatbot():
    return render_template('chatbot.html')



def chatbot(argument):
    openai.api_key = ""
    response = openai.Completion.create(
    model="text-davinci-003",
    prompt=argument,
    temperature=0.5,
    max_tokens=1024,
    top_p=1,
    frequency_penalty=0,
    presence_penalty=0
    )
    answer=str(response.choices[0].text)
    return answer

@app.route("/chat",methods=['POST'])
def chat():
      
  
    query= request.form['message']
    msg=chatbot(query)
    return render_template('chatbot.html',msg=msg)


@app.route('/logout')
def logout():
  session.pop('loggedin', None)
  session.pop('id', None)
  session.pop('email', None)
  return redirect(url_for('index'))
      

if __name__ =='__main__':
    app.run(host='0.0.0.0',debug=True)
