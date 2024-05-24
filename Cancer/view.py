from django.http import HttpResponse
from django.shortcuts import render
import mysql.connector
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.decomposition import PCA
import seaborn as sns
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import io
import base64
from django.db import connections
from datetime import date
import mysql.connector
con = mysql.connector.connect(
    host='localhost',
    user='root',
    password='',
    database='cancer_prediction'
)
cur = con.cursor()
from django.shortcuts import render, redirect

def safe_int(value, default=0):
    """Safely convert a value to int, returning a default if conversion fails."""
    try:
        return int(value)
    except (ValueError, TypeError):
        return default

def home(request):
    if not request.session.get('loguser'):
        return redirect('login') 
    if request.GET.get('submit'):
        n = request.GET.get("n", "")
        a = safe_int(request.GET.get("age", ""))
        p = request.GET.get("place", "")
        s = request.GET.get("Surgery", "")
        c = request.GET.get("Chemotherapy", "")
        r = request.GET.get("Radiation", "")
        i = request.GET.get("Immunotherapy", "")
        h = request.GET.get("Hormonetherapy", "")
        t = request.GET.get("TargetedTherapy", "")
        b = safe_int(request.GET.get("bn", ""))
        tn = request.GET.get("tfs", "")
        w = safe_int(request.GET.get("wm", ""))
        spm = safe_int(request.GET.get("spm", ""))
        qsbc = request.GET.get("qsbc", "")
        rsqpm = safe_int(request.GET.get("rsqpm", ""))
        bi = request.GET.get("bi", "")
        st = safe_int(request.GET.get("st", ""))
        wt = safe_int(request.GET.get("wt", ""))
        bath = safe_int(request.GET.get("bath", ""))
        rp = safe_int(request.GET.get("rp", ""))
        csl = safe_int(request.GET.get("csl", ""))
        cl = safe_int(request.GET.get("cl", ""))
        ho = safe_int(request.GET.get("ho", ""))
        bd = safe_int(request.GET.get("bd", ""))
        pa = safe_int(request.GET.get("pa", ""))
        ms = safe_int(request.GET.get("ms", ""))
        wp = safe_int(request.GET.get("wp", ""))
        ed = safe_int(request.GET.get("ed", ""))
        sl = safe_int(request.GET.get("sl", ""))
        bws = safe_int(request.GET.get("bws", ""))
        tjs = safe_int(request.GET.get("tjs", ""))
        dcs = safe_int(request.GET.get("dcs", ""))

        # Sleep efficiency
        try:
            sep = int(spm / (b - w) * 10)
        except ZeroDivisionError:
            sep = 0
        print("Sep value is = ", sep)
        if sep > 85:
            se = 0
        elif 75 <= sep <= 84:
            se = 1
        elif 65 <= sep <= 74:
            se = 2
        else:
            se = 3
        print("Sleep efficiency cal: ", sep, "with Sleep efficiency: ", se)

        # Sleep latency
        if tn == '<=15':
            fsr = 0
        elif tn == '16-30':
            fsr = 1
        elif tn == '31-60':
            fsr = 2
        else:
            fsr = 3

        st_int = safe_int(st)
        slc = fsr + st_int
        if slc == 0:
            sll = 0
        elif 1 <= slc <= 2:
            sll = 1
        elif 3 <= slc <= 4:
            sll = 2
        elif 5 <= slc <= 6:
            sll = 3
        print("Sleep latency cal: ", slc, "with Sleep latency: ", sll)

        # Sleep duration
        spm_int = safe_int(spm)
        if spm_int < 5:
            sd = 3
        elif 5 <= spm_int <= 6:
            sd = 2
        elif 6 < spm_int <= 7:
            sd = 1
        else:
            sd = 0
        print("Actual sleep:", spm, " with Sleep duration :", sd)

        # Sleep disturbance
        sdc = (safe_int(wt) + safe_int(bath) + safe_int(rp) + safe_int(csl) + 
               safe_int(cl) + safe_int(ho) + safe_int(bd) + safe_int(pa) + safe_int(ms))

        if sdc == 0:
            sde = 0
        elif 1 <= sdc <= 9:
            sde = 1
        elif 10 <= sdc <= 18:
            sde = 2
        elif 19 <= sdc <= 27:
            sde = 3
        else:
            sde = 4  
        print("Sleep Disturbance cal: ", sdc, "with Sleep disturbance : ", sde)

        # Daytime dysfunction
        dtf = safe_int(wp) + safe_int(ed)

        if dtf == 0:
            dd = 0
        elif 1 <= dtf <= 2:
            dd = 1
        elif 3 <= dtf <= 4:
            dd = 2
        elif 5 <= dtf <= 6:
            dd = 3
        print("Daytime Dysfunction Cal: ", dtf, "with Daytime Dysfunction: ", dd)

        # Subjective sleep quality
        print("Subjective Sleep Quality:", rsqpm)

        # Use of sleep medicine
        print("Use of sleep Medicine :", ms)

        # Calculate PSQI score
        Psqi = se + sll + sd + sde + dd + rsqpm + ms
        print("Psqi value is:", Psqi)
        # Initialize sq variable
        sq = "Bad"
        sq_message = ""

        # Determine sleep quality and corresponding message
        st=False
        if Psqi>=5:
            st=True
            
        if Psqi <= 5:
            sq = "Good"
            
            sq_message = "Congratulations! Your sleep quality is very good."
        elif 6 <= Psqi <= 10:
            sq = "Average"
            sq_message = "Your sleep quality is average. Please follow recommendations to improve your sleep health."
        elif 11 <= Psqi <= 15:
            sq = "Poor"
            sq_message = "Your sleep quality is poor. Please follow recommendations to improve your sleep health."
        elif 16 <= Psqi <= 20:
            sq = "Very Poor"
            sq_message = "Your sleep quality is very poor. Please follow recommendations to improve your sleep health."

        print("Sleep quality:", sq)
        print(sq_message)
        
        
        #Calculate the total score for         
        # Build result string with proper type conversion
        k = f"<h3>Name is = {n}"
        k += f"<br>Place is = {p}"
        k += f"<br>Age is = {a}"
        k += f"<br>Surgery is = {s}"
        k += f"<br>Chemotherapy is = {c}"
        k += f"<br>Radiation is = {r}"
        k += f"<br>Immunotherapy is = {i}"
        k += f"<br>Hormonetherapy is = {h}"
        k += f"<br>Targeted Therapy is = {t}"
        k += f"<br>Bedtime at night is = {b}"
        k += f"<br>Time you fall asleep is = {tn}"
        k += f"<br>Wakeup time in the morning = {w}"
        k += f"<br>Hours of actual sleep you get during past month = {spm}"
        k += f"<br>How was your sleep quality before cancer diagnosis = {qsbc}"
        k += f"<br>How would you rate your sleep quality during past month = {rsqpm}"
        k += f"<br>Behavioral issues in past before breast cancer diagnosis = {bi}"
        k += f"<br>How often have you had trouble sleeping = {st}"
        k += f"<br>You wakeup in the middle of the night or early morning = {wt}"
        k += f"<br>You have to get up to use the bathroom = {bath}"
        k += f"<br>You cannot breathe comfortably = {rp}"
        k += f"<br>You cough or snore loudly = {csl}"
        k += f"<br>You feel too cold = {cl}"
        k += f"<br>You feel too hot = {ho}"
        k += f"<br>You had bad dreams = {bd}"
        k += f"<br>You have pain = {pa}"
        k += f"<br>Have you taken medicine to help you sleep = {ms}"
        k += f"<br>Trouble staying awake while driving, eating meals, or engaging in social activity = {wp}"
        k += f"<br>You feel enough enthusiasm to get things done = {ed}"
        k += f"<br>You snore loudly = {sl}"
        k += f"<br>You have long pauses between breaths while asleep = {bws}"
        k += f"<br>You have legs twitching or jerking while asleep = {tjs}"
        k += f"<br>You have episodes of disorientation or confusion during sleep = {dcs}"
        un=request.session["loguser"]
        dt=date.today()
        with connections['default'].cursor() as cursor:
            cursor.execute("""
                INSERT INTO result (name, age, place, sp, se, slc, sll, spm, sd, sdc, sde, dtf, dd, rsqpm, ms, psqi, sq_msg,dt)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s,%s)
            """, [un, a, p, sep, se, slc, sll, spm, sd, sdc, sde, dtf, dd, rsqpm, ms, Psqi, sq,dt])

        return render(request, "result.html", {
            "name": n, "age": a, "place": p, 'Surgery': s,
            'Chemotherapy': c, 'Radiation': r, 'Immunotherapy': i, 'Hormonetherapy': h, 'TargetedTherapy': t,
            "sp": sep, "se": se, "slc": slc, "sll": sll, "spm": spm, "sd": sd, "sdc": sdc, "sde": sde,
            "dtf": dtf, "dd": dd, "rsqpm": rsqpm, "ms": ms, "Psqi": Psqi, "sq": sq,
            "sq_message": sq_message,"st":st
        })

    return render(request, "home.html")
def new_func(st):
    return st
        
def getKnn(request):
    data=pd.read_csv('C:\\Users\\HP\\Downloads\\Cancer Project\\Cancer Project\\Cancer\\Data_for_Model.csv')
    data.head(5)
    data.isnull().sum()
    data=data.drop(['Region',], axis=1)
    data.drop(columns=['Unnamed: 55','Unnamed: 54','Unnamed: 56'], inplace=True)
    data.isnull().sum()
    data.describe()
    data=data.drop(['Date',], axis=1)
    data.drop(columns=['SleepEff_Percentage'], inplace=True)
    data.drop(columns=['Other issues','Other_categories','Bed_Partner','Other_restlessness'], inplace=True)
    data
    data.drop(columns=['Bedtime'], inplace=True)
    data.drop(columns=['WakeUp_time',], inplace=True)
    X = np.array(data.iloc[:, 1:])
    y = np.array(data['PSQI Score'])
    X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size = 0.33, random_state = 42)
    knn = KNeighborsClassifier(n_neighbors = 13)
    knn.fit(X_train, y_train)
    knn.score(X_test, y_test)
    neighbors = []
    cv_scores = []
    from sklearn.model_selection import cross_val_score
# perform 10 fold cross validation
    for k in range(1, 51, 2):
        neighbors.append(k)
        knn = KNeighborsClassifier(n_neighbors = k)
        scores = cross_val_score(
        knn, X_train, y_train, cv = 10, scoring = 'accuracy')
        cv_scores.append(scores.mean())
    MSE = [1-x for x in cv_scores]
# determining the best k
    optimal_k = neighbors[MSE.index(min(MSE))]
    print('The optimal number of neighbors is % d ' % optimal_k)

# plot misclassification error versus k
    plt.figure(figsize = (10, 6))
    plt.plot(neighbors, MSE)
    plt.xlabel('Number of neighbors')
    plt.ylabel('Misclassification Error')
    
    # Convert the plot to a base64-encoded image string
    import io
    buffer = io.BytesIO()
    plt.savefig(buffer, format='png')
    buffer.seek(0)
    image_png = buffer.getvalue()
    buffer.close()
    import base64
    image_base64 = base64.b64encode(image_png).decode()
    context = {'knn_plot': image_base64}

    return render(request, 'knn.html',context)


def getCnn(request):
    data=pd.read_csv('Data_for_Model.csv')
    data
    data.head(5)
    data.isnull().sum()
    data=data.drop(['Region',], axis=1)
    data.drop(columns=['Unnamed: 55','Unnamed: 54','Unnamed: 56'], inplace=True)
    data.isnull().sum()
    #print(data)
    plt.figure(figsize=(8,20))
    sns.heatmap(data.isnull(), yticklabels = False)
    plt.xticks(rotation=90)
    plt.tick_params(labelsize=8)
    data.describe()
    data=data.drop(['Date',], axis=1)
    data.drop(columns=['SleepEff_Percentage'], inplace=True)
    data.drop(columns=['Other issues','Other_categories','Bed_Partner','Other_restlessness'], inplace=True)
    data
    data.drop(columns=['Bedtime'], inplace=True)
    data.drop(columns=['WakeUp_time',], inplace=True)

    corr_matrix=data.corr()
    corr_matrix
    plt.figure(figsize = (30,30))
    sns.heatmap(corr_matrix, annot=True)
    plt.xticks(rotation=90)
    plt.yticks(rotation=360)
    plt.tick_params(labelsize=8)
    data.hist(bins = 10, figsize = (30,30), color='blue')
    target_data=data['PSQI Score']
    input_data=data.drop(['PSQI Score'],axis=1)
    #print(target_data)
    #print(input_data)

    X = np.array(input_data).astype('float32')
    y = np.array(target_data).astype('float32')

    from sklearn.preprocessing import StandardScaler, MinMaxScaler
    scaler=StandardScaler()
    X=scaler.fit_transform(X)

    from sklearn.model_selection import train_test_split
    X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2)
    target_data = target_data.replace({18: 17, 19: 17})
    input_data_data = input_data.replace({18: 17, 19: 17})

    '''unique_values = target_data.unique()
    ("Unique Values:", unique_values)
    expected_classes = range(18)  # Expected range of classes
    unexpected_values = [value for value in unique_values if value not in expected_classes]
    if unexpected_values:
        print("Unexpected Values:", unexpected_values)
        # Handle unexpected values (e.g., remove them or map them to valid classes)
        # Example: target_df = target_df[target_df.isin(expected_classes)]
        # Example: target_df.replace(unexpected_values, valid_value, inplace=True)
    else:
        print("No unexpected values found.")'''
    from sklearn.preprocessing import LabelEncoder
    le = LabelEncoder()
    y_train = le.fit_transform(y_train)

    import xgboost as xgb
    from xgboost import XGBClassifier
    model = XGBClassifier(learning_rate = 0.1, max_depth = 50, n_estimators = 100)
    model.fit(X =X_train,y =y_train)
    from xgboost import XGBClassifier
    classifier = XGBClassifier()
    classifier.fit(X = X_train,y =  y_train)
    from sklearn.metrics import confusion_matrix, accuracy_score
    y_pred = classifier.predict(X_test)
    y_pred = le.inverse_transform(y_pred)
    cm = confusion_matrix(y_test, y_pred)
    cm
    print(accuracy_score(y_test, y_pred))
    #print(data)
    # Convert the plot to a base64-encoded image string
    import io
    buffer = io.BytesIO()
    plt.savefig(buffer, format='png')
    buffer.seek(0)
    image_png = buffer.getvalue()
    buffer.close()
    import base64
    image_base64 = base64.b64encode(image_png).decode()
    context = {'cnn_plot': image_base64}

    return render(request, 'cnn.html',context)
def getSvm(request):
    data=pd.read_csv('Data_for_Model.csv')
    data
    data.head(5)
    data.isnull().sum()
    data=data.drop(['Region',], axis=1)
    data.drop(columns=['Unnamed: 55','Unnamed: 54','Unnamed: 56'], inplace=True)
    data.isnull().sum()
    data.describe()
    data=data.drop(['Date',], axis=1)
    data.drop(columns=['SleepEff_Percentage'], inplace=True)
    data.drop(columns=['Other issues','Other_categories','Bed_Partner','Other_restlessness'], inplace=True)
    data
    data.drop(columns=['Bedtime'], inplace=True)
    data.drop(columns=['WakeUp_time',], inplace=True)
    X = data.iloc[:,2:40]
    print(X.shape)
    X.head()
    y = data['PSQI Score']
    print(y.shape)
    y.head()
    y_num = pd.get_dummies(y)
    y_num.tail()
    X.corr()
    plt.figure(figsize=(18, 12))
    # Train the SVM model
    svm = SVC()
    svm.fit(X, y)
    sns.heatmap(X.corr(), vmin=0.85, vmax=1, annot=True, cmap='YlGnBu', linewidths=.5)
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_scaled = pd.DataFrame(X_scaled)
    X_scaled_drop = X_scaled.drop(X_scaled.columns[[2, 3, 12, 13, 22, 23]], axis=1)
    pca = PCA(n_components=0.95)
    x_pca = pca.fit_transform(X_scaled_drop)
    x_pca = pd.DataFrame(x_pca)
    print("Before PCA, X dataframe shape = ",X.shape,"\nAfter PCA, x_pca dataframe shape = ",x_pca.shape)
    print(pca.explained_variance_ratio_)
    print(pca.explained_variance_ratio_.sum())
    colnames = ['PC1','PC2','PC3','PC4','PC5','PC6','PC7','PC8','PC9','PC10','PC11','PC12','PC13','PC14','PC15','PC16','PC17','PC18','PC19','PC20','PC21','PC22','PSQI Score']
    diag = data.iloc[:,1:2]
    Xy = pd.DataFrame(np.hstack([x_pca,diag.values]),columns=colnames)
    Xy.head()
    X=(Xy.iloc[:,0:11]).values
    #75:25 train:test data splitting
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)

    print("X_train shape ",X_train.shape)
    print("y_train shape ",y_train.shape)
    print("X_test shape ",X_test.shape)
    print("y_test shape ",y_test.shape)
    svc = SVC()
    svc.fit(X_train, y_train)
    y_pred_svc =svc.predict(X_test)
    y_pred_svc.shape
    cm = confusion_matrix(y_test, y_pred_svc)
    print("Confusion matrix:\n",cm)
     # Generate classification report
    creport = classification_report(y_test, y_pred_svc)
    print("Classification report:\n", creport)
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    # PCA
    pca = PCA(n_components=0.95)
    X_train_pca = pca.fit_transform(X_train_scaled)
    X_test_pca = pca.transform(X_test_scaled)
    # Plot the confusion matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(confusion_matrix(y_test, y_pred_svc), annot=True, fmt='d', cmap='YlGnBu')
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    
    # Convert the confusion matrix plot to a base64-encoded image string
    buffer = io.BytesIO()
    plt.savefig(buffer, format='png')
    buffer.seek(0)
    cm_image_png = buffer.getvalue()
    buffer.close()
    cm_base64 = base64.b64encode(cm_image_png).decode()

    # Plot the SVM correlation heatmap
    plt.figure(figsize=(12, 8))  # Adjust the figsize as needed
    sns.heatmap(pd.DataFrame(X_train_pca).corr(), vmin=0.85, vmax=1, annot=True, cmap='YlGnBu', linewidths=.5)
    plt.title('SVM Correlation Heatmap')

    
    # Convert X to a DataFrame
    X_df = pd.DataFrame(X)

    # Convert the SVM correlation heatmap plot to a base64-encoded image string
    buffer = io.BytesIO()
    plt.savefig(buffer, format='png')
    buffer.seek(0)
    svm_image_png = buffer.getvalue()
    buffer.close()
    svm_base64 = base64.b64encode(svm_image_png).decode()

    # Convert the classification report to HTML format
    creport_html = "<pre>" + creport.replace('\n', '<br>') + "</pre>"

    # Pass the SVM plot, confusion matrix, and classification report to the HTML template
    context = {'svm_plot': svm_base64, 'confusion_matrix': cm_base64, 'classification_report': creport_html}
    return render(request, 'svm.html', context)


def welcome(request):
    return render(request, 'welcome.html')

from django.shortcuts import render, redirect
import mysql.connector
con = mysql.connector.connect(
    host='localhost',
    user='root',
    password='',
    database='cancer_prediction'
)
cur = con.cursor()
def register(request):
    msg = ""
    if request.GET.get('submit'):
        n = request.GET.get('n')
        a = request.GET.get('a')
        p = request.GET.get('p')
        s = request.GET.get('s')
        e = request.GET.get('e')
        ps = request.GET.get('ps')

        if n and a and p and s and e and ps:
            try:
                a = int(a)  # Ensure age is an integer

                # Use parameterized query to prevent SQL injection
                query = """
                INSERT INTO registration (name, age, place, stage, email, pass) 
                VALUES (%s, %s, %s, %s, %s, %s)
                """
                cur.execute(query, (n, a, p, s, e, ps))
                con.commit()

                msg = "Account Created Successfully"
            except Exception as ex:
                msg = f"Error: {str(ex)}"
        else:
            msg = "All fields are required."

    return render(request, "register.html", {"msg": msg})


def user_login(request):
    if request.GET.get('submit'):
        e=request.GET.get('e')
        password=request.GET.get('password')
        s="select * from registration where email='"+e+"' and pass='"+password+"'"
        cur.execute(s)
        x=0
        for row in cur:
            x=x+1
        if x>0:
            request.session["loguser"]=e
            return redirect(link)
        else:
            return render(request,"login.html",{"msg":"Invalid Email or Password"})
    else:
        return render(request,"login.html")
    
def lout(request):
    if 'loguser' in request.session:
        del request.session['loguser']
    return redirect(user_login)

def link(request):
    return render(request,"link.html")

def srecord(request):
    un = request.session.get("loguser")
    s = f"SELECT * FROM result WHERE name='{un}'"
    cur.execute(s)
    dt = []
    ps = []

    for row in cur:
        dt.append(row[18])
        ps.append(row[16])

    if not dt or not ps:
        return render(request, 'srecord.html', {'error': 'No data found'})

    df1 = pd.DataFrame(ps, columns=['PSQI Score'])
    md = df1['PSQI Score'].median()

    dt.append("Predict PSQ")
    ps.append(round(md))

    if len(dt) != len(ps):
        return render(request, 'srecord.html', {'error': 'Invalid data length'})

    df = pd.DataFrame({'Date': dt, 'PSQI Score': ps})

    plt.figure(figsize=(10, 6))
    plt.plot(df['Date'], df['PSQI Score'], marker='o')
    plt.title('PSQI Score Over Time')
    plt.xlabel('Date')
    plt.ylabel('PSQI Score')
    plt.grid(True)

    # Add horizontal line for Predict PSQ value
    plt.axhline(y=round(md), color='red', linestyle='--', label='Predict PSQ')

    # Add legend
    plt.legend()

    # Convert the plot to a base64-encoded image string
    buffer = io.BytesIO()
    plt.savefig(buffer, format='png')
    buffer.seek(0)
    image_png = buffer.getvalue()
    buffer.close()
    image_base64 = base64.b64encode(image_png).decode()

    context = {'cnn_plot': image_base64, 'median': round(md)}

    return render(request, 'srecord.html',context)


def recom(request):
    un = request.session.get("loguser")
    
    psqi_score = 6  
    if psqi_score > 5:
        unrecommendation = True
    else:
        unrecommendation = False
    
    return render(request, "recom.html", {'username': un, 'recommendation': unrecommendation})

def recommendation(request):
    
    return render(request, "recommendation.html")