# -*- coding: utf-8 -*-
"""motwani_singh_p3_fa22.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1ZttCAtsnG4qiJbd_011aD6ZfUL1AdJp9
"""

# Commented out IPython magic to ensure Python compatibility.
# importing the necessary libraries
from mlxtend.plotting import plot_decision_regions
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import RandomizedSearchCV 
from sklearn.metrics import roc_auc_score  
from sklearn.metrics import accuracy_score
from sklearn.linear_model import SGDClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import roc_curve 
from sklearn.metrics import log_loss
import warnings
from time import time
from sklearn import preprocessing
from sklearn.feature_selection import chi2
from sklearn.preprocessing import LabelEncoder 
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC, LinearSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.multiclass import OneVsRestClassifier
from sklearn import metrics
from sklearn.metrics import confusion_matrix, classification_report, auc, f1_score, precision_score, recall_score
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.model_selection import cross_val_score
import pickle
plt.style.use('ggplot')
# %matplotlib inline
# Tools & Evaluation metrics
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import roc_curve, accuracy_score, precision_recall_curve 
from sklearn.metrics import f1_score, make_scorer
from sklearn.model_selection import train_test_split, cross_val_score
warnings.filterwarnings('ignore')

# Commented out IPython magic to ensure Python compatibility.
import sys,os
# Block which runs on both Google Colab and Local PC without any modification
if 'google.colab' in sys.modules:    
    project_path = "/content/drive/My Drive/"
    # Google Colab lib
    from google.colab import drive
    # Mount the drive
    drive.mount('/content/drive/', force_remount=True)
    sys.path.append(project_path)
#     %cd $project_path

# Let's look at the sys path
print('Current working directory', os.getcwd())

#Loading the dataset
cdi_data = pd.read_csv('/content/drive/My Drive/cdi.csv')

#Print the first 5 rows of the dataframe.
cdi_data.head()

print(cdi_data.columns)

cdi_data.info()

print(cdi_data.shape)
cdi_data.isnull().sum()

cdi_data.describe()

"""Exploratory Data Analysis (EDA)"""

def plotPerColumnDistribution(df, nGraphShown, nGraphPerRow):
    nunique = df.nunique()
    df = df[[col for col in df if nunique[col] > 1 and nunique[col] < 50]] # For displaying purposes, pick columns that have between 1 and 50 unique values
    nRow, nCol = df.shape
    columnNames = list(df)
    nGraphRow = (nCol + nGraphPerRow - 1) / nGraphPerRow
    plt.figure(num = None, figsize = (6 * nGraphPerRow, 8 * nGraphRow), dpi = 80, facecolor = 'w', edgecolor = 'k')
    for i in range(min(nCol, nGraphShown)):
        plt.subplot(nGraphRow, nGraphPerRow, i + 1)
        columnDf = df.iloc[:, i]
        if (not np.issubdtype(type(columnDf.iloc[0]), np.number)):
            valueCounts = columnDf.value_counts()
            valueCounts.plot.bar()
        else:
            columnDf.hist()
        plt.ylabel('counts')
        plt.xticks(rotation = 90)
        plt.title(f'{columnNames[i]} (column {i})')
    plt.tight_layout(pad = 1.0, w_pad = 1.0, h_pad = 1.0)
    plt.show()

def plotCorrelationMatrix(df, graphWidth):
    filename = df.dataframeName
    df = df.dropna('columns') # drop columns with NaN
    df = df[[col for col in df if df[col].nunique() > 1]] # keep columns where there are more than 1 unique values
    if df.shape[1] < 2:
        print(f'No correlation plots shown: The number of non-NaN or constant columns ({df.shape[1]}) is less than 2')
        return
    corr = df.corr()
    plt.figure(num=None, figsize=(graphWidth, graphWidth), dpi=80, facecolor='w', edgecolor='k')
    corrMat = plt.matshow(corr, fignum = 1)
    plt.xticks(range(len(corr.columns)), corr.columns, rotation=90)
    plt.yticks(range(len(corr.columns)), corr.columns)
    plt.gca().xaxis.tick_bottom()
    plt.colorbar(corrMat)
    plt.title(f'Correlation Matrix for {filename}', fontsize=15)
    plt.show()

# Scatter and density plots
def plotScatterMatrix(df, plotSize, textSize):
    df = df.select_dtypes(include =[np.number]) # keep only numerical columns
    # Remove rows and columns that would lead to df being singular
    df = df.dropna('columns')
    df = df[[col for col in df if df[col].nunique() > 1]] # keep columns where there are more than 1 unique values
    columnNames = list(df)
    if len(columnNames) > 10: # reduce the number of columns for matrix inversion of kernel density plots
        columnNames = columnNames[:10]
    df = df[columnNames]
    ax = pd.plotting.scatter_matrix(df, alpha=0.75, figsize=[plotSize, plotSize], diagonal='kde')
    corrs = df.corr().values
    for i, j in zip(*plt.np.triu_indices_from(ax, k = 1)):
        ax[i, j].annotate('Corr. coef = %.3f' % corrs[i, j], (0.8, 0.2), xycoords='axes fraction', ha='center', va='center', size=textSize)
    plt.suptitle('Scatter and Density Plot')
    plt.show()

plotPerColumnDistribution(cdi_data, 10, 5)

cdi_data.dataframeName = 'U.S._Chronic_Disease_Indicators.csv'
plotCorrelationMatrix(cdi_data, 8)

plotScatterMatrix(cdi_data, 20, 10)

# Dropping rows with muissing value in the Data Value column

cdi_data.dropna(subset=['DataValue'])

"""**BUSINESS DECISION 1:**
Which category of disease "Topic" has more questions and thus needs more policy clarification?

"""

CountStatus=cdi_data['Topic'].value_counts()
print (CountStatus)
CountStatus.plot.barh()

"""**BUSINESS DECISION 2:** Which questions pertain to the insurance policy amount and need to be revisited with respect to policy designing?"""

df_exp= cdi_data[cdi_data['DataValueUnit']=='$']
df_exp.info()

"""Filtering data to consider only United States locations"""

df=cdi_data
plt.figure(figsize=(16,6))
ax = sns.heatmap(df.isnull(),yticklabels = False, cbar = False, cmap='viridis')
ax.set(xlabel = 'Columns in the dataset', ylabel = 'Null Values', title = 'Null values in each column') 

df = df.drop(['YearEnd','DataSource','LowConfidenceLimit','HighConfidenceLimit','LocationID',
              'DataValueFootnoteSymbol','DatavalueFootnote','Response','ResponseID', 'StratificationCategory2',
              'StratificationCategory3','Stratification2','Stratification3', 'StratificationCategoryID2', 
              'StratificationCategoryID3','StratificationID2','StratificationID3'], axis = 1)

df['YearStart'] = pd.to_numeric(df['YearStart'])
df = df[(df['YearStart'] > 2009) & (df['YearStart'] < 2016)]

df['GeoLocation'] = df['GeoLocation'].str.replace(r"\(","")
df['GeoLocation'] = df['GeoLocation'].str.replace(r"\)","")
new = df['GeoLocation'].str.split(" ", n = 2, expand = True)
df["Latitude"] = new[2]
df["Longitude"] = new[1]
df.drop(columns=["GeoLocation"], inplace = True)
df.drop(df[df.LocationDesc == 'United States'].index,
            inplace = True)

df = df.dropna()

df_new=df
print(df_new.groupby('YearStart')['Topic'].count())
df_new.groupby(['Topic','YearStart'])['YearStart'].count()
df_new.groupby(['Topic','Question']).agg({'Question':'count'})
df[df['Topic']=='Alcohol'].groupby(['Topic','Question']).agg({'Question':'count'})
df_new.groupby(df_new['Topic']=='Alcohol')['Question'].apply(lambda x: list(np.unique(x))).count()
df_new.groupby('Topic')['Question'].nunique()
df.groupby(['Topic','Question']).nunique()
print(df[df['Topic']=='Diabetes'].groupby(['Topic','Question']).agg({'Question':'count'}))
print(df[df['Topic']=='Cardiovascular Disease'].groupby(['Topic','Question']).agg({'Question':'count'}))
print(df[df['Topic']=='Chronic Obstructive Pulmonary Disease'].groupby(['Topic','Question']).agg({'Question':'count'}))
#df_g2.head()
print(df.info())

df_by_topic = df_new.groupby('Topic')
#type(df_by_topic)
df_by_topic.describe().head()
df_by_topic_count = df_by_topic.count()
#df_by_topic_count.head()
df_by_topic_count.rename(columns={'YearStart':'count'}, inplace = True)
#print(df_by_topic_count.columns)
df_by_topic_count.sort_values(by=['count'], inplace=True)
#print(df_by_topic_count)
#plt.scatter(df_by_count.index, df_by_count)
sns.set_palette('Set3',6)
ax = sns.barplot(df_by_topic_count['count'], df_by_topic_count.index)
ax.set(xlabel = 'Count', ylabel = 'Type of disease', title = 'Distribution of data by topic')

p=sns.pairplot(df)

q3 = df.quantile(0.75)
q1 = df.quantile(0.25)
iqr = q3 - q1
print('IQR for data attributes')
print(iqr)

data_out = df[~((df < (q1 - 1.5 * iqr)) |(df> (q3 + 1.5 * iqr))).any(axis=1)]
print('{} points are outliers based on IQR'.format(df.shape[0] - data_out.shape[0]))

print(df.columns)
for col in df:
  print(df[col].unique())

# Import label encoder
from sklearn import preprocessing
  
# label_encoder object knows how to understand word labels.
label_encoder = preprocessing.LabelEncoder()
  
# Encode labels in column 'species'.
df['LocationAbbr']= label_encoder.fit_transform(df['LocationAbbr'])
df['LocationDesc']= label_encoder.fit_transform(df['LocationDesc'])
df['Question']= label_encoder.fit_transform(df['Question'])
df['DataValueUnit']= label_encoder.fit_transform(df['DataValueUnit'])
df['StratificationCategory1']= label_encoder.fit_transform(df['StratificationCategory1'])
df['Stratification1']= label_encoder.fit_transform(df['Stratification1'])
df['TopicID']= label_encoder.fit_transform(df['TopicID'])
df['QuestionID']= label_encoder.fit_transform(df['QuestionID'])
df['DataValueTypeID']= label_encoder.fit_transform(df['DataValueTypeID'])
df['StratificationCategoryID1']= label_encoder.fit_transform(df['StratificationCategoryID1'])
df['StratificationID1']= label_encoder.fit_transform(df['StratificationID1'])
df['Latitude']= label_encoder.fit_transform(df['Latitude'])
df['DataValueType']= label_encoder.fit_transform(df['DataValueType'])

df.head()

"""**Test Train Split and Cross Validation methods**
Train Test Split : To have unknown datapoints to test the data rather than testing with the same points with which the model was trained. This helps capture the model performance much better.



Cross Validation: When model is split into training and testing it can be possible that specific type of data point may go entirely into either training or testing portion. This would lead the model to perform poorly. Hence over-fitting and underfitting problems can be well avoided with cross validation techniques


"""

X =  pd.DataFrame(df.drop(["Topic"],axis = 1))
y = df.Topic

#importing train_test_split
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.4,random_state=42, stratify=y)

from sklearn.naive_bayes import GaussianNB

#Create a dataframe to capture the accuracy and f1-score of various models   
log_cols=["Classifier", "Precision", "Recall", "Accuracy", "f1_score", "Latency"]
log = pd.DataFrame(columns=log_cols)

from sklearn.model_selection import cross_val_score

# Create a function for various classification models
# A class that logs the time
class Timer():
    '''
    A generic class to log the time
    '''
    def __init__(self):
        self.start_ts = None
    def start(self):
        self.start_ts = time()
    def stop(self):
        return 'Time taken: %2fs' % (time()-self.start_ts)
    
timer = Timer()

# A method that plots the Precision-Recall curve
def plot_prec_recall_vs_thresh(precisions, recalls, thresholds):
    plt.figure(figsize=(10,5))
    plt.plot(thresholds, precisions[:-1], 'b--', label='precision')
    plt.plot(thresholds, recalls[:-1], 'g--', label = 'recall')
    plt.xlabel('Threshold')
    plt.legend()


# A method to train and test the model
def run_classification(estimator, X_train, X_test, y_train, y_test, clfname, arch_name=None, pipelineRequired=False, isDeepModel=False):
    global log
    timer.start()
    # train the model
    clf = estimator
    modelname = clfname
    scores = cross_val_score(clf, X_train, y_train, cv=10) 

    if pipelineRequired :
        clf = Pipeline([('vect', CountVectorizer()),
                     ('tfidf', TfidfTransformer()),
                     ('clf', estimator),
                     ])
      
    if isDeepModel :
      
        clf.fit(X_train, y_train, validation_data=(X_test, y_test),epochs=10, batch_size=128,verbose=1,callbacks=call_backs(arch_name))
        # predict from the classifier
        y_pred = clf.predict(X_test)
        y_pred = np.argmax(y_pred, axis=1)
        y_train_pred = clf.predict(X_train)
        y_train_pred = np.argmax(y_train_pred, axis=1)
    else :
        clf.fit(X_train, y_train)
        # predict from the classifier
        y_pred = clf.predict(X_test)
        y_train_pred = clf.predict(X_train)
    
    print('Estimator:', clf)
    print('='*80)
    print('Training accuracy: %.2f%%' % (accuracy_score(y_train,y_train_pred) * 100))
    print('Testing accuracy: %.2f%%' % (accuracy_score(y_test, y_pred) * 100))
    print('='*80)

    try:
      classes = y_train.unique().tolist()
    except:
      classes = np.unique(y_train).tolist()
    print('Confusion matrix:\n')
    hmap = sns.heatmap(pd.DataFrame(confusion_matrix(y_test, y_pred), index=classes, columns=classes), annot=True, fmt="d")
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    print('='*80)
    print('Classification report:\n %s' % (classification_report(y_test, y_pred)))

    print(timer.stop(), 'to run the model')
    time_taken=timer.stop()
    log_entry = pd.DataFrame([[modelname,precision_score(y_pred, y_test,average='weighted'),recall_score(y_pred, y_test,average='weighted'),
                               accuracy_score(y_pred, y_test),f1_score(y_pred, y_test,average='weighted'),time_taken]], columns=log_cols)
    log = log.append(log_entry)

run_classification(GaussianNB(), X_train, X_test, y_train, y_test,"GaussianNB-BOW & Hierarchical")

run_classification(LogisticRegression(n_jobs=1, C=1e5,class_weight='balanced'), X_train, X_test, y_train, y_test, "LogisticRegression-BOW & Hierarchical")

run_classification(RandomForestClassifier(max_depth=15, random_state=0, class_weight="balanced"), X_train, X_test, y_train, y_test, "RandomForest-BOW &Hierarchical")

run_classification(DecisionTreeClassifier(), X_train, X_test, y_train, y_test, "DecisionTree-BOW & Hierarchical")

log.set_index(["Classifier"]).sort_values(by=['f1_score'])

#pd.set_option('display.max_colwidth', None)
import matplotlib.pyplot as plt 
from matplotlib.ticker import MaxNLocator
import seaborn as sns
sns.set(style='whitegrid',palette='muted',font_scale=1.2)
HAPPY_COLORS_PALETTE = ["#01BEFE","#FFDD00","#FF7D00","#FF006D","#ADFF02","#8F00FF"]
sns.set_palette(sns.color_palette(HAPPY_COLORS_PALETTE))

log.set_index(["Classifier"]).sort_values(by=['f1_score']).plot(kind='barh',figsize=[7,6])

knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)
y_pred = knn.predict(X_test)
print(metrics.accuracy_score(y_test, y_pred))

import plotly.express as px
import pandas as pd

fig = px.scatter_geo(df,lat='Latitude',lon='Longitude', hover_name="Topic")
#fig.update_layout(title = 'World map', title_x=0.5)
fig.show()