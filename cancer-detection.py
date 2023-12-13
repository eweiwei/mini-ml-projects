import streamlit as st
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
from sklearn import datasets, linear_model, preprocessing
from sklearn.linear_model import LogisticRegression

st.write("""
	# Sample of **Wisconsin Breast Cancer Data**
	""")
st.write('---')

names = ['id','thick','size','shape','marg','cell_size','bare',
         'chrom','normal','mit','class']
df = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/' +
                 'breast-cancer-wisconsin/breast-cancer-wisconsin.data',
                names=names,na_values='?',header=None)
df = df.dropna()

st.write(df.head(5))

st.sidebar.header('Specify Input Parameters')

def user_input_features():
	THICK = st.sidebar.slider('THICK', 1, 10, int(df.thick.mean()))
	SIZE = st.sidebar.slider('SIZE', 1, 10, 5)
	SHAPE = st.sidebar.slider('SHAPE', 1, 10, 5)
	MARG = st.sidebar.slider('MARG', 1, 10, int(df.marg.mean()))
	CELL_SIZE = st.sidebar.slider('CELL_SIZE', 1, 10, int(df.cell_size.mean()))
	BARE = st.sidebar.slider('BARE', 1, 10, int(df.bare.mean()))
	CHROM = st.sidebar.slider('CHROM', 1, 10, int(df.chrom.mean()))
	NORMAL = st.sidebar.slider('NORMAL', 1, 10, int(df.normal.mean()))
	MIT = st.sidebar.slider('MIT', 1, 10, int(df.mit.mean()))
	data = {
		'THICK': THICK,
		'SIZE': SIZE,
		'SHAPE': SHAPE,
		'MARG': MARG,
		'CELL_SIZE': CELL_SIZE,
		'BARE': BARE,
		'CHROM': CHROM,
		'NORMAL': NORMAL,
		'MIT': MIT,
	}
	features = pd.DataFrame(data, index=[0])
	return features

dn = user_input_features()

st.header('SPECIFIED INPUT PARAMETERS')
st.write(dn)
st.write('---')

# Converting to a zero-one indicator.
yraw = np.array(df['class'])
BEN_VAL = 2   # value in the 'class' label for benign samples
MAL_VAL = 4   # value in the 'class' label for malignant samples
y = (yraw == MAL_VAL).astype(int)
Iben = (y==0)
Imal = (y==1)

# Get two predictors
xnames =['size','marg'] 
X = np.array(df[xnames])
    
# Compute the bin edges for the 2d histogram
x0val = np.array(list(set(X[:,0]))).astype(float)
x1val = np.array(list(set(X[:,1]))).astype(float)
x0, x1 = np.meshgrid(x0val,x1val)
x0e= np.hstack((x0val,np.max(x0val)+1))
x1e= np.hstack((x1val,np.max(x1val)+1))

def metrics(y, y_hat):
    # TODO: 2)
    # calculate TP, TN, FP and FN rates
    
    TP = np.sum((y + y_hat) == 2) # y = 1, yhat = 1
    TN = np.sum((y + y_hat) == 0) # y = 0, yhat = 0
    FP = np.sum((y - y_hat) == -1) # y = 0, yhat = 1
    FN = np.sum((y - y_hat) == 1) # y = 1, yhat = 0
    
    return TP, TN, FP, FN

clf = LogisticRegression(random_state=0)
clf.fit(X, y)
y_pred_clf = clf.predict(X)

st.write(y_pred_clf)

TP, TN, FP, FN = metrics(y, y_pred_clf)
# calculate the following metrics using TP, TN, FP, and FN
accuracy = (TP+TN)/(TP+TN+FP+FN)
sensitivity = TP/(TP+FN)
precision = TP/(TP+FP)

metricsData = {
	'ACCURACY': accuracy,
	'SENSITIVITY': sensitivity,
	'PRECISION': precision
}

modelMetrics = pd.DataFrame(metricsData, index=[0])

st.header('BREAST CANCER MODEL PREDICTION AND METRICS')

st.write(modelMetrics)

st.write('---')

# Make a plot for each class

st.header('BREAST CANCER GRAPHS')
plt.title('MARG VS SIZE | BENIGN AND MALIGN GRAPH')

yval = list(set(y))
color = ['g','r']
plt.figure(figsize=(8,6))
for i in range(len(yval)):
    I = np.where(y==yval[i])[0]
    count, x0e, x1e = np.histogram2d(X[I,0],X[I,1],[x0e,x1e])
    x0, x1 = np.meshgrid(x0val,x1val)
    plt.scatter(x0.ravel(), x1.ravel(), s=2*count.ravel(),alpha=0.5,
                c=color[i],edgecolors='none')
plt.ylim([0,14])
plt.legend(['benign','malign'], loc='upper right')
plt.xlabel(xnames[0], fontsize=16)
plt.ylabel(xnames[1], fontsize=16)
st.set_option('deprecation.showPyplotGlobalUse', False)
st.pyplot(bbox_inches='tight')

st.write('---')