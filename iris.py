import streamlit as st
import pandas as pd
import shap
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from sklearn import datasets,preprocessing
from sklearn.linear_model import LogisticRegression

st.write("""
# Simple Iris Flower Prediction App
This app predicts the **Iris flower** type!
""")

st.sidebar.header('User Input Parameters')

def user_input_features():
    sepal_length = st.sidebar.slider('Sepal length', 4.3, 7.9, 5.4)
    sepal_width = st.sidebar.slider('Sepal width', 2.0, 4.4, 3.4)
    petal_length = st.sidebar.slider('Petal length', 1.0, 6.9, 1.3)
    petal_width = st.sidebar.slider('Petal width', 0.1, 2.5, 0.2)
    data = {'sepal_length': sepal_length,
            'sepal_width': sepal_width,
            'petal_length': petal_length,
            'petal_width': petal_width}
    features = pd.DataFrame(data, index=[0])
    return features

df = user_input_features()

st.subheader('User Input parameters')
st.write(df)

iris = datasets.load_iris()
X = iris.data
Y = iris.target

clf = LogisticRegression()
clf.fit(X, Y)

prediction = clf.predict(df)
prediction_proba = clf.predict_proba(df)

st.subheader('Class labels and their corresponding index number')
st.write(iris.target_names)

st.subheader('Prediction')
st.write(iris.target_names[prediction])
#st.write(prediction)

st.subheader('Prediction Probability')
st.write(prediction_proba)
st.write('---')


st.header('How Different Iris Properties Contribute to the Classification')

def slider_features():
	sepal_length = st.slider('Sepal Length', 4.3, 7.9,5.4)
	sepal_width = st.slider('Sepal Width', 2.0, 4.4, 3.4)
	iris_data = {
	'sepal_length': sepal_length,
	'sepal_width': sepal_width}
	sliders = pd.DataFrame(iris_data, index=[0])
	return sliders

sf = slider_features()

st.subheader('User Input Parameters')
st.write(sf)

iris_graphs = datasets.load_iris()
x = iris_graphs.data[:,:2] # using only first two features (sepal length, sepal width)
y = iris_graphs.target



### CODE FROM SKLEARN IRIS DEMO ###

# Plot the decision boundary. For that, we will assign a color to each
# point in the mesh [x_min, x_max]x[y_min, y_max].
logreg = LogisticRegression(random_state=0)
logreg.fit(x, y)

predictions = logreg.predict(sf)
prediction_prob = logreg.predict_proba(sf)

st.subheader('Class labels and their corresponding index number')
st.write(iris_graphs.target_names)

st.subheader('Prediction')
st.write(iris_graphs.target_names[predictions])
#st.write(prediction)

st.subheader('Prediction Probability')
st.write(prediction_prob)
st.write('---')

plt.figure(figsize=(8,6))
x_min, x_max = x[:, 0].min() - .5, x[:, 0].max() + .5
y_min, y_max = x[:, 1].min() - .5, x[:, 1].max() + .5
h = .02  # step size in the mesh
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
Z = logreg.predict(np.c_[xx.ravel(), yy.ravel()])

# normalizing the data
Xs = preprocessing.scale(x)

# Plot it again and compare it with the data before normalization

# y==0 picks the indicies corresponding to y equal zero
plt.figure(figsize=(8,6))
plt.title('Sepal Width vs Sepal Length')
plt.rcParams['figure.figsize'] = [6, 4]
plt.plot(x[y==0,0], x[y==0,1], 'o', markerfacecolor=(1,0,0,1), markeredgecolor='black')
plt.plot(x[y==1,0], x[y==1,1], 'o', markerfacecolor=(0,1,0,1), markeredgecolor='black')
plt.plot(x[y==2,0], x[y==2,1], 'o', markerfacecolor=(0,0,1,1), markeredgecolor='black')
# remember this a 3-class classification problem!

plt.xlabel('Sepal Length')
plt.ylabel('Sepal Width')

st.set_option('deprecation.showPyplotGlobalUse', False)
st.pyplot(bbox_inches='tight')

# Put the result into a color plot
Z = Z.reshape(xx.shape)
plt.figure(1, figsize=(4, 3))
plt.title('Sepal Width vs Sepal Length with Classification')
plt.pcolormesh(xx, yy, Z, cmap=plt.cm.Paired, shading='auto')

# Plot also the training points
plt.scatter(x[:, 0], x[:, 1], c=y, edgecolors='k', cmap=plt.cm.Paired)
plt.xlabel('Sepal length')
plt.ylabel('Sepal width')

plt.xlim(xx.min(), xx.max())
plt.ylim(yy.min(), yy.max())
plt.xticks(())
plt.yticks(())

st.set_option('deprecation.showPyplotGlobalUse', False)
st.pyplot(bbox_inches='tight')

st.write('---')