import streamlit as st 
from sklearn import datasets
import numpy as np

from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPClassifier

#https://www.youtube.com/watch?v=Klqn--Mu2pE

st.title("Streamlit 示例")
st.write("教程"" [link](https://www.youtube.com/watch?v=Klqn--Mu2pE)")

st.write("""
	#探索不同的分类器
	哪一个最好
	""")

dataset_name = st.sidebar.selectbox("选择数据集", ("虹膜","乳腺癌","酒","糖尿病"))

classifier_name = st.sidebar.selectbox("选择分类器", ("KNN","SVM","随机森林分类器","神经网络"))


def get_dataset(dataset_name):
	if dataset_name == "虹膜":
		data = datasets.load_iris()
	elif dataset_name == "乳腺癌":
		data = datasets.load_breast_cancer()
	elif dataset_name == "酒":
		data = datasets.load_wine()
	else: #dataset_name == "糖尿病":
		data = datasets.load_diabetes()
	# else:
	# 	data = datasets.fetch_california_housing()
	X = data.data 
	y = data.target 
	return X,y

X,y = get_dataset(dataset_name)
st.write("数据集的形状", X.shape)
st.write("标签数量", len(np.unique(y)))

def add_parameter_ui(clf_name):
	params = dict()
	if clf_name =="KNN":
		K = st.sidebar.slider("K(最近邻的数量)",1,15)
		params["K"] = K 
	elif clf_name == "SVM":
		C = st.sidebar.slider("C",0.01,10.0)
		params["C"] = C
	elif clf_name == "神经网络":
		alpha = st.sidebar.slider("alpha",0.01,0.99)
		params["alpha"]=alpha
	else:
		max_depth = st.sidebar.slider("max_depth",2,15)
		n_estimators = st.sidebar.slider("n_estimators",1,100)
		params["max_depth"] = max_depth
		params["n_estimators"] = n_estimators
		
	return params 


params = add_parameter_ui(classifier_name)



def get_classifier(clf_name, params):
	if clf_name == "KNN":
		clf = KNeighborsClassifier(n_neighbors = params["K"]);
	elif clf_name == "SVM":
		clf = SVC(C=params["C"])
		# params["C"] = C
	elif clf_name == "神经网络":
		clf = MLPClassifier(solver='lbfgs', alpha=params["alpha"],
          					random_state=1)
	else:
		# max_depth = st.sidebar.slider("max_depth", 2,15)
		clf = RandomForestClassifier(n_estimators = params["n_estimators"],
			max_depth = params["max_depth"], random_state=1234)
	return clf

clf = get_classifier(classifier_name, params)

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state = 1234)

clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

acc = accuracy_score(y_test,y_pred)
st.write(f"分类器 = {classifier_name}")
st.write(f"准确性 = {acc}")

#plot 
pca = PCA(2)
X_projected = pca.fit_transform(X)

x1 = X_projected[:,0]
x2 = X_projected[:,1]

fig = plt.figure()
plt.scatter(x1,x2,c=y,alpha=0.8, cmap="viridis")
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.colorbar()

st.pyplot(fig)