from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
# import matplotlib.pyplot as plt



mnist = load_digits()

x = mnist.data
y = mnist.target


x_train,x_test,y_train,y_test = train_test_split(x,y)
mlp = MLPClassifier() # 0.98 
mlp.fit(x_train,y_train)
print(mlp.score(x_test,y_test))


'''

def plot(value):
	plt.matshow(x_train[value].reshape((8,8)),cmap=plt.cm.binary)m
	plt.xlabel(y_train[value])
	print("label -> ",y_train[value])
	plt.show()
	

plot(10)


'''










