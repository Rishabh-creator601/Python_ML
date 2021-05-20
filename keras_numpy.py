from keras.models import Sequential
from keras.layers import Dense
import numpy as np

#Loading Data

train = np.random.random((1000,3))
test = np.random.random((1000,3))
labels = np.random.randint(2,size=(1000,1))

#Model network

model = Sequential()
model.add(Dense(5,input_dim=3, activation='sigmoid'))
model.add(Dense(4,activation='sigmoid'))
model.add(Dense(1, activation='softmax'))

model.compile(loss="binary_crossentropy",optimizer="adam",metrics=["accuracy"])

model.fit(train,labels,epochs=10,batch_size=32)

#Result : accuracy

loss,acc = model.evaluate(test,labels)
print(acc * 100) 