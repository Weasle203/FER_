import numpy as np
from tensorflow.python.keras.layers import Conv2D,Reshape,MaxPooling2D
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import InputLayer
from tensorflow.python.keras.layers import Dense,Flatten
from tensorflow.python.keras.optimizers import Adam
from tensorflow.python.keras.layers import Dropout
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression


fer = np.load('inventory/fer_data.npz')
data,label = fer['data']/255,fer['label']
data = (data - np.mean(data))/np.std(data)
size_of_data = len(label)
code = np.zeros((size_of_data,7))
code[np.arange(size_of_data),label] = 1
print(label[140],code[140])
#Data Splitting
ratio = 10000
train_data =  data[:ratio]
train_cls = label[:ratio]
train_label = code[:ratio]
validation_data = data[30000:]
validation_cls = label[30000:]
validation_label = code[30000:]


#image features

img_size = 48
img_size_flat = 2304
img_shape = [48,48]
img_shape_full = [48,48,1]
num_classes = 7

def dataDistribution(label):
    '''  0: -4593 images- Angry
    1: -547 images- Disgust
    2: -5121 images- Fear
    3: -8989 images- Happy
    4: -6077 images- Sad
    5: -4002 images- Surprise
    6: -6198 images- Neutral'''
    figure = plt.figure()
    ax = figure.add_subplot(1,1,1)
    ax.hist(label, [0,1,2,3,4,5,6,7],rwidth = .25,align = 'left',color = '#CDCDEF')
    ax.set_xticks([0,1,2,3,4,5,6,7])
    ax.set_xticklabels(['Angry','Disgust','Fear','Happy','Sad','Surprise','Neutral'],
                        rotation = 30,fontsize = 'small',color = '#C200F1')
    
    
#constructiong the network
def simpleModel():
    #withh the following parameter accuracy found to be
    #acc_test = .446
    #acc_train = .4464
	model = Sequential()
	model.add(InputLayer(input_shape=(img_size_flat,)))
	model.add(Reshape(img_shape_full))
	model.add(Conv2D(filters=32,kernel_size=5,strides=(1,1),activation='relu',padding = 'same',name = 'layer_conv1'))
	model.add(MaxPooling2D(pool_size = (2,2),strides = (2,2)))
	model.add(Conv2D(filters = 64,kernel_size = 5,strides = (2,2),activation = 'relu',padding = 'same',name = 'layer_conv2' ))
	model.add(MaxPooling2D(pool_size = (2,2),strides = (2,2)))
	model.add(Conv2D(filters = 36,kernel_size = 3,strides = (2,2),activation = 'relu',padding = 'same',name = 'layer_conv3'))

	#model.add(MaxPooling2D(pool_size = (2,2),strides = (2,2)))
	model.add(Flatten())
	model.add(Dense(128,activation = 'relu'))
	model.add(Dense(num_classes,activation='softmax'))
	optimizer = Adam(.001)
	model.compile( optimizer = optimizer,loss = 'categorical_crossentropy',metrics = ['accuracy'])

	model.fit(x = train_data,y=train_label ,epochs = 3,batch_size = 64)
	result = model.evaluate(x = validation_data, y=validation_label)

	for name,value in zip(model.metrics_names,result):
	    print('{0}:  {1}'.format(name,value))


#simpleModel()

def copiedModel():
    model = Sequential()
    #model.add(InputLayer(input_shape=(img_size_flat,)))
    #print("shape outputted by the Input layer: ",model.output_shape)
    model.add(Reshape(img_shape_full))
    #print("shape outputted by the reshape layer: ",model.output_shape)
    model.add(Conv2D(filters = 64,kernel_size = 5,input_shape = img_shape_full,activation = 'relu',strides = (1,1),padding = 'same',name = 'layer_conv1'))
    #print("shape outputted by the first convolutional layer: ",model.output_shape)
    model.add(MaxPooling2D(pool_size = (3,3),strides = (2,2)))
    #print("shape outputted by the MaxPooling Layer layer: ",model.output_shape)
    model.add(Conv2D(filters = 64,kernel_size = 5,activation = 'relu',strides = (2,2),padding = 'same',name = 'layer_conv2'))
    #print("shape outputted by the second Convolutional layer: ",model.output_shape)
    model.add(MaxPooling2D(pool_size = (3,3),strides = (2,2)))
    #print("shape outputted by the second Maxpooling layer: ",model.output_shape)
    model.add(Conv2D(filters = 128,kernel_size = 4,activation = 'relu'))
    #print("shape outputted by the convolutional layer: ",model.output_shape)
    model.add(Dropout(rate = .3))
    #print("shape outputted by the Dropout layer: ",model.output_shape)
    model.add(Flatten())
    #print("shape outputted by the after flatten layer: ",model.output_shape)
    model.add(Dense(3072,activation = 'relu'))
    #print("shape outputted by the after Dense layer: ",model.output_shape)
    model.add(Dense(num_classes,activation = 'softmax'))
    optimizer = Adam(.001)
    model.compile(optimizer = optimizer,loss = 'categorical_crossentropy', metrics = ['accuracy'])
    model.fit(x= train_data,y=train_label,epochs = 18,batch_size = 50)
    result = model.evaluate(x = validation_data,y = validation_label)
    for name,value in zip(model.metrics_names,result):
        print('{0}: {1}'.format(name,value))
    return model

m = copiedModel()
#model.add(Regression(optimizer = 'momentum',loss = 'categoriacal_crossentropy'))   

#m.save('Models/fer_cnn_model_1.h5')
m.save_weights('Models/fer_cnn_model_2_weights.h5')
with open('Models/fer_cnn_model_2_json.json','w') as json_file:
    json_file.write(m.to_json())
#print(m.summary())



#np.savez('inventory/fer.npz',rtrain = rtrain,rtest = rtest,rcls = train_cls,tcls = validation_cls)
