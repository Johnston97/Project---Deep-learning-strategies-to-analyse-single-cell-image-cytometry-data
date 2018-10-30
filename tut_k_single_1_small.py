import cv2

import numpy as np
import os 
import keras




from keras import backend as K
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D, SeparableConv2D
from keras.utils import np_utils
import itertools

import matplotlib.pyplot as plt
import matplotlib
#from matplotlib import save_model_to_file
from keras.utils.vis_utils import plot_model

from numpy import *
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
import time
import keras.utils.vis_utils 
from keras.utils.vis_utils import plot_model


import keras.backend as K

config = K.tf.ConfigProto()
config.gpu_options.allow_growth = True
session = K.tf.Session(config=config)

start = time.time()


def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')


#May need to move all pics into a single folder
path1 = 'C:/Users/Luke/Documents/Computer_Science/CSC3095/Software/Data/train_channels_tsne/ch3'
path2 = 'C:/Users/Luke/Documents/Computer_Science/CSC3095/Software/Data/train_channels_tsne/ch4'
path3 = 'C:/Users/Luke/Documents/Computer_Science/CSC3095/Software/Data/train_channels_tsne/ch6'

#BLACK = [0,0,0]

#size of resized images
img_rows = 66
img_cols = 66


listing = os.listdir(path1)
#print(listing)
num_samples=size(listing)
#print(num_samples)


#0 rather than -1 loads the image as 8-bit rather than 16-bit
ch3_matrix = array([array(cv2.resize(cv2.copyMakeBorder(cv2.imread(path1 + '\\' + im2,-1),
										int((img_rows - (cv2.imread(path1 + '\\' + im2,-1).shape[0]))/2),
										int((img_rows - (cv2.imread(path1 + '\\' + im2,-1).shape[0]))/2),
										int((img_cols - (cv2.imread(path1 + '\\' + im2,-1).shape[1]))/2),
										int((img_cols - (cv2.imread(path1 + '\\' + im2,-1).shape[1]))/2),
										cv2.BORDER_CONSTANT, value=int(cv2.imread(path1 + '\\' + im2,-1)[0][0])),(img_rows,img_cols))).flatten()
				for im2 in listing])		
				
				
print(ch3_matrix.shape)
print(ch3_matrix[0][0])
	
				
				
mean = np.mean(ch3_matrix)
print("mean = ", mean)
sd = np.std(ch3_matrix)
print("std = ", sd)

ch3_matrix = ((ch3_matrix - mean) / sd)

print("ch3max value", np.amax(ch3_matrix))
print("ch3min value", np.amin(ch3_matrix))		

		
#print("Channel3 " , ch3_matrix)
print(ch3_matrix.shape)

ch4_matrix = array([array(cv2.resize(cv2.copyMakeBorder(cv2.imread(path2 + '\\' + im2,-1),
										int((img_rows - (cv2.imread(path2 + '\\' + im2,-1).shape[0]))/2),
										int((img_rows - (cv2.imread(path2 + '\\' + im2,-1).shape[0]))/2),
										int((img_cols - (cv2.imread(path2 + '\\' + im2,-1).shape[1]))/2),
										int((img_cols - (cv2.imread(path2 + '\\' + im2,-1).shape[1]))/2),
										cv2.BORDER_CONSTANT, value=int(cv2.imread(path2 + '\\' + im2,-1)[0][0])),(img_rows,img_cols))).flatten()
				for im2 in listing])
				
				
mean = np.mean(ch4_matrix)
print("mean = ", mean)
sd = np.std(ch4_matrix)
print("std = ", sd)

ch4_matrix = ((ch4_matrix - mean) / sd)
print("ch4max value", np.amax(ch4_matrix))
print("ch4min value", np.amin(ch4_matrix))
#print("Channel6 " , ch6_matrix)
print(ch4_matrix.shape)


ch6_matrix = array([array(cv2.resize(cv2.copyMakeBorder(cv2.imread(path3 + '\\' + im2,-1),
										int((img_rows - (cv2.imread(path3 + '\\' + im2,-1).shape[0]))/2),
										int((img_rows - (cv2.imread(path3 + '\\' + im2,-1).shape[0]))/2),
										int((img_cols - (cv2.imread(path3 + '\\' + im2,-1).shape[1]))/2),
										int((img_cols - (cv2.imread(path3 + '\\' + im2,-1).shape[1]))/2),
										cv2.BORDER_CONSTANT, value=int(cv2.imread(path3 + '\\' + im2,-1)[0][0])),(img_rows,img_cols))).flatten()
				for im2 in listing])
				
				
mean = np.mean(ch6_matrix)
print("mean = ", mean)
sd = np.std(ch6_matrix)
print("std = ", sd)

ch6_matrix = ((ch6_matrix - mean) / sd)

#print("Channel6 " , ch6_matrix)
print(ch6_matrix.shape)

print("ch6max value", np.amax(ch6_matrix))
print("ch6min value", np.amin(ch6_matrix))


#matrix of all channels per image
comb_matrix = np.zeros((num_samples,(img_rows * img_cols * 3)))

#print("Size = ", len(im_matrix))
for i in range(0,num_samples):
	comb_matrix[i] = np.concatenate((ch3_matrix[i],ch4_matrix[i], ch6_matrix[i]), axis =0)

print("Combined = " , comb_matrix)
print(comb_matrix.shape)
print(comb_matrix[60])



#img1 = comb_matrix[0].reshape(2, 66, 66)
#img1 = img1[0]
 
#plt.imshow(img1)
#plt.show()

#Can normalise here or for each matrix


#Label images
#create an array of 1s
label = np.ones((num_samples,), dtype = int)
label[0:14]=0 #Ana
label[14:114]=1 #G1
label[114:214]=2 #G2
label[214:282]=3 #Meta
label[282:382]=4 #Pro
label[382:482]=5 #S
label[482:509]=6 #Telo


data, Label = shuffle(comb_matrix, label, random_state=2)
#data, Label = comb_matrix, label
#train_data = [data,Label]

batch_size = 32
num_classes = 7
epochs = 15
input_shape = (3, img_rows, img_cols)


def base_model():
	model = Sequential()
	model.add(Conv2D(64, kernel_size=(3, 3),
                 activation='relu',
                 input_shape=input_shape,data_format='channels_first'))
	model.add(Conv2D(64, kernel_size=(3, 3),
                 activation='relu',
                 input_shape=input_shape,data_format='channels_first'))
	model.add(MaxPooling2D(pool_size=(2, 2), data_format='channels_first'))
	
	model.add(Conv2D(128, kernel_size=(3, 3),
                 activation='relu',
                 input_shape=input_shape,data_format='channels_first'))		
	model.add(Conv2D(128, kernel_size=(3, 3),
                 activation='relu',
                 input_shape=input_shape,data_format='channels_first'))
	model.add(MaxPooling2D(pool_size=(2, 2), data_format='channels_first'))
		
	model.add(Conv2D(256, kernel_size=(3, 3),
                 activation='relu',
                 input_shape=input_shape,data_format='channels_first'))		
	model.add(Conv2D(256, kernel_size=(3, 3),
                 activation='relu',
                 input_shape=input_shape,data_format='channels_first'))
	model.add(MaxPooling2D(pool_size=(2, 2), data_format='channels_first'))	
		
	model.add(Dropout(0.25))
	model.add(Flatten())
	
	model.add(Dense(512, activation='relu'))
	model.add(Dropout(0.5))
	model.add(Dense(num_classes, activation='softmax'))

	model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adadelta(),
              metrics=['accuracy'])
	return model

cnn_m = base_model()
cnn_m.summary()

#plot_model(cnn_m, to_file='model.png', show_shapes=True)




from sklearn.model_selection import KFold
kf = KFold(n_splits=5)

KFold(n_splits=5, random_state=5, shuffle=False)
average_accuracy = []
average_loss = []
sum_y_pred = []
sum_Y_test = []
it = 0

for train_index, test_index in kf.split(data):
		#print("TRAIN:", train_index, "TEST:", test_index)
		X_train, X_test = data[train_index], data[test_index]
		y_train, y_test = Label[train_index], Label[test_index]
		
		#if(y_train = 0 or y_train = 3 or y_train = 6):
		#	X_train.transform
		
		X_train = X_train.reshape(X_train.shape[0], 3, img_rows, img_cols)
		X_test = X_test.reshape(X_test.shape[0],3,img_rows,img_cols)
		X_train = X_train.astype('float32')
		X_test = X_test.astype('float32')
		#X_train /= 4096
		#X_test /= 4096
		Y_train = keras.utils.to_categorical(y_train, num_classes)
		Y_test = keras.utils.to_categorical(y_test, num_classes)

		
		cnn_m.fit(X_train, Y_train,
		batch_size=batch_size,
        epochs=epochs,
		verbose=2,
        validation_data=(X_test, Y_test))

		score = cnn_m.evaluate(X_test, Y_test, verbose=0)
		
		from sklearn.metrics import confusion_matrix
		Y_pred = cnn_m.predict(X_test,verbose=2)
		y_pred = np.argmax(Y_pred,axis=1)
		sum_y_pred.extend(y_pred)
		sum_Y_test.extend(Y_test)
		
		
		for ix in range(0):
			print (ix, confusion_matrix(np.argmax(Y_test,axis=1), y_pred)[ix].sum())
		print (confusion_matrix(np.argmax(Y_test,axis=1), y_pred))
		print('Test loss:', score[0])
		print('Test accuracy:', score[1])
		average_accuracy.append(score[0])
		average_loss.append(score[1])
#it = it + 1
		

average_accuracy = sum(average_accuracy) / len(average_accuracy)
average_loss = sum(average_loss) / len(average_loss)

print("Combined Confusion Matrix")
print("Combined Confusion Matrix")
#for ix in range(7):
	#print (ix, confusion_matrix(np.argmax(sum_Y_test,axis=1), sum_y_pred)[ix].sum())
	
#print("y_pred = ", sum_y_pred)
#print("y_test = ", sum_Y_test)


cm = confusion_matrix(np.argmax(sum_Y_test,axis=1), sum_y_pred)

plt.figure()
plot_confusion_matrix(cm, classes=["Ana","G1","G2","Meta","Pro","S","Telo"],
                      title='Confusion matrix, without normalization')

# Plot normalized confusion matrix
#plt.figure()
#plot_confusion_matrix(cnf_matrix, classes=class_names, normalize=True,
#                    title='Normalized confusion matrix')

plt.show()
print('Average test loss:', average_accuracy)
print('Average test accuracy:', average_loss)

cnn_m.save("model_small.h5")
end = time.time()
print(end-start)