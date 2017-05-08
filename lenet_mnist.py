# USAGE
# python lenet_mnist.py --save-model 1 --weights output/lenet_weights.hdf5
# python lenet_mnist.py --load-model 1 --weights output/lenet_weights.hdf5

# import the necessary packages
from lenet import LeNet
from sklearn.cross_validation import train_test_split
from sklearn import datasets
from keras.optimizers import SGD
from keras.utils import np_utils
import numpy as np
import argparse
import cv2
import pandas as pd
# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-s", "--save-model", type=int, default=-1,
	help="(optional) whether or not model should be saved to disk")
ap.add_argument("-l", "--load-model", type=int, default=-1,
	help="(optional) whether or not pre-trained model should be loaded")
ap.add_argument("-w", "--weights", type=str,
	help="(optional) path to weights file")
args = vars(ap.parse_args())

# grab the MNIST dataset (if this is your first time running this
# script, the download may take a minute -- the 55MB MNIST dataset
# will be downloaded)
#print("[INFO] downloading MNIST...")
#dataset = datasets.fetch_mldata("MNIST Original")
dataset = pd.read_csv('./data/train.csv')
data = dataset.ix[:,1:]
data = data.as_matrix()
# reshape the MNIST dataset from a flat list of 784-dim vectors, to
# 28 x 28 pixel images, then scale the data to the range [0, 1.0]
# and construct the training and testing splits
data = data.reshape(data.shape[0], 28, 28)
data = data[:, np.newaxis, :, :]
(trainData, testData, trainLabels, testLabels) = train_test_split(
	data / 255.0, dataset.ix[:,0], test_size=0.33)

# transform the training and testing labels into vectors in the
# range [0, classes] -- this generates a vector for each label,
# where the index of the label is set to `1` and all other entries
# to `0`; in the case of MNIST, there are 10 class labels
trainLabels = np_utils.to_categorical(trainLabels, 10)
testLabels = np_utils.to_categorical(testLabels, 10)

# initialize the optimizer and model
print("[INFO] compiling model...")
opt = SGD(lr=0.01)
model = LeNet.build(width=28, height=28, depth=1, classes=10,
	weightsPath=args["weights"] if args["load_model"] > 0 else None)
model.compile(loss="categorical_crossentropy", optimizer=opt,
	metrics=["accuracy"])

# only train and evaluate the model if we *are not* loading a
# pre-existing model
if args["load_model"] < 0:
	print("[INFO] training...")
	model.fit(trainData, trainLabels, batch_size=128, nb_epoch=20,
		verbose=1)

	# show the accuracy on the testing set
	print("[INFO] evaluating...")
	(loss, accuracy) = model.evaluate(testData, testLabels,
		batch_size=128, verbose=1)
	print("[INFO] accuracy: {:.2f}%".format(accuracy * 100))

# check to see if the model should be saved to file
if args["save_model"] > 0:
	print("[INFO] dumping weights to file...")
	model.save_weights(args["weights"], overwrite=True)

#	randomly select a few testing digits
# for i in np.random.choice(np.arange(0, len(testLabels)), size=(10,)):
# 	# classify the digit
# 	probs = model.predict(testData[np.newaxis, i])
# 	prediction = probs.argmax(axis=1)

# 	# resize the image from a 28 x 28 image to a 96 x 96 image so we
# 	# can better see it
# 	image = (testData[i][0] * 255).astype("uint8")
# 	image = cv2.merge([image] * 3)
# 	image = cv2.resize(image, (96, 96), interpolation=cv2.INTER_LINEAR)
# 	cv2.putText(image, str(prediction[0]), (5, 20),
# 		cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2)

# 	# show the image and prediction
# 	print("[INFO] Predicted: {}, Actual: {}".format(prediction[0],
# 		np.argmax(testLabels[i])))
# 	cv2.imshow("Digit", image)
# 	cv2.waitKey(0)


# test_data = pd.read_csv('./data/test.csv')
# test_data = test_data.as_matrix()
# test_data = test_data.reshape(test_data.shape[0],28,28)
# test_data = test_data[:, np.newaxis, :, :]
# test_data =test_data/255.0
# #for i in np.random.choice(np.arange(0,test_data.shape[0]), size=(10,)):
# probs = model.predict(test_data)
# prediction = probs.argmax(axis=1)
# output = pd.DataFrame(prediction,columns=['Label'])
# output['ImageId'] = np.arange(len(output['Label']))+1
# output = pd.DataFrame(output,columns=['ImageId','Label']) 
# output.to_csv('output.csv',index = False)
test_img = cv2.imread('yes.jpg',0)
test_img = test_img[np.newaxis, :, :]
test_img = test_img/255.0
probs = model.predict(test_img[np.newaxis,:])
prediction = probs.argmax(axis=1)
print(probs)
print(prediction)
