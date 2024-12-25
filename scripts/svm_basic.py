from sklearn import svm
import numpy as np
import joblib
import sklearn
# from sklearn.preprocessing import MinMaxScaler
import time

x = np.loadtxt("total_input_shuffle.txt", dtype = float, unpack = False)
# x = sklearn.preprocessing.scale(x)


scaler = sklearn.preprocessing.StandardScaler().fit(x)
# print scaler
# scaler = np.array(scaler)
#
# np.savetxt("scaler.txt", scaler)
x = scaler.transform(x)
# scalar = MinMaxScaler()
# x = scalar.fit_transform(x)
# min_max_scalar = sklearn.preprocessing.MinMaxScalar()
# sklearn.preprocessing
# x = min_max_scalar.fit_transform(x)
print("shape of x: ",x.shape)
# X = [[0,0],[1,1]]
y = np.loadtxt("total_label_v2_shuffle.txt",dtype = float, unpack = False)
y = y.astype(int)
print("shape of y: ",y.shape)

# x_shuffle, y_shuffle = sklearn.utils.shuffle(x, y, random_state = 1)
x_shuffle = x
y_shuffle = y


# np.savetxt("x_shuffle.txt", x_shuffle, fmt = '%10.4f')
# np.savetxt("y_shuffle.txt", y_shuffle, fmt = '%10.4f')
# np.savetxt("x_preprocessed.txt", x, fmt = '%10.4f')


# print x[1,:].dtype
# y = y.reshape(0)
# print y.shape
# y1 = np.loadtxt("label2.txt",dtype= float, unpack = False)

# input = np.loadtxt("ear_output.txt",dtype=float,unpack=False)

number_train = int(x_shuffle.shape[0] / 10 * 9)
number_test = x_shuffle[0] - number_train


print("number of training set ", number_train)

train = x_shuffle[0:number_train, :]
train_label = y_shuffle[0:number_train]

test = x_shuffle[number_train : , :]
test_label = y_shuffle[number_train : ]
# input = np.array(input)

print ("number of testing set: ", test_label.shape[0])


clf = svm.SVC(C = 50)

#=============================================#
#clf = svm.SVC(kernel = 'linear', C = 100)

# clf = svm.SVC(C = 10000000000)
# clf = svm.LinearSVC(C=100)
# result = clf.fit(x[:, -1], y[:,-1])

print ("=========start training========")
time1 = time.time()
result = clf.fit(train,train_label)

print (result)

time2 = time.time()

print ("training time(s):", time2 - time1)


# test = x[14544,:].reshape(1,13)
# test = train

#=============================================#

predict_test = clf.predict(test)
predict_train = clf.predict(train)

#=============================================#

index_1_in_test = np.where(test_label == 1)
index_1_in_test = np.array(index_1_in_test, dtype = int)
number_1_in_test = index_1_in_test.shape[1]


index_1_in_train = np.where(train_label == 1)
index_1_in_train = np.array(index_1_in_train, dtype = int)
number_1_in_train = index_1_in_train.shape[1]

print ("number of 1s in train dataset ",number_1_in_train)
print ("number of 1s in test dataset ",number_1_in_test)

#=============================================#

print (predict_test)
print (test_label)
# print x[14544, :]

predict_train = predict_train.astype(int)
predict_test = predict_test.astype(int)
np.savetxt("predicted.txt", predict_test, fmt = '%10.4f')

correct = 0
correct_train = 0

for i in range(0, predict_train.shape[0]):
    if str(train_label[i]) == str(predict_train[i]):
        if str(predict_train[i]) == "1":
            correct_train += 1


for i in range(0,predict_test.shape[0]):
    if str(test_label[i]) == str(predict_test[i]):
        if str(predict_test[i]) == "1":
            correct += 1
            # print "correct"

        # correct += 1
        # print "correct", correct
    # else:
        # print "false"

print ("\ntrain accuracy for '1': ", correct_train / float(number_1_in_train))
print ("test accuracy for '1': ", correct/ float(number_1_in_test))





print ("train accuracy for all: " + str(np.mean(predict_train.astype(np.float64) == train_label.astype(np.float64))))
print ("test accuracy for all: " + str(np.mean(predict_test.astype(np.float64) == test_label.astype(np.float64))))


joblib.dump(clf, 'my_model.pkl')
# joblib.dump(clf, 'filename.pkl')