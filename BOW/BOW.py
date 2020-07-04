import cv2 as cv
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import _pickle as Pickle


def load_images(path):
    imgs_list = []
    count = 0
    for currentpath, folders, files in os.walk(path):
        for folder in folders:
            # imgs_list[folder] = []
            for currentfol, subfolders, images in os.walk(os.path.join(currentpath, folder)):
                for img in images:
                    if folder == "Faces":
                        count += 1
                    image = plt.imread(os.path.join(currentfol, img))
                    imgs_list.append(
                        image)
    return imgs_list, count


def extract_features(image, extractor):
    keypoints, descriptors = extractor.detectAndCompute(image, None)
    return keypoints, descriptors


def generate_descriptors(train_set, extractor):
    # print(image_set)
    descriptors_list = []
    # print(train_set)
    img_features = []

    features = []
    for img in train_set:
        img = cv.normalize(img, None, 0, 255,
                           cv.NORM_MINMAX).astype('uint8')
        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        keypoints, descriptors = extract_features(gray, extractor)
        descriptors_list.extend(descriptors)
        features.append(descriptors)

    return descriptors_list, features


def train_kmeans(descriptors_list, path):
    kmeans = KMeans(n_clusters=100)
    kmeans.fit(descriptors_list)
    Pickle.dump(kmeans, open(path, 'wb'))


def generate_histograms(images_features, kmeans):

    visual_words = kmeans.cluster_centers_
    histograms = []
    for img in images_features:
        histogram = np.zeros(len(visual_words))
        img_features = kmeans.predict(img)

        for cluster in img_features:
            histogram[cluster] += 1
        # print(histogram)
        histograms.append(histogram)

    vStack = np.array(histograms[0])
    for remaining in histograms[1:]:
        vStack = np.vstack((vStack, remaining))

    return vStack


def predict(images,  kmeans, svm):
    extractor = cv.xfeatures2d.SIFT_create()
    descriptors_list, images_features = generate_descriptors(images, extractor)
    X_test = generate_histograms(images_features, kmeans)
    X_test = StandardScaler().fit_transform(X_test)

    print("Xtest shape", X_test.shape)

    Y_predicted = svm.predict(X_test)
    return Y_predicted


def train_BOW(path):

    #train_set, faces_count = load_images(".\BOW\images\\train 2")
    train_set, faces_count = load_images(path)
    extractor = cv.xfeatures2d.SIFT_create()
    descriptors_list, images_features = generate_descriptors(
        train_set, extractor)

    print("descriptor done")

    #train_kmeans(descriptors_list, "kmeans4.pkl")
    print("kmeans done")
    # change path of Kmeans model
    kmeans = Pickle.load(open("kmeans2.pkl", 'rb'))

    X_train = generate_histograms(images_features, kmeans)
    X_train = StandardScaler().fit_transform(X_train)
    print("Xtrain shape", X_train.shape)
    Y_train = np.zeros((X_train.shape[0]))
    Y_train[0:faces_count] = 1
    print("ytrain", Y_train)

    clf = SVC()  # make classifier object
    clf.fit(X_train, Y_train)
    Pickle.dump(clf, open("SVM.pkl", 'wb'))

    print("Training done")
    return kmeans, clf


kmeans, clf = train_BOW(".\BOW\images\\train 2")

# TEST
test_set, faces_count_test = load_images(".\BOW\images\\test 3")

Y_predicted = predict(test_set, kmeans, clf)

# Calculating accuracy
# Y_test = np.zeros((len(test_set)))
# Y_test[0:faces_count_test] = 1
# accuracy = accuracy_score(Y_test, Y_predicted, normalize=True)
# print("accuarcy=", accuracy)
# accuracy_face = accuracy_score(
#     Y_test[0:faces_count_test], Y_predicted[0:faces_count_test], normalize=True)
# print("accuracy face=", accuracy_face)
