# feature_vector -> X (Features vectors) 
# classes -> Y (Classes)

import re
import numpy as np
from sklearn.cross_validation import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, precision_score, recall_score, precision_recall_fscore_support
from random import randrange

f1 = []
f2 = []
f3 = []
f4 = []
f5 = []


f1f2 = list()
f1f2f3 = list()
f1f2f3f4 = list()
f1f2f3f4f5 = list()

# -------- Sentence Cleaning -------- 
def cleanSentence(sentence):	
	clean_sentence = ''
	clean_sentences = []
	for i in range(len(sentence)):
		clean_sentence = ''
		for j in range(len(sentence[i])):
			if sentence[i][j].isnumeric() or sentence[i][j].isalpha() or sentence[i][j]==' ':
				clean_sentence += sentence[i][j]
			elif j+1!=len(sentence[i]):
				if sentence[i][j+1].isalpha() or sentence[i][j+1].isnumeric():
					clean_sentence += ' '
					# continue
		clean_sentences.append(clean_sentence)
	return clean_sentences

# -------- 10-Fold Cross Validation -------- 

def cross_validation_split(dataset, datasetY, folds=3):
	dataset_split = list()
	datasetY_split = list()
	dataset_copy = list(dataset)
	datasetY_copy = list(datasetY)
	fold_size = int(len(dataset) / folds)
	for i in range(folds):
		fold = list()
		foldY = list()
		while len(fold) < fold_size:
			index = randrange(len(dataset_copy))
			fold.append(dataset_copy.pop(index))
			foldY.append(datasetY_copy.pop(index))
		dataset_split.append(fold)
		datasetY_split.append(foldY)
	return dataset_split, datasetY_split

# -------- Data Loading -------- 

words = []
ngrams = {}
clean_sentences = []

with open("corpus.txt", "r") as f:
	words = f.read().lower()
	words = re.split("\t|\n", words)
	# print(words)

sentences_zero = []
sentences_one = []
classes_zero = []
classes_one = []

# -------- Dividing the sentences in two classes -------- 

for i in range(0, len(words)-1, 2):
	if i%2==0:
		if words[i+1]=='0':
			sentences_zero.append(words[i])
			classes_zero.append(words[i+1])
		elif words[i+1]=='1':
			sentences_one.append(words[i])
			classes_one.append(words[i+1])

# Call cleanSentence
sentences_zero = cleanSentence(sentences_zero)
sentences_one = cleanSentence(sentences_one)

# -------- N-Gram -------- 

def ngram(n,f1,f2,f3,f4,f5):
	if n<6:
		print("\nFor N = ", n)

		# -------- List of all Unique words -------- 

		unique_words_dict = {}
		temp = ''

		# For Class = Zero

		for i in sentences_zero:
			temp1 = list(i.split())
			for j in range(len(temp1)):
				temp = ''
				if j+n-1<len(temp1) and n!=1:	
					for k in range(n):
						temp += temp1[j+k]
				elif temp=='' and n==1:
					temp = temp1[j]
				# if len(temp)>1:	
				if temp not in unique_words_dict.keys():
					unique_words_dict[temp] = []

		# For Class = One

		for i in sentences_one:
			temp1 = list(i.split())
			for j in range(len(temp1)):
				temp = ''
				if j+n-1<len(temp1) and n!=1:	
					for k in range(n):
						temp += temp1[j+k]
				elif temp=='' and n==1:
					temp = temp1[j]
				# if len(temp)>1:	
				if temp not in unique_words_dict.keys():
					unique_words_dict[temp] = []

		unique_words = list(unique_words_dict.keys())
		unique_words = sorted(unique_words)
		# print(unique_words)

		# -------- Feature Vector Initialization -------- 

		feature_vector = np.zeros((len(sentences_zero)+len(sentences_one),len(unique_words)))
		print(feature_vector.shape)

		# -------- Feature Vector Updation -------- 
		clean_sentences = sentences_zero + sentences_one
		classes = classes_zero + classes_one
		temp = ''
		for i in range(len(clean_sentences)):
			temp1 = clean_sentences[i].split()
			for j in range(len(temp1)):
				temp = ''
				if j+n-1<len(temp1) and n!=1:	
					for k in range(n):
						temp = temp + temp1[j+k]
					feature_vector[i][unique_words.index(temp)]+=1
				elif temp=='' and n==1:
					temp = temp1[j]
					# if len(temp)>1:
					feature_vector[i][unique_words.index(temp)]+=1


	feature_vector_zero = feature_vector[0:(len(feature_vector)//2)]
	feature_vector_one = feature_vector[(len(feature_vector)//2):(len(feature_vector))]

	# print(len(classes_zero))

	feature_vector_zero_10folds, classes_zero_10folds = cross_validation_split(feature_vector_zero, classes_zero, 10)
	feature_vector_one_10folds, classes_one_10folds = cross_validation_split(feature_vector_one, classes_one, 10)	

	# -------- Train Test Split --------
	feature_vector = []
	classes = []
	print('\n')
	# print(len(feature_vector_one_10folds))
	accuracy = []
	precision = []
	recall = []
	for i in range(0,10):
		X_train = []
		y_train = []
		X_test = []
		y_test = []
		for j in range(0, 10):
			if i==j:
				for k in range(len(feature_vector_zero_10folds[j])):
					# print(k)
					X_test.append(feature_vector_zero_10folds[j][k])
					X_test.append(feature_vector_one_10folds[j][k])
					y_test.append(classes_zero_10folds[j][k])
					y_test.append(classes_one_10folds[j][k])
			elif i!=j:
				for k in range(len(feature_vector_zero_10folds[j])):
					X_train.append(feature_vector_zero_10folds[j][k])
					X_train.append(feature_vector_one_10folds[j][k])
					y_train.append(classes_zero_10folds[j][k])
					y_train.append(classes_one_10folds[j][k])

		# -------- Fitting SVM to the Training set -------- 

		classifier = SVC(kernel = 'linear', random_state = 0)
		classifier.fit(X_train, y_train)

		# -------- Predicting the Test set results -------- 

		y_pred = classifier.predict(X_test)

		# -------- Making the Confusion Matrix --------

		cm = confusion_matrix(y_test, y_pred)
		print(cm)

		# -------- Metrics --------
		p, r, f1, support = list(precision_recall_fscore_support(y_test, y_pred, average='weighted'))


		accuracy.append(accuracy_score(y_test, y_pred))
		precision.append(p)
		recall.append(r)

		print("Accuracy: ", accuracy[i])
		print("Precision: ", precision[i])
		print("Recall: ", recall[i])

	return (sum(accuracy)/len(accuracy)), (sum(precision)/len(precision)), (sum(recall)/len(recall)), f1,f2,f3,f4,f5

for i in range(1,6):
	# n>5 is for f1f2 and so on
	accuracy, precision, recall,f1,f2,f3,f4,f5 = ngram(i,f1,f2,f3,f4,f5)
	# print(f1)
	print("\n")
	print("Accuracy for N = ", i, "is ", accuracy)
	print("Precision for N = ", i, "is ", precision)
	print("Recall for N = ", i, "is " , recall)