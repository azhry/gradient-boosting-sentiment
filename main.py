from entities.Storage import Storage
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import label_binarize, LabelEncoder
from sklearn.metrics import roc_curve, auc, confusion_matrix, classification_report
from scipy import interp
from itertools import cycle
import matplotlib.pyplot as plt
import numpy as np

s = Storage()
count_vect = CountVectorizer()
tfidf_transformer = TfidfTransformer()
gradboost = OneVsRestClassifier(GradientBoostingClassifier(random_state=7))
labels = ["Berdampak positif", "Berdampak negatif", "Netral"]
le = LabelEncoder()

all_predict_probas = np.array([])
all_tests = np.array([])
fpr = dict()
tpr = dict()
roc_auc = dict()

for i in range(10):
	train = s.load(f"data/folds/train{i + 1}.pckl")
	test = s.load(f"data/folds/test{i + 1}.pckl")

	train_vect = count_vect.fit_transform(train["Review"])
	train_tfidf = tfidf_transformer.fit_transform(train_vect)

	gradboost.fit(train_tfidf, train["Label"])

	test_vect = count_vect.transform(test["Review"])
	test_tfidf = tfidf_transformer.transform(test_vect)
	predicted = gradboost.predict_proba(test_tfidf)
	y_test = le.fit_transform(test["Label"])
	y_test = label_binarize(y_test, classes=[0, 1, 2])

	if i == 0:
		all_predict_probas = predicted
		all_tests = y_test
	else:
		all_predict_probas = np.append(all_predict_probas, predicted, 0)
		all_tests = np.append(all_tests, y_test, 0)

	print(all_predict_probas.shape)

for i in range(len(labels)):
	fpr[i], tpr[i], _ = roc_curve(all_tests[:, i], all_predict_probas[:, i])
	roc_auc[i] = auc(fpr[i], tpr[i])


# Compute micro-average ROC curve and ROC area
fpr["micro"], tpr["micro"], _ = roc_curve(all_tests.ravel(), all_predict_probas.ravel())
roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

n_classes = len(labels)

# First aggregate all false positive rates
all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))

# Then interpolate all ROC curves at this points
mean_tpr = np.zeros_like(all_fpr)
for i in range(n_classes):
    mean_tpr += interp(all_fpr, fpr[i], tpr[i])

# Finally average it and compute AUC
mean_tpr /= n_classes

fpr["macro"] = all_fpr
tpr["macro"] = mean_tpr
roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

plt.figure()
lw = 2
plt.plot(fpr["micro"], tpr["micro"],
         label='micro-average ROC curve (area = {0:0.2f})'
               ''.format(roc_auc["micro"]),
         color='deeppink', linestyle=':', linewidth=4)

plt.plot(fpr["macro"], tpr["macro"],
         label='macro-average ROC curve (area = {0:0.2f})'
               ''.format(roc_auc["macro"]),
         color='navy', linestyle=':', linewidth=4)

colors = cycle(['aqua', 'darkorange', 'cornflowerblue'])
for i, color in zip(range(n_classes), colors):
    plt.plot(fpr[i], tpr[i], color=color, lw=lw,
             label='ROC curve of class {0} (area = {1:0.2f})'
             ''.format(i, roc_auc[i]))

plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic curve')
plt.legend(loc="lower right")
plt.show()
