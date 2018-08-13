from entities.Storage import Storage
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import label_binarize
from sklearn.metrics import roc_curve, auc
from scipy import interp
import matplotlib.pyplot as plt
import numpy as np

s = Storage()
count_vect = CountVectorizer()
tfidf_transformer = TfidfTransformer()

tprs = []
aucs = []
mean_fpr = np.linspace(0, 1, 100)

# https://stackoverflow.com/questions/45332410/sklearn-roc-for-multiclass-classification
for i in range(10):
	train = s.load(f"data/folds/train{i + 1}.pckl")
	test = s.load(f"data/folds/test{i + 1}.pckl")

	train_vect = count_vect.fit_transform(train["Review"])
	train_tfidf = tfidf_transformer.fit_transform(train_vect)

	gradboost = GradientBoostingClassifier(random_state=7)
	gradboost.fit(train_tfidf, train["Label"])

	test_vect = count_vect.transform(test["Review"])
	test_tfidf = tfidf_transformer.transform(test_vect)
	score = gradboost.score(test_tfidf, test["Label"])
	print(score)

# 	probas_ = gradboost.predict_proba(test_tfidf)
# 	fpr, tpr, thresholds = roc_curve(test["Label"], probas_[:, 1])
# 	tprs.append(interp(mean_fpr, fpr, tpr))
# 	tprs[-1][0] = 0.0
# 	roc_auc = auc(fpr, tpr)
# 	aucs.append(roc_auc)
# 	plt.plot(fpr, tpr, lw=1, alpha=0.3, label="ROC fold %d (AUC = %0.2f)" % (i + 1, roc_auc))

# plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r',
#          label='Luck', alpha=.8)

# mean_tpr = np.mean(tprs, axis=0)
# mean_tpr[-1] = 1.0
# mean_auc = auc(mean_fpr, mean_tpr)
# std_auc = np.std(aucs)
# plt.plot(mean_fpr, mean_tpr, color='b',
#          label=r'Mean ROC (AUC = %0.2f $\pm$ %0.2f)' % (mean_auc, std_auc),
#          lw=2, alpha=.8)

# std_tpr = np.std(tprs, axis=0)
# tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
# tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
# plt.fill_between(mean_fpr, tprs_lower, tprs_upper, color='grey', alpha=.2,
#                  label=r'$\pm$ 1 std. dev.')

# plt.xlim([-0.05, 1.05])
# plt.ylim([-0.05, 1.05])
# plt.xlabel('False Positive Rate')
# plt.ylabel('True Positive Rate')
# plt.title('Receiver operating characteristic example')
# plt.legend(loc="lower right")
# plt.show()