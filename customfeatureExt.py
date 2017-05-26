import pandas as chen
from custompreprocess import preprocess as mypre
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from mybayes import NaiveBayes
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report
import numpy as np

iphonedataset = chen.read_csv('Datasets/iphonetweets.csv', sep=',', names=['text', 'label'])
stsgolddataset = chen.read_csv('Datasets/stsgoldtweets.csv', sep=';', names=['id', 'label', 'text'])
archeagedataset = chen.read_csv('Datasets/Archeagetweets.csv', sep=',', names=['text', 'label'])
hobbitdataset = chen.read_csv('Datasets/Hobbittweets.csv', sep=',', names=['text', 'label'])

print len(stsgolddataset['text'][stsgolddataset['label'] == 0])

# bu eleman veri seti bolunurken hangi index egitime hangi index teste atildi bilbilmek icin eklendi
index1 = np.arange(0, len(iphonedataset), 1)

feat_train_iphone, feat_test_iphone, label_train_iphone, label_test_iphone, label_train_index_iphone, label_test_index_iphone = train_test_split(
    iphonedataset['text'],
    iphonedataset['label'], index1,
    test_size=0.3)
index2 = np.arange(0, len(hobbitdataset), 1)
feat_train_hobbit, feat_test_hobbit, label_train_hobbit, label_test_hobbit, label_train_index_hobbit, label_test_index_hobbit = train_test_split(
    hobbitdataset['text'],
    hobbitdataset['label'], index2,
    test_size=0.3)
index3 = np.arange(0, len(archeagedataset), 1)
feat_train_arch, feat_test_arch, label_train_arch, label_test_arch, label_train_index_arch, label_test_index_arch = train_test_split(
    archeagedataset['text'],
    archeagedataset['label'], index3,
    test_size=0.3)
index4 = np.arange(0, len(stsgolddataset), 1)
feat_train_sts, feat_test_sts, label_train_sts, label_test_sts, label_train_index_sts, label_test_index_sts = train_test_split(
    stsgolddataset['text'],
    stsgolddataset['label'], index4,
    test_size=0.3)

# SVM Section

HandDefinedSVM = SVC(kernel='linear')

boruhatti_svm = Pipeline([
    ('bag_of_words_model', CountVectorizer(analyzer=mypre)),
    ('TF&IDF', TfidfTransformer()),
    ('classifier', HandDefinedSVM)
])
boruhatti_svm.fit(feat_train_iphone, label_train_iphone)
tahmin_iphone = boruhatti_svm.predict(feat_test_iphone)

print "iphone report SVM"
print classification_report(label_test_iphone, tahmin_iphone)

boruhatti_svm.fit(feat_train_hobbit, label_train_hobbit)
tahmin_hobbit = boruhatti_svm.predict(feat_test_hobbit)

print "hobbit report SVM"
print classification_report(label_test_hobbit, tahmin_hobbit)

boruhatti_svm.fit(feat_train_arch, label_train_arch)
tahmin_arch = boruhatti_svm.predict(feat_test_arch)

print "archage report SVM"
print classification_report(label_test_arch, tahmin_arch)

boruhatti_svm.fit(feat_train_sts, label_train_sts)
tahmin_sts = boruhatti_svm.predict(feat_test_sts)

print "sts report SVM"
print classification_report(label_test_sts, tahmin_sts)

# Custom Naive Bayes Section

# typeoffeature ile olasiligin nasil hesaplanacagini kontrol edilir
# 0 olursa multinominal naive hesaplama 1 olursa gaussian olasilik heaplama
typeoffeature = 1  # multinominal=0,   continues=1
clf = NaiveBayes(typeoffeature)

boruhatti_nb = Pipeline([
    ('bag_of_words_model', CountVectorizer(analyzer=mypre)),
    ('TF&IDF', TfidfTransformer()),  # TF&IDF  of TF
    ('classifier', clf)
])

boruhatti_nb.fit(feat_train_iphone, label_train_iphone)
tahmin_iphone = boruhatti_nb.predict(feat_test_iphone)

print "iphone report Custom Naive Bayes"
print classification_report(label_test_iphone, tahmin_iphone)

boruhatti_nb.fit(feat_train_hobbit, label_train_hobbit)
tahmin_hobbit = boruhatti_nb.predict(feat_test_hobbit)

print "hobbit report Custom Naive Bayes"
print classification_report(label_test_hobbit, tahmin_hobbit)

boruhatti_nb.fit(feat_train_arch, label_train_arch)
tahmin_arch = boruhatti_nb.predict(feat_test_arch)

print "archage report Custom Naive Bayes"
print classification_report(label_test_arch, tahmin_arch)

boruhatti_nb.fit(feat_train_sts, label_train_sts)
tahmin_sts = boruhatti_nb.predict(feat_test_sts)

print "sts report Custom Naive Bayes"
print classification_report(label_test_sts, tahmin_sts)

# Decision Tree Section

clf_dt = DecisionTreeClassifier()

boruhatti_dt = Pipeline([
    ('bag_of_words_model', CountVectorizer(analyzer=mypre)),
    ('TF&IDF', TfidfTransformer()),
    ('classifier', clf_dt)
])

boruhatti_dt.fit(feat_train_iphone, label_train_iphone)
tahmin_iphone = boruhatti_dt.predict(feat_test_iphone)

print "Iphone report Decision Tree"
print classification_report(label_test_iphone, tahmin_iphone)

boruhatti_dt.fit(feat_train_hobbit, label_train_hobbit)
tahmin_hobbit = boruhatti_dt.predict(feat_test_hobbit)

print "Hobbit report Decision Tree"
print classification_report(label_test_hobbit, tahmin_hobbit)

boruhatti_dt.fit(feat_train_arch, label_train_arch)
tahmin_arch = boruhatti_dt.predict(feat_test_arch)

print "Archage report Decision Tree"
print classification_report(label_test_arch, tahmin_arch)

boruhatti_dt.fit(feat_train_sts, label_train_sts)
tahmin_sts = boruhatti_dt.predict(feat_test_sts)

print "sts report Decision Tree"
print classification_report(label_test_sts, tahmin_sts)


# Linear Regression Section

def categorize_lr(predictions, unique_labels): #Linear reggresion icin kategorilestirme
    list_of_labels = []
    for elem in predictions:
        if elem < 0 and 0 in unique_labels :
            list_of_labels.append(0)
        elif elem > 0 and 4 in unique_labels:
            list_of_labels.append(4)
        elif elem < 0 and -1 in unique_labels:
            list_of_labels.append(-1)
        elif elem >0 and 1 in unique_labels:
            list_of_labels.append(1)
        else:
            print "Etiket bilgisi taninamadi."
            break

    return list_of_labels


rgr_lr = LinearRegression()

boruhatti_lr = Pipeline([
    ('bag_of_words_model', CountVectorizer(analyzer=mypre)),
    ('TF&IDF', TfidfTransformer()),
    ('regression', rgr_lr)
])

boruhatti_lr.fit(feat_train_iphone, label_train_iphone)
tahmin_iphone = boruhatti_lr.predict(feat_test_iphone)
lr_tahmin_labels = categorize_lr(tahmin_iphone, np.unique(label_train_iphone))

print "Iphone report Linear Regression"
print classification_report(label_test_iphone, lr_tahmin_labels)


boruhatti_lr.fit(feat_train_hobbit, label_train_hobbit)
tahmin_hobbit = boruhatti_lr.predict(feat_test_hobbit)
lr_tahmin_labels = categorize_lr(tahmin_hobbit, np.unique(label_train_hobbit))

print "Hobbit report Linear Regression"
print classification_report(label_test_hobbit, lr_tahmin_labels)

boruhatti_lr.fit(feat_train_arch, label_train_arch)
tahmin_arch = boruhatti_lr.predict(feat_test_arch)
lr_tahmin_labels = categorize_lr(tahmin_arch, np.unique(label_train_arch))

print "Archage report Linear Regression"
print classification_report(label_test_arch, lr_tahmin_labels)


boruhatti_lr.fit(feat_train_sts, label_train_sts)
tahmin_sts = boruhatti_lr.predict(feat_test_sts)
lr_tahmin_labels = categorize_lr(tahmin_sts, np.unique(label_train_sts))

print "Sts report Linear Regression"
print classification_report(label_test_sts, lr_tahmin_labels)

"""
#Sklearn Naive Bayes Section
boruhatti2 = Pipeline([
    ('bag_of_words_model', CountVectorizer(analyzer=mypre)),
    ('TF&IDF', TfidfTransformer()),
    ('classifier', MultinomialNB())
])

boruhatti2.fit(feat_train_iphone,label_train_iphone)
tahmin_iphone_MNB = boruhatti2.predict(feat_test_iphone)

print "Iphone report MNB"
print classification_report(label_test_iphone, tahmin_iphone_MNB)
"""
"""
iphoneText = (iphonedataset['text']).apply(mypre)
kelimecantasi = CountVectorizer(analyzer=mypre).fit(iphonedataset['text'])
iphoneTweet_kc = kelimecantasi.transform(iphonedataset['text'])
tfidf_donusturucu = TfidfTransformer().fit(iphoneTweet_kc)
iphoneTweet_tfidf = tfidf_donusturucu.transform(iphoneTweet_kc)

hobbitText = (hobbitdataset['text']).apply(mypre)
kelimecantasi = CountVectorizer(analyzer=mypre).fit(hobbitdataset['text'])
hobbitTweet_kc = kelimecantasi.transform(hobbitdataset['text'])
tfidf_donusturucu = TfidfTransformer().fit(hobbitTweet_kc)
hobbitTweet_tfidf = tfidf_donusturucu.transform(hobbitTweet_kc)
"""
"""
X = np.array([[0.694528102757],
              [0.527145544048],
              [0.455474073186],
              [0.24835626801]])

y = np.array([-1, 1, -1, 1])

test_X = np.array([[3.694528102757],
                  [0.527145544048],
                  [0.455474073186],
                  [0.24835626801]])

print "shape : ", len(test_X.shape)

clf = NaiveBayes()
clf.fit(X, y)
nbTahmin = clf.predict(test_X)

print nbTahmin
print classification_report(y,nbTahmin)
"""
"""
n_positive = label_train[label_train == "1"].count()
n_negative = label_train[label_train == "-1"].count()

total_lbls = label_train.count()

print n_negative
print n_positive

print total_lbls
"""
"""
print '\n\n\n'
print '\n\n\n'
print bow_transformer.get_feature_names()[1500]

ip30 = iphonedataset['text'][30]
print ip30
bow30 = bow_transformer.transform([ip30])
print bow30
tfidf30 = tfidf_transformer.transform(bow30)
print tfidf30
"""
"""
print "Tf-idf"
print iphoneTweet_tfidf
print '\n'
"""
"""
#Train TF&IDF - Iphone
iphone_NB_Train_Bag = CountVectorizer(analyzer=mypre).fit(feat_train_iphone)
iphone_NB_Train_KC = iphone_NB_Train_Bag .transform(feat_train_iphone)

iphoneNB_TFIDF_Train_Transformer = TfidfTransformer().fit(iphone_NB_Train_KC)
iphone_NB_TFIDF_Train = iphoneNB_TFIDF_Train_Transformer.transform(iphone_NB_Train_KC)

#Test TF&IDF - Iphone
iphone_NB_Test_Bag = CountVectorizer(analyzer=mypre).fit(feat_test_iphone)
iphone_NB_Test_KC = iphone_NB_Test_Bag .transform(feat_test_iphone)

iphoneNB_TFIDF_Test_Transformer = TfidfTransformer().fit(iphone_NB_Test_KC)
iphone_NB_TFIDF_Test = iphoneNB_TFIDF_Test_Transformer.transform(iphone_NB_Test_KC)

#Train TF&IDF - Hobbit
hobbit_NB_Train_Bag = CountVectorizer(analyzer=mypre).fit(feat_train_hobbit)
hobbit_NB_Train_KC = hobbit_NB_Train_Bag.transform(feat_train_hobbit)

hobbitNB_TFIDF_Train_Transformer = TfidfTransformer().fit(hobbit_NB_Train_KC)
hobbit_NB_TFIDF_Train = hobbitNB_TFIDF_Train_Transformer.transform(hobbit_NB_Train_KC)

#Test TF&IDF - Hobbit
hobbit_NB_Test_Bag = CountVectorizer(analyzer=mypre).fit(feat_test_hobbit)
hobbit_NB_Test_KC = hobbit_NB_Test_Bag.transform(feat_test_hobbit)

hobbitNB_TFIDF_Test_Transformer = TfidfTransformer().fit(hobbit_NB_Test_KC)
hobbit_NB_TFIDF_Test = hobbitNB_TFIDF_Test_Transformer.transform(hobbit_NB_Test_KC)
"""
"""
outerlist_Train = []

for i in range(len(hobbit_NB_TFIDF_Train_matrixed)):
     innerlist = []
     toplam = 0
     for elem in hobbit_NB_TFIDF_Train_matrixed[i][hobbit_NB_TFIDF_Train_matrixed[i] != 0]:
         toplam = toplam + elem
     innerlist.append(toplam)
     outerlist_Train.append(innerlist)

outerlist_Test = []

for i in range(len(hobbit_NB_TFIDF_Test_matrixed)):
    innerlist = []
    toplam = 0
    for elem in hobbit_NB_TFIDF_Test_matrixed[i][hobbit_NB_TFIDF_Test_matrixed[i] != 0]:
        toplam = toplam + elem
    innerlist.append(toplam)
    outerlist_Test.append(innerlist)
"""
