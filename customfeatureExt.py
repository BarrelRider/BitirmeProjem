import pandas as chen
from custompreprocess import preprocess as mypre
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report
from mybayes import NaiveBayes
import numpy as np

iphonedataset = chen.read_csv('Datasets/iphonetweets.csv', sep=',', names=['text', 'label'])
stsgolddataset = chen.read_csv('Datasets/stsgoldtweets.csv', sep=';', names=['id', 'label', 'text'])
archeagedataset = chen.read_csv('Datasets/Archeagetweets.csv', sep=',', names=['text', 'label'])
hobbitdataset = chen.read_csv('Datasets/Hobbittweets.csv', sep=',', names=['text', 'label'])

feat_train_iphone, feat_test_iphone, label_train_iphone, label_test_iphone = train_test_split(iphonedataset['text'],
                                                                    iphonedataset['label'],
                                                                    test_size=0.3)

feat_train_hobbit, feat_test_hobbit, label_train_hobbit, label_test_hobbit = train_test_split(hobbitdataset['text'],
                                                                    hobbitdataset['label'],
                                                                    test_size=0.3)

feat_train_arch, feat_test_arch, label_train_arch, label_test_arch = train_test_split(archeagedataset['text'],
                                                                    archeagedataset['label'],
                                                                    test_size=0.3)

feat_train_sts, feat_test_sts, label_train_sts, label_test_sts = train_test_split(stsgolddataset['text'],
                                                                    stsgolddataset['label'],
                                                                    test_size=0.3)

#SVM Section
HandDefinedSVM = SVC(kernel='linear')

boruhatti = Pipeline ([
    ('bag_of_words_model', CountVectorizer(analyzer=mypre)),
    ('TF&IDF', TfidfTransformer()),
    ('classifier', HandDefinedSVM)
])
boruhatti.fit(feat_train_iphone, label_train_iphone)
tahmin_iphone = boruhatti.predict(feat_test_iphone)

print "iphone report SVM"
print classification_report(label_test_iphone, tahmin_iphone)

boruhatti.fit(feat_train_hobbit, label_train_hobbit)
tahmin_hobbit = boruhatti.predict(feat_test_hobbit)

print "hobbit report SVM"
print classification_report(label_test_hobbit, tahmin_hobbit)

boruhatti.fit(feat_train_arch, label_train_arch)
tahmin_arch = boruhatti.predict(feat_test_arch)

print "archage report SVM"
print classification_report(label_test_arch, tahmin_arch)

boruhatti.fit(feat_train_sts, label_train_sts)
tahmin_sts = boruhatti.predict(feat_test_sts)

print "sts report SVM"
print classification_report(label_test_sts, tahmin_sts)


#Custom Naive Bayes Section
def getTFIDF(feature_vec):
    feature_NB_Bag = CountVectorizer(analyzer=mypre).fit(feature_vec)
    feature_NB_Kc = feature_NB_Bag.transform(feature_vec)

    feature_NB_Transformer = TfidfTransformer().fit(feature_NB_Kc)
    feature_NB_TFIDF = feature_NB_Transformer.transform(feature_NB_Kc)

    return feature_NB_TFIDF

def generateSpecialMatrix(matrixed_feature):
    outerList = []
    for i in range(len(matrixed_feature)):
        innerList = []
        toplam = 0
        for elem in matrixed_feature[i][matrixed_feature[i] != 0]:
            toplam = toplam + elem
        innerList.append(toplam/len(matrixed_feature[i][matrixed_feature[i] != 0]))
        outerList.append(innerList)
    return outerList


iphone_NB_TFIDF_Train_matrixed = np.array(getTFIDF(feat_train_iphone).toarray())
iphone_NB_TFIDF_Test_matrixed = np.array(getTFIDF(feat_test_iphone).toarray())
iphone_NB_label_train = np.array(label_train_iphone.as_matrix())
iphone_NB_label_test = np.array(label_test_iphone.as_matrix())

hobbit_NB_TFIDF_Train_matrixed = np.array(getTFIDF(feat_train_hobbit).toarray())
hobbit_NB_TFIDF_Test_matrixed = np.array(getTFIDF(feat_test_hobbit).toarray())
hobbit_NB_label_train = np.array(label_train_hobbit.as_matrix())
hobbit_NB_label_test = np.array(label_test_hobbit.as_matrix())

outerList_hobbit_Train = generateSpecialMatrix(hobbit_NB_TFIDF_Train_matrixed)
outerList_hobbit_Test = generateSpecialMatrix(hobbit_NB_TFIDF_Test_matrixed)


clf = NaiveBayes()
clf.fit(hobbit_NB_TFIDF_Train_matrixed, hobbit_NB_label_train)

tahmin_2 = clf.predict(hobbit_NB_TFIDF_Test_matrixed)

print "hobbit report Custom Naive Bayes"
print classification_report(hobbit_NB_label_test, tahmin_2)








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