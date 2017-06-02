import pandas as chen
from custompreprocess import preprocess as mypre
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from mybayes import NaiveBayes
from sklearn.tree import DecisionTreeClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report

iphonedataset = chen.read_csv('Datasets/iphonetweets.csv', sep=',', names=['text', 'label'])
stsgolddataset = chen.read_csv('Datasets/stsgoldtweets.csv', sep=';', names=['id', 'label', 'text'])
archeagedataset = chen.read_csv('Datasets/Archeagetweets.csv', sep=',', names=['text', 'label'])
hobbitdataset = chen.read_csv('Datasets/Hobbittweets.csv', sep=',', names=['text', 'label'])
moviesdataset = chen.read_csv('Datasets/movietweets.txt', sep="	", names=['label', 'text'])


feat_train_iphone, feat_test_iphone, label_train_iphone, label_test_iphone = train_test_split(
    iphonedataset['text'],
    iphonedataset['label'],
    test_size=0.3)
feat_train_hobbit, feat_test_hobbit, label_train_hobbit, label_test_hobbit = train_test_split(
    hobbitdataset['text'],
    hobbitdataset['label'],
    test_size=0.3)
feat_train_arch, feat_test_arch, label_train_arch, label_test_arch = train_test_split(
    archeagedataset['text'],
    archeagedataset['label'],
    test_size=0.3)
feat_train_sts, feat_test_sts, label_train_sts, label_test_sts = train_test_split(
    stsgolddataset['text'],
    stsgolddataset['label'],
    test_size=0.3)

feat_train_movies,feat_test_movies,label_train_movies,label_test_movies = train_test_split(
    moviesdataset['text'],
    moviesdataset['label'],
    test_size=0.3
)

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

boruhatti_svm.fit(feat_train_movies, label_train_movies)
tahmin_movies = boruhatti_svm.predict(feat_test_movies)

print "movies report SVM"
print classification_report(label_test_movies, tahmin_movies)

# Custom Naive Bayes Section
clf = NaiveBayes()

boruhatti_nb = Pipeline([
    ('bag_of_words_model', CountVectorizer(analyzer=mypre)),
    ('TF&IDF', TfidfTransformer()),
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

boruhatti_nb.fit(feat_train_movies,label_train_movies)
tahmin_movies = boruhatti_nb.predict(feat_test_movies)

print "movies report Custom Naive Bayes"
print classification_report(label_test_movies, tahmin_movies)

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

boruhatti_dt.fit(feat_train_movies, label_train_movies)
tahmin_movies = boruhatti_dt.predict(feat_test_movies)

print "movies report Decision Tree"
print classification_report(label_test_movies, tahmin_movies)