from nltk.corpus import stopwords
import pandas as chen
import string
import re

def preprocess(listoftext):


    customstopwords = stopwords.words('english')
    customstopwords.append('rt')
    emojipattern = re.compile("["
                              u"\U0001F600-\U0001F64F"
                              u"\U0001F300-\U0001F5FF"
                              u"\U0001F680-\U0001F6FF"
                              u"\U0001F1E0-\U0001F1FF"
                              u"\U00000430-\U00000648"
                              u"\U0000263a-\U0000fe0f"
                              u"\U0000201c-\U0000201d"
                              u"\U00002000-\U000020e3"
                              u"\U0000064a-\U0000064d"
                              u"\U0000ff00-\U0000ff09"
                              u"\U000fe520-\U000fe529"
                              u"\U0000221b-\U0000221e"
                              u"\U000feb90-\U000feb99"
                              u"\U00001d20-\U00001d4c"
                              u"\U00000080-\U000000FF"
                              u"\U00000100-\U00000139"
                              u"\U0000FF80-\U0001007F""]+", flags=re.UNICODE)
    httppattern = re.compile('http?(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+')

    punctuationremoved = [char for char in listoftext if char not in  string.punctuation]
    punctuationremoved = ''.join(punctuationremoved)

    advancedremoved = emojipattern.sub(r'', punctuationremoved.decode('utf-8'))
    advancedremoved = httppattern.sub(r'', advancedremoved)

    stopwordsremoved = [word for word in advancedremoved.split() if word.lower() not in customstopwords]
    return stopwordsremoved

if __name__ == '__main__':

    iphonedataset = chen.read_csv('Datasets/iphonetweets.csv', sep=',', names=['text', 'label'])
    stsgolddataset = chen.read_csv('Datasets/stsgoldtweets.csv', sep=';', names=['id', 'label', 'text'])
    archeagedataset = chen.read_csv('Datasets/Archeagetweets.csv', sep=',', names=['text', 'label'])
    hobbitdataset = chen.read_csv('Datasets/Hobbittweets.csv', sep=',', names=['text', 'label'])
    moviedataset = chen.read_csv('Datasets/movietweets.txt',sep="	",names=['label', 'text'])

    iphoneTextset = iphonedataset['text']
    stsgoldTextset = stsgolddataset['text']
    archeageTextset = archeagedataset['text']
    hobbitTextset = hobbitdataset['text']
    movieTextset = moviedataset['text']

    iphonedataset_pre = iphoneTextset.apply(preprocess)
    stsgolddataset_pre = stsgoldTextset.apply(preprocess)
    hobbitdataset_pre = hobbitTextset.apply(preprocess)
    archeagedataset_pre = archeageTextset.apply(preprocess)
    moviedataset_pre = movieTextset.apply(preprocess)

    print "Example"
    print movieTextset
    print "-------------"
    print moviedataset_pre





