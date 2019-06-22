import pandas as pd
import nltk
nltk.download('stopwords')
nltk.download('averaged_perceptron_tagger')
nltk.download('wordnet')
import string
from nltk import pos_tag
from nltk.corpus import stopwords
from nltk.tokenize import WhitespaceTokenizer
from nltk.stem import WordNetLemmatizer
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import seaborn as sns
from nltk.corpus import wordnet
import config
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.feature_extraction.text import TfidfVectorizer
import pickle

def filter_insignificant(chunk, tag_prefix):
    good = []

    for word, tag in chunk:
        ok = True
        for prefix in tag_prefix:
            if tag.startswith(prefix):
                ok = False
                break

        if ok:
            good.append(word)

    return good


def get_wordnet_pos(pos_tag):
    if pos_tag.startswith('J'):
        return wordnet.ADJ
    elif pos_tag.startswith('V'):
        return wordnet.VERB
    elif pos_tag.startswith('N'):
        return wordnet.NOUN
    elif pos_tag.startswith('R'):
        return wordnet.ADV
    else:
        return wordnet.NOUN


def clean_text(text):
    # lower text
    text = text.lower()
    # tokenize text and remove puncutation
    text = [word.strip(string.punctuation) for word in text.split(" ")]
    # remove words that contain numbers
    text = [word for word in text if not any(c.isdigit() for c in word)]
    # remove stop words
    stop = stopwords.words('english')
    text = [x for x in text if x not in stop]
    # remove empty tokens
    text = [t for t in text if len(t) > 0]
    # pos tag text
    pos_tags = pos_tag(text)
    pos_tags = filter_insignificant(pos_tags,'N')
    # lemmatize text
    text = [WordNetLemmatizer().lemmatize(t) for t in pos_tags]
    # remove words with only one letter
    text = [t for t in text if len(t) > 1]
    # join all
    text = " ".join(text)
    # print("cleaning complete for",text)
    return (text)


# wordcloud function
def show_wordcloud(data, title = None):
    wordcloud = WordCloud(
        background_color = 'white',
        max_words = 200,
        max_font_size = 40,
        scale = 3,
        random_state = 42
    ).generate(str(data))

    fig = plt.figure(1, figsize = (20, 20))
    plt.axis('off')
    if title:
        fig.suptitle(title, fontsize = 20)
        fig.subplots_adjust(top = 2.3)

    plt.imshow(wordcloud)
    plt.show()

def countVectorizer(reviews_df):
    bow_vectorizer = CountVectorizer(max_features=1500, min_df=5, max_df=0.7, stop_words=stopwords.words('english'),
                                     ngram_range=(1, 2))
    bow_result = bow_vectorizer.fit_transform(reviews_df["review_clean"]).toarray()
    bow_df = pd.DataFrame(bow_result, columns=bow_vectorizer.get_feature_names())
    bow_df.index = reviews_df.index
    bow_df_final = pd.concat([bow_df, reviews_df["review_status"]], axis=1)
    with open('..//model//count_vectorizer.pk', 'wb') as fin:
        pickle.dump(bow_vectorizer, fin)
    return bow_df_final

def tfidfVectorizer(reviews_df):
    tfidf_vectorizer = TfidfVectorizer(min_df=10)
    tfidf_result = tfidf_vectorizer.fit_transform(reviews_df["review_clean"]).toarray()
    tfidf_df = pd.DataFrame(tfidf_result, columns=tfidf_vectorizer.get_feature_names())
    tfidf_df.index = reviews_df.index
    tfidf_df_final = pd.concat([tfidf_df, reviews_df["review_status"]], axis=1)
    return tfidf_df_final

def model_train(final_df):
    # feature selection
    label = "review_status"
    ignore_cols = [label, "review", "review_clean"]
    features = [c for c in final_df.columns if c not in ignore_cols]

    # split the data into train and test
    X_train, X_test, y_train, y_test = train_test_split(final_df[features], final_df[label], test_size=0.20,
                                                        random_state=42)
    return X_train, X_test, y_train, y_test
    # train a random forest classifier


def training_report(rf,X_test,y_test):
    print("Accuracy Score {} ".format(rf.score(X_test, y_test)))
    y_pred = rf.predict(X_test)
    print("Confusion Matrix  {}".format(confusion_matrix(y_test, y_pred)))
    print("Classification report {} ".format(classification_report(y_test, y_pred)))
    print("Accuracy Score {}".format(accuracy_score(y_test, y_pred)))

# read data
def main():
    reviews_df = pd.read_csv(config.training_data)
    dmap = {"good": 1, "bad":-1,"neutral" : 0}
    reviews_df["review_status"] = reviews_df["classification"].replace(dmap)

    # select only relevant columns
    reviews_df = reviews_df[["review", "review_status"]]
    reviews_df = reviews_df.sample(frac=0.1, replace=False, random_state=42)
    reviews_df["review"] = reviews_df["review"].apply(
        lambda x: str(x).replace("No Negative", "").replace("No Positive", ""))
    # print wordcloud
    # clean text data
    reviews_df["review_clean"] = reviews_df["review"].apply(lambda x: clean_text(str(x)))
    show_wordcloud(reviews_df["review_clean"])

    bow_df_final = countVectorizer(reviews_df)
    X_train, X_test, y_train, y_test = model_train(bow_df_final)
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)
    with open("..\\model\\count_vect_clf.pk","wb") as fin:
        pickle.dump(rf,fin)
    print("###### Reports for Count Vectorizer #######")
    training_report(rf,X_test,y_test)

    tfidf_df_final =tfidfVectorizer(reviews_df)
    X_train, X_test, y_train, y_test = model_train(tfidf_df_final)
    rf2 = RandomForestClassifier(n_estimators=100, random_state=42)
    rf2.fit(X_train, y_train)
    with open("..\\model\\tfidf_vect_clf.pk","wb") as fin:
        pickle.dump(rf2,fin)

    print("###### Reports for TFIDF Vectorizer #######")
    training_report(rf2, X_test, y_test)


if __name__ == "__main__":
    main()