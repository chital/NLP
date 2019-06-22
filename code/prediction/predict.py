import config
import training
import pickle
import pandas as pd
import string
from nltk import pos_tag
from nltk.corpus import stopwords
from nltk.tokenize import WhitespaceTokenizer
from nltk.stem import WordNetLemmatizer

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

def main():
    predict_df = pd.read_csv(config.prediction_data)
    dmap = {"good": 1, "bad": -1, "neutral": 0}
    predict_df['review_status'] = predict_df['classification'].replace(dmap)
    reviews_df = predict_df[["review", "review_status"]]
    reviews_df["review_clean"] = reviews_df["review"].apply(lambda x: clean_text(str(x)))
    with open("..\\model\\count_vectorizer.pk", "rb") as f:
        bow_vectorizer = pickle.load(f)
    gen_result = bow_vectorizer.transform(reviews_df["review_clean"]).toarray()
    pred_df = pd.DataFrame(gen_result, columns=bow_vectorizer.get_feature_names())
    print(pred_df.head())
    pred_df.index = reviews_df.index
    pred_df_final = pd.concat([pred_df, reviews_df["review_status"]], axis=1)
    print(pred_df_final.head())
    gen_label = "review_status"
    ignore_cols = [gen_label, "review", "review_clean"]
    pred_features = [c for c in pred_df_final.columns if c not in ignore_cols]
    pred_df_final.fillna(0, inplace=True)
    pred_x = pred_df_final[pred_features]
    pred_y = pred_df_final[gen_label]
    with open("..\\model\\count_vect_clf.pk","rb") as f:
        rf = pickle.load(f)
    print(rf.score(pred_x, pred_y))

if __name__ == "__main__":
    main()


