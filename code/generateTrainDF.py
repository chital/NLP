import pandas as pd
import config
import numpy as np



clothing_df = pd.read_csv(config.clothing_review)

def cloth_label(rating):
    if rating > 3:
        return "good"
    elif rating == 3:
        return "neutral"
    else:
        return "bad"


clothing_df['classification'] = clothing_df['Rating'].apply(cloth_label)
clothing_df_final = clothing_df.groupby('classification').apply(lambda s : s.sample(min(len(s),3500)))[['review','classification']]

print(clothing_df_final['classification'].value_counts())

hotel_df = pd.read_csv(config.hotel_review)
# append the positive and negative text reviews
hotel_df["review"] = hotel_df["Negative_Review"] + hotel_df["Positive_Review"]
# create the label
hotel_df["classification"] = hotel_df["Reviewer_Score"].apply(lambda x: "bad" if x <=4 else "neutral" if x<=7 else "good")
# select only relevant columns
hotel_df = hotel_df[["review", "classification"]]
hotel_df_final = hotel_df.groupby('classification').apply(lambda s: s.sample(min(len(s),252320)))[['review','classification']]
print(hotel_df_final['classification'].value_counts())

airline_df = pd.read_csv(config.airline_review)

dmap = {'positive':'good','neutral':'neutral','negative':'bad'}
airline_df['review']=airline_df['text']

airline_df["classification"]=airline_df["airline_sentiment"].replace(dmap)

airline_df_final = airline_df[['review','classification']]
print(airline_df_final['classification'].value_counts())


final_df = pd.concat([clothing_df_final,hotel_df_final,airline_df_final],ignore_index=True)

print(final_df.shape)
print(final_df['classification'].value_counts())

msk = np.random.rand(len(final_df)) < 0.98
train_df = final_df[msk]
predict_df =final_df[~msk]

print(train_df.shape)
print(predict_df.shape)
train_df.to_csv(config.training_data,index=False)
predict_df.to_csv(config.prediction_data,index=False)
hotel_df_final.to_csv('D:\\Office Projects\\NLP\\data\\Hotel_reviews_updated.csv')
