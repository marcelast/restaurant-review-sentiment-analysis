import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
import re
import io

st.title("Restaurant Review Sentiment Analysis")

st.markdown("**Purpose of the Model:**")
st.markdown("The purpose of this model is to analyze restaurant reviews to assess the sentiments expressed by customers, classifying reviews as positive, negative, or neutral based on the text of the review.")

st.markdown("**Objective of the Model:**")
st.markdown("The objective of the model is to build an automated classification system that utilizes natural language processing (NLP) techniques and machine learning to predict the overall sentiment of the reviews. The model will assist in quickly identifying the sentiments expressed in customer reviews, allowing restaurants to respond appropriately to customer feedback and improve their overall experience.")

data = pd.read_csv('restaurantsReviews.csv')

location_to_name = {
    3643106: "La Sarkis",
    9593555: "Zaxi Fun & Finest",
    12794044: "Saperavi",
    23890224: "Little Napoli",
    1089531: "La Placinte",
    9729405: "OSHO bar&kitchen",
    2261775: "Pegas Restaurant & Terrace",
    21041974: "Fuior",
    23474553: "Divus Restaurant",
    25270571: "Charmat Prosecco Bar"
}

data['restaurant_name'] = data['locationId'].map(location_to_name)

st.write("Loaded data:")
st.write(data.head())

st.write("Data Overview:")
buf = io.StringIO()
data.info(buf=buf)
s = buf.getvalue()
st.text(s)  


# Clean the text data
def clean_text(text):
    text = re.sub(r'http\S+', '', text) 
    text = re.sub(r'[^a-zA-Z\s]', '', text) 
    return text.lower()

data['text'] = data['text'].apply(clean_text)

data['sentiment'] = data['rating'].apply(lambda x: 'positive' if x >= 4 else 'negative' if x <= 2 else 'neutral')

st.write("Cleaned data and sentiment classification:")
st.write(data[['text', 'sentiment']].head())

# Split data
X = data['text']  
y = data['sentiment'] 

# Split (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Vectorize the text data using TF-IDF
vectorizer = TfidfVectorizer()
X_train_vect = vectorizer.fit_transform(X_train) 
X_test_vect = vectorizer.transform(X_test) 

st.write("Training data shape:", X_train_vect.shape)
st.write("Testing data shape:", X_test_vect.shape)

# Regression model
model = LogisticRegression(max_iter=1000)  
model.fit(X_train_vect, y_train)

y_pred = model.predict(X_test_vect)
accuracy = accuracy_score(y_test, y_pred)
st.write("Model Accuracy:", accuracy)

# Conclusion
st.write("The logistic regression model achieved an accuracy of about 75.61% on the test data, indicating it can correctly predict the sentiment of most restaurant reviews.")
st.markdown("**Conclusion:**")
st.markdown("The logistic regression model achieved an accuracy of approximately 75.61% on the test dataset, indicating that it can correctly predict the sentiment of most restaurant reviews. This performance demonstrates the model's utility in sentiment analysis, thereby providing valuable insights for restaurants in managing customer relationships and enhancing the services offered.")

# 1. Distribution of Sentiment Classes
st.subheader("Distribution of Sentiment Classes")
sentiment_counts = data['sentiment'].value_counts()
plt.figure(figsize=(8, 5))
sns.barplot(x=sentiment_counts.index, y=sentiment_counts.values, palette='viridis')
plt.title('Distribution of Sentiment Classes')
plt.xlabel('Sentiment')
plt.ylabel('Count')
st.pyplot(plt)

# 2. Word Cloud for Positive Reviews
st.subheader("Word Cloud for Positive Reviews")
positive_reviews = ' '.join(data[data['sentiment'] == 'positive']['text'])
wordcloud_pos = WordCloud(width=800, height=400, background_color='white').generate(positive_reviews)
plt.figure(figsize=(10, 5))
plt.imshow(wordcloud_pos, interpolation='bilinear')
plt.axis('off')
st.pyplot(plt)

# 3. Word Cloud for Negative Reviews
st.subheader("Word Cloud for Negative Reviews")
negative_reviews = ' '.join(data[data['sentiment'] == 'negative']['text'])
wordcloud_neg = WordCloud(width=800, height=400, background_color='white').generate(negative_reviews)
plt.figure(figsize=(10, 5))
plt.imshow(wordcloud_neg, interpolation='bilinear')
plt.axis('off')
st.pyplot(plt)

# 4. Confusion Matrix
st.subheader("Confusion Matrix")
cm = confusion_matrix(y_test, y_pred, labels=model.classes_)
plt.figure(figsize=(8, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=model.classes_, yticklabels=model.classes_)
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('True')
st.pyplot(plt)


st.subheader("Top Restaurants based on Sentiment Score")

sentiment_scores = data.groupby('restaurant_name').agg(
    total_score=('sentiment', lambda x: (x == 'positive').sum() - (x == 'negative').sum())
).reset_index()

sentiment_scores = sentiment_scores.sort_values(by='total_score', ascending=False)

st.write(sentiment_scores[['restaurant_name', 'total_score']])

top_restaurants = sentiment_scores.head(10)

plt.figure(figsize=(12, 6))
sns.barplot(x='total_score', y='restaurant_name', data=top_restaurants, palette='viridis')
plt.xlabel('Scorul Total')
plt.ylabel('Restaurant')
st.pyplot(plt)



