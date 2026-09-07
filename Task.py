
# TASK 4 - SENTIMENT ANALYSIS OF SOCIAL MEDIA DATA

# 1. Import libraries
import pandas as pd
import matplotlib.pyplot as plt
import string
from wordcloud import WordCloud


# 2. LOAD DATASET

df = pd.read_csv("twitter_training.csv")

print("First 5 rows:")
print(df.head())

print("\nColumn names:")
print(df.columns)

print("\nDataset shape:")
print(df.shape)

# 3. RENAME COLUMNS
df.columns = ["ID", "Topic", "Sentiment", "Text"]

print("\nRenamed columns:")
print(df.columns)


# 4. CHECK MISSING VALUES

print("\nMissing values:")
print(df.isnull().sum())


# 5. DATA CLEANING

# Remove rows containing missing values
clean_df = df.dropna().copy()

print("\nShape after removing missing values:")
print(clean_df.shape)

# Convert text to lowercase
clean_df["Text"] = clean_df["Text"].str.lower()

# Remove punctuation
clean_df["Text"] = clean_df["Text"].str.replace(
    "[" + string.punctuation + "]",
    "",
    regex=True
)

# Remove extra spaces
clean_df["Text"] = clean_df["Text"].str.strip()

print("\nCleaned text:")
print(clean_df["Text"].head())


# 6. SENTIMENT DISTRIBUTION
sentiment_count = clean_df["Sentiment"].value_counts()

print("\nSentiment Counts:")
print(sentiment_count)

plt.figure(figsize=(8, 5))

sentiment_count.plot(kind="bar")

plt.title("Sentiment Distribution", fontweight="bold")
plt.xlabel("Sentiment")
plt.ylabel("Number of Posts")
plt.xticks(rotation=0)
plt.tight_layout()
plt.show()


# 7. SENTIMENT PERCENTAGE

sentiment_percentage = (
    clean_df["Sentiment"]
    .value_counts(normalize=True)
    * 100
)

print("\nSentiment Percentage:")
print(sentiment_percentage.round(2))


plt.figure(figsize=(7, 7))

plt.pie(
    sentiment_count,
    labels=sentiment_count.index,
    autopct="%1.1f%%",
    startangle=90
    )

plt.title("Sentiment Percentage", fontweight="bold")
plt.tight_layout()
plt.show()


# 8. TOP 10 MOST MENTIONED TOPICS / BRANDS

top_topics = clean_df["Topic"].value_counts().head(10)

print("\nTop 10 Topics / Brands:")
print(top_topics)


plt.figure(figsize=(9, 5))

top_topics.sort_values().plot(kind="barh")

plt.title(
    "Top 10 Topics / Brands by Mentions",
    fontweight="bold"
)

plt.xlabel("Number of Posts")
plt.ylabel("Topic / Brand")

plt.tight_layout()
plt.show()


# 9. SENTIMENT BY TOP 5 TOPICS / BRANDS

top_5_topics = clean_df["Topic"].value_counts().head(5).index

topic_sentiment = pd.crosstab(
    clean_df[clean_df["Topic"].isin(top_5_topics)]["Topic"],
    clean_df[clean_df["Topic"].isin(top_5_topics)]["Sentiment"],
    normalize="index"
) * 100

print("\nSentiment by Top 5 Topics:")
print(topic_sentiment.round(2))


# Create stacked bar chart
topic_sentiment.plot(
    kind="bar",
    stacked=True,
    figsize=(10, 6)
)

plt.title(
    "Sentiment Distribution by Top Topics",
    fontweight="bold"
)

plt.xlabel("Topic / Brand")
plt.ylabel("Percentage (%)")
plt.legend(title="Sentiment")

plt.tight_layout()
plt.show()


# 10. MOST COMMON WORDS - WORD CLOUD

all_text = " ".join(clean_df["Text"])

wordcloud = WordCloud(
    width=1200,
    height=600,
    background_color="white",
    max_words=100
).generate(all_text)


plt.figure(figsize=(12, 6))

plt.imshow(wordcloud, interpolation="bilinear")

plt.axis("off")

plt.title(
    "Most Common Words in Social Media Posts",
    fontweight="bold"
)

plt.tight_layout()
plt.show()


# 11. SAVE CLEANED DATA

clean_df.to_csv(
    "twitter_training_cleaned.csv",
    index=False
)

print("\nCleaned dataset saved successfully!")


# 12. FINAL SUMMARY

print("\n========== TASK 4 SUMMARY ==========")

print("Total posts analyzed:", len(clean_df))

print("\nSentiment counts:")
print(sentiment_count)

print("\nTop 5 topics / brands:")
print(clean_df["Topic"].value_counts().head(5))

print("\nTask 4 completed successfully!")
