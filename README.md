# Sentiment-Analysis-on-Barak-Obama-Tweets
To answer the question "Which category of emotion is most frequent for Barak Obama?" I've done sentiment Analysis on Barak Obama's Tweet.
I've categorised Tweets into 3 catergory as Positive, Negative and Neutral. To categorized the tweets I followed below steps.&lt;BR>
1. Tweet is a Positive Tweet if number of Positive words in a Tweet is greater than number of Negative words.&lt;BR> 
2. Tweet is a Negative tweet if Negative words are greater than Positive words. &lt;BR> 
3. If number of Positive and negative words are equal in a tweet then its a Neutral Tweet.&lt;BR>   

To do this I build a vocabulary of Positive and negative word list from Datasets provided below. 
1. Positive Words Dataset: https://gist.github.com/mkulakowski2/4289437 
2. Negative Words Dataset: https://gist.github.com/mkulakowski2/4289441count  
3. Twitter Sentiment Analysis Dataset: http://thinknook.com/twitter-sentiment-analysis-training-corpus-dataset-2012-09-22/  
Results: Positive Tweets are highest with count of 24331.  

To identify the Hypothesis: "Most of Barak Obama's tweets will be regarding healthcare".  
I scraped the health related words from "http://www.english-for-students.com/Health-Vocabulary.html" and created healthWordList.
And considered a tweet as a healtcare related tweet if atleast one word in a tweet is also is healthWordList.  
Results: The Hypothesis is false as only 6768 Tweets out of 27346 overall tweets are related to Health care.  

After completing this I checked the accuracy of categorised tweets using Naive Bayes Algorithm. 
I trained the model with Sentiment Analysis Dataset with different tweets and checked the accuracy of my categorization of tweets.
