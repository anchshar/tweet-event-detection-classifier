#Import the necessary methods from tweepy library
from tweepy.streaming import StreamListener
from tweepy import OAuthHandler
from tweepy import Stream

import os
#Variables that contains the user credentials to access Twitter API 
access_token = ""
access_token_secret = ""
consumer_key = ""
consumer_secret = ""

count = 0
#data_file = open("Marriage", "w+")
#data_file = open("Baby", "w+")
data_file = open("NewHouse", "w+")


#This is a basic listener that just prints received tweets to stdout.
class StdOutListener(StreamListener):
    
    def on_data(self, data):
        global count
        global data_file
        if count < 4000:
            data_file.write(data + "\n")
            count = count + 1
        print str(count) + " tweets written" 
        return True

    def on_error(self, status):
        print status


if __name__ == '__main__':

    #This handles Twitter authetification and the connection to Twitter Streaming API
    l = StdOutListener()
    auth = OAuthHandler(consumer_key, consumer_secret)
    auth.set_access_token(access_token, access_token_secret)
    stream = Stream(auth, l)
    #stream2 = Stream(auth,l)
    #This line filter Twitter Streams to capture data by the keywords: 'python', 'javascript', 'ruby'
    #stream.filter(track=['Wedding','Tied the knot','married','marriage','honeymoon','wed'])
    #stream.filter(track=['Baby boy','baby girl', 'new born','became a father','became a mother','am a father','am a mother'])
    stream.filter(track=['new house','moved in','bought a house','moved to','new place','new home'])
