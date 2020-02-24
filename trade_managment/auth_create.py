from yahoo_oauth import OAuth2
import json

creds = {'consumer_key': 'dj0yJmk9NU16SmEzZ0VjT3BBJmQ9WVdrOU5HNUpXSFZQTnpZbWNHbzlNQS0tJnM9Y29uc3VtZXJzZWNyZXQmc3Y9MCZ4PTQz', 
'consumer_secret': 'aa4a654d50e6f7df98376dd0ab6da75b5d8a5227'}
with open('oauth3.json', "w") as f:
   f.write(json.dumps(creds))
oauth = OAuth2(None, None, from_file='oauth3.json')