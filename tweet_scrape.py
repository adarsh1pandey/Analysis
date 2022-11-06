from unicodedata import name
import numpy as np
import json
import tweepy
import pandas as pd
from tweepy import OAuthHandler
from pprint import pprint

@classmethod
def parse(cls, api, raw):
    status = cls.first_parse(api, raw)
    setattr(status, 'json', json.dumps(raw))
    return status

class TweetScraper(object):
    def __init__(self):
        self.df = pd.read_csv('./Data/id_annotated.tsv', sep='\t', header=None)

        self.access_token = "227333278-qRu2VFBTKktO1orp7kotaaSx1brdAhqAhWyx5e3j"
        self.access_token_secret = "KToUaDqXNCCndtE14okptPeqdtlmIwzahjgrOHgJ2vm9t"
        self.consumer_key = "UVMu9FNSWco0Uerxw7AP4fs77"
        self.consumer_secret = "cPtrBRPFCIPLKryjzFWXjBZRqpFyWGcKZrygBykDBn7UH5B1k5"

        self.auth = OAuthHandler(self.consumer_key, self.consumer_secret)
        self.auth.set_access_token(self.access_token, self.access_token_secret)
        self.api = tweepy.API(self.auth, wait_on_rate_limit=True)

    def get_tweet_ids(self):
        tweets = self.df[0]
        tweets = tweets.values
        tweet_list = []
        for ix in tweets:
            tweet_list.append(ix)
        return tweet_list

    def scrape_tweet_metadata(self, tweet_list):
        final_dict = {}
        tweet_count = len(tweet_list)
        ctr = 0
        try:
            for i in range(tweet_count):
                ix = tweet_list[i]
                print(ctr)
                ctr += 1
                cur = self.api.statuses_lookup([ix], include_entities=True)
                if len(cur)>0:
                    pprint(cur[0]._json)
                    final_dict[str(ix)] = cur[0]._json
        except tweepy.error.TweepError:
            final_dict[str(ix)] = {}
            print('Something went wrong, quitting...')
        return final_dict

    def write_json(self, file_name, final_dict):
        with open('./Data/'+file_name+'.json', 'w') as fp:
            json.dump('./Data/'+final_dict, fp, indent=4)

    def generate_follower_matrix(self, tweet_metadata):
        all_users = []
        for tweet in tweet_metadata.values():
            all_users.append(tweet['user']['id'])
        all_users = set(all_users)
        print(len(all_users))

        user_followers = {}
        for tweet in tweet_metadata.values():
            try:
                user_id = tweet['user']['id']
                screen_name = tweet['user']['screen_name']
                user_followers[str(user_id)] = []
                print(user_id)
                follower_ids = []
                for page in tweepy.Cursor(self.api.followers_ids, screen_name=screen_name).pages():
                    follower_ids.extend(page)
                print(follower_ids)
                for fol_id in follower_ids:
                    if fol_id in all_users:
                        user_followers[str(user_id)].append(fol_id)
            except tweepy.error.TweepError:
                # fill_tweets.append([])
                user_followers[str(user_id)] = []
                print('Something went wrong')
        return user_followers

    def generate_retweet_matrix(self, tweet_metadata):
        all_users = []
        for tweet in tweet_metadata.values():
            all_users.append(tweet['user']['id'])
        all_users = set(all_users)
        print(len(all_users))


        user_retweets = {}
        for tweet_id in tweet_metadata:
            try:
                print(tweet_id)
                user_id = tweet_metadata[tweet_id]['user']['id']
                user_retweets[str(user_id)] = []
                retweet_ids = []
                res = self.api.retweets(id=tweet_id)
                for rt in res:
                    rt = rt._json
                    pprint(rt)
                    rt_id = rt['user']['id']
                    if rt_id in all_users:
                        user_retweets[user_id].append(rt_id)
            except tweepy.TweepError:
                user_retweets[str(user_id)] = []
                print('Something went wrong')
        return user_retweets

    def scrape_user_history_data(self, tweet_metadata):
        all_users = []
        user_ids = []
        for tweet in tweet_metadata.values():
            all_users.append(tweet['user']['screen_name'])
            user_ids.append(tweet['user']['id'])

        tweepy.models.Status.first_parse = tweepy.models.Status.parse
        tweepy.models.Status.parse = parse

        tweepy.models.User.first_parse = tweepy.models.User.parse
        tweepy.models.User.parse = parse

        history = {}
        ctr = 0
        for user, id in zip(all_users, user_ids):
            ctr += 1
            print(ctr)
            try:
                tweets = self.api.user_timeline(screen_name=user,
                                            page=1,
                                            count=20,
                                            tweet_mode='extended',
                                            full_text=True)
            except:
                pass
            page=2

            while (True):
                try:
                    more_tweets = self.api.user_timeline(screen_name=user,
                                                page=page,
                                                count=20,
                                                tweet_mode='extended',
                                                full_text=True)
                    # There are no more tweets
                    if (len(more_tweets) == 0):
                        break
                    else:
                        page = page + 1
                        tweets = tweets + more_tweets
                except:
                    pass

            history[id] = tweets
        return history

if __name__ == '__main__':
    tweet_scraper = TweetScraper()
    tweet_id_list = tweet_scraper.get_tweet_ids()
    print(tweet_id_list)
    tweet_metadata = tweet_scraper.scrape_tweet_metadata(tweet_id_list)
    tweet_scraper.write_json('tweet_data', tweet_metadata)
    follower_metadata = tweet_scraper.generate_follower_matrix(tweet_metadata)
    tweet_scraper.write_json('user_data', follower_metadata)
    retweet_metadata = tweet_scraper.generate_retweet_matrix(tweet_metadata)
    tweet_scraper.write_json('rt_data', retweet_metadata)
    history_data = tweet_scraper.scrape_user_history_data(tweet_metadata)
    tweet_scraper.write_json('history_data', history_data)



