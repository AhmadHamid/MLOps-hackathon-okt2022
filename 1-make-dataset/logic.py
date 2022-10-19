from functools import cache
from typing import Iterable
import pandas as pd
import tweepy
import praw
from tweepy.models import Status as Tweet
from praw.models.reddit.submission import Submission as RedditPost
import yaml

config = yaml.safe_load(open("secrets.yaml"))


@cache
def get_twitter_client():
    twitter_auth = tweepy.OAuth1UserHandler(
        config["twitter"]["API_key"],
        config["twitter"]["API_secret"],
        config["twitter"]["access_token"],
        config["twitter"]["access_secret"],
    )
    cli = tweepy.API(twitter_auth)
    return cli


def get_tweets(user, limit):
    cli = get_twitter_client()
    tweets = tweepy.Cursor(
        cli.user_timeline, screen_name=user, tweet_mode="extended"
    ).items(limit)
    return list(tweets)


def tweets_to_df(tweets: Iterable[Tweet]):
    return pd.DataFrame(
        {
            # "obj": tweet,
            "id": tweet.id_str,
            "created_at": tweet.created_at,
            "source": f"tweeter/{tweet.author.screen_name}",
            "text": tweet.full_text,
            "url": (tweet.entities["urls"] or [{}])[0].get("display_url"),
        }
        for tweet in tweets
    )


@cache
def get_reddit_client():
    return praw.Reddit(
        client_id=config["reddit"]["client_id"],
        client_secret=config["reddit"]["client_secret"],
        user_agent="random snouglou",
    )


def get_reddits_old(subreddit_name, feed, limit):
    assert feed in ["hot", "new", "top"]
    cli = get_reddit_client()
    subreddit = cli.subreddit(subreddit_name)
    return list(getattr(subreddit, feed)(limit=limit))  # subreddit.feed(limit)


def reddits_to_df_old(subms: Iterable[RedditPost]):
    return pd.DataFrame(
        {
            # "obj": subm,
            "id": subm.id,
            "created_at": subm.created_utc,
            "source": f"reddit/{subm.subreddit.display_name}",
            "text": subm.title,
            "url": subm.url,
        }
        for subm in subms
    )

def get_reddits(subreddit_name, feed, limit):
    sublist = list()
    api = get_reddit_client()
    subreddit = api.subreddit(subreddit_name)

    for subm in subreddit.top(limit=limit):
        if ((subm.link_flair_text == "Asshole") or (subm.link_flair_text == "Not the A-hole") or (subm.link_flair_text == "Everyone Sucks") or (subm.link_flair_text == "No A-holes here")):
            sublist.append(subm)
    
    return sublist

def reddits_to_df(subms: Iterable[RedditPost]):
    return pd.DataFrame(
            {
                "id": subm.id,
                "created_at": subm.created_utc,
                "source": subm.subreddit.display_name,
                "author": subm.author,
                "url": subm.url,
                "title": subm.title,
                "text": subm.selftext,
                "flair": subm.link_flair_text
            }
            for subm in subms
        )