import praw
from helpers import client_id_env,client_secret_env,user_agent_env
import json


class Redditinstance:
    def __init__(self):
        self.Reddit =self.create_reddit_instance()
    def create_reddit_instance(self):
        """Create and return a PRAW Reddit instance."""
        # Access environment variables
        client_id = client_id_env
        client_secret = client_secret_env
        user_agent = user_agent_env
        
        # Validate that required environment variables are set
        if not client_id or not client_secret or not user_agent:
            raise ValueError("Missing required Reddit API credentials in environment variables.")

        # Create and return the Reddit instance
        return praw.Reddit(
            client_id = client_id,
            client_secret = client_secret,
            user_agent=user_agent
        )
    
    def get_data(self,subredditlist: list):
        for subredditelement in subredditlist:
            subreddit = self.Reddit.subreddit(subredditelement)
            for submission in subreddit.hot(limit=20):
                if submission.stickied:
                    continue
                if not submission.selftext.strip():
                    continue
                if submission.author and submission.author.name == "AutoModerator":
                    continue
                post_id = submission.id
                title = submission.title
                body = submission.selftext
                score = submission.score
                num_comments = submission.num_comments


                submission.comments.replace_more(limit=0)
                comments = [comment.body for comment in submission.comments]
                

                post = json.dumps({
                'subreddit': subredditelement,
                'post_id': post_id,
                'title': title,
                'post_body': body,
                'score': score,
                'num_comments': num_comments,
                'comments': comments[1:] #Removing the first comment that repeats the 
            })
                yield post
