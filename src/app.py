from tools import *
from scripts.DataLoading import *

subrs = ['PoliticalDebate','NeutralPolitics','PoliticalDiscussion']
inst = Redditinstance()

postiterator = inst.get_data(subrs)

producer = Producer(bootstrap_servers="localhost:9092",topic_name="reddit")

for post in postiterator:
    producer.send("reddit",post)
    producer.flush()  

asyncio.run(retrieve_mongo_data("localhost:27007","TheSocialAnalysisdb","RedditPosts"))
    