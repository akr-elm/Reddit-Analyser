import os
import sys

# Add the project root to Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(project_root)

from sparkInstance import SparkStream

spark = SparkStream()
##Pipeline definition:
df = spark.read_from_kafka()
parsed_df = spark.tansform_stream(df)
cleaned_df = spark.cleaning(parsed_df)
spark.write_to_mongo(cleaned_df)
###Retrive data for the sentiment analysis
