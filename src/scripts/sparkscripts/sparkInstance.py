from pyspark.sql import SparkSession
from pyspark.sql.functions import col, from_json, expr, udf
from pyspark.sql.types import StructType, StructField, StringType, ArrayType, IntegerType
from pyspark.sql import functions as F
import os
import re


class SparkStream:
    def __init__(self,collection="RedditPosts", mongoURL = "mongodb://mongodb:27017"):
        self.kafkaserver = os.getenv("KAFKA_SERVER")
        self.collection = collection
        self.database = "TheSocialAnalysisdb"

        self.spark = SparkSession.builder \
            .appName("TheSocialAnalysis") \
            .config("spark.jars.packages",
                    "org.apache.spark:spark-sql-kafka-0-10_2.12:3.4.1,"
                    "org.mongodb.spark:mongo-spark-connector_2.12:10.2.1") \
            .config("spark.mongodb.read.connection.uri", mongoURL) \
            .config("spark.mongodb.write.connection.uri", mongoURL) \
            .config("spark.mongodb.output.database", self.database) \
            .config("spark.mongodb.output.collection", self.collection) \
            .getOrCreate()
        self.schema = StructType([
            StructField("subreddit", StringType(), True),
            StructField("post_id", StringType(), True),
            StructField("title", StringType(), True),
            StructField("post_body", StringType(),True),
            StructField("score", IntegerType() ,False),
            StructField("num_comments",IntegerType(),True),
            StructField("comments", ArrayType(StringType()),True)
        ])

        # Register UDF once during initialization
        
    
    def read_from_kafka(self):
        return self.spark.readStream \
                .format("kafka") \
                .option("kafka.bootstrap.servers", self.kafkaserver) \
                .option("subscribe", "reddit") \
                .option("startingOffsets", "earliest") \
                .load()
    
    def tansform_stream(self,df):
        parsed_df = df.select(
            from_json(col("value").cast("string"),self.schema).alias("data")
        ).select("data.*")

        parsed_df = parsed_df.withColumn("id", expr("uuid()"))

        return parsed_df
    
    @staticmethod
    def clean_text(text):
        if text is None:
            return text

        try:
            text = text.lower()
            patterns = [
                (r'https?://\S+|www\.\S+', ''),       # URLs
                (r'@\w+|\#\w+', ''),                  # Mentions and hashtags
                (r'[áàäâãåæ]', 'a'),                  # Accents
                (r'<[^>]+>', ''),                     # HTML tags
                (r'\s+', ' '),                        # Extra whitespace
                (r'^(\*\*r/.*|r/.*)', ''),            # Subreddit prefixes
                (r'\[.*?\]', ''),                     # Square brackets
                (r'[^\x00-\x7F]+', '')                # Remove unicode / emojis
            ]

            for pattern, repl in patterns:
                try:
                    text = re.sub(pattern, repl, text)
                except Exception:
                    continue

            return text.strip()

        except Exception:
            return text

    clean_text_udf = udf(clean_text.__func__, StringType())
    def cleaning(self, df):
        
        try:
            # Clean post_body
            cleaned_df = df.withColumn(
                "post_body",
                F.when(
                    F.col("post_body").isNotNull(),
                    self.clean_text_udf(F.col("post_body"))
                ).otherwise(F.col("post_body"))
            )

            # Clean comments array
            cleaned_df = cleaned_df.withColumn(
                "comments",
                F.when(
                    F.col("comments").isNotNull(),
                    F.transform(F.col("comments"), self.clean_text_udf)
                ).otherwise(F.col("comments"))
            )

            return cleaned_df
        except Exception as e:
            print(f"Error in cleaning: {str(e)}")
            return df
    


    
    def writing(self, df, batch_id):
   
        try:
            # Try to write to MongoDB
            df.write \
                .format("mongodb") \
                .mode("append") \
                .option("database", self.database) \
                .option("collection", self.collection) \
                .save()
            print(f"Successfully wrote batch {batch_id} to MongoDB")
            
        except Exception as e:
            if "Collection already exists" in str(e):
                # If collection exists, try to append to it
                try:
                    df.write \
                        .format("mongodb") \
                        .mode("append") \
                        .option("database", self.database) \
                        .option("collection", self.collection) \
                        .save()
                    print(f"Successfully appended batch {batch_id} to existing collection")
                except Exception as inner_e:
                    print(f"Error appending to existing collection: {str(inner_e)}")
                    raise inner_e
            else:
                print(f"Error writing batch {batch_id} to MongoDB: {str(e)}")
                raise e
        
    def write_to_mongo(self, df):
        query = df.writeStream \
            .foreachBatch(self.writing) \
            .option("checkpointLocation", "/tmp/spark-streaming-checkpoint") \
            .outputMode('append') \
            .start()
        query.awaitTermination()
