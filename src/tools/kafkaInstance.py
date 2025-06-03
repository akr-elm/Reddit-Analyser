from kafka.admin import KafkaAdminClient, NewTopic
from kafka import KafkaProducer
from kafka.errors import NoBrokersAvailable, TopicAlreadyExistsError
import time
import logging
from helpers.config import *




###SETUP THE
logger = logging.getLogger(__name__)

def CreatedTopic(bootstrap_servers:str,topic_name:str, num_partitions = 3, replication_factor=1):
    try:
        admin_client = KafkaAdminClient(bootstrap_servers=bootstrap_servers)
        topic = NewTopic(name = topic_name, num_partitions=num_partitions, replication_factor=replication_factor)
        admin_client.create_topics([topic])
        return True
    except NoBrokersAvailable:
        logger.warning("No broker Found, retrying in 5 seconds")
    except TopicAlreadyExistsError:
        logger.info("Topic already exists")
        return True


def Producer(bootstrap_servers,topic_name):
    """You can change the port to the one where you defined your kafka server
        By default it's 9092"""
    max_retries = 3
    retry_count = 0
    
    while retry_count < max_retries:
        if CreatedTopic(bootstrap_servers, topic_name):
            try:
                producer = KafkaProducer(
                    bootstrap_servers=bootstrap_servers,
                    value_serializer=lambda v: v.encode('utf-8')
                )
                logger.info("Successfully created Kafka producer")
                return producer
            except Exception as e:  
                logger.error(f"Failed to create producer: {str(e)}")
                retry_count += 1
                time.sleep(5)
        else:
            retry_count += 1
            time.sleep(5)
    
    raise Exception("Failed to create Kafka producer after maximum retries")