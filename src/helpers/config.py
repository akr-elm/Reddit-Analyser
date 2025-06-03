import logging 
import os
from logging.handlers import RotatingFileHandler


log_dir = os.path.join(os.path.dirname(__file__),'logs')
os.makedirs(log_dir, exist_ok=True)
log_file = os.path.join(log_dir, 'ailearning.log')


logger = logging.getLogger("AiLearning")
logger.setLevel(logging.INFO)
handler = RotatingFileHandler(log_file, maxBytes=5 * 1024 * 1024, backupCount=1, encoding='utf-8')
formatter = logging.Formatter("[%(asctime)s] %(levelname)s - %(message)s")
handler.setFormatter(formatter)
logger.handlers.clear()
logger.addHandler(handler)


client_id_env="CTTwsWgr3sFMS1Ba47bOEQ"
client_secret_env="ByXeNrRZ6f62lNqB14pqSWkiNYGSMA"
user_agent_env="intitaki01"
