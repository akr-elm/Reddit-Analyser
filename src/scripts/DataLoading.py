import asyncio
import json
import motor.motor_asyncio as mot


async def retrieve_mongo_data(host:str,db:str, collection:str):
    try:
        client = mot.AsyncIOMotorClient(host)

        database = client[db]
        collection = database[collection]

        cursor = collection.find({})
        documents = await cursor.to_list(length=None)
        savedir = "../Reddit-Analyser/data/raw"
        for doc in documents:
            with open(f'{savedir}/posts.json', 'a', encoding='utf-8') as f:
                f.write(json.dumps(doc, default=str) + '\n')

        return True

    except Exception as e:
        return False