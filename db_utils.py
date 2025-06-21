import pymongo
from urllib.parse import quote_plus

def get_mongo_client(username: str, password: str, cluster: str = "x4cluster.n6xsnhl.mongodb.net") -> pymongo.MongoClient:
    """
    Returns an authenticated MongoClient for the specified cluster.

    Args:
        username (str): MongoDB username.
        password (str): MongoDB password.
        cluster (str): MongoDB cluster host (default: "x4cluster.n6xsnhl.mongodb.net").

    Returns:
        pymongo.MongoClient: Authenticated MongoDB client instance.
    """
    encoded_username = quote_plus(username)
    encoded_password = quote_plus(password)
    uri = f"mongodb+srv://{encoded_username}:{encoded_password}@{cluster}/"
    return pymongo.MongoClient(uri)

def insert_dataframe_to_mongo(client, dataframe, collection_name: str, db_name: str = 'squadv2'):
    """
    Converts any ndarray in the DataFrame to lists and inserts it into the specified MongoDB collection.

    Args:
        client: pymongo.MongoClient instance.
        dataframe (pd.DataFrame): DataFrame to insert.
        collection_name (str): Name of the target collection.
        db_name (str): Name of the database.
    """
    def convert_ndarray_to_list(obj):
        if isinstance(obj, dict):
            return {k: convert_ndarray_to_list(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_ndarray_to_list(i) for i in obj]
        elif hasattr(obj, "tolist"):  # for numpy.ndarray
            return obj.tolist()
        else:
            return obj

    cleaned_df = dataframe.applymap(convert_ndarray_to_list)
    records = cleaned_df.to_dict("records")

    client[db_name][collection_name].insert_many(records)
    print(f"{len(records)} documents inserted into '{db_name}.{collection_name}'")

def drop_collection(client, collection_name: str, db_name: str = 'squadv2'):
    """
    Drops a collection from a MongoDB database.

    Args:
        client: pymongo.MongoClient instance.
        collection_name (str): Name of the collection to drop.
        db_name (str): Name of the database.
    """
    if collection_name in client[db_name].list_collection_names():
        client[db_name].drop_collection(collection_name)
        print(f"Collection '{db_name}.{collection_name}' dropped.")
    else:
        print(f"Collection '{db_name}.{collection_name}' does not exist.")

def read_collection(client, collection_name: str, db_name: str = 'squadv2', as_dataframe: bool = False, projection: dict = None):
    """
    Reads all documents from a MongoDB collection.

    Args:
        client: pymongo.MongoClient instance.
        db_name (str): Name of the database.
        collection_name (str): Name of the collection to read.
        as_dataframe (bool): If True, returns a DataFrame; otherwise, a list of dicts.
        projection (dict): Fields to include/exclude, e.g. {"_id": 0} to exclude the _id.

    Returns:
        list[dict] or pandas.DataFrame: Retrieved documents.
    """
    import pandas as pd

    collection = client[db_name][collection_name]
    cursor = collection.find({}, projection or {})

    data = list(cursor)
    print(f"{len(data)} documents read from '{db_name}.{collection_name}'")

    if as_dataframe:
        return pd.DataFrame(data)
    return data

def update_collection_from_dataframe(client, db_name: str, collection_name: str, dataframe, match_field: str):
    """
    Updates documents in a MongoDB collection using data from a DataFrame.

    Args:
        client: pymongo.MongoClient instance.
        db_name (str): Name of the database.
        collection_name (str): Name of the collection to update.
        dataframe (pd.DataFrame): DataFrame containing updated data.
        match_field (str): Unique field used to match documents to update.
    """
    def convert_ndarray_to_list(obj):
        if isinstance(obj, dict):
            return {k: convert_ndarray_to_list(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_ndarray_to_list(i) for i in obj]
        elif hasattr(obj, "tolist"):  # e.g. numpy.ndarray
            return obj.tolist()
        else:
            return obj

    cleaned_df = dataframe.applymap(convert_ndarray_to_list)
    collection = client[db_name][collection_name]

    updated_count = 0
    for record in cleaned_df.to_dict("records"):
        match_value = record.get(match_field)
        if match_value is None:
            continue  # Skip rows without a match field

        result = collection.update_one(
            {match_field: match_value},  # filter
            {"$set": record},            # update
            upsert=False                 # do not insert if not exists
        )
        if result.modified_count > 0:
            updated_count += 1

    print(f"{updated_count} documents updated in '{db_name}.{collection_name}'")

def list_databases_and_collections(client):
    """
    Prints all databases and collections available on the MongoDB client.

    Args:
        client: pymongo.MongoClient instance.
    """
    db_names = client.list_database_names()
    if not db_names:
        print("No databases found.")
        return

    print("Available databases and collections:")
    for db_name in db_names:
        db = client[db_name]
        collection_names = db.list_collection_names()
        print(f" {db_name}")
        for col in collection_names:
            print(f"   └── {col}")