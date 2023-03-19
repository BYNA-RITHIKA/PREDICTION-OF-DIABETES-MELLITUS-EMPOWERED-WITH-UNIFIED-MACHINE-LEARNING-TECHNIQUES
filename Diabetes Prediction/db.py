from flask import Flask
from flask_pymongo import pymongo
from app import app
CONNECTION_STRING = "mongodb+srv://Rithika:rithika@cluster0.b3kb3ec.mongodb.net/?retryWrites=true&w=majority"
client = pymongo.MongoClient(CONNECTION_STRING)
db = client.get_database('DiabetesPrediction')
user_collection = pymongo.collection.Collection(db, 'Users')

