# -*- coding:utf-8
from pymongo import MongoClient

class MongoUtil:
  
  def __init__(self, host, port, user, passwd, authenticate_dbName, find_dbName):
    self.client = MongoClient("%s:%d"%(host, port))
    self.client[authenticate_dbName].authenticate(user, passwd, authenticate_dbName, mechanism='SCRAM-SHA-1')
    self.db = self.client[find_dbName]

  def __del__(self):
    self.close()

  def close(self):
    self.client.close()

  def findAll(self, collectionName, condition=None, fields=None, sort=None):
    if condition is None and sort is None:
      return self.db[collectionName].find({}, fields)
    elif condition is None:
      return self.db[collectionName].find({}, fields).sort(sort)
    elif sort is None:
      return self.db[collectionName].find(condition, fields)
    else:
      return self.db[collectionName].find(condition, fields).sort(sort)

  def findOne(self, collectionName, condition=None, fields=None):
    if condition is None:
      return self.db[collectionName].find_one({}, fields)
    else:
      return self.db[collectionName].find_one(condition, fields)

  def count(self, collectionName, condition=None):
    if condition is None:
      return self.db[collectionName].count()
    else:
     return self.db[collectionName].count(condition)

  def distinct(self, collectionName, fields, query=None):
    if query is None:
      return self.db[collectionName].distinct(fields)
    else:
      return self.db[collectionName].distinct(fields, query)

  def insertOne(self, collectionName, dict):
    return self.db[collectionName].insert_one(dict).inserted_id

  def updateOne(self, collectionName, condition, update):
    return self.db[collectionName].update_one(condition, update).modified_count

  def updateMany(self, collectionName, condition, update):
    return self.db[collectionName].update_many(condition, update).modified_count

  def deleteMany(self, collectionName, codition):
    return self.db[collectionName].delete_many(codition).deleted_count
