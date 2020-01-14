from flask import Flask
from flask import  jsonify, request
from recommender.recommender import Recommender
import mysql.connector

import numpy as np

app = Flask(__name__)
mydb = mysql.connector.connect(
  host="localhost",
  user="phpmyadmin",
  passwd="Thanh!23",
  database='realEstateSchema'
)
cursor = mydb.cursor()

# # -------------------------- HELPERS ---------------------
# def getPurchasedItems(customer_name):
# 	customer = session.query(Customer).filter(Customer.name == customer_name)[:1]
# 	if customer:
# 		items_list = []
# 		purchases = session.query(Purchcase).filter(Purchcase.customer_id == customer[0].id)
# 		[items_list.append(item.item_id) for item in purchases]
# 		return items_list
# 	else:
# 		return None

# def countItems():
# 	return session.query(Item).count()

# # TODO: handle users without items
# def buildVector(array_num):
# 	total_items = array_num.size() +1 
# 	v = np.zeros(total_items)
# 	purchases = getPurchasedItems(customer_name)
# 	if purchases:
# 		v[purchases] = 1
# 		return v[1:]
# 	else:
# 		return None

def buildGlobalMatrix():
	query = ("select id,area_num,price_num from news_item202001")
	cursor.execute(query)
	vectors = []
	list_id = []
	for (id,area_num, price_num) in cursor:
		list_id.append(id)
		v = np.array([area_num,price_num])
		vectors.append(v)
	return np.array(vectors), np.array(list_id)

@app.route('/api/real_estate/predict')
def predictNearRealestate():
	vectors = []
	list_id = []
	for i in request.args.get('include').split(","):
		print(i)
		query = ("select id,area_num,price_num from news_item"+i)
		cursor.execute(query)
		for (id,area_num, price_num) in cursor:
			list_id.append(str(i)+"_"+str(id))
			v = np.array([area_num,price_num])
			vectors.append(v)
	r = Recommender(np.array(vectors)) # pass the matrix to the model
	r.fit()
	query = ("select id,area_num,price_num from news_"+request.args.get('prefix')+" where id="+request.args.get('id'))
	cursor.execute(query)
	list_return=[]
	for (id,area_num, price_num) in cursor:
		list_return=r.getKNN(np.array([area_num,price_num]))
	result={}
	result['data']=",".join([str(list_id[i]) for i in list_return[0]])
	result['description']="Before '_' is prefix of partition, after that is ID of real estate"
	result['total']=list_return.shape[1]
	return jsonify(result)

# def insertEvent():
# 	global events
# 	events = events +1
# 	if events >= buffer:
# 		global matrix
# 		buildGlobalMatrix()
# 		global r 
# 		r = Recommender(matrix)
# 		print "Updating matrix"

# ------------------------------------------------------
# matrix,list_id = buildGlobalMatrix() # build the initial matrix
# r = Recommender(matrix) # pass the matrix to the model
# r.fit() # pass the matrix to the model
if __name__ == "__main__":
	app.run(debug=True)
