from sklearn.neighbors import NearestNeighbors
import numpy as np
import random	
import os.path
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.externals import joblib
from sklearn.externals import joblib
import heapq

class Recommender:

	def __init__(self, train):
		self.train = train

	def vector_model(self,to_predict):
		cos = cosine_similarity(self.train,to_predict)
		return self.train[(heapq.nlargest(2,xrange(len(cos)),cos.take))[1:][0]]

	def fit(self):
		self.nbrs = NearestNeighbors(n_neighbors=10).fit(self.train) # choose the 5 nearest neighbors

	def predict(self, to_predict):
		indices,distances = self.getKNN(to_predict)
		return [self.train[i] for i in indices]

	def getKNN(self,to_predict):
		# distances, indices = self.nbrs.kneighbors(self.train , n_neighbors=10,return_distance=False) # but take only the closest
		return self.nbrs.kneighbors(to_predict.reshape(1, -1),return_distance=False)
		# his is the proof that the best match is in the cluster