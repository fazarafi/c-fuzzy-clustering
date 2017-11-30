from __future__ import division


class Fcm_cluster(object):
	def __init__(self, m, dataset, eps, num_cluster):
		self.m = m
		self.dataset = dataset
		self.centroids = []
		self.num_cluster = num_cluster
		self.num_dataset = len(dataset)
		self.eps = eps
		self.m_matrix = [[0 for i in range(self.num_cluster)] for j in range(self.num_dataset)]

	def init_matrix(self):
		for i in range(self.num_dataset):
			for j in range(self.num_cluster):
				self.m_matrix[i][j] = 0.1*j+0.2
		
	def iterate(self):
		centroids = [[] for i in range(self.num_cluster)]
		stop = False
		while not(stop):
			prev_mat = self.m_matrix
			for j in range(self.num_cluster):
				sig_data_md = []
				sig_md = 0
				for i in range(self.num_dataset):
					sig_data_md.append(self.time_att_md(self.dataset[i],(self.m_matrix[i][j]**self.m)))
					sig_md += self.m_matrix[i][j]**self.m
				
				centroids[j] = self.divide_att_md(self.accumulate_att(sig_data_md),sig_md)

			self.update_matrix(centroids)
			# print('max=',self.find_max_md())
			# print('eps=',self.eps)
			if (self.find_max_md()<self.eps):
				stop = True	
			stop = True


	def time_att_md(self, attributes, md):
		result = []
		for i in range(len(attributes)):
			result.append(attributes[i]*md)
		return result

	def divide_att_md(self, attributes, md):
		result = []
		for i in range(len(attributes)):
			result.append(attributes[i]/md)
		return result		

	def accumulate_att(self, arr):
		if len(arr)==0:
			return []
		else:
			result = [0 for i in range(len(arr[0]))]
			for i in range(len(arr)):
				for j in range(len(arr[0])):
					result[j] += arr[i][j]
			return result

	def update_matrix(self, centroids):	
		for i in range(self.num_dataset):
			for j in range(self.num_cluster):
				sig_dist = 0
				for k in range(self.num_cluster):
					sig_dist += (self.calc_dist(self.dataset[i],centroids[j])/
						self.calc_dist(self.dataset[i],centroids[k]))**(2/(self.m-1))

				self.m_matrix[i][j] = 1/sig_dist


	def find_max_md(self):
		max_md = 0
		for i in range(self.num_dataset):
			for j in range(self.num_cluster):
				if (self.m_matrix[i][j] > max_md):
					max_md = self.m_matrix[i][j]

		return max_md


	def main_process(self):
		self.init_matrix()
		self.iterate()


	
	def calc_dist(self, data, centroid):
		# TODO: cara cari distance

		return 0.5

	def readfile(filename):
		self.dataset = []
		with open(filename) as f:
    		content = f.readlines()
    		self.dataset.append(content)


import csv
import numpy

# filename = 'dataset\\CencusIncome.data.txt'
# raw_data = open(filename, 'rt')
# reader = csv.reader(raw_data, delimiter=',', quoting=csv.QUOTE_NONE)
# x = list(reader)
# print(x[0])

fcm = Fcm_cluster(m=2, dataset=[[1,2,3],[1,2,2]], eps=0.01, num_cluster=2)
fcm.readfile('dataset\\CencusIncome.data.txt')

print(self.dataset[0])

fcm.main_process()
arr = [[1,2,3],[1,2,2]]
for i in range(len(arr)):
	print(i)
print(arr[0][1]+arr[1][1])
