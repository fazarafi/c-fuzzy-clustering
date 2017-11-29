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
		for i in range(0,self.num_dataset):
			for j in range(0,self.num_cluster):
				self.m_matrix[i][j] = 0.1
		
	def iterate(self):
		centroids = [0 for i in range(self.num_cluster)]
		stop = False
		while not(stop):
			prev_mat = self.m_matrix
			for j in range(0,self.num_cluster):
				sig_data_md = 1
				sig_md = 1
				for i in range(0,self.num_dataset):
					sig_data_md += (self.m_matrix[i][j]**self.m)*self.dataset[i]
					sig_md += self.m_matrix[i][j]**self.m

				centroids[j] = sig_data_md/sig_md
			print(centroids)
			
			self.update_matrix(centroids)
			if (self.find_max_md()<self.eps):
				stop = True	
			stop = True


	def calc_centroid(self):
		return ''

	def update_matrix(self, centroids):	
		print('update')
		for i in range(0,self.num_dataset):
			for j in range(0,self.num_cluster):
				sig_dist = 0
				for k in range(0,self.num_cluster):
					sig_dist += (self.calc_dist(self.dataset[i],centroids[j])/
						self.calc_dist(self.dataset[i],centroids[k]))**(2/(self.m-1))

				self.m_matrix[i][j] = 1/sig_dist


	def find_max_md(self):
		max_md = 0
		for i in range(0,self.num_dataset):
			for j in range(0,self.num_cluster):
				if (self.m_matrix[i][j] > max_md):
					max_md = self.m_matrix[i][j]

		return max_md


	def main_process(self):
		self.init_matrix()
		self.iterate()


	
	def calc_dist(self, data, centroid):
		# TODO: cara cari distance
		return 0.5


fcm = Fcm_cluster(m=2, dataset=[0,2,3,2,1], eps=0.01, num_cluster=2)
fcm.main_process()





