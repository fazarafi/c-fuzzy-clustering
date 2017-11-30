from __future__ import division
import pandas as pd
import math
import numpy as np

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
    
    def manhattan_dist(data, centroid):
        total = 0.0
        # u can traverse data or centroid, it's the same
        i = 0
        for elem in centroid:
            total += abs(elem-data[i])
            i += 1
        # TODO: cara cari distance
        return total

    def euclidean_dist(data, centroid):
        total = 0.0
        # u can traverse data or centroid, it's the same
        i = 0
        for elem in centroid:
            total += math.pow(elem-data[i], 2)
            i += 1
        # TODO: cara cari distance
        return math.sqrt(total)

    def calc_dist(self, data, centroid):
        return Fcm_cluster.euclidean_dist(data, centroid)

    def read_file(self, filename):
        self.dataset = []
        file = open(filename)
        for line in file:
            content = line.split(', ')
            self.dataset.append(content)

    # bad : missing value gone
    def read_file2(self, filename):
        self.dataset = []
        file = open(filename)
        for line in file:
            instance = []
            content = line.split(', ')
            for elem in content:
                try:
                    instance.append(float(elem))
                except ValueError:
                    pass
            self.dataset.append(instance)

    # bad : space still exist
    def read_file3(self, filename):
        self.dataset = pd.read_csv(filename, header=None)

    def delete_colummn(self, column_idx):
        self.dataset = np.delete(self.dataset, np.s_[column_idx:column_idx+1], axis=1)
    
    def delete_columns(self, arr_column_idx):
        for idx in reversed(arr_column_idx):
            self.delete_colummn(idx)



fcm = Fcm_cluster(m=2, dataset=[[1,2,3],[1,2,2]], eps=0.01, num_cluster=2)
fcm.read_file('dataset\\nyoba.dat')

print(fcm.dataset)
# fcm.delete_colummn(1)
fcm.delete_columns([1, 3, 5, 6, 7, 8, 9, 13, 14])
print(fcm.dataset)

# fcm.main_process()
# print(Fcm_cluster.euclidean_dist([1,2,3],[1,0,1]))
# arr = [[1,2,3],[1,2,2]]
# for i in range(len(arr)):
#     print(i)
# print(arr[0][1]+arr[1][1])
