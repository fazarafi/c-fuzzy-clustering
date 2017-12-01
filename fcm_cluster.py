from __future__ import division
import pandas as pd
import math
import numpy as np
import random

class Fcm_cluster(object):
    def __init__(self, m, eps, num_cluster):
        self.m = m # parameter, has to be greater than 1
        self.dataset = None
        self.centroids = []
        self.num_cluster = num_cluster # int
        self.num_dataset = None # int
        self.eps = eps # parameter, positive near zero
        self.m_matrix = None # matrix of float, size = num_dataset * num_cluster
        self.means = None # array of float, len = num_atr
        self.standard_deviation = None # array of float, len = num_atr
        self.z_score = None # matrix of float, size = dataset's size
        self.cluster = None # array of integer, len = num_dataset

    def main_process(self):
        self.init_matrix()
        self.iterate()
    
    def init_matrix(self):
        for i in range(self.num_dataset):
            max = 1.0
            for j in range(self.num_cluster):
                if (j==self.num_cluster-1):
                    self.m_matrix[i][j] = max;
                else:
                    self.m_matrix[i][j] = random.uniform(0.0,max)
                max -= self.m_matrix[i][j]
        
    def iterate(self):
        centroids = [[] for i in range(self.num_cluster)]
        stop = False
        while not(stop):
            prev_mat = self.copy_matrix()
            for j in range(self.num_cluster):
                sig_data_md = []
                sig_md = 0
                for i in range(self.num_dataset):
                    sig_data_md.append(self.time_att_md(self.dataset[i],pow(self.m_matrix[i][j],self.m)))
                    sig_md += pow(self.m_matrix[i][j],self.m)
                
                centroids[j] = self.divide_att_md(self.accumulate_att(sig_data_md),sig_md)
            self.update_matrix(centroids)
            print('max=',self.find_max_md_diff(prev_mat))
            print('eps=',self.eps)
            if (self.find_max_md_diff(prev_mat)<self.eps):
                stop = True    
            # stop = True


    def copy_matrix(self):
        dup = [[0 for i in range(self.num_cluster)] for j in range(self.num_dataset)]
        for i in range(self.num_dataset):
            dup[0] = self.m_matrix[0][:]
            # for j in range(self.num_cluster):
            #     dup

        return dup

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
                    sig_dist += pow(self.calc_dist(self.dataset[i],centroids[j])/self.calc_dist(self.dataset[i],centroids[k]),2/(self.m-1))

                self.m_matrix[i][j] = 1/sig_dist
    
    def find_max_md_diff(self, prev_mat):
        max_md_diff = 0
        for i in range(self.num_dataset):
        	for j in range(self.num_cluster):
        		diff = abs(self.m_matrix[i][j]-prev_mat[i][j])
        		# print(self.m_matrix[i][j],prev_mat[i][j])
        		if (diff > max_md_diff):
        			max_md_diff = diff
        return max_md_diff

    def manhattan_dist(data, centroid):
        total = 0.0
        # u can traverse data or centroid, it's the same
        i = 0
        for elem in centroid:
            total += abs(elem-data[i])
            i += 1
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
        self.num_dataset = len(self.dataset)
        self.m_matrix = [[0 for i in range(self.num_cluster)] for j in range(self.num_dataset)]

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
        self.num_dataset = len(self.dataset)
        self.m_matrix = [[0 for i in range(self.num_cluster)] for j in range(self.num_dataset)]

    # bad : space still exist
    def read_file3(self, filename):
        self.dataset = pd.read_csv(filename, header=None)
        self.num_dataset = len(self.dataset)
        self.m_matrix = [[0 for i in range(self.num_cluster)] for j in range(self.num_dataset)]

    def delete_column(self, column_idx):
        self.dataset = np.delete(self.dataset, np.s_[column_idx:column_idx+1], axis=1)
    
    # pre requisite : self.dataset
    # to be deleted = [1, 3, 5, 6, 7, 8, 9, 13, 14]
    def delete_columns(self, arr_column_idx):
        for idx in reversed(arr_column_idx):
            self.delete_column(idx)

    # pre requisite : self.dataset only contains numeric, no nominal, no missing values
    def cast_dataset_to_float(self):
        new_dataset = []
        for ins in self.dataset:
            new_ins = []
            for elem in ins:
                new_ins.append(float(elem))
            new_dataset.append(new_ins)
        self.dataset = new_dataset

    # WARNING!!, dataset has to end with maximum 1 new line.
    # More new line at the end of file will cause the delete_columns to error
    def get_dataset(self, filename):
        self.read_file(filename)
        self.delete_columns([1, 3, 5, 6, 7, 8, 9, 13, 14])
        self.cast_dataset_to_float()

    # Z-score
    # mean
    def calculate_means(self):
        means = []
        # inisialisasi
        for atr in self.dataset[0]:
            means.append(0)

        # hitung total
        for ins in self.dataset:
            i = 0
            for elem in ins:
                means[i] += elem
                i += 1

        # dibagi jumlah dataset
        i = 0
        for elem in means:
            means[i] = elem / self.num_dataset
            i += 1

        self.means = means
        return self.means

    # sd
    # pre requisite : self.means
    def calculate_sd(self):
        sd = []
        # inisialisasi
        for atr in self.dataset[0]:
            sd.append(0)

        # hitung total (x-xbar)^2
        for ins in self.dataset:
            i = 0
            for elem in ins:
                sd[i] += math.pow(elem-self.means[i], 2)
                i += 1

        # dibagi jumlah dataset - 1, trus di akar
        i = 0
        for elem in sd:
            sd[i] = math.sqrt(elem / (self.num_dataset-1))
            i += 1

        self.standard_deviation = sd
        return self.standard_deviation

    # z-score
    # a function
    def calculate_z_score(self):
        means = self.calculate_means()
        sd = self.calculate_sd()

        z_score = []
        for ins in self.dataset:
            j = 0
            z_score_ins = []
            for elem in ins:
                z_score_ins.append((elem-means[j])/sd[j])
                j += 1
            z_score.append(z_score_ins)

        self.z_score = z_score
        return self.z_score

    def set_dataset_to_z_score(self):
        self.dataset = self.calculate_z_score()

    # pre requisite : self.m_matrix
    def get_clusters(self):
        cluster = []

        for ins in self.m_matrix:
            membership_max = 0
            idx_max = 0
            idx = 0
            for membership in ins:
                if (membership>membership_max):
                   idx_max = idx
                   membership_max = membership
                idx += 1
            cluster.append(idx_max)

        self.cluster = cluster
        return self.cluster

    # pre requisite : self.cluster
    def print_clusters(self):
        arr_cluster = [] # consist of array of idx_ins
        i = 0

        while (i < self.num_cluster):
            idx_ins = 0
            arr_member = []

            for idx_clus in self.cluster:
                if (idx_clus == i):
                    arr_member.append(idx_ins)
                idx_ins += 1

            arr_cluster.append(arr_member)
            i += 1

        print("all =", arr_cluster)
        print("")
        i = 0
        for ins_member in arr_cluster:
            print("cluster - ", i, "=", ins_member)
            i += 1
        print("")
        i = 0
        for ins_member in arr_cluster:
            print("total member cluster -", i, "=", len(ins_member))
            i += 1

fcm = Fcm_cluster(m=10, eps=0.5001, num_cluster=2)
fcm.get_dataset('dataset\\CencusIncome.data.txt') # ngambil data dari file, hapus yg nominal
fcm.set_dataset_to_z_score() # optional, kalo error, division by zero, berarti sd = 0
fcm.main_process() # clustering
fcm.get_clusters() # dapetin array of cluster (tiap ins)
fcm.print_clusters() # print hasil
