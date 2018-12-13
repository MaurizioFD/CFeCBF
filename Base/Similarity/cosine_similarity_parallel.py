#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 23/10/17

@author: Maurizio Ferrari Dacrema
"""

import numpy as np
import time, sys
import scipy.sparse as sps
import multiprocessing
from functools import partial

try:
    from Base.Cython.cosine_similarity import Cosine_Similarity
except ImportError:
    print("Unable to load Cython Cosine_Similarity, reverting to Python")
    from Base.cosine_similarity import Compute_Similarity



class Cosine_Similarity_Parallel:


    def __init__(self, dataMatrix, topK=100, shrink = 0, normalize = True,
                 mode = "cosine", row_weights = None, n_workers = 4):
        """
        Computes the cosine similarity on the columns of dataMatrix
        If it is computed on URM=|users|x|items|, pass the URM as is.
        If it is computed on ICM=|items|x|features|, pass the ICM transposed.
        :param dataMatrix:
        :param topK:
        :param shrink:
        :param normalize:           If True divide the dot product by the product of the norms
        :param row_weights:         Multiply the values in each row by a specified value. Array
        :param mode:    "cosine"    computes Cosine similarity
                        "adjusted"  computes Adjusted Cosine, removing the average of the users
                        "pearson"   computes Pearson Correlation, removing the average of the items
                        "jaccard"   computes Jaccard similarity for binary interactions using Tanimoto
                        "tanimoto"  computes Tanimoto coefficient for binary interactions

        """

        super(Cosine_Similarity_Parallel, self).__init__()

        self.topK = topK
        self.shrink = shrink
        self.normalize = normalize
        self.n_columns = dataMatrix.shape[1]
        self.n_rows = dataMatrix.shape[0]
        self.n_workers = n_workers

        # if self.n_columns < self.n_workers:
        #     print("Cosine_Similarity_Parallel: number of columns lower than number of workers, setting workers to number of columns")
        #     self.n_workers = self.n_columns

        self.mode = mode
        self.row_weights = row_weights

        self.dataMatrix = dataMatrix.copy()



    def compute_similarity(self):

        print("Cosine_Similarity_Parallel: begin computation")

        start_time = time.time()

        start_end_col_tuple_list = []

        # This way every worker has at least 1 column
        if self.n_columns < self.n_workers:
             self.n_workers = self.n_columns

        columns_per_worker = int(self.n_columns/self.n_workers)



        # if self.n_columns < columns_per_worker:
        #     columns_per_worker = self.n_columns
        #
        # num_blocks = int(self.n_columns/columns_per_worker)
        #
        #
        #
        # if num_blocks*columns_per_worker < self.n_columns:
        #     num_blocks += 1



        for worker_index in range(self.n_workers):

            start_col = worker_index * columns_per_worker
            end_col = min([(worker_index + 1) * columns_per_worker, self.n_columns])

            if worker_index == self.n_workers-1:
                end_col = self.n_columns

            start_end_col_tuple_list.append((start_col, end_col))



        global progress_status
        progress_status = Progress_Status(start_time)

        # run_block_partial = partial(self.run_block, progress_status = progress_status)

        pool = multiprocessing.Pool(processes=multiprocessing.cpu_count(), maxtasksperchild=1)
        resultList = pool.map(self.run_block, start_end_col_tuple_list)

        pool.close()


        self.W_sparse = sps.csr_matrix((self.n_columns, self.n_columns))


        for block_result in resultList:
            self.W_sparse += block_result



        end_time = time.time()
        columnPerSec = self.n_columns/(end_time-start_time)

        print("Cosine_Similarity_Parallel: Computation terminated, {:.2f} column/sec, elapsed time {:.2f} min".format(
            columnPerSec, (end_time-start_time)/60))

        return self.W_sparse







    def run_block(self, start_end_col_tuple):

        start_time = time.time()

        start_col = start_end_col_tuple[0]
        end_col = start_end_col_tuple[1]

        print("Cosine_Similarity_Parallel: Created worker for columns {}-{}".format(
            start_col, end_col))


        similarity_cython = Cosine_Similarity(self.dataMatrix, topK = self.topK,
                                             shrink=self.shrink, normalize = self.normalize,
                                             mode = self.mode, row_weights = self.row_weights)




        # Compute from start_col to end_col excluded
        W_sparse = similarity_cython.compute_similarity(start_col, end_col)

        end_time = time.time()
        columnPerSec = (end_col - start_col)/(end_time-start_time)

        print("Cosine_Similarity_Parallel: Computed similarity for columns {}-{}, {:.2f} column/sec, elapsed time {:.2f} min.".format(
            start_col, end_col, columnPerSec, (end_time-start_time)/60))


        progress_status.add_completed_columns(end_col-start_col)

        columnPerSec = progress_status.get_completed_columns_counter/(end_time-progress_status.get_start_time)

        print("Cosine_Similarity_Parallel: Overall progress is {:.2f} %, {:.2f} column/sec, elapsed time {:.2f} min.".format(
            progress_status.get_completed_columns_counter/self.n_columns*100, columnPerSec, (end_time-progress_status.get_start_time)/60))

        return W_sparse





class Progress_Status(object):

    def __init__(self, start_time):
        self.completed_columns_counter = multiprocessing.Value('i', 0)
        self.start_time = multiprocessing.Value('d', start_time)

        self.lock = multiprocessing.Lock()

    def add_completed_columns(self, col_number):
        with self.lock:
            self.completed_columns_counter.value += col_number

    @property
    def get_completed_columns_counter(self):
        return self.completed_columns_counter.value

    @property
    def get_start_time(self):
        return self.start_time.value