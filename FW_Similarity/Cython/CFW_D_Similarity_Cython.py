#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 08/09/17

@author: Maurizio Ferrari Dacrema
"""

from KNN.ItemKNNCustomSimilarityRecommender import ItemKNNCustomSimilarityRecommender

from Base.SimilarityMatrixRecommender import SimilarityMatrixRecommender
from Base.Incremental_Training_Early_Stopping import Incremental_Training_Early_Stopping
from Base.Recommender import Recommender
from Base.Recommender_utils import check_matrix
from Base.IR_feature_weighting import okapi_BM_25, TF_IDF


from Base.Similarity.Compute_Similarity import Compute_Similarity


import time, os, sys, subprocess
import numpy as np
import scipy.sparse as sps




class EvaluatorCFW_D_wrapper(object):

    def __init__(self, evaluator_object, ICM_target, model_to_use = "best"):

        self.evaluator_object = evaluator_object
        self.ICM_target = ICM_target.copy()

        assert model_to_use in ["best", "incremental"], "EvaluatorCFW_D_wrapper: model_to_use must be either 'best' or 'incremental'. Provided value is: ''".format(model_to_use)

        if model_to_use is "incremental":
            self.use_incremental = True
        else:
            self.use_incremental = False


    def evaluateRecommender(self, recommender_object):

        recommender_object.set_ICM_and_recompute_W(self.ICM_target, recompute_w = False)

        # Use either best model or incremental one
        recommender_object.compute_W_sparse(use_incremental = self.use_incremental)

        return self.evaluator_object.evaluateRecommender(recommender_object)







class CFW_D_Similarity_Cython(Incremental_Training_Early_Stopping, SimilarityMatrixRecommender, Recommender):

    RECOMMENDER_NAME = "CFW_D_Similarity_Cython"

    INIT_TYPE_VALUES = ["random", "one", "zero", "BM25", "TF-IDF"]


    def __init__(self, URM_train, ICM, S_matrix_target,
                 recompile_cython = False):


        super(CFW_D_Similarity_Cython, self).__init__()

        if (URM_train.shape[1] != ICM.shape[0]):
            raise ValueError("Number of items not consistent. URM contains {} but ICM contains {}".format(URM_train.shape[1],
                                                                                                          ICM.shape[0]))

        if(S_matrix_target.shape[0] != S_matrix_target.shape[1]):
            raise ValueError("Items imilarity matrix is not square: rows are {}, columns are {}".format(S_matrix_target.shape[0],
                                                                                                        S_matrix_target.shape[1]))

        if(S_matrix_target.shape[0] != ICM.shape[0]):
            raise ValueError("Number of items not consistent. S_matrix contains {} but ICM contains {}".format(S_matrix_target.shape[0],
                                                                                                          ICM.shape[0]))

        self.URM_train = check_matrix(URM_train, 'csr')
        self.S_matrix_target = check_matrix(S_matrix_target, 'csr')
        self.ICM = check_matrix(ICM, 'csr')


        self.n_items = self.URM_train.shape[1]
        self.n_users = self.URM_train.shape[0]
        self.n_features = self.ICM.shape[1]

        self.sparse_weights = True


        if recompile_cython:
            print("Compiling in Cython")
            self.runCompilationScript()
            print("Compilation Complete")



    def runCompilationScript(self):

        # Run compile script setting the working directory to ensure the compiled file are contained in the
        # appropriate subfolder and not the project root

        compiledModuleSubfolder = "/FW_Similarity/Cython"
        fileToCompile = 'CFW_D_Similarity_Cython_SGD.pyx'

        command = ['python',
                   'compileCython.py',
                   fileToCompile,
                   'build_ext',
                   '--inplace'
                   ]

        output = subprocess.check_output(' '.join(command), shell=True, cwd=os.getcwd() + compiledModuleSubfolder)


        try:

            command = ['cython',
                       fileToCompile,
                       '-a'
                       ]

            output = subprocess.check_output(' '.join(command), shell=True, cwd=os.getcwd() + compiledModuleSubfolder)

        except:
            pass

        print("Compiled module saved in subfolder: {}".format(compiledModuleSubfolder))

        # Command to run compilation script
        # python compileCython.py CFW_D_Similarity_Cython_SGD.pyx build_ext --inplace

        # Command to generate html report
        # cython -a CFW_D_Similarity_Cython_SGD.pyx






    def generateTrainData_low_ram(self):

        print(self.RECOMMENDER_NAME + ": Generating train data")

        start_time_batch = time.time()


        # Here is important only the structure
        self.similarity = Compute_Similarity(self.ICM.T, shrink=0, topK=self.topK, normalize=False)
        S_matrix_contentKNN = self.similarity.compute_similarity()
        S_matrix_contentKNN = check_matrix(S_matrix_contentKNN, "csr")


        self.writeLog(self.RECOMMENDER_NAME + ": Collaborative S density: {:.2E}, nonzero cells {}".format(
            self.S_matrix_target.nnz/self.S_matrix_target.shape[0]**2, self.S_matrix_target.nnz))

        self.writeLog(self.RECOMMENDER_NAME + ": Content S density: {:.2E}, nonzero cells {}".format(
            S_matrix_contentKNN.nnz/S_matrix_contentKNN.shape[0]**2, S_matrix_contentKNN.nnz))


        if self.normalize_similarity:

            # Compute sum of squared
            sum_of_squared_features = np.array(self.ICM.T.power(2).sum(axis=0)).ravel()
            sum_of_squared_features = np.sqrt(sum_of_squared_features)



        num_common_coordinates = 0

        estimated_n_samples = int(S_matrix_contentKNN.nnz*(1+self.add_zeros_quota)*1.2)

        self.row_list = np.zeros(estimated_n_samples, dtype=np.int32)
        self.col_list = np.zeros(estimated_n_samples, dtype=np.int32)
        self.data_list = np.zeros(estimated_n_samples, dtype=np.float64)

        num_samples = 0


        for row_index in range(self.n_items):

            start_pos_content = S_matrix_contentKNN.indptr[row_index]
            end_pos_content = S_matrix_contentKNN.indptr[row_index+1]

            content_coordinates = S_matrix_contentKNN.indices[start_pos_content:end_pos_content]

            start_pos_target = self.S_matrix_target.indptr[row_index]
            end_pos_target = self.S_matrix_target.indptr[row_index+1]

            target_coordinates = self.S_matrix_target.indices[start_pos_target:end_pos_target]

            # Chech whether the content coordinate is associated to a non zero target value
            # If true, the content coordinate has a collaborative non-zero value
            # if false, the content coordinate has a collaborative zero value
            is_common = np.in1d(content_coordinates, target_coordinates)

            num_common_in_current_row = is_common.sum()
            num_common_coordinates += num_common_in_current_row


            for index in range(len(is_common)):

                if num_samples == estimated_n_samples:
                    dataBlock = 1000000
                    self.row_list = np.concatenate((self.row_list, np.zeros(dataBlock, dtype=np.int32)))
                    self.col_list = np.concatenate((self.col_list, np.zeros(dataBlock, dtype=np.int32)))
                    self.data_list = np.concatenate((self.data_list, np.zeros(dataBlock, dtype=np.float64)))

                if is_common[index]:
                    # If cell exists in target matrix, add its value
                    # Otherwise it will remain zero with a certain probability

                    col_index = content_coordinates[index]

                    self.row_list[num_samples] = row_index
                    self.col_list[num_samples] = col_index

                    new_data_value = self.S_matrix_target[row_index, col_index]

                    if self.normalize_similarity:
                        new_data_value *= sum_of_squared_features[row_index]*sum_of_squared_features[col_index]

                    self.data_list[num_samples] = new_data_value

                    num_samples += 1


                elif np.random.rand() <= self.add_zeros_quota:

                    col_index = content_coordinates[index]

                    self.row_list[num_samples] = row_index
                    self.col_list[num_samples] = col_index
                    self.data_list[num_samples] = 0.0

                    num_samples += 1



            if time.time() - start_time_batch > 30 or num_samples == S_matrix_contentKNN.nnz*(1+self.add_zeros_quota):

                print(self.RECOMMENDER_NAME + ": Generating train data. Sample {} ( {:.2f} %) ".format(
                    num_samples, num_samples/ S_matrix_contentKNN.nnz*(1+self.add_zeros_quota) *100))

                sys.stdout.flush()
                sys.stderr.flush()

                start_time_batch = time.time()


        self.writeLog(self.RECOMMENDER_NAME + ": Content S structure has {} out of {} ( {:.2f}%) nonzero collaborative cells".format(
            num_common_coordinates, S_matrix_contentKNN.nnz, num_common_coordinates/S_matrix_contentKNN.nnz*100))



        # Discard extra cells at the left of the array
        self.row_list = self.row_list[:num_samples]
        self.col_list = self.col_list[:num_samples]
        self.data_list = self.data_list[:num_samples]


        data_nnz = sum(np.array(self.data_list)!=0)
        data_sum = sum(self.data_list)

        collaborative_nnz = self.S_matrix_target.nnz
        collaborative_sum = sum(self.S_matrix_target.data)

        self.writeLog(self.RECOMMENDER_NAME + ": Nonzero collaborative cell sum is: {:.2E}, average is: {:.2E}, "
                      "average over all collaborative data is {:.2E}".format(
                      data_sum, data_sum/data_nnz, collaborative_sum/collaborative_nnz))

        if self.evaluator_object is not None and self.show_max_performance:
            self.computeMaxTheoreticalPerformance()




    def computeMaxTheoreticalPerformance(self):

        # Max performance would be if we were able to learn the content matrix having for each non-zero cell exactly
        # the value that appears in the collaborative similarity

        print(self.RECOMMENDER_NAME + ": Computing collaborative performance")

        recommender = ItemKNNCustomSimilarityRecommender()
        recommender.fit(self.S_matrix_target, self.URM_train)

        results_run = self.evaluator_object(recommender)

        self.writeLog(self.RECOMMENDER_NAME + ": Collaborative performance is: {}".format(results_run))


        print(self.RECOMMENDER_NAME + ": Computing top structural performance")

        n_items = self.ICM.shape[0]

        S_optimal = sps.csr_matrix((self.data_list, (self.row_list, self.col_list)), shape=(n_items, n_items))
        S_optimal.eliminate_zeros()

        recommender = ItemKNNCustomSimilarityRecommender()
        recommender.fit(S_optimal, self.URM_train)

        results_run = self.evaluator_object(recommender)

        self.writeLog(self.RECOMMENDER_NAME + ": Top structural performance is: {}".format(results_run))



    def writeLog(self, string):

        print(string)
        sys.stdout.flush()
        sys.stderr.flush()

        if self.logFile is not None:
            self.logFile.write(string + "\n")
            self.logFile.flush()


    def compute_W_sparse(self, use_incremental = False):

        if use_incremental:
            feature_weights = self.D_incremental
        else:
            feature_weights = self.D_best


        self.similarity = Compute_Similarity(self.ICM.T, shrink=0, topK=self.topK,
                                            normalize=self.normalize_similarity, row_weights=feature_weights)

        self.W_sparse = self.similarity.compute_similarity()
        self.sparse_weights = True



    def set_ICM_and_recompute_W(self, ICM_new, recompute_w = True):

        self.ICM = ICM_new.copy()

        if recompute_w:
            self.compute_W_sparse(use_incremental = False)





    def fit(self, validation_every_n = 5, show_max_performance = False, logFile = None, precompute_common_features = True,
            learning_rate = 0.01, positive_only_weights = True, init_type ="zero", normalize_similarity = False,
            use_dropout = True, dropout_perc = 0.3,
            l1_reg = 0.0, l2_reg = 0.0, epochs = 50, topK = 300,
            add_zeros_quota = 0.0,
            sgd_mode = 'adagrad', gamma = 0.9, beta_1 = 0.9, beta_2 = 0.999,
            stop_on_validation = False, lower_validatons_allowed = 5, validation_metric = "MAP",
            evaluator_object = None):


        if init_type not in self.INIT_TYPE_VALUES:
           raise ValueError("Value for 'init_type' not recognized. Acceptable values are {}, provided was '{}'".format(self.INIT_TYPE_VALUES, init_type))


        # Import compiled module
        from FW_Similarity.Cython.CFW_D_Similarity_Cython_SGD import CFW_D_Similarity_Cython_SGD

        self.logFile = logFile

        if validation_every_n is not None:
            self.validation_every_n = validation_every_n
        else:
            self.validation_every_n = np.inf

        self.evaluator_object = evaluator_object


        self.show_max_performance = show_max_performance
        self.positive_only_weights = positive_only_weights
        self.normalize_similarity = normalize_similarity
        self.learning_rate = learning_rate
        self.add_zeros_quota = add_zeros_quota
        self.l1_reg = l1_reg
        self.l2_reg = l2_reg
        self.epochs = epochs
        self.topK = topK


        self.generateTrainData_low_ram()

        weights_initialization = None

        if init_type == "random":
            weights_initialization = np.random.normal(0.001, 0.1, self.n_features).astype(np.float64)
        elif init_type == "one":
            weights_initialization = np.ones(self.n_features, dtype=np.float64)
        elif init_type == "zero":
            weights_initialization = np.zeros(self.n_features, dtype=np.float64)
        elif init_type == "BM25":
            weights_initialization = np.ones(self.n_features, dtype=np.float64)
            self.ICM = self.ICM.astype(np.float32)
            self.ICM = okapi_BM_25(self.ICM)

        elif init_type == "TF-IDF":
            weights_initialization = np.ones(self.n_features, dtype=np.float64)
            self.ICM = self.ICM.astype(np.float32)
            self.ICM = TF_IDF(self.ICM)

        else:
            raise ValueError("CFW_D_Similarity_Cython: 'init_type' not recognized")


        # Instantiate fast Cython implementation
        self.FW_D_Similarity = CFW_D_Similarity_Cython_SGD(self.row_list, self.col_list, self.data_list,
                                                      self.n_features,
                                                      self.ICM,
                                                      precompute_common_features = precompute_common_features,
                                                      non_negative_weights = self.positive_only_weights,
                                                      weights_initialization = weights_initialization,
                                                      use_dropout = use_dropout, dropout_perc = dropout_perc,
                                                      learning_rate=learning_rate,
                                                      l1_reg=l1_reg,
                                                      l2_reg=l2_reg,
                                                      sgd_mode=sgd_mode,
                                                      gamma = gamma, beta_1=beta_1, beta_2=beta_2)



        print(self.RECOMMENDER_NAME + ": Initialization completed")


        self._train_with_early_stopping(epochs, validation_every_n, stop_on_validation,
                                    validation_metric, lower_validatons_allowed, evaluator_object,
                                    algorithm_name = self.RECOMMENDER_NAME)


        self.compute_W_sparse()

        sys.stdout.flush()





    def _initialize_incremental_model(self):
        self.D_incremental = self.FW_D_Similarity.get_weights()
        self.D_best = self.D_incremental.copy()


    def _update_incremental_model(self):
        self.D_incremental = self.FW_D_Similarity.get_weights()


    def _update_best_model(self):
        self.D_best = self.D_incremental.copy()


    def _run_epoch(self, num_epoch):
        self.loss = self.FW_D_Similarity.fit()
        self.D_incremental = self.FW_D_Similarity.get_weights()









    def saveModel(self, folder_path, file_name = None):

        import pickle

        if file_name is None:
            file_name = self.RECOMMENDER_NAME

        print("{}: Saving model in file '{}'".format(self.RECOMMENDER_NAME, folder_path + file_name))

        dictionary_to_save = {
            "D_best":self.D_best,
            "topK":self.topK,
            "sparse_weights":self.sparse_weights,
            "W_sparse":self.W_sparse,
            "normalize_similarity":self.normalize_similarity
        }

        pickle.dump(dictionary_to_save,
                    open(folder_path + file_name, "wb"),
                    protocol=pickle.HIGHEST_PROTOCOL)


        print("{}: Saving complete".format(self.RECOMMENDER_NAME))






