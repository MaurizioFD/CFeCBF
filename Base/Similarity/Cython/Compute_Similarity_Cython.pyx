"""
Created on 23/10/17

@author: Maurizio Ferrari Dacrema
"""

#cython: boundscheck=False
#cython: wraparound=True
#cython: initializedcheck=False
#cython: language_level=3
#cython: nonecheck=False
#cython: cdivision=True
#cython: unpack_method_calls=True
#cython: overflowcheck=False


import time, sys

import numpy as np
cimport numpy as np

from cpython.array cimport array, clone

from libc.math cimport sqrt



import scipy.sparse as sps
from Base.Recommender_utils import check_matrix

#
# ctypedef struct data_pointer_s:
#     long start_position
#     long num_elements




cdef class Compute_Similarity_Cython:

    cdef int TopK
    cdef long n_columns, n_rows

    cdef double[:] this_item_weights
    cdef int[:] this_item_weights_mask, this_item_weights_id
    cdef int this_item_weights_counter

    cdef int[:] user_to_item_row_ptr, user_to_item_cols
    cdef int[:] item_to_user_rows, item_to_user_col_ptr
    cdef double[:] user_to_item_data, item_to_user_data
    cdef double[:] sumOfSquared, sumOfSquared_to_1_minus_alpha, sumOfSquared_to_alpha
    cdef int shrink, normalize, adjusted_cosine, pearson_correlation, tanimoto_coefficient, asymmetric_cosine, dice_coefficient, tversky_coefficient
    cdef float asymmetric_alpha, tversky_alpha, tversky_beta

    cdef int use_row_weights
    cdef double[:] row_weights

    cdef double[:,:] W_dense

    def __init__(self, dataMatrix, topK = 100, shrink=0, normalize = True,
                 asymmetric_alpha = 0.5, tversky_alpha = 1.0, tversky_beta = 1.0,
                 similarity = "cosine", row_weights = None):
        """
        Computes the cosine similarity on the columns of dataMatrix
        If it is computed on URM=|users|x|items|, pass the URM as is.
        If it is computed on ICM=|items|x|features|, pass the ICM transposed.
        :param dataMatrix:
        :param topK:
        :param shrink:
        :param normalize:           If True divide the dot product by the product of the norms
        :param row_weights:         Multiply the values in each row by a specified value. Array
        :param asymmetric_alpha     Coefficient alpha for the asymmetric cosine
        :param similarity:  "cosine"        computes Cosine similarity
                            "adjusted"      computes Adjusted Cosine, removing the average of the users
                            "asymmetric"    computes Asymmetric Cosine
                            "pearson"       computes Pearson Correlation, removing the average of the items
                            "jaccard"       computes Jaccard similarity for binary interactions using Tanimoto
                            "dice"          computes Dice similarity for binary interactions
                            "tversky"       computes Tversky similarity for binary interactions
                            "tanimoto"      computes Tanimoto coefficient for binary interactions

        """
        """
        Asymmetric Cosine as described in: 
        Aiolli, F. (2013, October). Efficient top-n recommendation for very large scale binary rated datasets. In Proceedings of the 7th ACM conference on Recommender systems (pp. 273-280). ACM.
        
        """

        super(Compute_Similarity_Cython, self).__init__()

        self.n_columns = dataMatrix.shape[1]
        self.n_rows = dataMatrix.shape[0]
        self.shrink = shrink
        self.normalize = normalize
        self.asymmetric_alpha = asymmetric_alpha
        self.tversky_alpha = tversky_alpha
        self.tversky_beta = tversky_beta

        self.adjusted_cosine = False
        self.asymmetric_cosine = False
        self.pearson_correlation = False
        self.tanimoto_coefficient = False
        self.dice_coefficient = False
        self.tversky_coefficient = False

        if similarity == "adjusted":
            self.adjusted_cosine = True
        elif similarity == "asymmetric":
            self.asymmetric_cosine = True
        elif similarity == "pearson":
            self.pearson_correlation = True
        elif similarity == "jaccard" or similarity == "tanimoto":
            self.tanimoto_coefficient = True
            # Tanimoto has a specific kind of normalization
            self.normalize = False

        elif similarity == "dice":
            self.dice_coefficient = True
            self.normalize = False

        elif similarity == "tversky":
            self.tversky_coefficient = True
            self.normalize = False

        elif similarity == "cosine":
            pass
        else:
            raise ValueError("Cosine_Similarity: value for paramether 'mode' not recognized."
                             " Allowed values are: 'cosine', 'pearson', 'adjusted', 'asymmetric', 'jaccard', 'tanimoto',"
                             "dice, tversky."
                             " Passed value was '{}'".format(similarity))


        self.TopK = min(topK, self.n_columns)
        self.this_item_weights = np.zeros(self.n_columns, dtype=np.float64)
        self.this_item_weights_id = np.zeros(self.n_columns, dtype=np.int32)
        self.this_item_weights_mask = np.zeros(self.n_columns, dtype=np.int32)
        self.this_item_weights_counter = 0

        # Copy data to avoid altering the original object
        dataMatrix = dataMatrix.copy()





        if self.adjusted_cosine:
            dataMatrix = self.applyAdjustedCosine(dataMatrix)
        elif self.pearson_correlation:
            dataMatrix = self.applyPearsonCorrelation(dataMatrix)
        elif self.tanimoto_coefficient or self.dice_coefficient or self.tversky_coefficient:
            dataMatrix = self.useOnlyBooleanInteractions(dataMatrix)



        # Compute sum of squared values to be used in normalization
        self.sumOfSquared = np.array(dataMatrix.power(2).sum(axis=0), dtype=np.float64).ravel()

        # Tanimoto does not require the square root to be applied
        if not (self.tanimoto_coefficient or self.dice_coefficient or self.tversky_coefficient):
            self.sumOfSquared = np.sqrt(self.sumOfSquared)

        if self.asymmetric_cosine:
            self.sumOfSquared_to_1_minus_alpha = np.power(self.sumOfSquared, 2 * (1 - self.asymmetric_alpha))
            self.sumOfSquared_to_alpha = np.power(self.sumOfSquared, 2 * self.asymmetric_alpha)


        # Apply weight after sumOfSquared has been computed but before the matrix is
        # split in its inner data structures
        self.use_row_weights = False

        if row_weights is not None:

            if dataMatrix.shape[0] != len(row_weights):
                raise ValueError("Cosine_Similarity: provided row_weights and dataMatrix have different number of rows."
                                 "Row_weights has {} rows, dataMatrix has {}.".format(len(row_weights), dataMatrix.shape[0]))


            self.use_row_weights = True
            self.row_weights = np.array(row_weights, dtype=np.float64)





        dataMatrix = check_matrix(dataMatrix, 'csr')

        self.user_to_item_row_ptr = dataMatrix.indptr
        self.user_to_item_cols = dataMatrix.indices
        self.user_to_item_data = np.array(dataMatrix.data, dtype=np.float64)

        dataMatrix = check_matrix(dataMatrix, 'csc')
        self.item_to_user_rows = dataMatrix.indices
        self.item_to_user_col_ptr = dataMatrix.indptr
        self.item_to_user_data = np.array(dataMatrix.data, dtype=np.float64)




        if self.TopK == 0:
            self.W_dense = np.zeros((self.n_columns,self.n_columns))





    cdef useOnlyBooleanInteractions(self, dataMatrix):
        """
        Set to 1 all data points
        :return:
        """

        cdef long index

        for index in range(len(dataMatrix.data)):
            dataMatrix.data[index] = 1

        return dataMatrix



    cdef applyPearsonCorrelation(self, dataMatrix):
        """
        Remove from every data point the average for the corresponding column
        :return:
        """

        cdef double[:] sumPerCol
        cdef int[:] interactionsPerCol
        cdef long colIndex, innerIndex, start_pos, end_pos
        cdef double colAverage


        dataMatrix = check_matrix(dataMatrix, 'csc')


        sumPerCol = np.array(dataMatrix.sum(axis=0), dtype=np.float64).ravel()
        interactionsPerCol = np.diff(dataMatrix.indptr)


        #Remove for every row the corresponding average
        for colIndex in range(self.n_columns):

            if interactionsPerCol[colIndex]>0:

                colAverage = sumPerCol[colIndex] / interactionsPerCol[colIndex]

                start_pos = dataMatrix.indptr[colIndex]
                end_pos = dataMatrix.indptr[colIndex+1]

                innerIndex = start_pos

                while innerIndex < end_pos:

                    dataMatrix.data[innerIndex] -= colAverage
                    innerIndex+=1


        return dataMatrix



    cdef applyAdjustedCosine(self, dataMatrix):
        """
        Remove from every data point the average for the corresponding row
        :return:
        """

        cdef double[:] sumPerRow
        cdef int[:] interactionsPerRow
        cdef long rowIndex, innerIndex, start_pos, end_pos
        cdef double rowAverage

        dataMatrix = check_matrix(dataMatrix, 'csr')

        sumPerRow = np.array(dataMatrix.sum(axis=1), dtype=np.float64).ravel()
        interactionsPerRow = np.diff(dataMatrix.indptr)


        #Remove for every row the corresponding average
        for rowIndex in range(self.n_rows):

            if interactionsPerRow[rowIndex]>0:

                rowAverage = sumPerRow[rowIndex] / interactionsPerRow[rowIndex]

                start_pos = dataMatrix.indptr[rowIndex]
                end_pos = dataMatrix.indptr[rowIndex+1]

                innerIndex = start_pos

                while innerIndex < end_pos:

                    dataMatrix.data[innerIndex] -= rowAverage
                    innerIndex+=1


        return dataMatrix





    cdef int[:] getUsersThatRatedItem(self, long item_id):
        return self.item_to_user_rows[self.item_to_user_col_ptr[item_id]:self.item_to_user_col_ptr[item_id+1]]

    cdef int[:] getItemsRatedByUser(self, long user_id):
        return self.user_to_item_cols[self.user_to_item_row_ptr[user_id]:self.user_to_item_row_ptr[user_id+1]]



    #
    # cdef data_pointer_s getUsersThatRatedItem_pointer(self, long item_id):
    #
    #     cdef data_pointer_s pointer = data_pointer_s()
    #
    #     pointer.start_position = self.item_to_user_col_ptr[item_id]
    #     pointer.num_elements = self.item_to_user_col_ptr[item_id+1] - pointer.start_position
    #
    #     return pointer
    #
    # cdef data_pointer_s getItemsRatedByUser_pointer(self, long user_id):
    #
    #
    #     cdef data_pointer_s pointer = data_pointer_s()
    #
    #     pointer.start_position = self.user_to_item_row_ptr[user_id]
    #     pointer.num_elements = self.user_to_item_row_ptr[user_id+1] - pointer.start_position
    #
    #     return pointer





    cdef computeItemSimilarities(self, long item_id_input):
        """
        For every item the cosine similarity against other items depends on whether they have users in common. The more
        common users the higher the similarity.
        
        The basic implementation is:
        - Select the first item
        - Loop through all other items
        -- Given the two items, get the users they have in common
        -- Update the similarity for all common users
        
        That is VERY slow due to the common user part, in which a long data structure is looped multiple times.
        
        A better way is to use the data structure in a different way skipping the search part, getting directly the
        information we need.
        
        The implementation here used is:
        - Select the first item
        - Initialize a zero valued array for the similarities
        - Get the users who rated the first item
        - Loop through the users
        -- Given a user, get the items he rated (second item)
        -- Update the similarity of the items he rated
        
        
        """

        # Create template used to initialize an array with zeros
        # Much faster than np.zeros(self.n_columns)
        #cdef array[double] template_zero = array('d')
        #cdef array[double] result = clone(template_zero, self.n_columns, zero=True)


        cdef long user_index, user_id, item_index, item_id, item_id_second

        cdef int[:] users_that_rated_item = self.getUsersThatRatedItem(item_id_input)
        cdef int[:] items_rated_by_user

        cdef double rating_item_input, rating_item_second, row_weight

        # Clean previous item
        for item_index in range(self.this_item_weights_counter):
            item_id = self.this_item_weights_id[item_index]
            self.this_item_weights_mask[item_id] = False
            self.this_item_weights[item_id] = 0.0

        self.this_item_weights_counter = 0



        # Get users that rated the items
        for user_index in range(len(users_that_rated_item)):

            user_id = users_that_rated_item[user_index]
            rating_item_input = self.item_to_user_data[self.item_to_user_col_ptr[item_id_input]+user_index]

            if self.use_row_weights:
                row_weight = self.row_weights[user_id]
            else:
                row_weight = 1.0

            # Get all items rated by that user
            items_rated_by_user = self.getItemsRatedByUser(user_id)

            for item_index in range(len(items_rated_by_user)):

                item_id_second = items_rated_by_user[item_index]

                # Do not compute the similarity on the diagonal
                if item_id_second != item_id_input:
                    # Increment similairty
                    rating_item_second = self.user_to_item_data[self.user_to_item_row_ptr[user_id]+item_index]

                    self.this_item_weights[item_id_second] += rating_item_input*rating_item_second*row_weight


                    # Update global data structure
                    if not self.this_item_weights_mask[item_id_second]:

                        self.this_item_weights_mask[item_id_second] = True
                        self.this_item_weights_id[self.this_item_weights_counter] = item_id_second
                        self.this_item_weights_counter += 1




    def compute_similarity(self, start_col=None, end_col=None):
        """
        Compute the similarity for the given dataset
        :param self:
        :param start_col: column to begin with
        :param end_col: column to stop before, end_col is excluded
        :return:
        """

        cdef int print_block_size = 500

        cdef int itemIndex, innerItemIndex, item_id, local_topK
        cdef long long topKItemIndex

        cdef long long[:] top_k_idx

        # Declare numpy data type to use vetor indexing and simplify the topK selection code
        cdef np.ndarray[long, ndim=1] top_k_partition, top_k_partition_sorting
        cdef np.ndarray[np.float64_t, ndim=1] this_item_weights_np
        #cdef double[:] this_item_weights

        cdef long processedItems = 0

        # Data structure to incrementally build sparse matrix
        # Preinitialize max possible length
        cdef double[:] values = np.zeros((self.n_columns*self.TopK))
        cdef int[:] rows = np.zeros((self.n_columns*self.TopK,), dtype=np.int32)
        cdef int[:] cols = np.zeros((self.n_columns*self.TopK,), dtype=np.int32)
        cdef long sparse_data_pointer = 0

        cdef int start_col_local = 0, end_col_local = self.n_columns

        cdef array[double] template_zero = array('d')



        if start_col is not None and start_col>0 and start_col<self.n_columns:
            start_col_local = start_col

        if end_col is not None and end_col>start_col_local and end_col<self.n_columns:
            end_col_local = end_col






        start_time = time.time()
        last_print_time = start_time

        itemIndex = start_col_local

        # Compute all similarities for each item
        while itemIndex < end_col_local:

            processedItems += 1

            if processedItems % print_block_size==0 or processedItems==end_col_local:

                current_time = time.time()

                # Set block size to the number of items necessary in order to print every 30 seconds
                itemPerSec = processedItems/(time.time()-start_time)

                print_block_size = int(itemPerSec*30)

                if current_time - last_print_time > 30  or processedItems==end_col_local:

                    print("Similarity column {} ( {:2.0f} % ), {:.2f} column/sec, elapsed time {:.2f} min".format(
                        processedItems, processedItems*1.0/(end_col_local-start_col_local)*100, itemPerSec, (time.time()-start_time) / 60))

                    last_print_time = current_time

                    sys.stdout.flush()
                    sys.stderr.flush()


            # Computed similarities go in self.this_item_weights
            self.computeItemSimilarities(itemIndex)


            # Apply normalization and shrinkage, ensure denominator != 0
            if self.normalize:
                for innerItemIndex in range(self.n_columns):

                    if self.asymmetric_cosine:
                        self.this_item_weights[innerItemIndex] /= self.sumOfSquared_to_alpha[itemIndex] * self.sumOfSquared_to_1_minus_alpha[innerItemIndex]\
                                                             + self.shrink + 1e-6

                    else:
                        self.this_item_weights[innerItemIndex] /= self.sumOfSquared[itemIndex] * self.sumOfSquared[innerItemIndex]\
                                                             + self.shrink + 1e-6

            # Apply the specific denominator for Tanimoto
            elif self.tanimoto_coefficient:
                for innerItemIndex in range(self.n_columns):
                    self.this_item_weights[innerItemIndex] /= self.sumOfSquared[itemIndex] + self.sumOfSquared[innerItemIndex] -\
                                                         self.this_item_weights[innerItemIndex] + self.shrink + 1e-6

            elif self.dice_coefficient:
                for innerItemIndex in range(self.n_columns):
                    self.this_item_weights[innerItemIndex] /= self.sumOfSquared[itemIndex] + self.sumOfSquared[innerItemIndex] +\
                                                         self.shrink + 1e-6

            elif self.tversky_coefficient:
                for innerItemIndex in range(self.n_columns):
                    self.this_item_weights[innerItemIndex] /= self.this_item_weights[innerItemIndex] + \
                                                              (self.sumOfSquared[itemIndex]-self.this_item_weights[innerItemIndex])*self.tversky_alpha + \
                                                              (self.sumOfSquared[innerItemIndex]-self.this_item_weights[innerItemIndex])*self.tversky_beta +\
                                                              self.shrink + 1e-6

            elif self.shrink != 0:
                for innerItemIndex in range(self.n_columns):
                    self.this_item_weights[innerItemIndex] /= self.shrink


            if self.TopK == 0:

                for innerItemIndex in range(self.n_columns):
                    self.W_dense[innerItemIndex,itemIndex] = self.this_item_weights[innerItemIndex]

            else:

                # Sort indices and select TopK
                # Using numpy implies some overhead, unfortunately the plain C qsort function is even slower
                #top_k_idx = np.argsort(this_item_weights) [-self.TopK:]

                # Sorting is done in three steps. Faster then plain np.argsort for higher number of items
                # because we avoid sorting elements we already know we don't care about
                # - Partition the data to extract the set of TopK items, this set is unsorted
                # - Sort only the TopK items, discarding the rest
                # - Get the original item index
                #



                #this_item_weights_np = clone(template_zero, self.this_item_weights_counter, zero=False)
                this_item_weights_np = np.zeros(self.n_columns, dtype=np.float64)

                # Add weights in the same ordering as the self.this_item_weights_id data structure
                for innerItemIndex in range(self.this_item_weights_counter):
                    item_id = self.this_item_weights_id[innerItemIndex]
                    this_item_weights_np[innerItemIndex] = - self.this_item_weights[item_id]


                local_topK = min([self.TopK, self.this_item_weights_counter])

                # Get the unordered set of topK items
                top_k_partition = np.argpartition(this_item_weights_np, local_topK-1)[0:local_topK]
                # Sort only the elements in the partition
                top_k_partition_sorting = np.argsort(this_item_weights_np[top_k_partition])
                # Get original index
                top_k_idx = top_k_partition[top_k_partition_sorting]



                # Incrementally build sparse matrix, do not add zeros
                for innerItemIndex in range(len(top_k_idx)):

                    topKItemIndex = top_k_idx[innerItemIndex]

                    item_id = self.this_item_weights_id[topKItemIndex]

                    if self.this_item_weights[item_id] != 0.0:

                        values[sparse_data_pointer] = self.this_item_weights[item_id]
                        rows[sparse_data_pointer] = item_id
                        cols[sparse_data_pointer] = itemIndex

                        sparse_data_pointer += 1


            itemIndex += 1

        # End while on columns


        if self.TopK == 0:

            return np.array(self.W_dense)

        else:

            values = np.array(values[0:sparse_data_pointer])
            rows = np.array(rows[0:sparse_data_pointer])
            cols = np.array(cols[0:sparse_data_pointer])

            W_sparse = sps.csr_matrix((values, (rows, cols)),
                                    shape=(self.n_columns, self.n_columns),
                                    dtype=np.float32)

            return W_sparse





def cosine_common(X):
    """
    Function that pairwise cosine similarity of the columns in X.
    It takes only the values in common between each pair of columns
    :param X: instance of scipy.sparse.csc_matrix
    :return:
        the result of co_prodsum
        the number of co_rated elements for every column pair
    """

    X = check_matrix(X, 'csc')

    # use Cython MemoryViews for fast access to the sparse structure of X
    cdef int [:] indices = X.indices
    cdef int [:] indptr = X.indptr
    cdef float [:] data = X.data

    # initialize the result variables
    cdef int n_cols = X.shape[1]
    cdef np.ndarray[np.float32_t, ndim=2] result = np.zeros([n_cols, n_cols], dtype=np.float32)
    cdef np.ndarray[np.int32_t, ndim=2] common = np.zeros([n_cols, n_cols], dtype=np.int32)

    # let's declare all the variables that we'll use in the loop here
    # NOTE: declaring the type of your variables makes your Cython code run MUCH faster
    # NOTE: Cython allows cdef's only in the main scope
    # cdef's in nested codes will result in compilation errors
    cdef int current_col, second_col, n_i, n_j, ii, jj, n_common
    cdef float ii_sum, jj_sum, ij_sum, x_i, x_j

    for current_col in range(n_cols):
        n_i = indptr[current_col+1] - indptr[current_col]
        # the correlation matrix is symmetric,
        # let's compute only the values for the upper-right triangle
        for second_col in range(current_col+1, n_cols):
            n_j = indptr[second_col+1] - indptr[second_col]

            ij_sum, ii_sum, jj_sum = 0.0, 0.0, 0.0
            ii, jj = 0, 0
            n_common = 0

            # here we exploit the fact that the two subvectors in indices are sorted
            # to compute the dot product of the rows in common between i and j in linear time.
            # (indices[indptr[i]:indptr[i]+n_i] and indices[indptr[j]:indptr[j]+n_j]
            # contain the row indices of the non-zero items in columns i and j)
            while ii < n_i and jj < n_j:
                if indices[indptr[current_col] + ii] < indices[indptr[second_col] + jj]:
                    ii += 1
                elif indices[indptr[current_col] + ii] > indices[indptr[second_col] + jj]:
                    jj += 1
                else:
                    x_i = data[indptr[current_col] + ii]
                    x_j = data[indptr[second_col] + jj]
                    ij_sum += x_i * x_j
                    ii_sum += x_i ** 2
                    jj_sum += x_j ** 2
                    ii += 1
                    jj += 1
                    n_common += 1

            if n_common > 0:
                result[current_col, second_col] = ij_sum / np.sqrt(ii_sum * jj_sum)
                result[second_col, current_col] = result[current_col, second_col]
                common[current_col, second_col] = n_common
                common[second_col, current_col] = n_common

    return result, common



###################################################################################################################
#########################
#########################       ARGSORT
#########################




from libc.stdlib cimport malloc, free#, qsort

# Declaring QSORT as "gil safe", appending "nogil" at the end of the declaration
# Otherwise I will not be able to pass the comparator function pointer
# https://stackoverflow.com/questions/8353076/how-do-i-pass-a-pointer-to-a-c-function-in-cython
cdef extern from "stdlib.h":
    ctypedef void const_void "const void"
    void qsort(void *base, int nmemb, int size,
            int(*compar)(const_void *, const_void *)) nogil



# Node struct
ctypedef struct matrix_element_s:
    long coordinate
    double data


cdef int compare_struct_on_data(const void * a_input, const void * b_input):
    """
    The function compares the data contained in the two struct passed.
    If a.data > b.data returns >0  
    If a.data < b.data returns <0      
    
    :return int: +1 or -1
    """

    cdef matrix_element_s * a_casted = <matrix_element_s *> a_input
    cdef matrix_element_s * b_casted = <matrix_element_s *> b_input

    if (a_casted.data - b_casted.data) > 0.0:
        return +1
    else:
        return -1





cdef long[:] argsort(double[:] this_item_weights, int TopK):

    #start_time = time.time()
    cdef array[long] template_zero = array('l')
    cdef array[long] result = clone(template_zero, TopK, zero=False)
    #print("clone {} sec".format(time.time()-start_time))

    cdef matrix_element_s *matrix_element_array
    cdef int index, num_elements

    num_elements = len(this_item_weights)

    # Allocate vector that will be used for sorting
    matrix_element_array = < matrix_element_s *> malloc(num_elements * sizeof(matrix_element_s))

    #start_time = time.time()

    # Fill vector wit pointers to list elements
    for index in range(num_elements):
        matrix_element_array[index].coordinate = index
        matrix_element_array[index].data = this_item_weights[index]

    #print("Init {} sec".format(time.time()-start_time))

    #start_time = time.time()
    # Sort array elements on their data field
    qsort(matrix_element_array, num_elements, sizeof(matrix_element_s), compare_struct_on_data)
    #print("qsort {} sec".format(time.time()-start_time))

    #start_time = time.time()
    # Sort is from lower to higher, therefore the elements to be considered are from len-topK to len
    for index in range(TopK):

        result[index] = matrix_element_array[num_elements - index - 1].coordinate
    #print("result {} sec".format(time.time()-start_time))

    free(matrix_element_array)



    return result

