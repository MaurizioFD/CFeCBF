#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

@author: Maurizio Ferrari Dacrema
"""

import numpy as np
import scipy.sparse as sps
import time
import os

def check_matrix(X, format='csc', dtype=np.float32):
    if format == 'csc' and not isinstance(X, sps.csc_matrix):
        return X.tocsc().astype(dtype)
    elif format == 'csr' and not isinstance(X, sps.csr_matrix):
        return X.tocsr().astype(dtype)
    elif format == 'coo' and not isinstance(X, sps.coo_matrix):
        return X.tocoo().astype(dtype)
    elif format == 'dok' and not isinstance(X, sps.dok_matrix):
        return X.todok().astype(dtype)
    elif format == 'bsr' and not isinstance(X, sps.bsr_matrix):
        return X.tobsr().astype(dtype)
    elif format == 'dia' and not isinstance(X, sps.dia_matrix):
        return X.todia().astype(dtype)
    elif format == 'lil' and not isinstance(X, sps.lil_matrix):
        return X.tolil().astype(dtype)
    else:
        return X.astype(dtype)


def similarityMatrixTopK(item_weights, forceSparseOutput = True, k=100, verbose = False, inplace=True):
    """
    The function selects the TopK most similar elements, column-wise

    :param item_weights:
    :param forceSparseOutput:
    :param k:
    :param verbose:
    :param inplace: Default True, WARNING matrix will be modified
    :return:
    """

    assert (item_weights.shape[0] == item_weights.shape[1]), "selectTopK: ItemWeights is not a square matrix"

    start_time = time.time()

    if verbose:
        print("Generating topK matrix")

    nitems = item_weights.shape[1]
    k = min(k, nitems)

    # for each column, keep only the top-k scored items
    sparse_weights = not isinstance(item_weights, np.ndarray)

    if not sparse_weights:

        idx_sorted = np.argsort(item_weights, axis=0)  # sort data inside each column

        if inplace:
            W = item_weights
        else:
            W = item_weights.copy()

        # index of the items that don't belong to the top-k similar items of each column
        not_top_k = idx_sorted[:-k, :]
        # use numpy fancy indexing to zero-out the values in sim without using a for loop
        W[not_top_k, np.arange(nitems)] = 0.0

        if forceSparseOutput:
            W_sparse = sps.csr_matrix(W, shape=(nitems, nitems))

            if verbose:
                print("Sparse TopK matrix generated in {:.2f} seconds".format(time.time() - start_time))

            return W_sparse

        if verbose:
            print("Dense TopK matrix generated in {:.2f} seconds".format(time.time()-start_time))

        return W

    else:
        # iterate over each column and keep only the top-k similar items
        data, rows_indices, cols_indptr = [], [], []

        item_weights = check_matrix(item_weights, format='csc', dtype=np.float32)

        for item_idx in range(nitems):

            cols_indptr.append(len(data))

            start_position = item_weights.indptr[item_idx]
            end_position = item_weights.indptr[item_idx+1]

            column_data = item_weights.data[start_position:end_position]
            column_row_index = item_weights.indices[start_position:end_position]

            non_zero_data = column_data!=0

            idx_sorted = np.argsort(column_data[non_zero_data])  # sort by column
            top_k_idx = idx_sorted[-k:]

            data.extend(column_data[non_zero_data][top_k_idx])
            rows_indices.extend(column_row_index[non_zero_data][top_k_idx])


        cols_indptr.append(len(data))

        # During testing CSR is faster
        W_sparse = sps.csc_matrix((data, rows_indices, cols_indptr), shape=(nitems, nitems), dtype=np.float32)
        W_sparse = W_sparse.tocsr()

        if verbose:
            print("Sparse TopK matrix generated in {:.2f} seconds".format(time.time() - start_time))

        return W_sparse




def areURMequals(URM1, URM2):

    if(URM1.shape != URM2.shape):
        return False

    return (URM1-URM2).nnz ==0


def removeTopPop(URM_1, URM_2=None, percentageToRemove=0.2):
    """
    Remove the top popular items from the matrix
    :param URM_1: user X items
    :param URM_2: user X items
    :param percentageToRemove: value 1 corresponds to 100%
    :return: URM: user X selectedItems, obtained from URM_1
             Array: itemMappings[selectedItemIndex] = originalItemIndex
             Array: removedItems
    """


    item_pop = URM_1.sum(axis=0)  # this command returns a numpy.matrix of size (1, nitems)

    if URM_2 != None:

        assert URM_2.shape[1] == URM_1.shape[1], \
            "The two URM do not contain the same number of columns, URM_1 has {}, URM_2 has {}".format(URM_1.shape[1], URM_2.shape[1])

        item_pop += URM_2.sum(axis=0)


    item_pop = np.asarray(item_pop).squeeze()  # necessary to convert it into a numpy.array of size (nitems,)
    popularItemsSorted = np.argsort(item_pop)[::-1]

    numItemsToRemove = int(len(popularItemsSorted)*percentageToRemove)

    # Choose which columns to keep
    itemMask = np.in1d(np.arange(len(popularItemsSorted)), popularItemsSorted[:numItemsToRemove],  invert=True)

    # Map the column index of the new URM to the original ItemID
    itemMappings = np.arange(len(popularItemsSorted))[itemMask]

    removedItems = np.arange(len(popularItemsSorted))[np.logical_not(itemMask)]

    return URM_1[:,itemMask], itemMappings, removedItems

#
#
# def load_edges (filePath, header = False):
#
#     values, rows, cols = [], [], []
#
#     fileHandle = open(filePath, "r")
#     numCells = 0
#
#     if header:
#         fileHandle.readline()
#
#     for line in fileHandle:
#         numCells += 1
#         if (numCells % 1000000 == 0):
#             print("Processed {} cells".format(numCells))
#
#         if (len(line)) > 1:
#             line = line.split(",")
#
#             value = line[2].replace("\n", "")
#
#             if not value == "0" and not value == "NaN":
#                 rows.append(int(line[0]))
#                 cols.append(int(line[1]))
#                 values.append(float(value))
#
#     return  sps.csr_matrix((values, (rows, cols)), dtype=np.float32)



def addZeroSamples(S_matrix, numSamplesToAdd):

    n_items = S_matrix.shape[1]

    S_matrix_coo = S_matrix.tocoo()

    row_index = list(S_matrix_coo.row)
    col_index = list(S_matrix_coo.col)
    data = list(S_matrix_coo.data)

    existingSamples = set(zip(row_index, col_index))

    addedSamples = 0
    consecutiveFailures = 0

    while (addedSamples < numSamplesToAdd):

        item1 = np.random.randint(0, n_items)
        item2 = np.random.randint(0, n_items)

        if (item1 != item2 and (item1, item2) not in existingSamples):

            row_index.append(item1)
            col_index.append(item2)
            data.append(0)

            existingSamples.add((item1, item2))

            addedSamples += 1
            consecutiveFailures = 0

        else:
            consecutiveFailures += 1

        if (consecutiveFailures >= 100):
            raise SystemExit(
                "Unable to generate required zero samples, termination at 100 consecutive discarded samples")

    return row_index, col_index, data


def reshapeSparse(sparseMatrix, newShape):

    if sparseMatrix.shape[0] > newShape[0] or sparseMatrix.shape[1] > newShape[1]:
        ValueError("New shape cannot be smaller than SparseMatrix. SparseMatrix shape is: {}, newShape is {}".format(
            sparseMatrix.shape, newShape))


    sparseMatrix = sparseMatrix.tocoo()
    newMatrix = sps.csr_matrix((sparseMatrix.data, (sparseMatrix.row, sparseMatrix.col)), shape=newShape)

    return newMatrix
