#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 22/11/18

@author: Maurizio Ferrari Dacrema
"""

import numpy as np




def split_big_CSR_in_columns(sparse_matrix_to_split, num_split = 2):
    """
    The function returns a list of split for the given matrix
    :param sparse_matrix_to_split:
    :param num_split:
    :return:
    """

    assert sparse_matrix_to_split.shape[1]>0, "split_big_CSR_in_columns: sparse_matrix_to_split has no columns"
    assert num_split>=1 and num_split <= sparse_matrix_to_split.shape[1], "split_big_CSR_in_columns: num_split parameter not valid, value must be between 1 and {}, provided was {}".format(sparse_matrix_to_split.shape[1], num_split)


    if num_split == 1:
        return [sparse_matrix_to_split]



    n_column_split = int(sparse_matrix_to_split.shape[1]/num_split)

    sparse_matrix_split_list = []

    for num_current_split in range(num_split):

        start_col = n_column_split*num_current_split

        if num_current_split +1 == num_split:
            end_col = sparse_matrix_to_split.shape[1]
        else:
            end_col = n_column_split*(num_current_split + 1)

        print("split_big_CSR_in_columns: Split {}, columns: {}-{}".format(num_current_split, start_col, end_col))

        sparse_matrix_split_list.append(sparse_matrix_to_split[:,start_col:end_col])

    return sparse_matrix_split_list









def remove_empty_rows_and_cols(URM, ICM = None):

    URM = check_matrix(URM, "csr")
    rows = URM.indptr
    numRatings = np.ediff1d(rows)
    user_mask = numRatings >= 1

    URM = URM[user_mask,:]

    cols = URM.tocsc().indptr
    numRatings = np.ediff1d(cols)
    item_mask = numRatings >= 1

    URM = URM[:,item_mask]

    removedUsers = np.arange(0, len(user_mask))[np.logical_not(user_mask)]
    removedItems = np.arange(0, len(item_mask))[np.logical_not(item_mask)]

    if ICM is not None:

        ICM = ICM[item_mask,:]

        return URM.tocsr(), ICM.tocsr(), removedUsers, removedItems


    return URM.tocsr(), removedUsers, removedItems





from Data_manager.IncrementalSparseMatrix import IncrementalSparseMatrix


def load_CSV_into_SparseBuilder (filePath, header = False, separator="::"):


    matrixBuilder = IncrementalSparseMatrix(auto_create_col_mapper = True, auto_create_row_mapper = True)

    fileHandle = open(filePath, "r")
    numCells = 0

    if header:
        fileHandle.readline()

    for line in fileHandle:
        numCells += 1
        if (numCells % 1000000 == 0):
            print("Processed {} cells".format(numCells))

        if (len(line)) > 1:
            line = line.split(separator)

            line[-1] = line[-1].replace("\n", "")

            try:
                user_id = line[0]
                item_id = line[1]


                try:
                    value = float(line[2])

                    if value != 0.0:

                        matrixBuilder.add_data_lists([user_id], [item_id], [value])

                except ValueError:
                    print("load_CSV_into_SparseBuilder: Cannot parse as float value '{}'".format(line[2]))


            except IndexError:
                print("load_CSV_into_SparseBuilder: Index out of bound in line '{}'".format(line))


    fileHandle.close()



    return  matrixBuilder.get_SparseMatrix(), matrixBuilder.get_column_token_to_id_mapper(), matrixBuilder.get_row_token_to_id_mapper()







import time, sys, os

def urllretrieve_reporthook(count, block_size, total_size):

    global start_time_urllretrieve

    if count == 0:
        start_time_urllretrieve = time.time()
        return

    duration = time.time() - start_time_urllretrieve + 1

    progress_size = int(count * block_size)
    speed = int(progress_size / (1024 * duration))
    percent = min(float(count*block_size*100/total_size),100)

    sys.stdout.write("\rDataReader: Downloaded {:.2f}%, {:.2f} MB, {:.0f} KB/s, {:.0f} seconds passed".format(
                    percent, progress_size / (1024 * 1024), speed, duration))

    sys.stdout.flush()




from Base.Recommender_utils import check_matrix



def removeFeatures(ICM, minOccurrence = 5, maxPercOccurrence = 0.30, reconcile_mapper = None):
    """
    The function eliminates the values associated to feature occurring in less than the minimal percentage of items
    or more then the max. Shape of ICM is reduced deleting features.
    :param ICM:
    :param minPercOccurrence:
    :param maxPercOccurrence:
    :param reconcile_mapper: DICT mapper [token] -> index
    :return: ICM
    :return: deletedFeatures
    :return: DICT mapper [token] -> index
    """

    ICM = check_matrix(ICM, 'csc')

    n_items = ICM.shape[0]

    cols = ICM.indptr
    numOccurrences = np.ediff1d(cols)

    feature_mask = np.logical_and(numOccurrences >= minOccurrence, numOccurrences <= n_items*maxPercOccurrence)

    ICM = ICM[:,feature_mask]

    deletedFeatures = np.arange(0, len(feature_mask))[np.logical_not(feature_mask)]

    print("RemoveFeatures: removed {} features with less then {} occurrencies, removed {} features with more than {} occurrencies".format(
        sum(numOccurrences < minOccurrence), minOccurrence,
        sum(numOccurrences > n_items*maxPercOccurrence), int(n_items*maxPercOccurrence)
    ))

    if reconcile_mapper is not None:
        reconcile_mapper = reconcile_mapper_with_removed_tokens(reconcile_mapper, deletedFeatures)

        return ICM, deletedFeatures, reconcile_mapper


    return ICM, deletedFeatures



def reconcile_mapper_with_removed_tokens(mapper_dict, indices_to_remove):
    """

    :param mapper_dict: must be a mapper of [token] -> index
    :param indices_to_remove:
    :return:
    """

    # When an index has to be removed:
    # - Delete the corresponding key
    # - Decrement all greater indices

    indices_to_remove = set(indices_to_remove)
    removed_indices = []

    # Copy key set
    dict_keys = list(mapper_dict.keys())

    # Step 1, delete all values
    for key in dict_keys:

        if mapper_dict[key] in indices_to_remove:

            removed_indices.append(mapper_dict[key])
            del mapper_dict[key]


    removed_indices = np.array(removed_indices)


    # Step 2, decrement all remaining indices to fill gaps
    # Every index has to be decremented by the number of deleted tokens with lower index
    for key in mapper_dict.keys():

        lower_index_elements = np.sum(removed_indices<mapper_dict[key])
        mapper_dict[key] -= lower_index_elements


    return mapper_dict




def downloadFromURL(URL, folder_path, file_name):

    import urllib
    from urllib.request import urlretrieve

    # If directory does not exist, create
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

    print("Downloading: {}".format(URL))
    print("In folder: {}".format(folder_path + file_name))

    try:

        urlretrieve (URL, folder_path + file_name, reporthook=urllretrieve_reporthook)

    except urllib.request.URLError as urlerror:

        print("Unable to complete atuomatic download, network error")
        raise urlerror




    sys.stdout.write("\n")
    sys.stdout.flush()








def invert_dictionary(id_to_index):

    index_to_id = {}

    for id in id_to_index.keys():
        index = id_to_index[id]
        index_to_id[index] = id

    return index_to_id