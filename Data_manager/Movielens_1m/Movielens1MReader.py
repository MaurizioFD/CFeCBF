#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 14/09/17

@author: Maurizio Ferrari Dacrema
"""



import scipy.sparse as sps
import zipfile

from Data_manager.DataReader import DataReader
from Data_manager.DataReader_utils import downloadFromURL, load_CSV_into_SparseBuilder




class Movielens1MReader(DataReader):

    DATASET_URL = "http://files.grouplens.org/datasets/movielens/ml-1m.zip"
    DATASET_SUBFOLDER = "Movielens_1m/"
    AVAILABLE_ICM = []
    DATASET_SPECIFIC_MAPPER = []

    IS_IMPLICIT = True


    def __init__(self):
        super(Movielens1MReader, self).__init__()


    def _get_dataset_name_root(self):
        return self.DATASET_SUBFOLDER



    def _load_from_original_file(self):
        # Load data from original

        print("Movielens1MReader: Loading original data")

        zipFile_path =  self.DATASET_SPLIT_ROOT_FOLDER + self.DATASET_SUBFOLDER

        try:

            dataFile = zipfile.ZipFile(zipFile_path + "ml-1m.zip")

        except (FileNotFoundError, zipfile.BadZipFile):

            print("Movielens1MReader: Unable to fild data zip file. Downloading...")

            downloadFromURL(self.DATASET_URL, zipFile_path, "ml-1m.zip")

            dataFile = zipfile.ZipFile(zipFile_path + "ml-1m.zip")


        URM_path = dataFile.extract("ml-1m/ratings.dat", path=zipFile_path + "decompressed/")

        self.URM_all, self.item_original_ID_to_index, self.user_original_ID_to_index = load_CSV_into_SparseBuilder(URM_path, separator="::")


        print("Movielens1MReader: cleaning temporary files")

        import shutil

        shutil.rmtree(zipFile_path + "decompressed", ignore_errors=True)

        print("Movielens1MReader: loading complete")

