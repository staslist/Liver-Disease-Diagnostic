# -*- coding: utf-8 -*-
"""
Created on Mon Jun  3 17:06:52 2019

@author: Stanislav Listopad
"""

'''SOME PARTS OF THE CODEBASE HAVE BEEN OBSCURED WITH '...' TO REMOVE SENSITIVE INFORMATION.'''

''' Style conventions 
Variables, functions, methods, packages, modules: this_is_a_variable
Classes and exceptions: CapWords
Protected methods and internal functions: _single_leading_underscore
Private methods: __double_leading_underscore
Constants: CAPS_WITH_UNDERSCORES
Indent: Tab (4 Spaces)
Line Length: 79 chars.
Surround top-level function and class definitions with two blank lines.
Use spaces around operators. 
Keep comments meticilously updated. Avoid inline comments.
'''

'''
Values
"Build tools for others that you want to be built for you." - Kenneth Reitz
"Simplicity is alway better than functionality." - Pieter Hintjens
"Fit the 90% use-case. Ignore the nay sayers." - Kenneth Reitz
"Beautiful is better than ugly." - PEP 20
Build for open source (even for closed source projects).
General Development Guidelines
"Explicit is better than implicit" - PEP 20
"Readability counts." - PEP 20
"Anybody can fix anything." - Khan Academy Development Docs
Fix each broken window (bad design, wrong decision, or poor code) as soon as it is discovered.
"Now is better than never." - PEP 20
Test ruthlessly. Write docs for new features.
Even more important that Test-Driven Development--Human-Driven Development
These guidelines may--and probably will--change.
'''

'''
Current Goal: Set up efficient and streamlined workflow for analysis of 
blood and liver tissue samples from within the AH-Project. 
The analysis has two major components:
1) Feature selection. Filter, Filter + Wrapper (Hybrid), Embedded (RF/RFE).
2) Classification. Simple ML models.
'''

'''
Goals: Identify best features and best classification rate. 

Alcoholic Hepatitis Feature Selection / Classification Project
RNA-Seq Data Only
FPKM (Geometric Normalization) counts from multiple pipelines.

The following data layout is used.
Root_Directory/ contains all of the pipeline subdirectories.
Each pipeline name is made from reference, aligner, and annotation.
Example: hg19_Tuxedo_Refflat. Aligner and annotation are capitalized.
Each pipeline subdirectory contains folders with cuffnorm and cuffdiff 
runs over the entire data (all samples). The cuffnorm folders are named
Cuffnorm + _ + all capitals normalization + _ + optional PRELIM flag. 
Example: Cuffnorm_GEOM.
The cuffdiff files are named Cuffdiff + _ all capitals normalization + 
_ all capitals dispersion + _ optional PRELIM flag.
Example: Cuffdiff_UQ_POOL_PRELIM.
Additionally each pipeline directory contains a FOLD10 and/or FOLD5 folders.
These folders contains results of Cuffdiff execution over each fold. 
In FOLD5/FOLD10 folder each subfolder is named Cuffdiff + _ + all capitals normalization 
+ _ + all capitals dispersion + FOLD + fold number.
Example: Cuffdiff_GEOM_COND_FOLD8.
Each terminal Cuffdiff subdirectory contains gene_exp.diff, gene_exp.txt (gene_exp.diff 
filterd by A-Lister), genes.read_group_tracking, and read_groups.info. 
Each terminal Cuffnorm subdirectory contains genes.fpkm table and samples.table.
'''
def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn

import csv
import time
import math
import unittest
import numpy as np
import pandas
import seaborn
import os
import itertools
import sys
import random
import threading
import gseapy as gp
import re
from copy import deepcopy

from IPython.display import display 
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.svm import LinearSVC
from sklearn.neural_network import MLPClassifier
from sklearn.feature_selection import mutual_info_classif
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
from sklearn.feature_selection import RFE

from scipy import interp
from sklearn.utils import shuffle
from sklearn.metrics import roc_curve, auc

import matplotlib.pyplot as plt

import time

'''Sample Layouot Used for Testing.'''
samples_AH_TEST = ['ABC', 'BCE', 'DFG', 'MLP']
samples_CT_TEST = ['1', '2', '3']
samples_DA_TEST = ['!', '@']
cond_samples_TEST = {'AH': samples_AH_TEST, 'CT': samples_CT_TEST, 'DA': samples_DA_TEST}

'''PBMC (Blood) Sample IDS from the AH-Project.'''

samples_AH_PB = ['...']
        
samples_CT_PB = ['...']

samples_DA_PB = ['...']

samples_AA_PB = ['...']

samples_NF_PB = ['...']

samples_HP_PB = ['...']

samples_HP_PB_excluded = ['...']

cond_samples = {"AH":samples_AH_PB, "CT": samples_CT_PB, "DA" : samples_DA_PB, "AA" : samples_AA_PB, 
                "NF": samples_NF_PB, "HP": samples_HP_PB}

cond_samples_excluded = {"AH":samples_AH_PB, "CT": samples_CT_PB, "DA" : samples_DA_PB, "AA" : samples_AA_PB, 
                         "NF": samples_NF_PB, "HP": samples_HP_PB_excluded}

'''Liver tissue samples from the AH-Project '''

samples_AH_LV = ['...']

samples_AH_LV_excluded = ['...']

samples_CT_LV = ['...']

samples_AC_LV = ['...']

samples_NF_LV = ['...']

samples_HP_LV = ['...']

cond_samples_LV = {'AH': samples_AH_LV, 'CT': samples_CT_LV, 'AC': samples_AC_LV,
                   'NF': samples_NF_LV, 'HP': samples_HP_LV}

cond_samples_LV_excluded = {'AH': samples_AH_LV_excluded, 'CT': samples_CT_LV, 'AC': samples_AC_LV,
                            'NF': samples_NF_LV, 'HP': samples_HP_LV}

pipelines_ALL = ['hg19_Hisat2_Curated', 'hg19_Hisat2_Ensembl', 'hg19_Hisat2_Gencode', 'hg19_Hisat2_Refflat', 'hg19_Starcq_Curated',
                 'hg19_Starcq_Ensembl', 'hg19_Starcq_Gencode', 'hg19_Starcq_Refflat', 'hg19_Tuxedo_Curated', 'hg19_Tuxedo_Ensembl',
                 'hg19_Tuxedo_Gencode', 'hg19_Tuxedo_Refflat', 'hg38_Hisat2_Curated', 'hg38_Hisat2_Ensembl', 'hg38_Hisat2_Gencode',
                 'hg38_Hisat2_Refflat', 'hg38_Starcq_Curated', 'hg38_Starcq_Ensembl', 'hg38_Starcq_Gencode', 'hg38_Starcq_Refflat',
                 'hg38_Tuxedo_Curated', 'hg38_Tuxedo_Ensembl', 'hg38_Tuxedo_Gencode', 'hg38_Tuxedo_Refflat']

models_ALL = ['log_reg', 'kNN', 'GNB', 'SVM', 'NN', 'RF']

# HELPER/UTILITY FUNCTIONS
def compare_float_lists(list1:list, list2:list):
    iter1 = iter(list1)
    iter2 = iter(list2)
    if(len(list1) != len(list2)):
        return False
    else:
        while True:
            try:
                float_val1 = next(iter1)
                float_val2 = next(iter2)
                if(float_val1 != float_val2):
                    return False
            except StopIteration:
                return True

def write_list_to_file(filename:str, to_write:list):
    with open(filename, 'w') as writer:
        for ele in to_write:
            writer.write(str(ele))
            writer.write('\n')

def generate_filtered_csv_file_one_col(file1:str, delim:str, taboo_list:list, out_dir:str):
    '''Assume a one column csv_file.'''
    to_write = []
    with open(file1) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=delim)
        for row in csv_reader:
            write = True
            for tl in taboo_list:
                if(tl in row[0]):
                    write = False
            if(write):
                to_write.append(row[0])
                
    filename = out_dir + '/' + 'filtered.csv'
    write_list_to_file(filename, to_write)
    
    return to_write
    
def compare_csv_file_contents(file1:str, file2:str, delim:str):
    '''Returns true if both csv files contain exactly same data.'''
    values = []
    filenames = [file1, file2]
    for filename in filenames:
        with open(filename) as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=delim)
            temp = []
            for row in csv_reader:
                if(len(row) >= 1):
                    temp.append(row)
                else:
                    print("Encountered an empty row in csv file.")
            values.append(temp)
    
    file_len1 = len(values[0])
    file_len2 = len(values[1])
    if(file_len1 != file_len2):
        print("Different number of lines in files 1 and 2.")
        return False
    
    i = 0
    while i < file_len1:
        if(values[0][i] != values[1][i]):
            print('Difference on line: ', i)
            print("File1: ", values[0][i])
            print("File2: ", values[1][i])
            return False
        i = i + 1
    return True

def count_nonmRNAs(genes:list)->int:
    nonmRNA_count = 0
    for gene in genes:
        tests = ['SNOR', 'MIR', 'RNU', 'RNY', '.']
        for test in tests:
            if(test in gene.upper()):
                nonmRNA_count += 1
                break
            
    return nonmRNA_count

def read_in_csv_file_one_column(filename:str, column:int, delim:str)->list:
    temp = []
    with open(filename) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=delim)
        for row in csv_reader:
            try:
                temp.append(row[column])
            except ValueError:
                pass
    return temp

def compare_one_column_csv_file_contents(file1:str, file2:str, delim:str):
    '''Assume that both csv files have one column with string values.
    Compute intersection between files. Return the intersection, difference, and union,
    between both files.'''
    values = []
    filenames = [file1, file2]
    for filename in filenames:
        with open(filename) as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=delim)
            temp = set()
            for row in csv_reader:
                if(len(row) >= 1):
                    temp.add(row[0])
                else:
                    print("Encountered an empty row in csv file.")
            values.append(temp)
        
    v0 = set(values[0])
    v1 = set(values[1])
    intersection = v0 & v1
    union = v0 | v1
    #print("File 1, but not File 2: ")
    difference1 = v0 - v1
    #print(len(difference1))
    #print(difference1)
    #print("File 2, but not File 1: ")
    difference2 = v1 - v0
    #print(len(difference2))
    #print(difference2)
    difference = difference1 | difference2
    #print('a & b: ', len(intersection))
    #print('(a - b) or (b - a): ', difference)

    return intersection, difference, union

def compare_one_column_txt_file_contents(file1:str, file2:str, num_read:int = -1):
    '''Assume that both txt files have one column with string values.
    Compute intersection between files. Return the intersection, difference, and union,
    between both files.
    num_read: specify the number of values to read in from each file. -1 means read in all the values.'''
    values = []
    filenames = [file1, file2]
    for filename in filenames:
        with open(filename) as reader:
            i = 0
            lines = reader.readlines()
            temp = set()
            for line in lines:
                if(len(line) >= 1):
                    if(num_read == -1 or i < num_read):
                        temp.add(line.replace('\n', ''))
                        i += 1
                    else:
                        break
                else:
                    print("Encountered an empty row in txt file.")
            values.append(temp)
    
    v0 = set(values[0])
    v1 = set(values[1])
    intersection = v0 & v1
    union = v0 | v1
    #print("File 1, but not File 2: ")
    difference1 = v0 - v1
    #print(len(difference1))
    #print(difference1)
    #print("File 2, but not File 1: ")
    difference2 = v1 - v0
    #print(len(difference2))
    #print(difference2)
    difference = difference1 | difference2
    #print('a & b: ', len(intersection))
    #print('(a - b) or (b - a): ', difference)

    return intersection, difference, union

def two_dim_list_len(two_dim_list:list)->int:
    ''' Find the length of a 2-dimensional list. 
    Each element in outer list must be a list. 
    Each element in inner lists must be a string.'''
    length = 0
    for sub_list in two_dim_list:
        if(isinstance(sub_list, list)):
            length = len(sub_list) + length
            for element in sub_list:
                if(isinstance(element, str)):
                    pass
                else:
                    raise ValueError("Each element in inner lists must be a string")
        else:
            raise ValueError("Each element in outer list be must a list.")
            
    return length

def filter_cuffdiff_file_by_gene_list(gene_list_file:str, cuffdiff_file:str, output_dir:str,
                                      out_fname:str = 'filtered.diff'):
    '''Filter a cuffdiff differential expression file by a list of genes 
    and output the differential expression data for those genes only in 
    another file.'''
    gene_list = []
    to_write = []
    with open(gene_list_file) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter='\t')
        for row in csv_reader:
            gene_list.append(row[0])
            
    #print(gene_list)
    with open(cuffdiff_file) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter='\t')
        line_count = 0
        for row in csv_reader:
            if(line_count == 0):
                to_write.append(row)
            elif(row[2] in gene_list):
                to_write.append(row)
            line_count += 1
    
    filename = output_dir + '/' + out_fname
    with open(filename, 'w') as writer:
        for line in to_write:
            for ele in line:
                writer.write(str(ele) + '\t')
            writer.write('\n')
            
def filter_binary_cuffdiff_file_by_gene_list(gene_list_file:str, cuffdiff_file:str, upregulated:bool, sample1:str,
                                             sample2:str, output_dir:str, out_fname:str = 'filtered.txt'):
    ''' Filter a cuffdiff differential expression file by a list of genes 
    and output either upregulated or downregulated genes. Only works as intended in case of a 2 condition 
    cuffdiff file. '''
    gene_list = []
    to_write = []
    with open(gene_list_file) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter='\t')
        for row in csv_reader:
            #print(row)
            gene_list.append(row[0])
            
    #print(gene_list)
    with open(cuffdiff_file) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter='\t')
        line_count = 0
        for row in csv_reader:
            if(line_count == 0):
                pass
            elif(row[2] in gene_list):
                # If this is a correct pairwise comparison
                if(row[4] in [sample1, sample2] and row[5] in [sample1, sample2]):
                    if(upregulated and float(row[9]) >= 0):
                        to_write.append(row[2])
                    elif(not upregulated and float(row[9]) < 0):
                        to_write.append(row[2])
                    else:
                        pass
            line_count += 1
    
    filename = output_dir + '/' + out_fname
    with open(filename, 'w') as writer:
        for gene_name in to_write:
            writer.write(str(gene_name) + '\n')

def generate_kfolds(samples:list, k:int):
    ''' Separates samples into k-folds (lists). '''
    if(k > len(samples)):
        raise ValueError("The number of folds must be <= than the number of samples.")
    folds = []
    i = 0
    while i < k:
        folds.append([])
        i = i + 1
        
    i = 0
    for sample in samples:
        folds[i % k].append(sample)
        i = i + 1
            
    return folds
    
# CORE FUNCTIONS
    
def generate_top_genes_RF(root_dir:str, pipeline:str, normalization:str, dispersion:str, num_DEG:int, num_samples:dict, 
                          n_estimators:int, num_runs:int, v_transform:bool=True, write:bool=True, fold:int = 0, 
                          num_folds:int = 10, tissue = 'PB', counts_format = 'Cuffnorm'):
    '''This function generates a list of top genes ranked by the random forest algorithm. '''
    if(pipeline not in pipelines_ALL):
        raise ValueError("This pipeline is not one of the pre-defined pipelines for the AH-Project.")
    if(normalization not in ['UQ', 'GEOM']):
        raise ValueError("The normalization must be either UQ = Upper Quartile or GEOM = Geometric.")
    if(dispersion not in ['POOL', 'COND']):
        raise ValueError("The dispersion must be either COND = Per Condtion or POOL = Pooled.")
    if(fold > num_folds):
        raise ValueError("The fold index must be smaller than total number of folds.")
    
    
    if(counts_format not in ['Cuffnorm', 'Cuffdiff']):
        raise ValueError("The counts format has to be Cuffnorm or Cuffdiff.")
    
    if(counts_format == 'Cuffnorm'):
        if(fold > 0):
            out_dir = root_dir + pipeline + '/FOLD' + str(num_folds) + '/Cuffnorm_' + normalization + '_FOLD' + str(fold) + '/'
        elif(fold == 0):
            out_dir = root_dir + pipeline + '/Cuffnorm_' + normalization + '/'
        else:
            raise ValueError('Fold number must be >= 0.')
    else:
        if(fold > 0):
            out_dir = root_dir + pipeline + '/FOLD' + str(num_folds) + '/Cuffdiff_'
            out_dir += normalization + '_' + dispersion + '_FOLD' + str(fold) + '/'
        elif(fold == 0):
            out_dir = root_dir + pipeline + '/Cuffdiff_' + normalization + '_' + dispersion + '/'
        else:
            raise ValueError('Fold number must be >= 0.')
    
    X, Y, gene_names = generate_data(num_samples, v_transform, root_dir, pipeline, normalization, dispersion,
                                     fold, num_folds, tissue, counts_format)
    
    i = 0
    RFs = []
    while i < num_runs:
        model = RandomForestClassifier(n_estimators = n_estimators, criterion = 'gini')
        clf = model.fit(X, Y)
        # print(clf.score(X, Y))
        RF = clf.feature_importances_
        RFs.append(RF)
        i = i + 1
    
    average_rankings = {}
    for RF in RFs:
        if(len(RF) != len(gene_names)):
            print("Error, the length of RF array does not equal to length of gene names list.")
        
        # Note, that there is an assumption here that the ordering of gene_names matches the ordering of columns in X matrix.
            
        j = 0
        while j < len(RF):
            try:
                average_rankings[gene_names[j]] = average_rankings[gene_names[j]] + RF[j]
            except KeyError:
                average_rankings[gene_names[j]] = RF[j]
            j += 1
    
    counter = 0
    for key in average_rankings:
        average_rankings[key] = average_rankings[key] / num_runs
        if(average_rankings[key] > 0):
            counter = counter + 1
    s_gene_ig = sorted(average_rankings.items(), key = lambda x: abs(x[1]), reverse = True)
    
    if(counter < num_DEG):
        num_DEG = counter
    
    top_DEGs = []
    i = 0
    for kv_tup in s_gene_ig:
        top_DEGs.append(str(kv_tup[0]))
        #print(kv_tup[0], kv_tup[1])
        i += 1
        if(i == num_DEG):
            break
    
    if(write):
        filename = 'top_RF_genes.txt'
        # if(v_transform):
        #     filename += '_vtrans'
        # filename += '.txt'
        
        filepath = out_dir + filename
        with open(filepath, 'w') as writer:
            k = 0
            while k < len(top_DEGs):
                writer.write(top_DEGs[k] + '\n')
                k = k + 1
                
    return top_DEGs
    
def generate_top_genes_IG(root_dir: str, pipeline:str, normalization:str, dispersion:str, num_DEG:int, num_samples:dict, 
                          v_transform:bool=True, write:bool=True, reverse:bool = True, fold:int = 0, num_folds:int = 10,
                          tissue:str = 'PB', counts_format:str = 'Cuffnorm'):
    '''This function generates a list of top genes ranked by the information gain algorithm.
    The text file containing top genes is placed either into a Cuffnorm or Cuffdiff directory.'''
    
    if counts_format not in ['Cuffnorm', 'Cuffdiff']:
        raise ValueError("The counts format has to be Cuffnorm or Cuffdiff.")
    
    if(pipeline not in pipelines_ALL):
        raise ValueError("This pipeline is not one of the pre-defined pipelines for the AH-Project.")
    if(normalization not in ['UQ', 'GEOM']):
        raise ValueError("The normalization must be either UQ = Upper Quartile or GEOM = Geometric.")
    if(fold > num_folds):
        raise ValueError("The fold index must be smaller than total number of folds.")
        
    if(counts_format == 'Cuffnorm'):
        if(fold > 0):
            out_dir = root_dir + pipeline + '/FOLD' + str(num_folds) + '/Cuffnorm_' + normalization + '_FOLD' + str(fold) + '/'
        elif(fold == 0):
            out_dir = root_dir + pipeline + '/Cuffnorm_' + normalization + '/'
        else:
            raise ValueError('Fold number must be >= 0.')
    else:
        if(fold > 0):
            out_dir = root_dir + pipeline + '/FOLD' + str(num_folds) + '/Cuffdiff_'
            out_dir += normalization + '_' + dispersion + '_FOLD' + str(fold) + '/'
        elif(fold == 0):
            out_dir = root_dir + pipeline + '/Cuffdiff_' + normalization + '_' + dispersion + '/'
        else:
            raise ValueError('Fold number must be >= 0.')
    
    X, Y, gene_names = generate_data(num_samples, v_transform, root_dir, pipeline, normalization,
                                     dispersion, fold, num_folds, tissue, counts_format)
    IG = mutual_info_classif(X, Y)
    gene_ig = {}
    if(len(IG) != len(gene_names)):
        print("Error, the length of information gain array does not equal to length of gene names list.")
    i = 0
    while i < len(IG):
        #print(gene_names[i], IG[i])
        gene_ig[gene_names[i]] = IG[i]
        i += 1
    s_gene_ig = sorted(gene_ig.items(), key = lambda x: abs(x[1]), reverse = reverse)
    top_DEGs = []
    i = 0
    for kv_tup in s_gene_ig:
        top_DEGs.append(str(kv_tup[0]))
        i += 1
        if(i == num_DEG):
            break
    
    if(write):
        filename = 'top_IG_genes'
        # if(v_transform):
        #     filename += '_vtrans'
        if(not reverse):
            filename += '_reverse'
        filename += '.txt'
        
        filepath = out_dir + filename
        with open(filepath, 'w') as writer:
            for gene in top_DEGs:
                writer.write(gene + '\n')
    return top_DEGs

def generate_top_genes_DE(root_dir: str, pipeline:str, normalization:str, dispersion:str, num_DEG:int,
                          num_samples:dict, scheme:str = 'MAX', weights:list = [], write:bool=True, validate:bool=True,
                          reverse:bool = True, fold:int = 0, num_folds:int = 10, counts_format = 'Cuffdiff'):
    '''Go through A-Lister filtered gene_exp.txt file and select top N genes.
    Then write the gene names into a txt file called top_N_genes.
    This must be done using only the pairwise comparisons that can be generated from conditions specified 
    in num_samples. Scheme: 'MEAN', 'MAX', 'W-MEAN', 'Pairwise' '''
    
    '''This function generates a list of top genes ranked by the information gain algorithm. '''
    if(pipeline not in pipelines_ALL):
        raise ValueError("This pipeline is not one of the pre-defined pipelines for the AH-Project.")
    if(normalization not in ['UQ', 'GEOM']):
        raise ValueError("The normalization must be either UQ = Upper Quartile or GEOM = Geometric.")
    if(dispersion not in ['POOL', 'COND']):
        raise ValueError("The dispersion must be POOL = pooled or COND = per condition.")
    if(fold > num_folds):
        raise ValueError("The fold index must be smaller than total number of folds.")
        
    if(counts_format not in ['Cuffnorm', 'Cuffdiff']):
        raise ValueError("Counts format must be Cuffnorm or Cuffdiff.")
    
        
    if(counts_format == 'Cuffnorm'):
        if(fold > 0):
            out_dir = root_dir + pipeline + '/FOLD' + str(num_folds) + '/Cuffnorm_' + normalization + '_FOLD' + str(fold) + '/'
        elif(fold == 0):
            out_dir = root_dir + pipeline + '/Cuffnorm_' + normalization + '/'
        else:
            raise ValueError('Fold number must be >= 0.')
    else:
        if(fold > 0):
            out_dir = root_dir + pipeline + '/FOLD' + str(num_folds) + '/Cuffdiff_' + normalization 
            out_dir += '_' + dispersion + '_FOLD' + str(fold) + '/'
        elif(fold == 0):
            out_dir = root_dir + pipeline + '/Cuffdiff_' + normalization + '_' + dispersion + '/'
        else:
            raise ValueError('Fold number must be >= 0.')
    
    if(validate):
        fname = root_dir +'/' + pipeline + '/Cuffdiff_' + normalization + '_' + dispersion + '/' + 'genes.read_group_tracking'
        gene_names = read_cuffdiff_gene_names(fname)
        
    #Translate condition names (ex: AH) to replicate condition names (ex: q1, q2, etc.)
    conditions = num_samples.keys()
    temp_dir = root_dir +'/' + pipeline + '/Cuffdiff_' + normalization + '_' + dispersion + '/'
    cond_to_rep_cond_map = generate_cond_name_to_rep_name_map(temp_dir, 1)
    rep_cond_names = []
    for cond in conditions:
        rep_cond_names.append(cond_to_rep_cond_map[cond])
    
    id_col = 2
    
    if(fold == 0):
        directory = root_dir + '/' + pipeline+'/Cuffdiff_'+normalization+'_'+dispersion
    else:
        directory = root_dir + '/' + pipeline+'/FOLD'+str(num_folds)+'/Cuffdiff_'+normalization+'_'+dispersion+'_FOLD' + str(fold)
    if(num_DEG == 'ALL' or num_DEG >= 2250):
        raise ValueError("Number of DEGs must be < 2250 for DE feature selection.")
    else:
        # Use DEGs pre-filtered by q_value < 0.05 and FPKM value1/value2 > 1.0;
        num_header_lines = 5
        sum_var = 0
        for i in range(1, len(rep_cond_names), 1):
            sum_var += i
        num_header_lines += sum_var
        filename = directory + "/gene_exp.txt"
        top_DEGs = select_top_cuffdiff_DEs(filename, num_header_lines, num_DEG, rep_cond_names, id_col,
                                           scheme, weights)
        top_DEGs_v = top_DEGs
        if(validate and 'Gencode' in pipeline):
            top_DEGs_v = select_top_cuffdiff_DEs(filename, num_header_lines, num_DEG, rep_cond_names, 0,
                                                 scheme, weights)
        
    if(validate):    
        # Validate top DEGs.
        num_top_DEGs_miss_in_cuffnorm_counts = 0
        for DEG in top_DEGs_v:
            if(DEG not in gene_names):
                num_top_DEGs_miss_in_cuffnorm_counts += 1
                print("This gene name from cuffdiff gene_exp file is missing in cuffnorm fpkm_table file: ", str(DEG))
        if(num_top_DEGs_miss_in_cuffnorm_counts/num_DEG > 0.01):
            raise ValueError("More than 1% of top DEGs cannot be found in Cuffnorm counts.")
    
    if(write):
        filename = 'top_' + str(num_DEG) + '_DE_' + scheme
        if(not reverse):
            filename += '_reverse'
        filename += '.txt'
        # filename = 'test.txt'
        
        filepath = out_dir + filename
        with open(filepath, 'w') as writer:
            for DEG in top_DEGs:
                writer.write(str(DEG) + '\n')
            
    return top_DEGs

def compare_classification_accuracy_all_pipelines():
    ''' 1)Pick best performing model(s). If there is a tie use time and space complexity to pick a winner. 
        2)For the given model(s) calculate the mean accuracy across feature sizes. If there is a tie compare low 
        feature sizes (10, 30, and 50) and medium feature sizes (100, 150, 200, 250) separetely. 
        3)Use this (these) measure(s) to compare the pipelines.'''
    max_avg = 0
    min_avg = 100
    best_pipeline = ""
    worst_pipeline = ""
    for pipeline in pipelines_ALL:
        accuracy_file = ('...'
                         + "Tuned_Perf_Curve_" + pipeline + "_GEOM_POOL_True.csv")
        perf = calculate_pipeline_accuracy_metrics(accuracy_file)
        print(pipeline, perf)
        if(perf > max_avg):
            max_avg = perf
            best_pipeline = pipeline
        if(perf < min_avg):
            min_avg = perf
            worst_pipeline = pipeline
            
    print(best_pipeline, max_avg)
    print(worst_pipeline, min_avg)

def calculate_pipeline_accuracy_metrics(filename:str, verbose:bool = False):
    ''' Takes in csv accuracy file (tuned models) for a given pipeline. And computes  
    various interesting statistics. 
    How to compare accuracies across pipelines? To save time lets use simple accuracy values, 
    already generated for 10, 30, 50, 100, 150, 200, and 250 features. 
    Potential metrics to consider. Average accuracy across models for given feature size.
    Maximum accuracy across models for given feature size.
    Minimum accuracy across models for given feature size.
    Average, max, and min accuracies for a model across feature sizes.
    Standard deviation across feature sizes for given feature model.
    '''
    #print(filename)
    
    acc_matrix = np.zeros((7, 5))
    
    with open(filename) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        row_count = 0
        for row in csv_reader:
            if(row_count > 0):
                col_count = 0
                for acc in row:
                    if(col_count > 0):
                        acc_matrix[row_count-1, col_count-1] = acc
                    col_count += 1
            row_count += 1
    if(verbose):
        print(acc_matrix)
    
    log_reg_avg = np.mean(acc_matrix[:,0])
    kNN_avg = np.mean(acc_matrix[:,1])
    GNB_avg = np.mean(acc_matrix[:,2])
    SVM_avg = np.mean(acc_matrix[:,3])
    NN_avg = np.mean(acc_matrix[:,4])
      
    model_avgs = np.array([log_reg_avg, kNN_avg, GNB_avg, SVM_avg, NN_avg])
    model_avgs_alt = [log_reg_avg, kNN_avg, GNB_avg, SVM_avg, NN_avg]
    
    feature_sizes = [10, 30, 50, 100, 150, 200, 250]
    i = 0
    log_reg_points, kNN_points, GNB_points, SVM_points, NN_points = 0,0,0,0,0
    for feature_size in feature_sizes:
        j = 0
        max_acc = 0
        best_models = []
        for acc in acc_matrix[i, :]:
            if(acc > max_acc):
                max_acc = acc
                best_models = [models_ALL[j]]
            elif(acc == max_acc):
                best_models.append(models_ALL[j])
            j += 1
        i += 1
        if('log_reg' in best_models):
            log_reg_points += 1
        if('kNN' in best_models):
            kNN_points += 1
        if('GNB' in best_models):
            GNB_points += 1
        if('SVM' in best_models):
            SVM_points += 1
        if('NN' in best_models):
            NN_points += 1
    
    point_totals = [log_reg_points, kNN_points, GNB_points, SVM_points, NN_points]
    
    return np.round(np.mean(model_avgs), 2), point_totals, model_avgs_alt
        
def generate_alister_batch_file_windows(root_dir:str, pipeline:str, normalization:str, dispersion:str, folds:int = 0):
    '''This function generates windows .bat batch files that execute A-Lister in order to filter 
    the gene_exp.diff files produced by the Cuffdiff according to fold change, q-value, and etc.'''
    to_write_temp = 'python ALister_CLI.py diff-expression '
    if(folds > 0):
        full_dir = root_dir + '/' + pipeline + '/' + 'FOLD' + str(folds) + '/' 'Cuffdiff_' + normalization + '_' + dispersion + '_FOLD!@#@!/'
    else:
        full_dir = root_dir + '/' + pipeline + '/' 'Cuffdiff_' + normalization + '_' + dispersion + '/'  
    full_dir_windows = full_dir.replace('/', '\\')
    to_write_temp += full_dir
    to_write_temp += 'gene_exp.diff' + ' -pc "q1->AH,q2->CT" -dq "AH*CT" ' '-o ' + full_dir + ' -n "gene" '
    to_write_temp += '-s1 "sample_1" -s2 "sample_2" -f "q_value:lt0.05,value_1:gt1.0,value_2:gt1.0"'
    to_write_temp += '\nmove ' + full_dir_windows + 'FilteredDEFiles\Query0\DEFile0.txt ' + full_dir_windows + 'gene_exp.txt'
    to_write_temp += '\nrmdir /Q /S ' + full_dir_windows + 'FilteredDEFiles'
    to_write_temp += '\ndel /f ' + full_dir_windows + 'result.txt'
    to_write_temp += '\ndel /f ' + full_dir_windows + 'data_dump.txt'
    if(folds > 0):
        filename = 'TOP_DEs_' + pipeline + '_' + normalization + '_' + dispersion + '_FOLD' + str(folds) + '.bat'
        with open(filename, 'w') as writer:
            i = 1
            while i <= folds:
                to_write = to_write_temp.replace('!@#@!', str(i))
                writer.write(to_write)
                writer.write('\n')
                i += 1
    else:
        filename = 'TOP_DEs_' + pipeline + '_' + normalization + '_' + dispersion + '.bat'
        with open(filename, 'w') as writer:
            writer.write(to_write_temp)

def generate_cuffnorm_or_cuffdiff_batch_file_HPC(reference_genome:str, aligner:str, annotation: str, normalization_method:str,
                                                 conditions:list, mode:str = "Cuffnorm", folds = 1, dispersion_method:str = ""
                                                 , tissue:str = 'PB'):
    ''' Generate the batch files for running cuffnorm or cuffdiff over AH project data on HPC. '''
    # Mode can be Cuffnorm or Cuffdiff
    # reference genome, aligner, and annotation must be lower case
    # normalization_method must be UQ or GEOM
    # dispersion_method must be POOL or COND
    
    if(tissue == 'PB'):
        cond_samples_tissue = cond_samples
    elif(tissue == 'PB_Excluded'):
        cond_samples_tissue = cond_samples_excluded
    elif(tissue == 'LV'):
        cond_samples_tissue = cond_samples_LV
    elif(tissue == 'LV_Excluded'):
        cond_samples_tissue = cond_samples_LV_excluded
    elif(tissue == 'TEST'):
        cond_samples_tissue = cond_samples_TEST
    else:
        raise ValueError("Unrecognized tissue argument.")
    
    Cond_Folds_Dict = {}
    for condition in conditions:
        if(condition not in ["AH", "AC", "CT", "AA", "DA", "NF", "HP"]):
            raise ValueError("Unknown condition.")
        Cond_Folds = generate_kfolds(cond_samples_tissue[condition], folds)
        Cond_Folds_Dict[condition] = Cond_Folds
        #print('Cond_Folds_Dict[', condition, ']: ', Cond_Folds_Dict)
        #print()
    
    fold_index = 0
    
    while fold_index < folds:
        Training_Folds = {}
        for condition in conditions:
            Training_Folds[condition] = []
        i = 0
        # Select training folds, leave out validation fold
        
        while i < folds:
            if((i == fold_index) and (folds > 1)):
                #print("Validation Fold: ", str(i))
                pass
            else:
                for condition in conditions:
                    #print("Training Fold: ", str(i), ' ', Cond_Folds_Dict[condition][i])
                    Training_Folds[condition].append(Cond_Folds_Dict[condition][i])

            i = i + 1
        
        part1 = reference_genome + "_" + aligner + "_" + annotation
        part1_capitalized = reference_genome + "_" + aligner.capitalize() + "_" + annotation.capitalize()
        
        part2 = normalization_method
        if(mode == "Cuffdiff"):
            part2 += "_" + dispersion_method
        if(folds > 1):
            part2 += "_FOLD" + str(fold_index + 1)
            
        temp = "SL_"
        r = 0
        num_conds = len(conditions)
        for condition in conditions:
            temp += str(len(cond_samples_tissue[condition])) + condition
            if(r < (num_conds - 1)):
                temp += "_vs_"
            else:
                if(tissue in ['PB', 'PB_Excluded']):
                    temp += "_PB_"
                elif (tissue in ['LV', 'LV_Excluded', 'TEST']):
                    temp += '_LV_'
            r += 1
            
        filename = temp + part1 + "_" + part2 + '.sh'
        jobname = temp + part1 + "_" + part2
    
        with open(filename, 'w') as writer:
            '...'
       
            conds_versus = ""
            i = 0
            num_conds = len(conditions)
            for condition in conditions:
                conds_versus += condition
                if(i < (num_conds - 1)):
                    conds_versus += "_"
                i = i + 1
                
            '...'
            writer.write("\n")
            if(mode == "Cuffnorm"):
                writer.write("cuffnorm -p 64 ")
            elif(mode == "Cuffdiff"):
                writer.write("cuffdiff -p 64 --max-bundle-frags 1000000000 ")
            else:
                raise ValueError("Mode must be Cuffnorm or Cuffdiff.")
            
            writer.write("--library-norm-method ")
            if(normalization_method == "UQ"):
                writer.write("quartile ")
            elif(normalization_method == "GEOM"):
                writer.write("geometric ")
            else:
                raise ValueError("Invalid normalization method value.")
                
            if(mode == "Cuffdiff"):
                if(dispersion_method == "POOL"):
                    pass
                elif(dispersion_method == "COND"):
                    writer.write("--dispersion-method per-condition ")
                else:
                    raise ValueError("Invalid dispersion method value.")
                
            '...'
            writer.write(part1_capitalized + "/" + mode +"_" + part2 + "/ " + reference_genome + "_" + annotation + ".gtf ")
            
            for k,Cond_Training_Folds in Training_Folds.items():
                #print('Condition: ', k)
                #print('Cond_Training_Folds: ', Cond_Training_Folds)
                num_Samples = two_dim_list_len(Cond_Training_Folds)
                #print("Num_Samples: ", str(num_Samples))
                k = 0
                for Fold in Cond_Training_Folds:
                    for Sample in Fold:
                        writer.write(Sample + "." + part1 + ".geneexp.cxb")
                        k = k + 1
                        if(k < num_Samples):
                            writer.write(",")
                        else:
                            writer.write(" ")
                            
        fold_index = fold_index + 1
    return Cond_Folds_Dict

def get_cuffdiff_gene_exp_file_fcs(filename:str, num_header_lines:int, id_col:int,
                                   rep_cond_names:list)->dict:
    '''Returns a dictionary mapping gene names to fold changes for each pairwise comparison 
    as specified via the rep_cond_names parameter.'''
    
    if (len(rep_cond_names) < 2):
        raise ValueError("Must provide at least two replicate condition names.")
    
    gene_exp_alt = {}
    with open(filename) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter='\t')
        line_count = 0
        lines_read = 0
        for row in csv_reader:
            line_count = line_count + 1
            #print(line_count)
            if(line_count <= num_header_lines):
                continue
            if(len(row) == 14):
                # Map each gene name to a list of fold changes 
                # (one for each pairwise comparison, such as q1_q2, q1_q3, and q2_q3).
                
                # Only record the fold change value stored in this line if it belongs to a valid 
                # pairwise comparison.
                #print((row[4] in rep_cond_names) and (row[5] in rep_cond_names))
                if((row[4] in rep_cond_names) and (row[5] in rep_cond_names)):
                    lines_read += 1
                    temp = gene_exp_alt.get(row[id_col])
                    if(temp == None):
                        gene_exp_alt[row[id_col]] = [abs(float(row[9]))]
                    else:
                        gene_exp_alt[row[id_col]].append(abs(float(row[9])))
            else:
                if(len(row) == 0):
                    # This is just an empty row.
                    pass
                else:
                    print("Warning. This row is missing values.")
                    print("Row: " + str(row))
                    
    #Remove any gene that does not have fold changes for all valid pairwise comparisons.
    num_valid_pairs = 0
    num_valid_conds = len(rep_cond_names)
    i = num_valid_conds - 1
    while i != 0:
        num_valid_pairs += i
        i -= 1
        
    gene_exp_clean = {}
    for k,v in gene_exp_alt.items():
        if(len(v) == num_valid_pairs):
            gene_exp_clean[k] = v
            
    return gene_exp_clean

def read_in_cuffdiff_gene_exp_file(filename:str, num_header_lines:int, N:"'ALL' or int", id_col:int,
                                   rep_cond_names:list, scheme:str, weights:list = [])->list:
    '''Read in Cuffdiff's gene_exp.diff file. Read in gene names and corresponding 
    absolute(fold change). Then return the top n genes sorted according to the scheme.
    Replicate condition names look like this: q1, q2, etc.
    If scheme = 'MEAN' --> Average absolute fold changes for each gene.
    Else if scheme - 'MAX' --> Pick maximum absolute fold change for each gene.
    Else if scheme - 'W-MEAN' --> Do weighted average of fold changes for each gene.
    '''
    
    gene_exp_clean = get_cuffdiff_gene_exp_file_fcs(filename, num_header_lines, id_col, 
                                                    rep_cond_names)
    gene_exp = []
    
    if(scheme == 'MEAN'):
        for gene_name,fold_changes in gene_exp_clean.items():
            fcs = np.array(fold_changes)
            avg_fc = np.mean(fcs)
            gene_exp.append((gene_name, avg_fc))
    elif(scheme == 'W-MEAN'):
        for gene_name,fold_changes in gene_exp_clean.items():
            if(len(weights) != len(fold_changes)):
                raise ValueError("The number of selected pairwise comparisons does not match number of weights.")
            s = 0
            i = 0
            for fc in fold_changes:
                s += fc * weights[i]
                i += 1
            w_avg_fc = s / i
            gene_exp.append((gene_name, w_avg_fc))
    elif(scheme == 'MAX'):
        for gene_name,fold_changes in gene_exp_clean.items():
            fcs = np.array(fold_changes)
            max_fc = np.max(fcs)
            gene_exp.append((gene_name, max_fc))
    else:
        raise ValueError("Scheme parameter must be 'W-MEAN', 'MEAN', or 'MAX'.")
        
    #Sort by absolute of fold change. 
    gene_exp.sort(key = sort_second_abs_float, reverse = True)
    #print(gene_exp)
    
    if(type(N) == int and N > len(gene_exp)):
        #print("Length of gene_exp: ", len(gene_exp))
        raise ValueError("N must be smaller than total number of genes read in.")

    top_N_genes = []
    limit = len(gene_exp)
    if(type(N) == int):
        limit = N
        
    i = 0
    while i < limit:
        top_N_genes.append(gene_exp[i][0])
        i = i + 1
    
    return top_N_genes

def read_in_cuffdiff_gene_exp_file_pairwise(filename: str, num_header_lines:int, N:int,
                                            id_col:int, rep_cond_names: list):
    '''Read in Cuffdiff's gene_exp.diff file. Read in gene names and corresponding 
    absolute(fold change). Then return the top N genes such that similar number of
    genes is selected from each pairwise comparison.'''
    
    if (len(rep_cond_names) < 2):
        raise ValueError("Must provide at least two replicate condition names.")
    
    gene_exp = []
    gene_exp_alt = {}
    with open(filename) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter='\t')
        line_count = 0
        lines_read = 0
        for row in csv_reader:
            line_count = line_count + 1
            #print(line_count)
            if(line_count <= num_header_lines):
                continue
            # Append gene name and log2(fold_change)
            if(len(row) == 14):
                # Map each pairwise comparison to a list of tuples that contain gene name and corresponding 
                # fold change.
                
                # Only record the fold change value stored in this line if it belongs to a valid 
                # pairwise comparison.
                #print((row[4] in rep_cond_names) and (row[5] in rep_cond_names))
                if((row[4] in rep_cond_names) and (row[5] in rep_cond_names)):
                    lines_read += 1
                    temp = gene_exp_alt.get( row[4]+row[5] )
                    if(temp == None):
                        gene_exp_alt[row[4] + row[5]] = [ (row[id_col], abs(float(row[9]))) ]
                    else:
                        gene_exp_alt[row[4] + row[5]].append( (row[id_col], abs(float(row[9]))) )
            else:
                if(len(row) == 0):
                    # This is just an empty row.
                    pass
                else:
                    print("Warning. This row is missing values.")
                    print("Row: " + str(row))
    
    for pc_name, gene_exp in gene_exp_alt.items():
        #print('pc_name: ', pc_name)
        #Sort by absolute of fold change. 
        gene_exp.sort(key = sort_second_abs_float, reverse = True)
        # print(len(gene_exp))

    top_genes = []
    iters = []
    for gene_exp in gene_exp_alt.values():
        iters.append(iter(gene_exp))
    
    # Keep track of which iterators (pairwise comparisons) have been depleted
    depletion_status = []
    i = 0
    while i < len(iters):
        depletion_status.append(False)
        i += 1
    
    cycles = 0
    while len(top_genes) < N:
        cycles += 1
        iter_index = -1
        for gene_exp in iters:
            iter_index += 1
            # If any of the pairwise comparisons become depleted, just skip them.
            try:
                gene = next(gene_exp)[0]
            except StopIteration:
                depletion_status[iter_index % len(iters)] = True
                if(all(depletion_status)):
                    raise StopIteration
                continue
            if(gene not in top_genes):
                top_genes.append(gene)
            if(len(top_genes) >= N):
                break
    #print('Cycles:', cycles)
    # print('top_genes: ', top_genes)
    return top_genes

def select_top_cuffdiff_DEs(filename:str, num_header_lines:int, N:"'ALL' or int",
                            rep_cond_names: list, id_col:int = 2, scheme:str = 'MAX', weights = [])->list:
    '''Return top N best genes. Use only the pairwise comparisons, composed of the provided
    condition replicate names (ex: q1, q2). Scheme: 'MEAN', 'MAX', 'W-MEAN', 'Pairwise'.
    '''
    if(scheme != 'MEAN' and scheme != 'MAX' and scheme != 'W-MEAN' and scheme != 'Pairwise'):
        raise ValueError('Scheme parameter must be "Pairwise", "MEAN", "MAX", or "W-MEAN".')
        
    if(type(N) != int and N != 'ALL'):
        raise TypeError("Parameter N must either be 'ALL' or an integer.")
    
    if(scheme in ['MEAN', 'W-MEAN', 'MAX']):
        top_genes = read_in_cuffdiff_gene_exp_file(filename, num_header_lines, N, id_col,
                                                     rep_cond_names, scheme, weights)
    elif(scheme == 'Pairwise'):
        if(type(N) != int):
            raise TypeError("N must be an integer when pairwise scheme is selected.")
        top_genes = read_in_cuffdiff_gene_exp_file_pairwise(filename, num_header_lines, N, id_col,
                                                              rep_cond_names)
        
    return top_genes
            
def sort_second_abs_float(value:tuple):
    return abs(float(value[1]))

def sort_third_float(value:tuple):
    return float(value[2])

def generate_cond_name_to_rep_name_map(file_dir:str, file_option:int = 0):
    '''Generates a dictionary that maps condition names (ex: AH) to replicate condition names (ex: q1).
    file_option: either 0 = Cuffnorm samples.table or 1 = Cuffdiff read_groups.info.
    If reverse = True, map replicate condition name to condition names instead (ex: q1 -> AH).'''
    result = {}
    filename = file_dir
    if(file_option == 0):
        filename += 'samples.table'
    elif(file_option == 1):
        filename += 'read_groups.info'
    else:
        raise ValueError("Invalid file option.")
        
    if(file_option == 0):
        with open(filename) as csv_file:
            csv_reader = csv.reader(csv_file, delimiter='\t')
            line_count = 0
            for row in csv_reader:
                if(line_count == 0):
                    pass
                else:
                    rep_cond_name = row[0][0:2]
                    cond_name = row[1][0:2]
                    result[cond_name] = rep_cond_name
                line_count += 1
    else:
        with open(filename) as csv_file:
            csv_reader = csv.reader(csv_file, delimiter='\t')
            line_count = 0
            for row in csv_reader:
                if(line_count == 0):
                    pass
                else:
                    rep_cond_name = row[1]
                    cond_name = row[0][0:2]
                    result[cond_name] = rep_cond_name
                line_count += 1
    
    return result
    

def compare_cuffnorm_sample_table_files(files:list)->bool:
    ''' Compare whether all of the given sample.table file contain the same assignment of 
    filenames to condition_sample number. '''
    
    i = 1
    while i < len(files):
        flag = compare_csv_files(files[0], files[i], [0, 1])
        if(not flag):
            return False
        i = i + 1
        
    return True
            
def compare_cuffdiff_read_groups_info_files(files:list)->bool:
    ''' Compare whether all of the given read_group.info files contain the same assignment 
    of filenames '...' to condition (ex: q1) 
    and replicate number (ex: 0) '''
    
    i = 1
    while i < len(files):
        flag = compare_csv_files(files[0], files[i], [0, 1, 2])
        if(not flag):
            return False
        i = i + 1
        
    return True

def compare_cuffdiff_mappings_to_cuffnorm_mappings(read_group_info:str, sample_table:str):
    ''' Identify whether the cuffnorm and cuffdiff files contain the same mapping of .cxb filenames to 
    condition_sample labels '...'. 
    Cuffdiff mappings are contained in read_group.info files and cuffnorm mappings in samples.table files.'''
    
    # Generate mappings for cuffdiffs read group info file.
    
    mappings = {}
    with open(read_group_info) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter='\t')
        line_count = 0
        for row in csv_reader:
            if line_count == 0:
                pass
            else:
                mappings[row[0]] = row[1] + "_" + row[2]
            line_count += 1
    
    mappings2 = {}
    with open(sample_table) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter='\t')
        line_count = 0
        for row in csv_reader:
            if line_count == 0:
                pass
            else:
                mappings2[row[1]] = row[0]
            line_count += 1
    
    error = False
    if(len(mappings) != len(mappings2)):
        error = True
    for key,value in mappings.items():
        try:
            if(value != mappings2[key]):
                error = True
        except:
            error = True
            
    if(error):
        raise ValueError("The provided cuffdiff and cuffnorm files have different file to condition_sample mappings.")
        
    return True
    
def compare_csv_files(file1:str, file2:str, column_indeces:list, delim1:str = '\t', delim2:str = '\t')->bool:
    ''' Compares whether the given columns contain identical values within both files. '''
    file1_contents = []
    with open(file1) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=delim1)
        for row in csv_reader:
            sublist = []
            for column_index in column_indeces:
                if(column_index >= len(row)):
                    raise ValueError("File1 has less columns than the following column index: ", str(column_index))
                sublist.append(row[column_index])
            file1_contents.append(sublist)
            
    file2_contents = []
    with open(file2) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=delim2)
        for row in csv_reader:
            sublist = []
            for column_index in column_indeces:
                if(column_index >= len(row)):
                    raise ValueError("File2 has less columns than the following column index: ", str(column_index))
                sublist.append(row[column_index])
            file2_contents.append(sublist)
    
    if(len(file1_contents) != len(file2_contents)):
        return False
    limit = len(file1_contents)
    
    i = 0
    while i < limit:
        if(file1_contents[i] != file2_contents[i]):
            return False
        i = i + 1
        
    return True

def read_cuffnorm_samples_table_filenames(filename:str)->list:
    ''' Read in and store all the .cxb sample filenames into a list.'''
    toReturn = []
    with open(filename) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter='\t')
        line_count = 0
        for row in csv_reader:
            if line_count == 0:
                pass
            else:
                toReturn.append(row[1])
            line_count += 1
    return toReturn

def read_cuffdiff_group_info_filenames(filename: str)->list:
    ''' Read in and store all the .cxb sample filenames into a list.'''
    toReturn = []
    with open(filename) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter='\t')
        line_count = 0
        for row in csv_reader:
            if line_count == 0:
                pass
            else:
                toReturn.append(row[0])
            line_count += 1
    return toReturn
        
def read_cuffdiff_counts(filename: str)->dict:
    ''' Stores the RNA-seq counts within a dictionary. 
    The keys are genename_category_samplenumber. Example: KLHL25_q1_34. 
    The values are floats.'''
    counts = {}
    with open(filename) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter='\t')
        line_count = 0
        for row in csv_reader:
            if line_count == 0:
                pass
            else:
                counts[str(row[0]) + "_" + (str(row[1]) + "_" + str(row[2]))] = float(row[6])
            line_count += 1
    return counts

def read_cuffnorm_gene_names(filename: str)->list:
    '''Reads in all the gene names into a list from a cuffnorm genes.fpkm_table file. 
    The order is preserved.'''
    gene_names = []
    with open(filename) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter='\t')
        line_count = 0
        for row in csv_reader:
            if line_count == 0:
                pass
            else:
                gene_name = row[0]
                gene_names.append(gene_name)
            line_count += 1
    return gene_names

def read_cuffdiff_gene_names(filename: str)->list:
    '''Reads in all the gene names into a list from a cuffnorm genes.read_group_tracking file. 
    The order is preserved.'''
    gene_names = []
    prev_gene_name = ''
    gene_name = ''
    with open(filename) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter='\t')
        line_count = 0
        for row in csv_reader:
            if line_count == 0:
                pass
            else:
                gene_name = row[0]
                #print('Current gene_name: ', row[0])
                if(gene_name != prev_gene_name):
                    #print('Current gene_name does not match prev_gene_name.')
                    gene_names.append(gene_name)
                    #print(gene_names)
            line_count += 1
            prev_gene_name = gene_name
    
    #print(gene_names)
    return gene_names

def read_cuffnorm_counts(filename: str)->dict:
    '''Reads in counts from cuffnorm genes.fpkm_table. Stores the RNA-seq counts within a dictionary. 
    The keys are genename_category_samplenumber. Example: KLHL25_q1_34. 
    The values are floats.'''
    counts = {}
    with open(filename) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter='\t')
        line_count = 0
        for row in csv_reader:
            if line_count == 0:
                headers = row
                pass
            else:
                gene_name = row[0]
                i = 1
                while i < len(row):
                    counts[gene_name + "_" + str(headers[i])] = float(row[i])
                    #print(gene_name + "_" + str(headers[i]) + "->" + str(row[i]))
                    i = i + 1
            line_count += 1
    return counts

def read_cuffnorm_counts2(filename: str, conditions:"list or 'ALL'" = 'ALL', log=False)->dict:
    '''Reads in counts from cuffnorm genes.fpkm_table. Only reads the counts for specified conditions.
    If conditions = ALL, read in all counts. Stores the cuffnorm RNA-seq counts in a dictionary.
    Genes are keys. Values are lists of two valued tuples. Each tuple contains the replicate name 
    (ex: q1_0) and the count value.
    Important: assuming Python >= 3.7 the ordering of genes is preserved and is consistent with the ordering within 
    the input Cuffnorm file.'''
    # Conditions are provided in the replicate condition name form (ex: q1, q2, q3, etc...)
    counts = {}
    headers = []
    with open(filename) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter='\t')
        line_count = 0
        for row in csv_reader:
            if line_count == 0:
                headers = row
            else:
                i = 1
                gene_name = row[0]
                counts[gene_name] = []
                while i < len(row):
                    if(log == False):
                        if(conditions == 'ALL'):
                            counts[gene_name].append((headers[i], float(row[i])))
                        elif(headers[i][0:2] in conditions):
                            #Only read in this count if its in one of the selected conditions.
                            counts[gene_name].append((headers[i], float(row[i])))
                    else:
                        if(conditions == 'ALL'):
                            counts[gene_name].append((headers[i], math.log(1 + float(row[i]))))
                        elif(headers[i][0:2] in conditions):
                            counts[gene_name].append((headers[i], math.log(1 + float(row[i]))))
                    i = i + 1
            line_count += 1
    
    return counts

def read_cuffdiff_counts2(filename: str, conditions:"list or 'ALL'" = 'ALL', log=False)->dict:
    '''Reads in counts from cuffdiff genes.read_group_tracking. Only reads the counts for specified conditions.
    If conditions = ALL, read in all counts. Stores the cuffnorm RNA-seq counts in a dictionary.
    Genes are keys. Values are lists of two valued tuples. Each tuple contains the replicate name 
    (ex: q1_0) and the count value.
    Important: assuming Python >= 3.7 the ordering of genes is preserved and is consistent with the ordering within 
    the input Cuffdiff file.'''
    counts = {}
    with open(filename) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter='\t')
        line_count = 0
        for row in csv_reader:
            if line_count == 0:
                pass
            else:
                gene_name = row[0]
                cond_name = row[1]
                rep_name = row[2]
                if(conditions == 'ALL' or (cond_name in conditions)):
                    try:
                        if(log):
                            counts[gene_name].append( (cond_name + '_' + rep_name, math.log(1 + float(row[6]))) )
                        else:
                            counts[gene_name].append( (cond_name + '_' + rep_name, float(row[6])) )
                    except KeyError:
                        if(log):
                            counts[gene_name] = [ (cond_name + '_' + rep_name, math.log(1 + float(row[6]))) ]
                        else:
                            counts[gene_name] = [ (cond_name + '_' + rep_name, float(row[6])) ]
            line_count += 1
    return counts


def read_cuffdiff_counts_mean(filename: str)->dict:
    '''Creates a dictionary which maps genename@categorylabel to a mean gene expression of that gene 
    for that condition (category / class). Example: KLHL25_q1:0.67234. The data is read from 
    Cuffdiff's genes.read_group_tracking files. '''
    counts = {}
    counts_lengths = {}
    with open(filename) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter='\t')
        line_count = 0
        for row in csv_reader:
            if line_count == 0:
                pass
            else:
                if((str(row[0]) + "@" + str(row[1])) in counts.keys()):
                    counts[str(row[0]) + "@" + str(row[1])] = counts[str(row[0]) + "@" + str(row[1])] + float(row[6])
                    counts_lengths[str(row[0]) + "@" + str(row[1])] = counts_lengths[str(row[0]) + "@" + str(row[1])] + 1
                else:
                    counts[str(row[0]) + "@" + str(row[1])] = float(row[6])
                    counts_lengths[str(row[0]) + "@" + str(row[1])] = 1
            line_count += 1
    for key in counts.keys():
        i = key.index("@")
        if(key[i+1:] in ['q1', 'q2', 'q3', 'q4', 'q5', 'q6', 'q7', 'q8', 'q9']):
            counts[key] = counts[key] / counts_lengths[key]
        else:
            message = ("This cuffdiff file should contain category labels q1 - q9 only as per " 
                       "Cuffdiff's genes.read_group_tracking file format.")
            raise ValueError(message)
    return counts

def read_cuffdiff_counts_mean_variance(filename: str, log=False)->dict:
    '''Returns a dictionary that maps each gene name to a 
    list of means and standard deviations. There is one mean and one standard deviation for each condition. 
    The means are listed first and the standard deviations are listed second.
    Assume that conditions are listed in order within the Cuffnorm file (q1, q2, ...). The individual replicates 
    within conditions are not listed in order. Assume that there maybe up to 9 conditions (q1 - q9).'''
    counts = read_cuffdiff_counts2(filename, 'ALL', log)
    counts_out = {}
    for k,v in counts.items():
        cond_counts = {}
        lengths = {}
        means = {}
        stds = {}
        sums = {}
        means_variance = []
        for cond_count_tuple in v:
            temp = cond_count_tuple[0].split('_')
            cond = temp[0]
            count = cond_count_tuple[1]
            try:
                cond_counts[cond].append(count)
                sums[cond] = sums[cond] + count
                lengths[cond] += 1
            except KeyError:
                cond_counts[cond] = [count]
                lengths[cond] = 1
                means[cond] = 0
                stds[cond] = 0
                sums[cond] = count
                
        for k2 in means.keys():
            means[k2] = sums[k2] / lengths[k2]
            
        for k3,v3 in cond_counts.items():
            for value in v3:
                stds[k3] = stds[k3] + (value - means[k3])**2
            stds[k3] = math.sqrt(stds[k3]/lengths[k3])
            
        for mean in means.values():
            means_variance.append(mean)
        
        for std in stds.values():
            means_variance.append(std)
        
        counts_out[k] = means_variance
        
    return counts_out

def read_cuffnorm_counts_mean_variance(filename: str, log=False)->dict:
    '''Returns a dictionary that maps each gene name to a 
    list of means and standard deviations. There is one mean and one standard deviation for each condition. 
    The means are listed first and the standard deviations are listed second.
    Assume that conditions are listed in order within the Cuffnorm file (q1, q2, ...). The individual replicates 
    within conditions are not listed in order. Assume that there maybe up to 9 conditions (q1 - q9).'''
    counts = {}
    with open(filename) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter='\t')
        line_count = 0
        for row in csv_reader:
            if line_count == 0:
                headers = row
                pass
            else:
                gene_name = row[0]
                i = 1
                means_variance = []
                
                sums = {}
                lengths = {}
                cond_counts = {}
                means = {}
                stds = {}
                
                while i < len(row):
                    if(headers[i][0:2] in ['q1', 'q2', 'q3', 'q4', 'q5', 'q6', 'q7', 'q8', 'q9']):
                        if(headers[i][0:2] not in sums.keys()):
                            sums[headers[i][0:2]] = 0
                            lengths[headers[i][0:2]] = 0
                            cond_counts[headers[i][0:2]] = []
                            means[headers[i][0:2]] = 0
                            stds[headers[i][0:2]] = 0
                        if(log):
                            sums[headers[i][0:2]] = sums[headers[i][0:2]] + math.log(1 + float(row[i]))
                            cond_counts[headers[i][0:2]].append(math.log(1 + float(row[i])))
                        else:
                            sums[headers[i][0:2]] = sums[headers[i][0:2]] + float(row[i])
                            cond_counts[headers[i][0:2]].append(float(row[i]))
                        lengths[headers[i][0:2]] = lengths[headers[i][0:2]] + 1
                    else:
                        raise ValueError("This script can only accept Cuffnorm files with up to 9 (q1 - q9) conditions.")
                    i = i + 1
                    
                for k in means.keys():
                    means[k] = sums[k] / lengths[k]
                
                for k,v in cond_counts.items():
                    for value in v:
                        stds[k] = stds[k] + (value - means[k])**2
                    stds[k] = math.sqrt(stds[k]/lengths[k])
                
                for mean in means.values():
                    means_variance.append(mean)
                
                for std in stds.values():
                    means_variance.append(std)
                    
                counts[gene_name] = means_variance
            line_count += 1
    return counts

def compare_cuffnorm_cuffdiff(cuffdiff_counts:str, cuffdiff_info:str, cuffnorm_counts:str, cuffnorm_info:str):
    ''' Comparison of Cuffnorm and Cuffdiff (Upper Quantile or Geometric Normalized) 'fpkm'
    counts with each other that were ran over identical .cxb files. '''
    
    compare_cuffdiff_mappings_to_cuffnorm_mappings(cuffdiff_info, cuffnorm_info)
    
    cuffdiff_counts = read_cuffdiff_counts(cuffdiff_counts)
    cuffnorm_counts = read_cuffnorm_counts(cuffnorm_counts)
    
    i = 0 
    inc_entries = 0
    for key,value in cuffdiff_counts.items():
        if(value != cuffnorm_counts[key]):
            inc_entries += 1
        i = i + 1
    
    #print("Cuffnorm to Cuffdiff: Percentage of incongruent entries: ")
    #print(str(inc_entries/i * 100) + '%')
    #print()
    
    return (inc_entries/i * 100)
    
def compare_cuffnorm_cuffnorm(file1:str, file2:str, margin:float):
    ''' Compare two non-identical cuffnorm count files with each other. 
    If the mean expressions and standard deviation of expression for each gene for each condition are 
    within margin of each other than the expression of two genes is considered similar. 
    Margin should be between 0 and 0.5.'''
    counts1 = read_cuffnorm_counts_mean_variance(file1)
    counts2 = read_cuffnorm_counts_mean_variance(file2)
    inc_entries = 0
    i = 0
    for key,value in counts1.items():
        num_conditions = len(value) // 2
        j = 0
        while j < num_conditions*2:
            try:
                if(math.isclose(counts2[key][j], value[j], rel_tol = margin)):
                    j += 1
                else:
                    inc_entries = inc_entries + 1
                    #print("Gene Name: ", key, "; File1 value: ", value, "; File2 value: ", counts2[key])
                    print("Gene Name (incongruent): ", key)
                    break
            except KeyError:
                print("Warning! Gene: ", key, "found in file1, but not file2.")
                break
        i = i + 1
    
    return (inc_entries/i * 100)
    
def compare_cuffdiff_cuffdiff(file1:str, file2:str, margin:float):
    ''' Compare two non-identical cuffdiff count files with each other. 
    If the mean expressions for each gene for each condition are within margin of each other 
    than the expression of two genes is considered similar. Margin should be between 0 and 0.5.'''
    counts1 = read_cuffdiff_counts_mean(file1)
    counts2 = read_cuffdiff_counts_mean(file2)
    inc_entries = 0
    i = 0
    for key,value in counts1.items():
        try:
            if(math.isclose(counts2[key], value, rel_tol = margin)):
                pass
            else:
                inc_entries = inc_entries + 1
                print("File1 value: ", value, "; File2 value: ", counts2[key])
        except KeyError:
            print("Warning! Gene: ", key, "found in file1, but not file2.")
        i = i + 1
    
    return (inc_entries/i * 100)


def validate_cuffnorm_cuffdiff_pipeline_files_one_setting(root_dir:str, pipeline:str, num_folds:int, normalization:str, dispersion:str):
    # The Cuffdiff and Cuffnorm RNA-seq counts attained from the same samples should be nearly identical.
    inc1 = compare_cuffnorm_cuffdiff(root_dir + '/' + pipeline + '/Cuffdiff_' + normalization + '_' + dispersion + '/genes.read_group_tracking', 
                                    root_dir + '/' + pipeline + '/Cuffdiff_' + normalization + '_' + dispersion + '/read_groups.info',
                                    root_dir + '/' + pipeline + '/Cuffnorm_' + normalization + '/genes.fpkm_table', 
                                    root_dir + '/' + pipeline + '/Cuffnorm_' + normalization + '/samples.table') 
    if(inc1 > 1):
        #if >1% incongruent at 0% margin
        raise ValueError("Error. Incongruency cuffnorm and cuffdiff counts.")
    
    # The cuffdiff RNA-seq counts per fold should be similar to the cuffdiff RNA-seq counts for the entire data.
    i = 1
    while i <= num_folds:
        file = root_dir + '/' + pipeline + '/FOLD'+str(num_folds)+'/Cuffdiff_' + normalization + '_' + dispersion + '_FOLD' + str(i) + '/genes.read_group_tracking'
        inc2 = compare_cuffdiff_cuffdiff(root_dir + '/' + pipeline + '/Cuffdiff_' + normalization + '_' + dispersion + '/genes.read_group_tracking', file, 0.2) 
        
        if(num_folds == 5):
            if(inc2 > 10):
                #>10% incongruent at 20% margin (5 FOLDS)
                print(inc2)
                raise ValueError("Root dir: " + root_dir + "; Pipeline: " + pipeline +
                                 "; Folds: 5. Incongruent folded vs non-folded cuffdiff counts.")
        elif(num_folds == 10):
            if(inc2 > 5):
                #>5% incongruent at 20% margin (10 FOLDS) 
                raise ValueError("Root dir: " + root_dir + "; Pipeline: " + pipeline +
                                 ". Folds: 10. Incongruent folded vs non-folded cuffdiff counts.")
        i = i + 1
    return True
    
def validate_cuffnorm_cuffdiff_pipeline_files_all_settings(root_dir:str, pipeline:str, num_folds:int):
    
    # Note on validation functions: cuffnorm to cuffdiff function checks that sample mappings are identical and 
    # for how many counts are identical. Cuffdiff to cuffdiff and cuffnorm to cuffnorm comparison functions 
    # check that the mean expression (count) for each gene are similar. These two functions do not check 
    # for identical mapping. There are additional functions to validate that compare_cuffdiff_read_groups_info_files 
    # & compare_cuffnorm_sample_table_files.
    
    # Validate that all Cuffnorm files were ran over the same samples.
    
    result = compare_cuffnorm_sample_table_files([root_dir + '/' + pipeline + '/Cuffnorm_GEOM/samples.table',
                                                  root_dir + '/' + pipeline + '/Cuffnorm_UQ/samples.table'])
            
    if(result != True):
        raise ValueError("Error. Runs with different cuffnorm settings contain different samples.")
    
    # Validate that all non-folded Cuffdiff instances have been ran over the same samples.
    
    result = compare_cuffdiff_read_groups_info_files([root_dir + '/' + pipeline + '/Cuffdiff_GEOM_COND/read_groups.info',
                                                      root_dir + '/' + pipeline + '/Cuffdiff_GEOM_POOL/read_groups.info',
                                                      root_dir + '/' + pipeline + '/Cuffdiff_UQ_COND/read_groups.info',
                                                      root_dir + '/' + pipeline + '/Cuffdiff_UQ_POOL/read_groups.info'])
            
    if(result != True):
        raise ValueError("Error. Runs with different cuffdiff settings contain different samples.")
    
    # Validate that each fold was ran over the same samples across different set ups (ex: samples in GEOM_COND_FOLD1 should be 
    # the same as samples in GEOM_POOL_FOLD1).
    
    i = 1
    while i <= num_folds:
        file1 = root_dir + '/' + pipeline + '/FOLD'+str(num_folds)+'/Cuffdiff_GEOM_COND_FOLD' + str(i) + '/read_groups.info'
        file2 = root_dir + '/' + pipeline + '/FOLD'+str(num_folds)+'/Cuffdiff_GEOM_POOL_FOLD' + str(i) + '/read_groups.info'
        file3 = root_dir + '/' + pipeline + '/FOLD'+str(num_folds)+'/Cuffdiff_UQ_COND_FOLD' + str(i) + '/read_groups.info'
        file4 = root_dir + '/' + pipeline + '/FOLD'+str(num_folds)+'/Cuffdiff_UQ_POOL_FOLD' + str(i) + '/read_groups.info'
        
        result = compare_cuffdiff_read_groups_info_files([file1, file2, file3, file4])  
        if(result != True):
            raise ValueError("Error. Two identically numbered folds contain different samples.")
        i = i + 1
    
    # Validate the pipeline against itself
    
    # Comparison of geometric and upper quantile normalization
    
    inc = compare_cuffnorm_cuffnorm(root_dir + '/' + pipeline + '/Cuffnorm_GEOM/genes.fpkm_table',
                                    root_dir + '/' + pipeline + '/Cuffnorm_UQ/genes.fpkm_table', 0.1) 
    if(inc > 1):
        #if >1% incongruent at 10% margin
        raise ValueError("Error. Incongruency between genes_GEOM and genes_UQ fpkm_table files.")
    
    #Upper Quartile Pooled Self Validation
    
    validate_cuffnorm_cuffdiff_pipeline_files_one_setting(root_dir, pipeline, num_folds, 'UQ', 'POOL')
    
    # Geometric Per-Condition Self Validation
    
    validate_cuffnorm_cuffdiff_pipeline_files_one_setting(root_dir, pipeline, num_folds, 'GEOM', 'COND')
    
    # Geometric Pooled Self Validation
    
    validate_cuffnorm_cuffdiff_pipeline_files_one_setting(root_dir, pipeline, num_folds, 'GEOM', 'POOL')
    
    # Upper Quartile Per-Condition Validation
    
    validate_cuffnorm_cuffdiff_pipeline_files_one_setting(root_dir, pipeline, num_folds, 'UQ', 'COND')
    
def generate_variance_mean_plots(filename:str, out_fname:str, v_transform:bool, counts_format:str):
    '''Plots mean vs variance graph of all genes for each condition in the provided fpkm_table or read_group_tracking file.
    The mean refers to mean gene expression and variance refers to variance of gene expression. '''
    
    if(counts_format == 'Cuffnorm'):
        counts = read_cuffnorm_counts_mean_variance(filename, v_transform)
    elif(counts_format == 'Cuffdiff'):
        counts = read_cuffdiff_counts_mean_variance(filename, v_transform)
    else:
        raise ValueError('Counts format should be Cuffnorm or Cuffdiff.')
    
    # counts layout 
    # key = gene names
    # value = list of all the condition means followed by all the condition standard deviations
    # (ex: [q1_mean, q2_mean, ... , qn_mean, q1_std, q2_std, ... , qn_std])
    
    cond_means = {}
    cond_stds = {}
    num_conds = 0
    
    for key,value in counts.items():
        num_conds = (len(value) // 2)
        i = 0
        while i < num_conds:
            cond_name = 'q' + str(i+1)
            
            if(cond_name not in cond_means.keys()):
                cond_means[cond_name] = []
                cond_stds[cond_name] = []
            
            cond_means[cond_name].append(value[i])
            cond_stds[cond_name].append(value[i + num_conds])
            i = i + 1
    
    #Lets plot mean vs variance graph for all genes. Lets do this for each condition.
    #This will result in up to 9 plots (read_cuffnorm_counts_mean_variance supports up to 9 conditions).
    
    i = 1
    while i <= num_conds:
        cond_name = 'q' + str(i)
        plt.plot(cond_means[cond_name], cond_stds[cond_name], 'ro')
        plt.ylabel('RNA-seq Counts Variance ' + cond_name, fontsize=14)
        plt.xlabel('RNA-seq Counts Mean '  + cond_name, fontsize=14)
        plt.savefig(out_fname + '_' + cond_name + '.png')
        # plt.show()
        i = i + 1
    return cond_means, cond_stds
    
    
# MACHINE LEARNING SETUP FUNCTIONS
    
def filenames_to_replicate_names_cuffnorm(samples_table_fname:str, names:list, reverse:bool = False)->list:
    ''' Convert .cxb filenames into condition_sample (replicate) names according to Cuffnorm sample.table file.
    Ignore filenames not listed in the file.'''
    old_name_to_new_name = {}
    with open(samples_table_fname) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter='\t')
        line_count = 0
        for row in csv_reader:
            if line_count == 0:
                pass
            else:
                if(not reverse):
                    old_name_to_new_name[row[1]] = row[0]
                else:
                    old_name_to_new_name[row[0]] = row[1]
            line_count = line_count + 1
    
    new_names = []
    absent = []
    for name in names:
        if(name in old_name_to_new_name.keys()):
            new_names.append(old_name_to_new_name[name])
        else:
            absent.append(name)
            
    # print(len(absent), " (rep or file) names not not listed in the sample.table file.")
        
    return new_names

def filenames_to_replicate_names_cuffdiff(read_groups_fname:str, names:list, reverse:bool = False)->list:
    ''' Convert .cxb filenames into condition_sample (replicate) names according to Cuffdiff read_groups.info file.
    Ignore filenames not listed in the file.'''
    old_name_to_new_name = {}
    with open(read_groups_fname) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter='\t')
        line_count = 0
        for row in csv_reader:
            if line_count == 0:
                pass
            else:
                if(not reverse):
                    old_name_to_new_name[row[0]] = row[1] + '_' + row[2] 
                else:
                    old_name_to_new_name[row[1] + '_' + row[2]] = row[0]
            line_count = line_count + 1
    
    new_names = []
    absent = []
    for name in names:
        if(name in old_name_to_new_name.keys()):
            new_names.append(old_name_to_new_name[name])
        else:
            absent.append(name)
            
    # print(len(absent), " (rep or file) names not not listed in the sample.table file.")
        
    return new_names

def generate_data(num_samples:dict, v_transform:bool, root_dir:str, pipeline:str, normalization:str, dispersion:str,
                  fold:int = 0, num_folds:int = 10, tissue:str = 'PB', counts_format:str = 'Cuffnorm', verbose:bool = False):
    '''Generates and returns X & Y using count data. Only conditions specified in num_samples are included
    in X & Y matrices. Also return the list of gene names aligned in parallel 
    with count data matrices. For simplicity sake only supports up to 7 pre-defined conditions
    (AH, AC, CT, DA, AA, NF, HP) as of now.'''

    if(counts_format not in ['Cuffnorm', 'Cuffdiff']):
        raise ValueError("Counts format should be either Cuffnorm or Cuffdiff.")

    if(tissue == 'PB'):
        cond_samples_tissue = cond_samples
    elif(tissue == 'PB_Excluded'):
        cond_samples_tissue = cond_samples_excluded
    elif(tissue == 'LV'):
        cond_samples_tissue = cond_samples_LV
    elif(tissue == 'LV_Excluded'):
        cond_samples_tissue = cond_samples_LV_excluded
    else:
        raise ValueError("Unrecognized tissue argument.")
    # Read in the counts. Maintain the gene name and condition/sample
    # identifiers for each count. (ex: ATRAC q1_0 : 5.42)
    
    num_samples_int = 0
    for v in num_samples.values():
        num_samples_int += v
    
    if(fold == 0):
        fpkm_table = root_dir + '/' + pipeline + '/Cuffnorm_' + normalization + '/genes.fpkm_table'
        group_tracking = root_dir + '/' + pipeline + '/Cuffdiff_' + normalization + '_' + dispersion + '/genes.read_group_tracking'
        samples_table = root_dir + '/' + pipeline + '/Cuffnorm_' + normalization + '/samples.table'
        groups_info = root_dir + '/' + pipeline + '/Cuffdiff_' + normalization + '_' + dispersion + '/read_groups.info'
    else:
        fpkm_table = root_dir + '/' + pipeline + '/FOLD' + str(num_folds) + '/Cuffnorm_' + normalization + '_FOLD' + str(fold) + '/genes.fpkm_table'
        group_tracking = root_dir + '/' + pipeline + '/FOLD' + str(num_folds) + '/Cuffdiff_' 
        group_tracking += normalization + '_' + dispersion + '_FOLD' + str(fold) + '/genes.read_group_tracking'
        samples_table = root_dir + '/' + pipeline + '/FOLD' + str(num_folds) + '/Cuffnorm_' + normalization + '_FOLD' + str(fold) + '/samples.table'
        groups_info = root_dir + '/' + pipeline + '/FOLD' + str(num_folds) + '/Cuffdiff_' + normalization 
        groups_info += '_' + dispersion + '_FOLD' + str(fold) + '/read_groups.info'
        
    #Translate condition names (ex: AH) to replicate condition names (ex: q1, q2, etc.)
    conditions = num_samples.keys()
    if(counts_format == 'Cuffnorm'):
        temp_dir = root_dir +'/' + pipeline + '/Cuffnorm_' + normalization + '/'
        cond_to_rep_cond_map = generate_cond_name_to_rep_name_map(temp_dir)
    else:
        temp_dir = root_dir +'/' + pipeline + '/Cuffdiff_' + normalization + '_' + dispersion + '/'
        cond_to_rep_cond_map = generate_cond_name_to_rep_name_map(temp_dir, 1)
    rep_cond_names = []
    for cond in conditions:
        if(cond_to_rep_cond_map[cond] not in rep_cond_names):
            rep_cond_names.append(cond_to_rep_cond_map[cond])
    
    if(counts_format == 'Cuffnorm'):
        counts = read_cuffnorm_counts2(fpkm_table, rep_cond_names, v_transform)
        gene_names = read_cuffnorm_gene_names(fpkm_table)
    else:
        counts = read_cuffdiff_counts2(group_tracking, rep_cond_names, v_transform)
        gene_names = read_cuffdiff_gene_names(group_tracking)
    
    # This will be useful at the end of the function.
    # We need to translate the conditions in num_samples variables into Y matrix integer labels.
    # We first translate condition names into replicate condition names (ex: AH -> q1).
    # Afterward we map each condition replicate name to a label integer for the Y matrix (ex: q1 -> 0).
    # This must be done such that each replicate condition name
    # maps to a successive integer starting at 0 (ex: q1 -> 0, q3 -> 1)
    cond_rep_name_to_label = {}
    label = 0
    # for cond in conditions:
    #     cond_rep_name_to_label[cond_to_rep_cond_map[cond]] = label
    #     label += 1
    for rep_cond_name in rep_cond_names:
        cond_rep_name_to_label[rep_cond_name] = label
        label += 1
        
    if(verbose):
        print("Condition replicate name to class label map: ", cond_rep_name_to_label)

    all_samples_cond_filenames = {'AH':[], 'AC':[], 'CT':[], 'DA':[], 'AA':[], 'NF':[], 'HP':[]}
    
    conditions = []
    for cond in num_samples.keys():
        conditions.append(cond)
        for name in cond_samples_tissue[cond]:
            all_samples_cond_filenames[cond].append(name + '.'+str.lower(pipeline)+'.geneexp.cxb')
             
    counts2 = {}                     
    gene_names2 = []
    for key, value in counts.items():
        gene_names2.append(key)
        for v in value:
            try:
                counts2[v[0]].append(v[1])
            except KeyError:
                counts2[v[0]] = [v[1]]
                
    if(gene_names != gene_names2):
        raise ValueError("The ordering of gene names is not consistent.")
    
    if(verbose):
        print("Number of genes: ", len(gene_names))
        print("Number of keys in counts2: ", len(counts2))
    
    if(len(counts2.keys()) != num_samples_int):
        raise ValueError("Error in data generation process.")
                
    # Validate counts. Each sample should have the same number of recorded (genes/counts).
    i = 0
    prev_value_length = 0
    for key,value in counts2.items():
        if(i == 0):
            prev_value_length = len(value)
        else:
            if(len(value) != prev_value_length):
                raise ValueError("Not all samples have the same number of recorded gene counts.")
            prev_value_length = len(value)
        i = i + 1
        
    if(verbose):
        print("The number of genes per sample: ", prev_value_length)

    # Use the samples.table files to correctly align X & Y data matrices
    
    cond_samples_filenames = {'AH':[], 'AC':[], 'CT':[], 'DA':[], 'AA':[], 'NF':[], 'HP':[]}
    cond_samples_repnames = {'AH':[], 'AC':[], 'CT':[], 'DA':[], 'AA':[], 'NF':[], 'HP':[]}
    
    if(counts_format == 'Cuffnorm'):
        data_filenames = read_cuffnorm_samples_table_filenames(samples_table)
    else:
        data_filenames = read_cuffdiff_group_info_filenames(groups_info)
        
    if(verbose):
        print("Filenames in this fold (or total): ", data_filenames)
    
    for fname in data_filenames:
        if(fname[0:2] in ['AH', 'AC', 'CT', 'DA', 'AA', 'NF', 'HP']):
            cond_samples_filenames[fname[0:2]].append(fname)
        else:
            raise ValueError("This filename belongs to an unkown conditions. Only AH, AC, CT, DA, AA, NF, and HP are supported.")
            
    # Check that every read in filename is valid.
    for cond in conditions:
        for fname in cond_samples_filenames[cond]:
            if(fname not in all_samples_cond_filenames[cond]):
                raise ValueError("This filename: " + fname + " is not valid.")
        
    for cond in conditions:
        if(counts_format == 'Cuffnorm'):
            cond_samples_repnames[cond] = filenames_to_replicate_names_cuffnorm(samples_table,
                                                                                cond_samples_filenames[cond])
        else:
            cond_samples_repnames[cond] = filenames_to_replicate_names_cuffdiff(groups_info,
                                                                                cond_samples_filenames[cond])
    
    X= []
    Y = []
    for cond in conditions:
        for rep_name in cond_samples_repnames[cond]:
            if(rep_name in counts2.keys()):
                if(verbose):
                    print("Added repname: ", rep_name, " to X & Y.")
                X.append(counts2[rep_name])
                if(rep_name[0:2] in cond_rep_name_to_label.keys()):
                    Y.append(cond_rep_name_to_label[rep_name[0:2]])
                else:
                    raise ValueError("The following replicate " + rep_name + " is not in " + cond_rep_name_to_label.keys() + ".")
            else:
                raise KeyError("The following replicate name was not found in count file.")
        
    # Convert the aligned X and Y lists into numpy arrays.
    X = np.array(X)
    Y = np.array(Y).reshape(len(Y), 1)
    
    if(X.shape[0] != num_samples_int):
        raise ValueError("Number of samples in data (X) matrix does not equal number of total samples.")
    if(Y.shape[0] != num_samples_int):
        raise ValueError("Number of samples in target labels (Y) does not equal number of total samples.")
    
    Y = np.ravel(Y)
    
    return X, Y, gene_names

def generate_train_validate_data(num_DEG:int, num_folds:int, num_samples:dict, root_dir:str, pipeline:str,
                                 normalization:str, dispersion:str, counts:dict, gene_names:list, features_file:str,
                                 taboo_list:list, tissue:str, out_dir:str, gene_lists:bool, counts_format:str,
                                 verbose:bool, testing:bool = False):
    '''Generates and returns X_trains, X_validates, Y_trains, Y_validates for each 
    fold using RNA-seq counts. Only conditions specified in num_samples are included in X & Y matrices.
    You are currently required to use all of the samples alloted to each condition.
    For simplicity sake only supports up to 7 pre-defined conditions 
    (AH, AC, CT, DA, AA, NF, HP) as of now. This function is filled with validation checks that test 
    the intermediary variables.'''
    num_samples_int = 0
    for v in num_samples.values():
        num_samples_int += v
        
    if(counts_format not in ['Cuffnorm', 'Cuffdiff']):
        raise ValueError("Counts format must be either Cuffnorm or Cuffdiff.")
        
    if(tissue == 'PB'):
        cond_samples_tissue = cond_samples
    elif(tissue == 'PB_Excluded'):
        cond_samples_tissue = cond_samples_excluded
    elif(tissue == 'LV'):
        cond_samples_tissue = cond_samples_LV
    elif(tissue == 'LV_Excluded'):
        cond_samples_tissue = cond_samples_LV_excluded
    else:
        raise ValueError("Unrecognized tissue argument.")
        
    assert(type(num_DEG) == int)
    assert(type(num_folds) == int and num_folds > 1)
    assert(num_folds <= num_samples_int)
    assert(num_DEG <= len(counts.keys()))
    assert(len(counts.keys()) == len(gene_names))
    if(not testing):
        assert(pipeline in pipelines_ALL)
    
    all_y_validates_rnames = []
    all_y_validates_snames = []
    
    X_trains = []
    X_validates = []
    Y_trains = []
    Y_validates = []
    
    # This will be useful at the end of the function.
    # We need to translate the conditions in num_samples variables into Y matrix integer labels.
    # We first translate condition names into replicate condition names (ex: AH -> q1).
    # Afterward we map each condition replicate name to a label integer for the Y matrix (ex: q1 -> 0).
    # This must be done such that each replicate condition name
    # maps to a successive integer starting at 0 (ex: q1 -> 0, q3 -> 1)
    if(counts_format == 'Cuffnorm'):
        temp_dir = root_dir + '/' + pipeline + '/Cuffnorm_' + normalization + '/'
        cond_to_rep_cond_map = generate_cond_name_to_rep_name_map(temp_dir, 0)
    else:
        temp_dir = root_dir + '/' + pipeline + '/Cuffdiff_' + normalization + '_' + dispersion + '/'
        cond_to_rep_cond_map = generate_cond_name_to_rep_name_map(temp_dir, 1)
    conditions = num_samples.keys()
    
    rep_cond_names = []
    for cond in conditions:
        if(cond_to_rep_cond_map[cond] not in rep_cond_names):
            rep_cond_names.append(cond_to_rep_cond_map[cond])
    
    cond_rep_name_to_label = {}
    label = 0
    for rep_cond_name in rep_cond_names:
        cond_rep_name_to_label[rep_cond_name] = label
        label += 1
        
    if(verbose):
        print("Condition replicate name to class label map: ", cond_rep_name_to_label)
    
    # Read in the counts. Maintain the gene name and condition/sample
    # identifiers for each count. (ex: ATRAC q1_0 : 5.42)
    samples_cond_filenames_ALL = {'AH':[], 'AC':[], 'CT':[], 'DA':[], 'AA':[], 'NF':[], 'HP':[]}
    samples_cond_filenames = {'AH':[], 'AC':[], 'CT':[], 'DA':[], 'AA':[], 'NF':[], 'HP':[]}
    samples_cond_repnames = {'AH':[], 'AC':[], 'CT':[], 'DA':[], 'AA':[], 'NF':[], 'HP':[]}
    
    conditions = []
    for cond in num_samples.keys():
        conditions.append(cond)
        for name in cond_samples_tissue[cond]:
            samples_cond_filenames_ALL[cond].append(name + '.'+str.lower(pipeline)+'.geneexp.cxb')
    
    if(counts_format == 'Cuffnorm'):
        samples_table = root_dir + '/' + pipeline + '/Cuffnorm_' + normalization + '/samples.table'
        filenames = read_cuffnorm_samples_table_filenames(samples_table)
    else:
        groups_info = root_dir + '/' + pipeline + '/Cuffdiff_' + normalization + '_' + dispersion + '/read_groups.info'
        filenames = read_cuffdiff_group_info_filenames(groups_info)
    # Sort all the filenames by the condition.
    for filename in filenames:
        samples_cond_filenames[filename[0:2]].append(filename)
    
    # Example replicate name: q1_15 -> condition 1, 15th sample within that condition. 
    possible_replicate_names = []
    for cond,fnames in samples_cond_filenames_ALL.items():
        if(counts_format == 'Cuffnorm'):
            rep_names = filenames_to_replicate_names_cuffnorm(samples_table, fnames)
        else:
            rep_names = filenames_to_replicate_names_cuffdiff(groups_info, fnames)
        for r_name in rep_names:
            possible_replicate_names.append(r_name)
    possible_replicate_names.sort()
    
    # While the uniquness (non-duplication) of each replicate is guranteed as long as Cuffnorm/Cuffdiff files are valid.
    # Here is a short check to ensure this in case the Cuffnorm/Cuffdiff samples table / read groups info file is invalid.
    temp = list(set(possible_replicate_names))
    temp.sort()
    if(possible_replicate_names != temp):
        raise ValueError("The replicate names are not unique.")
    
    i = 0
    # Due to the structure of cuffnorm/cuffdiff genes_fpkm / read_group_tracking
    # file we know that all genes will have same sets of replicates.
    # So we only need to verify that the replicates for the first gene look valid. 
    all_replicate_names_test = []
    for gene_feature in counts.values():
        if(len(gene_feature) != num_samples_int):
            raise ValueError("Each value in counts must be compromised of all replicates (= # of samples).")
        for rep_count_tuple in gene_feature:
            rep_name = rep_count_tuple[0]
            all_replicate_names_test.append(rep_name)
        
        # all_replicate_names_test should be a subset of possible_replicate_names
        for rep_name in all_replicate_names_test:
            if(rep_name not in possible_replicate_names):
                raise ValueError("The following replicate: " + rep_name + " is not valid.")
        break
   
    for cond in conditions:
        if(counts_format == 'Cuffnorm'):
            samples_cond_repnames[cond] = filenames_to_replicate_names_cuffnorm(samples_table,
                                                                                samples_cond_filenames[cond]) 
        else:
            samples_cond_repnames[cond] = filenames_to_replicate_names_cuffdiff(groups_info,
                                                                                samples_cond_filenames[cond])
    
    # Generate rep_name_verification dictionary which will be used to verify that every replicate is used in 
    # training data (num_folds - 1) times and in validation data 1 time.
    rep_name_verification = {}
    for r_name in all_replicate_names_test:
        rep_name_verification[r_name] = {'Train_Replicate' : 0, 'Validation_Replicate' : 0}
    
    if(verbose):
        print("Conditions: ", conditions)
        #print("Filenames: ", samples_cond_filenames)
        print("Replicate names: ", all_replicate_names_test)
    
    fold = 1
    while fold <= num_folds:
        if(verbose):
            print("Current fold: ", fold)
        
        top_genes = []
        removed_genes = []
        
        if(counts_format == 'Cuffnorm'):
            filename = root_dir + '/' + pipeline+'/FOLD'+str(num_folds)+'/Cuffnorm_'+normalization+'_FOLD' + str(fold)
            filename += '/' + features_file
        else:
            filename = root_dir + '/' + pipeline+'/FOLD'+str(num_folds)+'/Cuffdiff_'+normalization+ '_' + dispersion 
            filename += '_FOLD' + str(fold) + '/' + features_file
            
        with open(filename) as csv_file:
            csv_reader = csv.reader(csv_file, delimiter='\t')
            z = 0
            for row in csv_reader:
                if(len(row) >= 1):
                    # If this gene is not on the taboo list, read it in.
                    if row[0].upper() in taboo_list:
                        removed_genes.append(row[0])
                    else:
                        top_genes.append(row[0])
                        z += 1
                else:
                    print("Encountered an empty row in csv file.")
                if(z >= num_DEG):
                    break
        
        if(verbose):
            print("Top Genes: ", top_genes)
            print("Removed Genes: ", removed_genes)
        
        if(gene_lists):
            # Put the top genes into a file and place it into output directory.
            temp = out_dir.split('/')
            run_name = temp[-2]
            
            out_gene_list = out_dir + 'GeneLists/'
            if(not os.path.isdir(out_gene_list)):
                os.mkdir(out_gene_list)
            f = open(out_gene_list + 'gene_list_fold' + str(fold) + '_' + run_name + '.txt', "w")
            for gene in top_genes:
                f.write(gene + '\n')
            f.close()
            
            # Now generate a filtered .diff file in the same output directory.
            gene_list_name = out_gene_list + 'gene_list_fold' + str(fold) + '_' + run_name + '.txt'
            cuffdiff_name = root_dir + '/' + pipeline+'/FOLD'+str(num_folds)+'/Cuffdiff_'+normalization
            cuffdiff_name += '_' + dispersion + '_FOLD' + str(fold) + '/' + 'gene_exp.diff'
            out_fname = 'filtered_fold' + str(fold) + '_' + run_name + '.diff'
            filter_cuffdiff_file_by_gene_list(gene_list_name, cuffdiff_name, out_gene_list, out_fname)
        
        # Validate top genes.
        num_top_genes_miss_in_counts = 0
        for gene in top_genes:
            if(gene not in gene_names):
                num_top_genes_miss_in_counts += 1
                if(counts_format == 'Cuffnorm'):
                    print("This top gene is missing in cuffnorm fpkm_table file: ", str(gene))
                else:
                    print("This top gene is missing in cuffdiff read_group_tracking file: ", str(gene))
        if(num_top_genes_miss_in_counts/num_DEG > 0.01):
            if(counts_format == 'Cuffnorm'):
                raise ValueError("More than 1% of top genes cannot be found in Cuffnorm counts.")
            else:
                raise ValueError("More than 1% of top genes cannot be found in Cuffdiff counts.")
            
        # Filter counts data by selected genes.
        filtered_counts_temp = {}
        for gene in top_genes:
            if(gene in counts.keys()):
                filtered_counts_temp[gene] = counts[gene]
                
        if(verbose):
            #print("Filtered Counts Intermideary: ", filtered_counts_temp
            print("Filtered Counts Intermideary (keys): ", filtered_counts_temp.keys())
        
        # Its crucial that genes counts are ordered identically for each replicate, starting with topmost gene, and 
        # ending with bottom most gene according to the features files.
        filtered_counts = {}
        top_genes_verify = []
        p = 0             
        for gene, gene_feature in filtered_counts_temp.items():
            top_genes_verify.append(gene)
            # if(verbose):
            #     print("Gene Name: ", gene)
            if(p == 0):
                for rep_count_tuple in gene_feature:
                    rep_name = rep_count_tuple[0]
                    filtered_counts[rep_name] = []
            for rep_count_tuple in gene_feature:
                rep_name = rep_count_tuple[0]
                gene_count = rep_count_tuple[1]
                filtered_counts[rep_name].append(gene_count)
            p += 1
            
        # Generate per-sample counts heatmap for this fold.
        if(gene_lists):
            if(counts_format == 'Cuffnorm'):
                plot_per_sample_counts_heatmap(filtered_counts, top_genes_verify, samples_table, fold, out_gene_list, counts_format)
            else:
                plot_per_sample_counts_heatmap(filtered_counts, top_genes_verify, groups_info, fold, out_gene_list, counts_format)
        
        # Validate that the genes were added to filtered counts in the same order as that 
        # of the genes in the features file.
        rr = 0
        while rr < len(top_genes_verify):
            if(top_genes_verify[rr] != top_genes[rr]):
                raise ValueError("There is an issue with the ordering of top genes within the X matrices.")
            rr += 1
        
        # if(verbose):
            #print("Filtered Counts: ", filtered_counts)
            # print("Filtered Counts (keys): ", filtered_counts.keys())
            #print("Filtered Counts len(keys): ", len(filtered_counts.keys()))
                    
        # Validate filtered counts. Each sample should have the same number of recorded (genes/counts).
        i = 0
        previous_value_length = 0
        for value in filtered_counts.values():
            if(i == 0):
                previous_value_length = len(value)
            else:
                if(len(value) != previous_value_length):
                    raise ValueError("Not all samples have the same number of recorded gene counts.")
                previous_value_length = len(value)
            i = i + 1
    
        # Use the samples.table and read_groups.info files to correctly align and separate sample data 
        # into training and validation data.
        
        fold_train_cond_filenames = {'AH':[], 'AC':[], 'CT':[], 'DA':[], 'AA':[], 'NF':[], 'HP':[]}
        fold_train_cond_repnames = {'AH':[], 'AC':[], 'CT':[], 'DA':[], 'AA':[], 'NF':[], 'HP':[]}
        
        if(counts_format == 'Cuffnorm'):
            # Confirm that generated training fold filenames are identical to the ones in the respective cuffnorm fold's sample table.
            samples_table_fold = root_dir + '/' + pipeline + '/FOLD' +str(num_folds)
            samples_table_fold += '/Cuffnorm_' + normalization + '_FOLD' + str(fold) + '/samples.table'
            train_fold_filenames = read_cuffnorm_samples_table_filenames(samples_table_fold)
        else:
            groups_info_fold = root_dir + '/' + pipeline + '/FOLD' +str(num_folds) + '/Cuffdiff_' + normalization + '_'
            groups_info_fold += dispersion + '_FOLD' + str(fold) + '/read_groups.info'
            train_fold_filenames = read_cuffdiff_group_info_filenames(groups_info_fold)
            
        for value in train_fold_filenames:
            if(value[0:2] in ['AH', 'AC', 'CT', 'DA', 'AA', 'NF', 'HP']):
                fold_train_cond_filenames[value[0:2]].append(value)
            else:
                raise ValueError("Unrecognized condition. Only AH, CT, DA, AA, NF, and HP .cxb files are permitted.")            
        
        for cond in conditions:
            if(counts_format == 'Cuffnorm'):
                fold_train_cond_repnames[cond] = filenames_to_replicate_names_cuffnorm(samples_table, fold_train_cond_filenames[cond])
            else:
                fold_train_cond_repnames[cond] = filenames_to_replicate_names_cuffdiff(groups_info, fold_train_cond_filenames[cond])
        if(verbose):
            print("Train fold filenames: ", fold_train_cond_filenames)
            print("Train fold repnames: ", fold_train_cond_repnames)
        
        fold_validate_cond_filenames = {'AH':[], 'AC':[], 'CT':[], 'DA':[], 'AA':[], 'NF':[], 'HP':[]}
        fold_validate_cond_repnames = {'AH':[], 'AC':[], 'CT':[], 'DA':[], 'AA':[], 'NF':[], 'HP':[]}
        
        for cond in conditions:
            for fname in samples_cond_filenames[cond]:
                if(fname in fold_train_cond_filenames[cond]):
                    pass
                else:
                    fold_validate_cond_filenames[cond].append(fname)
                
        for cond in conditions:
            if(counts_format == 'Cuffnorm'):
                fold_validate_cond_repnames[cond] = filenames_to_replicate_names_cuffnorm(samples_table,
                                                                                          fold_validate_cond_filenames[cond])
            else:
                fold_validate_cond_repnames[cond] = filenames_to_replicate_names_cuffdiff(groups_info,
                                                                                          fold_validate_cond_filenames[cond])
        if(verbose): 
            print("Validate fold filenames: ", fold_validate_cond_filenames)
            print("Validate fold repnames: ", fold_validate_cond_repnames)
            num_validation_replicates = 0
            for v in fold_validate_cond_repnames.values():
                num_validation_replicates += len(v)
            proportion_validation =  num_validation_replicates / num_samples_int
            print("Propotion of validation replicates: ", proportion_validation)
            if(not math.isclose(proportion_validation, 1/num_folds, abs_tol = 0.04)):
               raise ValueError("The validation group size is invalid.")
            
        
        X_train = []
        X_validate = []
        Y_train = []
        Y_validate = []
        
        for cond in conditions:
            if(verbose):
                print("Condition: ", cond, "; Replicates: ", samples_cond_repnames[cond])
            for rep_name in samples_cond_repnames[cond]:
                if(rep_name in filtered_counts.keys()): 
                    if(rep_name in fold_train_cond_repnames[cond]):
                        # if(verbose):
                        #     print("Condition: ", cond, "; Training Replicate: ", rep_name)
                        X_train.append(filtered_counts[rep_name])
                        rep_name_verification[rep_name]['Train_Replicate'] += 1
                        if(rep_name[0:2] in cond_rep_name_to_label.keys()):
                            Y_train.append(cond_rep_name_to_label[rep_name[0:2]]) # Aka q1 -> 0; q2 -> 1 ... q9 -> 8
                        else:
                            raise ValueError("The following replicate " + rep_name + " is not in " + cond_rep_name_to_label.keys() + ".")
                    elif(rep_name in fold_validate_cond_repnames[cond]):
                        if(verbose):
                            print("Condition: ", cond, "; Validation Replicate: ", rep_name)
                        X_validate.append(filtered_counts[rep_name])
                        rep_name_verification[rep_name]['Validation_Replicate'] += 1
                        if(rep_name[0:2] in cond_rep_name_to_label.keys()):
                            Y_validate.append(cond_rep_name_to_label[rep_name[0:2]]) # Aka q1 -> 0; q2 -> 1 ... q9 -> 8
                            # Store repnames of validation folds for downstream data analysis
                            all_y_validates_rnames.append(rep_name)
                        else:
                            raise ValueError("The following replicate " + rep_name + " is not in " + cond_rep_name_to_label.keys() + ".")
                    else:
                        raise ValueError("The following replicate " + rep_name + " was not found in train/validate replicates.")
                else:
                    raise KeyError("The following replicate name was not found in count file.")
            
        # Convert the aligned X_train, X_validate, Y_train, Y_validate lists into numpy lists.
        
        X_train = np.array(X_train)
        X_validate = np.array(X_validate)
        Y_train = np.array(Y_train).reshape(len(Y_train),)
        Y_validate = np.array(Y_validate).reshape(len(Y_validate),)
        
        num_total_samples = 0
        for num in num_samples.values():
            num_total_samples += num
        
        if((X_train.shape[0] + X_validate.shape[0]) != num_total_samples):
            raise ValueError("Number of training + validation samples does not equal number of total samples.")
        if((Y_train.shape[0] + Y_validate.shape[0]) != num_total_samples):
            raise ValueError("Number of training + validation samples does not equal number of total samples.")
        
        X_trains.append(X_train)
        X_validates.append(X_validate)
        Y_trains.append(Y_train)
        Y_validates.append(Y_validate)
        
        fold += 1
    
    # print(rep_name_verification)
    for k,v in rep_name_verification.items():
        if(v != {'Train_Replicate': (num_folds - 1), 'Validation_Replicate': 1}):
            raise ValueError("Generated training and validation data matrices are invalid.")
    
    if(counts_format == 'Cuffnorm'):
        all_y_validates_snames = filenames_to_replicate_names_cuffnorm(samples_table, all_y_validates_rnames, True)
    else:
        all_y_validates_snames = filenames_to_replicate_names_cuffdiff(groups_info, all_y_validates_rnames, True)
    k = 0
    while k < len(all_y_validates_snames):
        all_y_validates_snames[k] = all_y_validates_snames[k].split('.')[0]
        k += 1
    if(verbose):
       print("The sample names of composed validation matrix: " + str(all_y_validates_snames))
       print("The repnames of composed validation matrix: " + str(all_y_validates_rnames))
        
    # print(cond_rep_name_to_label)
    return X_trains, X_validates, Y_trains, Y_validates, cond_rep_name_to_label

def tune_ML_model_helper(x_trains:list, x_validates:list, y_trains:list, y_validates:list, model:'scikit_model',
                         fold:int, run:int, accuracies_fold:list, y_validate_full:list, y_hat_full:list, 
                         probabilities:list, gen_dec_bounds:bool, decision_boundary_data:dict, rfe_features_data:dict,
                         param_tuple:tuple, rfe:bool, target_size:int, gene_lists:bool, classifier:str, verbose:bool):
    x_train, y_train = shuffle(x_trains[fold-1], y_trains[fold-1])
    x_validate, y_validate = x_validates[fold-1], y_validates[fold-1]
    # RFE Starts
    num_degs = x_train.shape[1]
    if((not rfe) or (num_degs <= target_size)):
        x_train2 = x_train
        x_validate2 = x_validate
    else:
        step_size = (num_degs - target_size) // 5
        rfe = RFE(estimator=model, n_features_to_select=target_size, step=step_size)
        rfe.fit(x_train, y_train)
        # print('Feature Ranking According to RFE: ', rfe.ranking_)
        # Create new x_train, y_train, x_validate, and y_validate using only selected features
        bool_mask = []
        for ele in rfe.ranking_:
            if(ele != 1):
                bool_mask.append(False)
            else:
                bool_mask.append(True)
        bool_mask = np.array(bool_mask)
        if(verbose):
            print('Boolean Mask: ', bool_mask)
            
        best_feature_set = []
        index = 0
        for v in bool_mask:
            if(v):
                best_feature_set.append(index)
            index += 1
            
        try:
            rfe_features_data[param_tuple].append( (run, fold, best_feature_set) )
        except KeyError:
            rfe_features_data[param_tuple] = [(run, fold, best_feature_set)]
        
        x_train2 = x_train[:, bool_mask]
        x_validate2 = x_validate[:, bool_mask]
        # print('X_Train_RFE.shape: ', x_train2.shape)
        # print('X_Validate_RFE.shape: ', x_validate2.shape)
    
    if(rfe):
        if((x_train2.shape[1] != target_size) or (x_validate2.shape[1] != target_size)):
            raise ValueError('The training data is not of appropriate size post RFE pruning.')
        
    # RFE Over
    clf = model.fit(x_train2, y_train)
    y_hat = clf.predict(x_validate2)
    
    accuracy = metrics.accuracy_score(y_validate, y_hat)
    probs = clf.predict_proba(x_validate2)
    
    accuracies_fold.append(accuracy)
    y_validate_full.append(y_validate)
    y_hat_full.append(y_hat)
    probabilities.append(probs)
    
    if(gen_dec_bounds):
        sub_dict = {'model': model, 'x': x_train2[:,0:2], 'y': y_train}
        try:
            decision_boundary_data[param_tuple].append(sub_dict)
        except KeyError:
            decision_boundary_data[param_tuple] = [sub_dict]
            
    # print("Model Feature Coefficients: ", clf.coef_)
    return clf
    

def tune_ML_model(x_trains:list, x_validates:list, y_trains:list, y_validates:list, num_folds:int,
                  num_runs:int, num_samples:dict, hyper_param_names:list, hyper_param_values:list,
                  classifier:str, out_dir:str, search_heuristic:str, wrapper:str, target_size:int,
                  sfs_replace:bool, gen_dec_bounds:bool, gene_lists:bool, verbose:bool):
    
    if(search_heuristic not in ['Grid', 'Random']):
        raise ValueError("The search heuristic must be 'Grid' or 'Random'.")
    if(wrapper not in ['RFE', 'SFS', None]):
        raise ValueError("Wrapper must be None, 'RFE', or 'SFS'.")
    
    max_accuracy_config = 0
    decision_boundary_data = {}
    rfe_features_data = {}
    config_data = {}
    
    to_eval = "itertools.product("
    i = 0
    while i < len(hyper_param_names):
        to_eval += "hyper_param_values[" + str(i) + "]"
        if(i < (len(hyper_param_names) - 1)):
            to_eval += ", "
        i += 1
    to_eval += ")"
    
    hyper_param_v_tuples = [item for item in (eval(to_eval))]
    
    if(search_heuristic == 'Random'):
        # Randomly remove some of the hyperparameter combinations
        num_params = len(hyper_param_v_tuples)
        factor = 20
        sample_size = max(1, num_params // factor)
        hyper_param_v_tuples = random.sample(hyper_param_v_tuples, sample_size)
    
    for param_tuple in hyper_param_v_tuples:
        #print("Current Config: ", param_tuple)
        avg_accuracies_runs = []
        
        stabilities_fold = []
        all_runs_accuracies_fold = []
        
        for run in range(num_runs):
            
            accuracies_fold = []
            fold = 1
            probabilities = []
            y_validate_full = []
            y_hat_full = []
            
            to_eval = "classifier("
            i = 0
            while i < len(hyper_param_names):
                to_eval += hyper_param_names[i] + " = param_tuple[" + str(i) + "]"
                if(i < (len(hyper_param_names)-1)):
                    to_eval += ','
                i += 1
            to_eval += ")"
            model = eval(to_eval)            
            while fold <= num_folds:
                rfe = False
                if(wrapper == 'RFE'):
                    rfe = True
                tune_ML_model_helper(x_trains, x_validates, y_trains, y_validates, model, fold, run,
                                     accuracies_fold, y_validate_full, y_hat_full, probabilities,
                                     gen_dec_bounds, decision_boundary_data, rfe_features_data, param_tuple, rfe,
                                     target_size, gene_lists, classifier, verbose)
                        
                fold += 1
            
            y_validate_full = np.concatenate(y_validate_full, axis = 0)
            y_hat_full = np.concatenate(y_hat_full, axis = 0)
            probs_full = np.concatenate(probabilities, axis = 0)
            
            sub_dict = {'y_valid_full': y_validate_full, 'y_hat_full': y_hat_full, 'probs_full': probs_full,
                        'model': model}
            try:
                config_data[param_tuple].append(sub_dict)
            except KeyError:
                config_data[param_tuple] = [sub_dict]
                
            avg_accuracy_run = metrics.accuracy_score(y_validate_full, y_hat_full)
            avg_accuracies_runs.append(avg_accuracy_run)
            
            all_runs_accuracies_fold.append(accuracies_fold)
            accuracies_fold = np.array(accuracies_fold)
            stability_fold = np.std(accuracies_fold)
            stabilities_fold.append(stability_fold)
            if(verbose):
                print('Config: ', param_tuple)
                print('Y_Hat:      ', y_hat_full)
        
        accuracy_config = np.mean(np.array(avg_accuracies_runs))
        
        if(verbose):
            print("This Config Accuracy: ", accuracy_config)
        
        if(accuracy_config > max_accuracy_config):
            max_accuracy_config = accuracy_config
            max_accuracies_fold = all_runs_accuracies_fold.copy()
            best_config_stabilities_fold = stabilities_fold.copy()
            best_configuration = param_tuple
     
    fold_sizes = []
    for y_validate in y_validates:
        fold_sizes.append(y_validate.shape[0])
            
    best_config_mean_stability_fold = np.mean(np.array(best_config_stabilities_fold))
    
    heuristic = 'binary'
    multi_class = 'raise'
    aggregated = False
    axis = None
    if(len(num_samples) > 2):
        if(aggregated):
            heuristic = 'weighted'
        else:
            heuristic = None
            axis = 0
        multi_class = 'ovo'
        
    # These metrics need to be computed regardless
    metrics_run_avg = {'Accuracy':[], 'Precision':[], 'Recall':[], 'ROC_AUC':[], 'Balanced_Accuracy':[],
                       'F1':[], 'Confusion_Matrix': []}
    metrics_config = {'Accuracy':[], 'Precision':[], 'Recall':[], 'ROC_AUC':[], 'Balanced_Accuracy':[],
                       'F1':[], 'Confusion_Matrix': []}
    probabilities_run = []
    best_y_hats = []
    
    # These metrics will need to be recomputed only if SFS wrapper is used.
    all_runs_accuracies_fold = []
    stabilities_fold = []
        
    if(len(config_data[best_configuration]) != num_runs):
        raise ValueError("The classifier predictions should be recorded for each run once.")
        
    config_dati = config_data[best_configuration]
    for run in range(num_runs):
        #print('Metrics Computation Run: ', run)
        y_validate_full = config_dati[run]['y_valid_full']
        y_hat_full = config_dati[run]['y_hat_full']
        probs_full = config_dati[run]['probs_full']
        
        if (gene_lists and wrapper == 'RFE'):
            fold = 1
            while fold <= num_folds:
                # Write out the RFE selected genes into the GeneLists folder.
                best_feature_set_tuples = rfe_features_data[best_configuration]
                if(len(best_feature_set_tuples) != (num_runs * num_folds)):
                    raise ValueError("Error in tracking best features selected by RFE.")
                best_feature_set = []
                for best_feature_set_tuple in best_feature_set_tuples:
                    if best_feature_set_tuple[0] == run and best_feature_set_tuple[1] == fold:
                        best_feature_set = best_feature_set_tuple[2]
                        break
                if(len(best_feature_set) == 0):
                    raise ValueError("Missing best feature set produced by RFE.")
                total_feature_pool_size = x_trains[0].shape[1]
                temp = out_dir.split('/')
                run_name = temp[-2]
                filename = out_dir + 'GeneLists/gene_list_fold' + str(fold) + '_' + run_name + '.txt'
                with open(filename) as reader:
                    lines = reader.readlines()
                rfe_lines = [lines[i] for i in best_feature_set]
                out_filename = out_dir + 'GeneLists/' + 'RFE_' + str(classifier.__name__) + '/'
                if(not os.path.isdir(out_filename)):
                    os.mkdir(out_filename)
                out_filename += 'gene_list_RFE_fold' + str(fold) + '_'
                out_filename += 'fpool_' + str(total_feature_pool_size) + '_tsize_' + str(target_size)
                out_filename +=  '_run_' + str(run) + '_' + run_name + '.txt'
                f = open(out_filename, "w")
                for rfe_line in rfe_lines:
                    f.write(rfe_line)
                f.close()
                
                fold += 1
        
        if(verbose and wrapper == 'SFS'):
            print('Y_Hat_Full Before SFS: ', y_hat_full)
            print('Number of features before SFS: ', x_trains[0].shape[1])
            print('Accuracy before SFS: ', metrics.accuracy_score(y_validate_full, y_hat_full))
        
        if(wrapper == 'SFS'):
            y_validate_full = []
            y_hat_full = []
            probs_full = []
            accuracies_fold = []
            
            # Perform Forward SFS Using Best Configuration of Model
            model = config_dati[run]['model']
            total_feature_pool_size = x_trains[0].shape[1]
            fold = 1
            while fold <= num_folds:
                print('SFS Computation Fold: ', fold)
                x_train, y_train_sfs = shuffle(x_trains[fold-1], y_trains[fold-1])
                x_validate, y_validate_sfs = x_validates[fold-1], y_validates[fold-1]
                best_feature_set = []
                current_feature_set = []
                current_feature_index = -1
                while len(current_feature_set) < target_size:
                    max_accuracy_sfs = 0
                    current_feature_set.append(-1)
                    current_feature_index += 1
                    print('Current Feature Set: ', current_feature_set, '; Current Feature Index: ', current_feature_index)
                    while current_feature_set[current_feature_index] < (total_feature_pool_size-1):
                        current_feature_set[current_feature_index] += 1
                        print('Current Feature Set: ', current_feature_set)
                        if(not sfs_replace):
                            if (current_feature_set[current_feature_index] in best_feature_set):
                                #If this feature has already been used, move onto the next feature
                                print('Feature ', current_feature_set[current_feature_index], ' has already been used.')
                                print('Best Feature Set For Reference: ', best_feature_set)
                                continue
                        x_train_sfs, x_validate_sfs = x_train[:,current_feature_set], x_validate[:,current_feature_set]
                        clf = model.fit(x_train_sfs, y_train_sfs)
                        y_hat_sfs = clf.predict(x_validate_sfs)
                        accuracy_sfs = metrics.accuracy_score(y_validate_sfs, y_hat_sfs)
                        print('x_train_sfs shape: ', x_train_sfs.shape, '; x_validate_sfs shape: ', x_validate_sfs.shape)
                        print('accuracy_sfs: ', accuracy_sfs)
                        if(accuracy_sfs > max_accuracy_sfs):
                            max_accuracy_sfs = accuracy_sfs
                            best_feature_set = deepcopy(current_feature_set)
                        print('max_accuracy_sfs: ', max_accuracy_sfs)
                        print('best_feature_set: ', best_feature_set)
                    current_feature_set = deepcopy(best_feature_set)
                # Now that we have selecteed the best feature set for this fold, compute the y_hat & probs
                if(verbose):
                    print("Best SFS features for fold ", fold, " are: ", best_feature_set)
                    
                if(gene_lists):
                    # Write out the SFS selected genes into the GeneLists folder.
                    temp = out_dir.split('/')
                    run_name = temp[-2]
                    filename = out_dir + 'GeneLists/gene_list_fold' + str(fold) + '_' + run_name + '.txt'
                    with open(filename) as reader:
                        lines = reader.readlines()
                    sfs_lines = [lines[i] for i in best_feature_set]
                    out_filename = out_dir + 'GeneLists/' + 'SFS_' + str(classifier.__name__) + '/'
                    if(not os.path.isdir(out_filename)):
                        os.mkdir(out_filename)
                    out_filename += 'gene_list_SFS_fold' + str(fold) + '_'
                    out_filename += 'fpool_' + str(total_feature_pool_size) + '_tsize_' + str(target_size)
                    out_filename +=  '_run_' + str(run) + '_' + run_name + '.txt'
                    f = open(out_filename, "w")
                    for sfs_line in sfs_lines:
                        f.write(sfs_line)
                    f.close()
                    
                x_train_sfs, x_validate_sfs = x_train[:,best_feature_set], x_validate[:,best_feature_set]
                clf = model.fit(x_train_sfs, y_train_sfs)
                y_hat_sfs = clf.predict(x_validate_sfs)
                print("Within fold y_hat_sfs: ", y_hat_sfs)
                probs_sfs = clf.predict_proba(x_validate_sfs)
                y_validate_full.append(y_validate_sfs)
                y_hat_full.append(y_hat_sfs)
                probs_full.append(probs_sfs)
                
                accuracy_fold = metrics.accuracy_score(y_validate_sfs, y_hat_sfs)
                accuracies_fold.append(accuracy_fold)
                
                fold += 1
            
            y_validate_full = np.concatenate(y_validate_full, axis = 0)
            y_hat_full = np.concatenate(y_hat_full, axis = 0)
            probs_full = np.concatenate(probabilities, axis = 0)
            all_runs_accuracies_fold.append(accuracies_fold)
            accuracies_fold = np.array(accuracies_fold)
            stability_fold = np.std(accuracies_fold)
            stabilities_fold.append(stability_fold)
        
        if(verbose and wrapper == 'SFS'):
            print('Y_Hat_Full After SFS: ', y_hat_full)
            print('Number of Features After SFS: ', x_train_sfs.shape[1])
            print('Accuracy After SFS: ', metrics.accuracy_score(y_validate_full, y_hat_full))
        
        probabilities_run.append(probs_full)
        best_y_hats.append(y_hat_full)
        
        avg_accuracy_run = metrics.accuracy_score(y_validate_full, y_hat_full)
        metrics_run_avg['Accuracy'].append(avg_accuracy_run)
        avg_precision_run = metrics.precision_score(y_validate_full, y_hat_full, average = heuristic)
        metrics_run_avg['Precision'].append(avg_precision_run)
        avg_recall_run = metrics.recall_score(y_validate_full, y_hat_full, average = heuristic)
        metrics_run_avg['Recall'].append(avg_recall_run)
        if(len(num_samples) > 2):
            avg_ROC_AUC_run = metrics.roc_auc_score(y_validate_full, probs_full, multi_class = multi_class)
            metrics_run_avg['ROC_AUC'].append(avg_ROC_AUC_run)
        avg_balanced_accuracy_run = metrics.balanced_accuracy_score(y_validate_full, y_hat_full)
        metrics_run_avg['Balanced_Accuracy'].append(avg_balanced_accuracy_run)
        avg_F1_run = metrics.f1_score(y_validate_full, y_hat_full, average = heuristic)
        metrics_run_avg['F1'].append(avg_F1_run)
        avg_confusion_matrix_run = metrics.confusion_matrix(y_validate_full, y_hat_full)
        metrics_run_avg['Confusion_Matrix'].append(avg_confusion_matrix_run)
    
    metrics_config['Accuracy'] = np.mean(np.array(metrics_run_avg['Accuracy']), axis = axis)
    metrics_config['Precision'] = np.mean(np.array(metrics_run_avg['Precision']), axis = axis)
    metrics_config['Recall'] = np.mean(np.array(metrics_run_avg['Recall']), axis = axis)
    if(len(num_samples) > 2):
        metrics_config['ROC_AUC'] = np.mean(np.array(metrics_run_avg['ROC_AUC']), axis = axis)
    metrics_config['Balanced_Accuracy'] = np.mean(np.array(metrics_run_avg['Balanced_Accuracy']))
    metrics_config['F1']= np.mean(np.array(metrics_run_avg['F1']), axis = axis)
    metrics_config['Confusion_Matrix'] = np.zeros((metrics_run_avg['Confusion_Matrix'][0].shape[0], metrics_run_avg['Confusion_Matrix'][0].shape[0]))
    for confusion_matrix in metrics_run_avg['Confusion_Matrix']:
        metrics_config['Confusion_Matrix'] = np.add(metrics_config['Confusion_Matrix'], confusion_matrix)
    metrics_config['Confusion_Matrix'] = np.divide(metrics_config['Confusion_Matrix'], num_runs)
    
    stability_run = np.std(np.array(metrics_run_avg['Accuracy']))
    if(wrapper == 'SFS'):
        # Recompute fold stabilities
        best_config_mean_stability_fold = np.mean(np.array(stabilities_fold))
        max_accuracies_fold = all_runs_accuracies_fold
    
    avg_probabilities_config = np.zeros((probabilities_run[0].shape[0], probabilities_run[0].shape[1]))
    for prob_matrix in probabilities_run:
        avg_probabilities_config = np.add(avg_probabilities_config, prob_matrix)
    best_probabilities = np.divide(avg_probabilities_config, num_runs)
    
    if(verbose):
        print('Classifier: ' + str(classifier.__name__) + '; Accuracy: ' + str(round(metrics_config['Accuracy'], 4)))
        print('Confusion_Matrix: ' + str(metrics_config['Confusion_Matrix']))
        print('Precision: ' + str(metrics_config['Precision']))
        print('Recall: ' + str(metrics_config['Recall']))
        print()
        print("Fold Scores: " + str(max_accuracies_fold))
        print("Fold Sizes: " + str(fold_sizes))
        print()
        print("Y True (Validate): ", end=' ')
        print(*y_validate_full)
        r = 0
        print('Num Runs: ', num_runs)
        for y_hat_run in best_y_hats:
            print("Run: ", r)
            print('Y Hat (Predicted): ', end=' ')
            print(*y_hat_run)
            r += 1
        print("Best Configuration: ", best_configuration)
    
    if(gen_dec_bounds and (wrapper == None) ):
        # This only works correctly when not using a wrapper.
        temp = out_dir.split('/')
        run_name = temp[-2]
        
        fig = plt.figure(figsize=(10, 8))
        
        num_features = x_trains[0].shape[1]
        i = random.randint(0, num_folds*num_runs - 1)
        fold_index = 1 + i % num_folds
        run_index = 1 + i // num_folds
         
        # Draw the decision boundary of the classifier
        x = decision_boundary_data[best_configuration][i]['x']
        y = decision_boundary_data[best_configuration][i]['y']
        model = decision_boundary_data[best_configuration][i]['model']
        
        # Read in top 2 genes from the respective gene list file.
        gene_list_dir = out_dir + 'GeneLists/'
        gene_list_fname = gene_list_dir + 'gene_list_fold' + str(fold-1) + '_' + run_name + '.txt'
        with open(gene_list_fname) as reader:
            lines = reader.readlines()
            top_genes = [lines[0], lines[1]]
        
        clf = model.fit(x, y)
        
        # Plotting decision regions
        x_min, x_max = x[:, 0].min() - 1, x[:, 0].max() + 1
        y_min, y_max = x[:, 1].min() - 1, x[:, 1].max() + 1
        xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1),
                              np.arange(y_min, y_max, 0.1))
        
        Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
        Z = Z.reshape(xx.shape)
    
        plt.contourf(xx, yy, Z, alpha=0.4)
        plt.scatter(x[:, 0], x[:, 1], c=y, s=40, edgecolor='k')
        plt.title(classifier.__name__)
        plt.xlabel(top_genes[0])
        plt.ylabel(top_genes[1])
        
        out_dir = out_dir + 'DecBounds/'
        if(not os.path.isdir(out_dir)):
            os.mkdir(out_dir)
        out_fname = out_dir + 'bound_' + str(classifier.__name__) + '_' + str(num_features) + '_features'
        out_fname += '_fold_' + str(fold_index) + '_run_' + str(run_index) + '_' + run_name + '.png'
        fig.savefig(out_fname)
    
    perf_metrics = {'Accuracy': metrics_config['Accuracy'], 'Run_Stability':round(stability_run, 4),
                   'Fold_Mean_Stability':round(best_config_mean_stability_fold, 4), 
                   'Precision':metrics_config['Precision'], 'Recall':metrics_config['Recall'],
                   'ROC_AUC':metrics_config['ROC_AUC'], 'Balanced_Accuracy':metrics_config['Balanced_Accuracy'],
                   'F1':metrics_config['F1'], 'Confusion_Matrix': metrics_config['Confusion_Matrix']}
    
    return perf_metrics, y_validate_full, best_probabilities
        
def train_validate_ML_models(num_samples:dict, class_labels:list, pipeline:str, normalization:str,
                             dispersion:str, root_dir:str, features_file:str, out_dir:str, num_features:list,
                             classifiers:list, num_folds:int, num_runs:int, tissue:str, perf_metrics:list,
                             taboo_list:list, wrapper:str, target_size:int, sfs_replace:bool, search_heuristic:str,
                             gene_lists:bool, conf_matrices:bool, dec_boundaries:bool, ROC_plot:bool,
                             v_transform:bool, counts_format:str, verbose:bool):
    '''Examines performance of multiple machine learning models as the number of features is varied.
    Generates console output and also creates csv files containing the accuracies of different classifiers. '''
    fs_type = None
    if('IG' in features_file):
        fs_type = 'IG'
    elif('RF' in features_file):
        fs_type = 'RF'
    elif('DE' in features_file):
        fs_type = 'DE'
    else:
        raise ValueError("The features file must have IG, RF, or DE in its name.")
    
    # Performance file stores verbose output that is useful for debugging and validation.
    temp = out_dir.split('/')
    run_name = temp[-2]
    perf_filename = 'Performance_' + run_name + '.txt'
    if(not os.path.isdir(out_dir)):
        os.mkdir(out_dir)
    f = open(out_dir + perf_filename, 'w')
    original_stdout = sys.stdout
    sys.stdout = f # Change the standard output to the file we created.
    
    accuracy_table = []
    rows_to_write = {'Accuracy':[], 'Run_Stability':[], 'Fold_Mean_Stability':[], 'Precision':[],
                       'Recall':[], 'ROC_AUC':[], 'Balanced_Accuracy':[], 'F1':[]}
    metrics_to_write = {'Accuracy':{}, 'Run_Stability':{}, 'Fold_Mean_Stability':{}, 'Precision':{},
                       'Recall':{}, 'ROC_AUC':{}, 'Balanced_Accuracy':{}, 'F1':{}}
    fieldnames = ['Features']
    for classifier in classifiers:
        fieldnames.append(classifier)
    accuracy_table.append(fieldnames)
        
    fpkm_table = root_dir + '/' + pipeline + '/Cuffnorm_' + normalization + '/genes.fpkm_table'
    read_groups = root_dir + '/' + pipeline + '/Cuffdiff_' + normalization + '_' + dispersion + '/genes.read_group_tracking'
    
    #Translate condtion names (ex: AH) to replicate condition names (ex: q1, q2, etc.)
    conditions = num_samples.keys()
    if(counts_format == 'Cuffnorm'):
        temp_dir = root_dir + '/' + pipeline + '/Cuffnorm_' + normalization + '/'
        cond_to_rep_cond_map = generate_cond_name_to_rep_name_map(temp_dir)
    else:
        temp_dir = root_dir + '/' + pipeline + '/Cuffdiff_' + normalization + '_' + dispersion + '/'
        cond_to_rep_cond_map = generate_cond_name_to_rep_name_map(temp_dir, 1)
    rep_cond_names = []
    for cond in conditions:
        if(cond_to_rep_cond_map[cond] not in rep_cond_names):
            rep_cond_names.append(cond_to_rep_cond_map[cond])
    
    if(counts_format == 'Cuffnorm'):
        counts = read_cuffnorm_counts2(fpkm_table, rep_cond_names, v_transform)
        gene_names = read_cuffnorm_gene_names(fpkm_table)
    else:
        counts = read_cuffdiff_counts2(read_groups, rep_cond_names, v_transform)
        gene_names = read_cuffdiff_gene_names(read_groups)
    
    num_total_samples = 0
    for value in num_samples.values():
        num_total_samples += value
    
    max_features = max(num_features)
    X_trains,X_validates,Y_trains,Y_validates,label_map = generate_train_validate_data(max_features, num_folds, num_samples,
                                                                                       root_dir, pipeline, normalization, 
                                                                                       dispersion, counts, gene_names,
                                                                                       features_file, taboo_list, tissue, out_dir,
                                                                                       gene_lists, counts_format, verbose)
    classifier_name_map = {'log reg': LogisticRegression, 'kNN':KNeighborsClassifier, 'GNB':GaussianNB,
                           'SVM':SVC, 'NN':MLPClassifier, 'RF':RandomForestClassifier}
    for num_DEG in num_features:
        for perf_metric in perf_metrics:
            metrics_to_write[perf_metric]['Features'] = num_DEG
        
        sub_X_trains = []
        sub_X_validates = []
        for X_train in X_trains:
            sub_X_train = X_train[:,0:num_DEG]
            sub_X_trains.append(sub_X_train)
        for X_validate in X_validates:
            sub_X_validate = X_validate[:,0:num_DEG]
            sub_X_validates.append(sub_X_validate)
        
        if(verbose):
            print("Num Features: " + str(num_DEG))
        best_y_trues_list = []
        best_probabilities_list = []
        for classifier in classifiers:
            if(classifier == "log reg"):
                
                grid_params = ["C", "class_weight", "solver"]
                grid_options = [[0.5, 1.0, 2.0, 3.0, 4.0, 5.0], [None, 'balanced'],
                                        ['newton-cg', 'lbfgs', 'liblinear', 'saga']]
                
                random_params = ["C", "class_weight", "solver", 'tol', 'max_iter']
                random_options = [[0.1, 0.5, 1.0, 2.0, 3.0, 4.0, 5.0, 10.0], [None, 'balanced'],
                                          ['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga'], [0.0001, 0.00001, 0.001],
                                          [75, 100, 150]]
                
            elif(classifier == "kNN"):
                
                grid_params = ["n_neighbors", "weights", "p"]
                grid_options = [[3,5,7,9,15], ['uniform', 'distance'], [1, 1.5, 2.0]]
                
                random_params = ["n_neighbors", "weights", 'algorithm', "p"]
                random_options = [[3,5,7,9,15,21], ['uniform', 'distance'], ['ball_tree', 'kd_tree', 'brute'],
                                      [1, 1.25, 1.5, 1.75, 2.0]]
                
            elif(classifier == "GNB"):
                grid_params, grid_options = [], []
                random_params, random_options = [], []
            elif(classifier == "SVM"):
                
                grid_params = ["kernel", "C", "class_weight", "probability", "degree", "gamma"]
                grid_options = [['linear', 'poly', 'rbf'] ,[0.5, 1.0, 2.0, 3.0, 4.0, 5.0],
                                    [None, 'balanced'], [True], [1, 3, 5], ['scale', 'auto']]
                
                random_params = ["kernel", "C", "class_weight", "cache_size", "probability", "degree", "gamma", 'tol']
                random_options = [['linear', 'poly', 'rbf', 'sigmoid'] ,[0.5, 1.0, 2.0, 3.0, 4.0, 5.0],
                                     [None, 'balanced'], [1000], [True], [1, 2, 3, 4, 5], ['scale', 'auto'],
                                     [0.001, 0.01, 0.0001]]
            elif(classifier == "NN"):
                
                grid_params = ["hidden_layer_sizes", "activation", "solver", "alpha"]
                grid_options = [[(math.ceil(num_total_samples / (2*num_DEG)), ), (10,), (25,), (50,), (100,)],
                                   ['logistic', 'tanh', 'relu'], ['lbfgs'],[0.00001, 0.0001, 0.001, 0.01, 0.1]]
                
                random_params = ["hidden_layer_sizes", "activation", "solver", "alpha", 'learning_rate']
                random_options = [[(math.ceil(num_total_samples / (2*num_DEG)), ), (10,), (25,), (50,), (100,), 
                                    (150,), (200,)], ['logistic', 'tanh', 'relu'], ['lbfgs', 'sgd', 'adam'],
                                    [0.00001, 0.0001, 0.001, 0.01, 0.1],['constant', 'invscaling', 'adaptive']]      
            elif(classifier == "RF"):
                grid_params = ["n_estimators", "criterion", "class_weight"]
                grid_options = [[50, 500], ['gini', 'entropy'], [None, 'balanced', 'balanced_subsample']]
                
                random_params = ["n_estimators", "criterion", "class_weight", "min_samples_split",
                                    'min_samples_leaf', 'max_features', 'min_impurity_decrease']
                random_options = [[50, 500], ['gini', 'entropy'], [None, 'balanced', 'balanced_subsample'],
                                     [2,3,4,5], [1,2], ['auto', 'log2', None], [0, 0.05, 0.1]]
            else:
                raise ValueError("An invalid classifier argument.")
                
            if(search_heuristic == 'Random'):
                params = random_params
                options = random_options
            elif(search_heuristic == 'Grid'):
                params = grid_params
                options = grid_options
            
            scores, y_trues, probs  = tune_ML_model(sub_X_trains, sub_X_validates, Y_trains,
                                                    Y_validates, num_folds, num_runs, num_samples,
                                                    params, options, classifier_name_map[classifier], out_dir, 
                                                    search_heuristic, wrapper, target_size, sfs_replace,
                                                    dec_boundaries, gene_lists, verbose)    
            
            best_y_trues_list.append(y_trues)
            best_probabilities_list.append(probs)
            
            for perf_metric in perf_metrics:
                if(isinstance(scores[perf_metric], float)):
                    if(perf_metric in ['Fold_Mean_Stability', 'Run_Stability']):
                        metrics_to_write[perf_metric][classifier] = round(scores[perf_metric], 4)
                    elif(perf_metric in ['Accuracy', 'Precision', 'Recall', 'Balanced_Accuracy', 'F1', 'ROC_AUC']):
                        # If this metric makes sense to be reported as % then report it as such.
                        metrics_to_write[perf_metric][classifier] = str(round(scores[perf_metric]*100, 2)) + '%'
                    else:
                        raise ValueError('Unknown performance metric.')
                else:
                    # This metric is a numpy array, a per-class metric (precision, recall, and f1 for multiclass)
                    # We will format it later.
                    metrics_to_write[perf_metric][classifier] = scores[perf_metric]
            # print("metrics_to_write: ", metrics_to_write)
            
            if(conf_matrices):
                mtx_dir = out_dir + 'ConfMatrices/'
                if(not os.path.isdir(mtx_dir)):
                    os.mkdir(mtx_dir)
                plot_confusion_matrix(scores['Confusion_Matrix'], class_labels, classifier, num_DEG,
                                      mtx_dir, fs_type, scores)
        
        for perf_metric in perf_metrics:
            row = {}
            i = 0
            for fieldname in fieldnames: 
                row[fieldname] = metrics_to_write[perf_metric][fieldname]
                i += 1
            # print('metrics_to_write: ', metrics_to_write)
            # print('row: ', row)
            rows_to_write[perf_metric].append(row)
            # print("rows_to_write (in loop): ", rows_to_write)
        # print("rows_to_write: ", rows_to_write)
        
        # Clear metrics_to_write
        metrics_to_write = {'Accuracy':{}, 'Run_Stability':{}, 'Fold_Mean_Stability':{}, 'Precision':{},
                            'Recall':{}, 'ROC_AUC':{}, 'Balanced_Accuracy':{}, 'F1':{}}
        
        # ROC Curves Work for Binary Case Only
        if(ROC_plot and len(num_samples) == 2):
            roc_dir = out_dir + '/ROCCurves/'
            if(not os.path.isdir(roc_dir)):
                os.mkdir(roc_dir)
            
            out_file = (roc_dir + 'ROC_curve_' + str(num_DEG) + '_features_' + run_name + '.png')
            generate_mean_ROC_curves(best_y_trues_list, best_probabilities_list, classifiers, num_DEG, num_folds, out_file)
        if(verbose):
            print('\n')
    
    mtrcs_dir = out_dir + '/Metrics/'
    if(not os.path.isdir(mtrcs_dir)):
        os.mkdir(mtrcs_dir)
    for perf_metric in perf_metrics:
        with open(mtrcs_dir + perf_metric + '_' + run_name + '.csv', 'w', newline='') as perf_curves_file:
            writer = csv.DictWriter(perf_curves_file, fieldnames=fieldnames)
            writer.writeheader()
            for row in rows_to_write[perf_metric]:
                #Row is a dictionary that maps column headers to column values.
                #If any given value in a column is a numpy array it is because it is a per class metric (precision, recall, F1)
                #In that case we want to decorate them first.
                for k,v in row.items():
                    if(isinstance(v, np.ndarray)):
                        text = ''
                        i = 0
                        for val in v:
                            text += class_labels[i] + ':' + str(round(val*100, 2)) + '% '
                            i += 1
                        row[k] = text
                writer.writerow(row)
    
    sys.stdout.close()
    sys.stdout = original_stdout # Reset the standard output to its original value
    
    print("Restored sys.stdout!")
    if(verbose):
        identify_misclassified_samples(out_dir + perf_filename, out_dir)
    
    return rows_to_write
            
def generate_mean_ROC_curves(best_y_trues_list:list, best_probabilities_list:list, classifiers:list, num_DEG:int, num_folds:int, out_file:str):
    plt.figure()
    
    index = 0
    colors = ['b', 'g', 'c', 'm', 'y', 'k']
    if(len(classifiers) > 6):
        raise ValueError("Can only plot ROCs for at most 6 classifiers at a time.")
    colors = colors[0:len(classifiers)]
    for classifier in classifiers:
        mean_fpr = np.linspace(0, 1, 100)
        
        best_y_trues = best_y_trues_list[index]
        best_probabilities = best_probabilities_list[index]
        
        fpr, tpr, thresholds = roc_curve(best_y_trues, best_probabilities[:,1])
        #plt.plot(fpr, tpr, lw=1, alpha=0.3, label='ROC fold %d (AUC = %0.2f)' % (fold, roc_auc))

        mean_tpr = interp(mean_fpr, fpr, tpr)
        mean_tpr[0] = 0.0
        mean_tpr[-1] = 1.0
        mean_auc = auc(mean_fpr, mean_tpr)
        plt.plot(mean_fpr, mean_tpr, color=colors[index],
                 label=r'Mean ROC ' + classifier + ' (AUC = %0.2f)' % (mean_auc),
                 lw=2, alpha=.8)
        index += 1
    
    plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r', label='Chance', alpha=.8)
    plt.xlabel('False positive rate')
    plt.ylabel('True positive rate')
    plt.title('ROC curve ' + str(num_DEG) + " features")
    plt.legend(loc='best')
    plt.savefig(out_file)
    # plt.show()
    
def plot_per_condition_counts_heatmap(counts:dict, top_genes:list, conditions:list, fold:int, out_dir:str,
                                      num_genes:int = 30):
    # Draw a per-replicate heatmap
    temp = out_dir.split('/')
    run_name = temp[-3]
    
    array = []
    for gene_name,means in counts.items():
        array.append(means)
    array = np.array(array)
    
    # We limit number of top genes to num_genes for the heatmap.
    if(array.shape[0] > num_genes):
        array = array[0:num_genes, :]
        top_genes = top_genes[0:num_genes]
        
    df_cm = pandas.DataFrame(array, index = top_genes,
                             columns = conditions)
    
    width = len(conditions) * 3
    height = len(top_genes) // 3
    plt.figure(figsize=(width,height))
    # 'BrBG'
    seaborn.heatmap(df_cm, cmap = 'Blues')
    plt.xlabel('Conditions', fontsize = 14)
    plt.ylabel('Genes', fontsize = 14)
    if(fold > 0):
        plt.title('FOLD' + str(fold), fontsize = 16)
    else:
        plt.title('Total Cuffnorm Counts', fontsize = 16)
    out_file = out_dir + 'FOLD' + str(fold) + '_' + run_name + '.png'
    plt.savefig(out_file, bbox_inches="tight")
    # plt.show()
    
def plot_per_sample_counts_heatmap(counts:dict, top_genes:list, samples_file:str, fold:int, out_dir:str,
                                   counts_format:str, num_genes:int = 30):
    # Draw a per-replicate heatmap
    temp = out_dir.split('/')
    run_name = temp[-3]
    
    array = []
    rep_names = []
    for rep_name,values in counts.items():
        rep_names.append(rep_name)
        array.append(values)
    if(counts_format == 'Cuffnorm'):
        filenames = filenames_to_replicate_names_cuffnorm(samples_file, rep_names, True)
    else:
        filenames = filenames_to_replicate_names_cuffdiff(samples_file, rep_names, True)
    sample_names = []
    for filename in filenames:
        fname = filename.split('.')
        sample_names.append(fname[0])
    array = np.array(array)
    
    # We limit number of top genes to num_genes for the heatmap.
    if(array.shape[1] > num_genes):
        array = array[:, 0:num_genes]
        top_genes = top_genes[0:num_genes]
        
    df_cm = pandas.DataFrame(array, index = sample_names,
                             columns = top_genes)
    
    width = len(top_genes) // 3
    height = len(rep_names) // 3
    plt.figure(figsize=(width,height))
    # 'BrBG'
    seaborn.heatmap(df_cm, cmap = 'Blues')
    plt.xlabel('Genes', fontsize = 14)
    plt.ylabel('Replicates', fontsize = 14)
    if(fold > 0):
        plt.title('FOLD' + str(fold), fontsize = 16)
    else:
        plt.title('Total Counts', fontsize = 16)
    out_file = out_dir + 'FOLD' + str(fold) + '_' + run_name + '.png'
    plt.savefig(out_file, bbox_inches="tight")
    # plt.show()
    
def plot_confusion_matrix(array: 'numpy_array', cm_labels:list, classifier:str, num_DEG:int, out_dir:str, fs_type:str, 
                          perf_metrics:dict):
    temp = out_dir.split('/')
    run_name = temp[-3]
    
    accuracy = perf_metrics['Accuracy']
    # run_stability = perf_metrics['Run_Stability']
    # fold_mean_stability = perf_metrics['Fold_Mean_Stability']
    precision = perf_metrics['Precision']
    recall = perf_metrics['Recall']
    ROC_AUC = perf_metrics['ROC_AUC']
    balanced_accuracy = perf_metrics['Balanced_Accuracy']
    f1 = perf_metrics['F1']
    
    acc_text = 'Total Accuracy: ' + str(int(accuracy*100)) + '%'
    balanced_acc_text = 'Balanced Accuracy: ' + str(int(balanced_accuracy*100)) + '%'
    
    #If this is a binary confusion matrix report total precision, recall, balanced accuracy, and F1 score.
    #If this is a multiclass confusion matrix report per class precision, recall, balanced accuracy, and F1 score.
    
    if(len(cm_labels) == 2):
        precision_text = 'Precision: ' + str(int(precision*100)) + '%'
        recall_text = 'Recall: ' + str(int(recall*100)) + '%'
        f1_text = 'F1: ' + str(int(f1*100)) + '%'
    elif(len(cm_labels) > 2):
        roc_auc_text = 'ROC_AUC: ' + str(int(ROC_AUC))
        precision_text = 'Precision: '
        recall_text = 'Recall: '
        f1_text = 'F1: '
        i = 0
        while i < len(cm_labels):
            precision_text += cm_labels[i] + ': ' +  str(int(precision[i]*100)) + '% '
            recall_text += cm_labels[i] + ': ' +  str(int(recall[i]*100)) + '% '
            f1_text += cm_labels[i] + ': ' +  str(int(f1[i]*100)) + '% '
            i += 1
            if(i % 3 == 0):
                precision_text += '\n    '
                recall_text += '\n    '
                f1_text += '\n    '
    else:
        raise ValueError("The number of labels must be >= 2.")
    
    df_cm = pandas.DataFrame(array, index = cm_labels,
                             columns = cm_labels)
    annotation = array.tolist()
    # Add per class accuracy on diagonal entries.
    i = 0
    while i < array.shape[0]:
        j = 0
        while j < array.shape[1]:
            annotation[i][j] = str(int(array[i][j])) if array[i][j].is_integer() else str(array[i][j])
            if(i == j):
                annotation[i][j] += '\n' + str( int((array[i][j]/np.sum(array[i,:]))*100) ) + '%'
            j += 1
        i += 1
    
    plt.figure()
    
    seaborn.heatmap(df_cm, cmap = 'Blues', annot=annotation, fmt='', annot_kws={"fontsize":12})
    plt.xlabel('Predicted Classes', fontsize = 14)
    plt.ylabel('Actual Classes', fontsize = 14)
    plt.title(classifier + ' ' + fs_type + ' ' + str(num_DEG) + ' ' + run_name, fontsize = 16)
    
    if(len(cm_labels) > 2):
        plt.figtext(0.1, -0.05, acc_text, horizontalalignment='left', fontsize = 12) 
        # plt.figtext(0.1, -0.1, balanced_acc_text, horizontalalignment='left', fontsize = 12) 
        plt.figtext(0.1, -0.2, precision_text, horizontalalignment='left', fontsize = 12) 
        plt.figtext(0.1, -0.3, recall_text, horizontalalignment='left', fontsize = 12)
        # plt.figtext(0.1, -0.4, f1_text, horizontalalignment='left', fontsize = 12) 
        # plt.figtext(0.1, -0.45, roc_auc_text, horizontalalignment='left', fontsize = 12) 
    elif(len(cm_labels) == 2):
        plt.figtext(0.1, -0.05, acc_text, horizontalalignment='left', fontsize = 12) 
        plt.figtext(0.1, -0.1, precision_text, horizontalalignment='left', fontsize = 12) 
        plt.figtext(0.1, -0.15, recall_text, horizontalalignment='left', fontsize = 12)
    
    out_file = out_dir + classifier + '_' + fs_type + '_' + str(num_DEG) + '_' + run_name + '.png'
    plt.savefig(out_file, bbox_inches="tight")
    # plt.show()
    
def identify_misclassified_samples(filename:str, out_dir:str):
    '''Parse through the performance log. Identify which sample IDs were classified correctly 
    and which were misclassified.'''
    
    # Store feature size 'Num Features'
    # For every feature size store 'The sample names of composed validation matrix'
    # Store classifier 'Classifier'
    # For every feature size for every classifier store 'Y True (Validate)' and 'Y Hat (Predicted)'
    # For each classifier identify misclassified indeces.
    # Report which sample ids were classified correctly vs incorrectly.
    
    temp = out_dir.split('/')
    run_name = temp[-2]
    
    num_degs_lines = []
    num_runs_lines = []
    sample_id_lines = []
    classifier_lines = []
    y_true_valid_lines = []
    y_hat_valid_lines = []
    
    with open(filename) as reader:
        lines = reader.readlines()
        for line in lines:
            if 'Num Features:' in line:
                num_degs_lines.append(line)
            elif 'Num Runs:' in line:
                num_runs_lines.append(line)
            elif 'The sample names of composed validation matrix:' in line:
                sample_id_lines.append(line)
            elif 'Classifier:' in line:
                classifier_lines.append(line)
            elif 'Y True (Validate):' in line:
                y_true_valid_lines.append(line)
            elif 'Y Hat (Predicted):' in line:
                y_hat_valid_lines.append(line)
    
    if(len(classifier_lines) != len(y_true_valid_lines)):
        raise ValueError("Issue with the performance log (B).")
        
    num_degs = []
    for line in num_degs_lines:
        num_degs.append(int(line[14:]))
        
    line = num_runs_lines[0]
    num_runs = (int(line[11:]))
        
    valid_sample_ids = []
    sample_id_line = sample_id_lines[0]
    index1 = sample_id_line.find('[')
    index2 = sample_id_line.find(']')
    sample_id_line = sample_id_line[index1+1:index2]
    sample_id_line = sample_id_line.replace("'", "")
    sample_id_line = sample_id_line.replace(" ", "")
    valid_sample_ids = sample_id_line.split(',')
    num_samples = len(valid_sample_ids)
    
    classifiers = []
    for line in classifier_lines:
        semicolon_index = line.find(';')
        classifiers.append(line[12:semicolon_index])
        
    y_true_valids = []
    for line in y_true_valid_lines:
        temp = line[20:].replace('\n', '')
        values = temp.split(' ')
        y_true_valids.append(values)
    
    y_hat_valids = []
    for line in y_hat_valid_lines:
        temp = line[20:].replace('\n', '')
        values = temp.split(' ')
        y_hat_valids.append(values)
    
    for row in y_true_valids:
        if(len(row) != num_samples):
            raise ValueError("Issue with the performance log (D).")
    for row in y_hat_valids:
        if(len(row) != num_samples):
            raise ValueError("Issue with the performance log (E).")
            
    i = 0
    misclassified_indeces = []
    while i < len(y_hat_valids):
        # Go through each y_hat and identify all incorrect predictions.
        # The number of y_hats is number of feature sets * number of classifiers * number of runs.
        y_actual = y_true_valids[0]
        y_predicted = y_hat_valids[i]
        j = 0
        sub_matrix = []
        while j < len(y_actual):
            if(y_actual[j] != y_predicted[j]):
                sub_matrix.append(j)
            j += 1
        i += 1
        misclassified_indeces.append(sub_matrix)
        
    misclassified_samples = []
    for row in misclassified_indeces:
        submatrix = []
        for index in row:
            submatrix.append(valid_sample_ids[index])
        misclassified_samples.append(submatrix)
            
    num_classifiers = int( len(classifiers) / len(num_degs) )
    
    f = open(out_dir + 'miss_samples_' + run_name + '.txt', "w")
    f.write('Number Samples: ' + str(num_samples) + '\n')
    i = 0
    while i < len(num_degs):
        f.write("Number Features: " + str(num_degs[i]) + '\n')
        j = 0
        while j < num_classifiers:
            f.write('Classifier: ' + classifiers[j] + '\n')
            f.write('Number of Runs: ' + str(num_runs) + '\n')
            k = 0
            while k < num_runs:
                index = i*num_classifiers*num_runs + j*num_runs + k
                f.write('Run #: ' + str(k) + '\n')
                f.write('Number of Misclassified Samples: ' + str(len(misclassified_samples[index])) + '\n')
                f.write("Misclassified Samples: " + str(misclassified_samples[index]) + '\n')
                k += 1
            j += 1
        i += 1
    f.close()
    
def process_misclassified_samples(filename:str, cutoff:int)->dict:
    '''Take misclassified samples file as input. Output a dictionary containing the count of how many times each sample
    appeared within the file. Also output names of samples that were misclassified in more than cutoff number of lines.'''
    result = {}
    all_sample_names = set(samples_AH_PB) | set(samples_DA_PB) | set(samples_AA_PB) | set(samples_NF_PB)
    all_sample_names = all_sample_names | set(samples_HP_PB) | set(samples_CT_PB) | set(samples_AH_LV)
    all_sample_names = all_sample_names | set(samples_AC_LV) | set(samples_NF_PB) | set(samples_HP_LV)
    all_sample_names = all_sample_names | set(samples_CT_LV)
    
    with open(filename) as reader:
        lines = reader.readlines()
        for line in lines:
            words = line.split("'")
            for word in words:
                if word in all_sample_names:
                    if word in result:
                        result[word] = result[word] + 1
                    else:
                        result[word] = 1
                  
    sorted_result = {k: v for k, v in sorted(result.items(), key=lambda item: item[1], reverse = True)}
    #print(sorted_result)
    
    highly_misclassified = set()
    for k,v in sorted_result.items():
        if v > cutoff:
            highly_misclassified.add(k)
    
    return sorted_result, highly_misclassified
  
def filter_cuffnorm_counts(filename:str):
    ''' Filter out genes for which majority of the replicates have 0 counts. '''
    counts2 = read_cuffnorm_counts2(filename)
    filtered_counts = {}
    eliminated_genes = []
    for key,value in counts2.items():
        zero_counter = 0
        counter = 0
        for v in value:
            if(v[1] == 0):
                zero_counter += 1
            counter += 1
        if(zero_counter / counter < 0.5):
            filtered_counts[key] = value
        else:
            eliminated_genes.append(key)
    return filtered_counts, eliminated_genes

def detect_outlier_features_by_std(fname:str, num_conditions:int, out_dir:str, counts_format:str, threshold:float = 3.5):
    '''Detects features (genes) whose counts for at least one sample are > than (mean + std * treshold) or 
    < than (mean - std * treshold). '''
    if(counts_format == 'Cuffnorm'):
        counts = read_cuffnorm_counts_mean_variance(fname, True)
        counts2 = read_cuffnorm_counts2(fname, 'ALL', True)
    else:
        counts = read_cuffdiff_counts_mean_variance(fname, True)
        counts2 = read_cuffdiff_counts2(fname, 'ALL', True)
    top_genes = []
    
    for gene in counts2.keys():
        top_genes.append(gene)
            
    top_genes_outlier = {}
    for top_gene in top_genes:
        top_genes_outlier[top_gene] = False
        
    for k,v in counts2.items():
        means_stds = counts[k]
        #print("Feature: ", k)
        #print("Means & STDs: ", str(means_stds))
        for rep_count_tuple in v:
            repname = rep_count_tuple[0]
            count = rep_count_tuple[1]
            cond = repname[0:2]
            mean_index = int(cond[1]) - 1
            std_index = mean_index + num_conditions
            #mean = counts[k][mean_index]
            #std = counts[k][std_index]
            mean = means_stds[mean_index]
            std = means_stds[std_index]
            if(count > (mean + std*threshold) or count < (mean - std*threshold)):
                top_genes_outlier[k] = True
                #print('Outlier Detected.')
                #print('Condition:', cond)
                #print('Count: ', str(count))
                
    taboo_list = []
                
    for k,v in top_genes_outlier.items():
        if(v == True):
            # print(k)
            taboo_list.append(k)
            
    print(len(taboo_list))
            
    filename = out_dir + '/' + 'std_taboo_list.csv'
    with open(filename, 'w') as writer:
        for ele in taboo_list:
            writer.write(str(ele))
            writer.write('\n')
    
    return taboo_list
    
def biological_validation(gene_list:'filename,list,etc.', run_name:str, out_dir:str, data_type:str = 'Pathway'):   
    '''
    data_type must be 'Pathway', 'Tissue', or 'Disease'
    '''
    # gene_list = pandas.read_csv(fname, header=None, sep="\t")
    # glist = gene_list.squeeze().str.strip().tolist()
    # print(glist[:10])
    
    # names = gp.get_library_name() # default: Human
    # print(names)
    
    # The key issue is that we only care about gene sets that map differentially expressed genes 
    # to pathways/diseases/tissues, and etc. We do not care about mapping of gene variants to 
    # pathways/disease/tissues. Therefore, we must choose the sets that focus on expression data.
    
    # Initial List
    # Some issues with this list.
    # 1) To correctly use up and down sets I likely need to separate my gene lists into up and down gene lists.
    # This is problematic because it forces me to use differential expression data wherein I would need to use it 
    # otherwise (Information Gain, Random Forest selection).
    # 2) The virus pertubations data sets are only relevant for hepatitis-C condition.
    # 3) I likely do not need multiple pathway and tissue datasets. 
    
    # gene_sets = ['BioPlanet_2019', 'WikiPathways_2019_Human', 'KEGG_2019_Human', 'Reactome_2016',
    #             'Panther_2016', 'BioCarta_2016', 'GO_Biological_Process_2018', 'Jensen_TISSUES',
    #             'Jensen_COMPARTMENTS', 'ARCHS4 Tissues', 'Human_Gene_Atlas', 'Table_Mining_of_CRISPR_Studies', 
    #             'GTEx_Tissue_Sample_Gene_Expression_Profiles_down',
    #             'GTEx_Tissue_Sample_Gene_Expression_Profiles_up','Disease_Perturbations_from_GEO_up',
    #             'Disease_Perturbations_from_GEO_down', 'RNA-Seq_Disease_Gene_and_Drug_Signatures_from_GEO',
    #             'Virus_Perturbations_from_GEO_down', 'Virus_Perturbations_from_GEO_up']
    
    # gene_sets2 = ['BioPlanet_2019', 'WikiPathways_2019_Human', 'KEGG_2019_Human', 'Reactome_2016',
    #              'Panther_2016', 'BioCarta_2016', 'GO_Biological_Process_2018', 'Jensen_TISSUES',
    #              'Jensen_COMPARTMENTS', 'ARCHS4 Tissues', 'Human_Gene_Atlas', 'Table_Mining_of_CRISPR_Studies']
    
    # gene_sets3 = ['BioPlanet_2019', 'WikiPathways_2019_Human', 'KEGG_2019_Human', 'Reactome_2016',
    #              'Panther_2016', 'BioCarta_2016', 'GO_Biological_Process_2018', 'ARCHS4_Tissues', 'Human_Gene_Atlas']
    
    selected_gene_set = []
    selected_regex = ''
    
    gene_sets_pathways = ['BioPlanet_2019', 'WikiPathways_2019_Human', 'KEGG_2019_Human', 
                          'GO_Biological_Process_2018']
    
    gene_sets_tissues = ['ARCHS4_Tissues', 'Human_Gene_Atlas']
    
    gene_sets_diseases = ['Disease_Perturbations_from_GEO_up','Disease_Perturbations_from_GEO_down']
    
    disease_regex = 'hepa|liver|cirrhosis|NAFLD|liver fibrosis|NASH|steatohepatitis|HCV|'
    disease_regex += 'alcohol|sepsis|septic shock|hypercholesterolemia|'
    disease_regex += 'hyperlipidemia|obesity'
    
    tissue_regex = 'Blood|Macrophage|Erythro|Platelet|Basophil|Neutrophil|Eosinophil|Cytokine|Tumor Necrosis Factor|'
    tissue_regex += 'Monocyte|Lymphocyte|Granulocyte|Dendritic|Megakaryocyte|T Cell|B Cell|NK Cell|Toll-like receptor|'
    tissue_regex += 'Fc receptor|Liver|Hepatocyte|Stellate|Kupffer|Sinusoidal Endothelial Cells|'
    tissue_regex += 'CD34+|Natural Killer Cell|PBMC|Tcell|Bcell|lymphoblast|CD8+|CD19+|CD4+|CD71+|Omentum'
    
    pathway_regex = 'Interferon|Immun|Interleukin|Prolactin|Complement|Chemokine|Oncostatin M|Rejection|Inflamma|' 
    pathway_regex += 'IL-1|IL1|IL-|selenium|osteopontin|circulation|coagulation|clotting|biosynthesis|'
    pathway_regex += 'degradation|cholesterol|lipid|TNF|steroid|metal ion|heme|metallo|CXCR|LDL|'
    pathway_regex += 'Phagocytosis|metabolism|TYROBP|AP-1|' + disease_regex + '|' + tissue_regex
    
    
    if(data_type == 'Pathway'):
        selected_gene_set = gene_sets_pathways
        selected_regex = pathway_regex
    elif(data_type == 'Tissue'):
        selected_gene_set = gene_sets_tissues
        selected_regex = tissue_regex
    elif(data_type == 'Disease'):
        selected_gene_set = gene_sets_diseases
        selected_regex = disease_regex
    else:
        raise ValueError("Invalid data type. Must be Pathway, Tissue, or Disease.")
        
    
    enr = gp.enrichr(gene_list = gene_list,
                     gene_sets = selected_gene_set,
                     description=run_name,
                     outdir = out_dir,
                     no_plot=True,
                     cutoff=0.05 # test dataset, use lower value from range(0,1)
                    )
    
    result = enr.results
    result1 = result.loc[result['P-value'] < 0.05]
    result2 = result.loc[result['Adjusted P-value'] < 0.05]
    result3 = result2.loc[result2['Term'].str.contains(selected_regex, case = False)]
    
    return result1, result2, result3
# TESTS
class TestAHProjectCodeBase(unittest.TestCase):    
    
    # Unit Tests
    
    # Primary Helper / Utility Functions
    
    def test_generate_kfolds(self):
        samples = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', 
                   '13', '14', '15', '16', '17', '18']
        
        folds = generate_kfolds(samples, 2)
        self.assertEqual(folds[0], ['1', '3', '5', '7', '9', '11', '13', '15', '17'])
        self.assertEqual(folds[1], ['2', '4', '6', '8', '10', '12', '14', '16', '18'])
        
        folds = generate_kfolds(samples, 5)
        self.assertEqual(folds[0], ['1', '6', '11', '16'])
        self.assertEqual(folds[1], ['2', '7', '12', '17'])
        self.assertEqual(folds[2], ['3', '8', '13', '18'])
        self.assertEqual(folds[3], ['4', '9', '14'])
        self.assertEqual(folds[4], ['5', '10', '15'])
        
        folds = generate_kfolds(samples, 10)
        self.assertEqual(folds[0], ['1', '11'])
        self.assertEqual(folds[1], ['2', '12'])
        self.assertEqual(folds[2], ['3', '13'])
        self.assertEqual(folds[3], ['4', '14'])
        self.assertEqual(folds[4], ['5', '15'])
        self.assertEqual(folds[5], ['6', '16'])
        self.assertEqual(folds[6], ['7', '17'])
        self.assertEqual(folds[7], ['8', '18'])
        self.assertEqual(folds[8], ['9'])
        self.assertEqual(folds[9], ['10'])
        
        folds = generate_kfolds(samples, 18)
        self.assertEqual(folds[0], ['1'])
        self.assertEqual(folds[1], ['2'])
        self.assertEqual(folds[2], ['3'])
        self.assertEqual(folds[3], ['4'])
        self.assertEqual(folds[4], ['5'])
        self.assertEqual(folds[5], ['6'])
        self.assertEqual(folds[6], ['7'])
        self.assertEqual(folds[7], ['8'])
        self.assertEqual(folds[8], ['9'])
        self.assertEqual(folds[9], ['10'])
        self.assertEqual(folds[10], ['11'])
        self.assertEqual(folds[11], ['12'])
        self.assertEqual(folds[12], ['13'])
        self.assertEqual(folds[13], ['14'])
        self.assertEqual(folds[14], ['15'])
        self.assertEqual(folds[15], ['16'])
        self.assertEqual(folds[16], ['17'])
        self.assertEqual(folds[17], ['18'])
        
        try:
            folds = generate_kfolds(samples, 22)
            self.assertEqual(True, False)
        except ValueError:
            pass
    
    def test_two_dim_list_len(self):
        two_dim_list = [['1', '2', '3'], ['4', '5', '6', '7'], ['8', '9', '10']]
        self.assertEqual(two_dim_list_len(two_dim_list), 10)
        
        two_dim_list2 = [['1'], ['2'], ['3'], ['4']]
        self.assertEqual(two_dim_list_len(two_dim_list2), 4)
        
        two_dim_list3 = [['1', '2'], 'hello', []]
        try:
            two_dim_list_len(two_dim_list3)
            self.assertEqual(True, False)
        except ValueError:
            pass
        
        two_dim_list3 = [['1', '2'], [5]]
        try:
            two_dim_list_len(two_dim_list3)
            self.assertEqual(True, False)
        except ValueError:
            pass
        
    def test_compare_one_column_csv_file_contents(self):
        root = os.getcwd()
        root = root.replace("\\", '/')
        delim = '\t'
        inter, diff, union = compare_one_column_csv_file_contents(root + "/TestInput/File1.txt", root + "/TestInput/File2.txt", delim)
        self.assertEqual(inter, {'WKMNF5', 'QPKSMC', 'PLKAN', 'SNOR10', 'MN', 'ABC', 'MIR55', 'QL', 'QPKLK', 'FOINEP3.10', 'BCO5'})
        self.assertEqual(diff, {'LC14', 'JIEMD', 'CKHJOQQQKK24', 'QPPAG', 'OIUYR', 'DFGB', 'LKWJMC'})
        self.assertEqual(union, {'WKMNF5', 'QPKSMC', 'PLKAN', 'SNOR10', 'MN', 'ABC', 'MIR55', 'QL', 'QPKLK', 'FOINEP3.10', 'BCO5',
                                 'LC14', 'JIEMD', 'CKHJOQQQKK24', 'QPPAG', 'OIUYR', 'DFGB', 'LKWJMC'})
        
    def test_compare_one_column_txt_file_contents(self):
        root = os.getcwd()
        root = root.replace("\\", '/')
        inter, diff, union = compare_one_column_txt_file_contents(root + "/TestInput/File1.txt", root + "/TestInput/File2.txt")
        self.assertEqual(inter, {'WKMNF5', 'QPKSMC', 'PLKAN', 'SNOR10', 'MN', 'ABC', 'MIR55', 'QL', 'QPKLK', 'FOINEP3.10', 'BCO5'})
        self.assertEqual(diff, {'LC14', 'JIEMD', 'CKHJOQQQKK24', 'QPPAG', 'OIUYR', 'DFGB', 'LKWJMC'})
        self.assertEqual(union, {'WKMNF5', 'QPKSMC', 'PLKAN', 'SNOR10', 'MN', 'ABC', 'MIR55', 'QL', 'QPKLK', 'FOINEP3.10', 'BCO5',
                                 'LC14', 'JIEMD', 'CKHJOQQQKK24', 'QPPAG', 'OIUYR', 'DFGB', 'LKWJMC'})
        inter, diff, union = compare_one_column_txt_file_contents(root + "/TestInput/File1.txt", root + "/TestInput/File2.txt", 10)
        self.assertEqual(inter, {'ABC', 'QL', 'WKMNF5', 'QPKLK', 'MN', 'BCO5', 'PLKAN'})
        self.assertEqual(diff, {'JIEMD', 'LKWJMC', 'CKHJOQQQKK24', 'DFGB', 'OIUYR', 'SNOR10'})
        self.assertEqual(union, {'ABC', 'QL', 'WKMNF5', 'QPKLK', 'MN', 'BCO5', 'PLKAN', 'JIEMD', 
                                 'LKWJMC', 'CKHJOQQQKK24', 'DFGB', 'OIUYR', 'SNOR10'})
        
    def test_compare_csv_file_contents(self):
        root = os.getcwd()
        root = root.replace("\\", '/')
        delim = '\t'
        self.assertEqual(True, compare_csv_file_contents(root + "/TestInput/gene_exp.diff",
                                                         root + "/TestInput/gene_exp_Copy.diff", delim))

    def test_filter_cuffdiff_file_by_gene_list(self):
        root = os.getcwd()
        root = root.replace("\\", '/')
        delim = '\t'
        gene_list_file = root + "/TestInput/top_200_genes_hg38_Starcq_Ensembl.txt"
        cuffdiff_file = root + "/TestInput/gene_exp.diff"
        output_dir = root + "/TestOutput"
        filter_cuffdiff_file_by_gene_list(gene_list_file, cuffdiff_file, output_dir)
        self.assertTrue(compare_csv_file_contents(root + "/TestInput/TopGenesDE/top_200_genes_hg38_Starcq_Ensembl_DE_expected.diff",
                                                  root + "/TestOutput/filtered.diff", delim))
       
    def test_compare_cuffdiff_mappings_to_cuffnorm_mappings(self):
        root = '...'
        cuffdiff_file = root + 'read_groups.info'
        cuffdiff_file2 = root + 'read_groups2.info'
        cuffdiff_file3 = root + 'read_groups3.info'
        cuffdiff_file4 = root + 'read_groups4.info'
        cuffdiff_file5 = root + 'read_groups5.info'
        cuffnorm_file = root + 'samples.table'
        cuffnorm_file2 = root + 'samples2.table'
        check = compare_cuffdiff_mappings_to_cuffnorm_mappings(cuffdiff_file, cuffnorm_file)
        self.assertTrue(check)
        
        try:print(compare_cuffdiff_mappings_to_cuffnorm_mappings(cuffdiff_file2, cuffnorm_file))
        except ValueError:pass 
        try:print(compare_cuffdiff_mappings_to_cuffnorm_mappings(cuffdiff_file3, cuffnorm_file))
        except ValueError:pass
        try:print(compare_cuffdiff_mappings_to_cuffnorm_mappings(cuffdiff_file4, cuffnorm_file))
        except ValueError:pass
        try:print(compare_cuffdiff_mappings_to_cuffnorm_mappings(cuffdiff_file5, cuffnorm_file))
        except ValueError:pass
        try:print(compare_cuffdiff_mappings_to_cuffnorm_mappings(cuffdiff_file, cuffnorm_file2))
        except ValueError:pass
        try:print(compare_cuffdiff_mappings_to_cuffnorm_mappings(cuffdiff_file2, cuffnorm_file2))
        except ValueError:pass
        try:print(compare_cuffdiff_mappings_to_cuffnorm_mappings(cuffdiff_file3, cuffnorm_file2))
        except ValueError:pass
        try:print(compare_cuffdiff_mappings_to_cuffnorm_mappings(cuffdiff_file4, cuffnorm_file2))
        except ValueError:pass
        try:print(compare_cuffdiff_mappings_to_cuffnorm_mappings(cuffdiff_file5, cuffnorm_file2))
        except ValueError:pass
        
    def test_read_cuffdiff_counts(self):
        file = '...'
        file += '...'
        counts = read_cuffdiff_counts(file)
        counts_exp = {'5S_rRNA_q1_0':0, '5S_rRNA_q2_12':0, '5S_rRNA_q2_18':0, '5S_rRNA_q2_14':0,
                      '5S_rRNA_q3_0':0, '5S_rRNA_q3_16':0, 'A1BG_q1_0':14.6984, 'A1BG_q2_12':10.8018, 
                      'A1BG_q2_18':44.3107, 'A1BG_q2_14':20.8957, 'A1BG_q3_0':18.3356, 'A1BG_q3_16':10.248,
                      'A1BG-AS1_q1_0':2.04062, 'A1BG-AS1_q2_12':2.99613, 'A1BG-AS1_q2_18':5.85184,
                      'A1BG-AS1_q2_14':2.34695, 'A1BG-AS1_q3_0':2.63584, 'A1BG-AS1_q3_16':2.6892}
        for k,v in counts.items():
            self.assertAlmostEqual(v, counts_exp[k])
            
    def test_read_cuffdiff_counts_mean(self):
        file = '...'
        file += '...'
        counts = read_cuffdiff_counts_mean(file)
        counts_exp = {'5S_rRNA@q1':0, '5S_rRNA@q2':0, '5S_rRNA@q3':0, 'A1BG@q1':14.6984,
                      'A1BG@q2':25.336066666, 'A1BG@q3':14.291799999, 'A1BG-AS1@q1':2.04062, 'A1BG-AS1@q2':3.73164,
                      'A1BG-AS1@q3':2.662519999}
        self.assertEqual(len(counts), len(counts_exp))
        for k,v in counts.items():
            self.assertAlmostEqual(v, counts_exp[k])
        
    def test_read_cuffnorm_counts(self):
        root = '...'
        file = root + 'genes_mini.fpkm_table'
        counts = read_cuffnorm_counts(file)
        counts_exp = {'A1BG_q1_25':22.8237, 'A1BG_q1_23':11.5761, 'A1BG_q2_3':15.9426,
                      'A1BG_q3_3':18.4931, 'A1BG-AS1_q1_25':4.83705, 'A1BG-AS1_q1_23':4.55384,
                      'A1BG-AS1_q2_3':3.89096, 'A1BG-AS1_q3_3':4.06879, 'A1CF_q1_25':0.000875093,
                      'A1CF_q1_23':0.00503351, 'A1CF_q2_3':0, 'A1CF_q3_3':0.00144417,
                      'A2M_q1_25':1.33351, 'A2M_q1_23':0.214789, 'A2M_q2_3':0.894067,
                      'A2M_q3_3':0.343044}
        for k,v in counts.items():
            self.assertAlmostEqual(v, counts_exp[k])
            
    def test_read_cuffdiff_counts_mean_variance(self):
        root_dir = '...'
        filename = root_dir + 'genes_mini2.read_group_tracking'
        mean_std = read_cuffdiff_counts_mean_variance(filename)
        mean_std_log = read_cuffdiff_counts_mean_variance(filename, True)
        mean_std_exp = {'A1BG': [17.1999, 15.9426, 18.4931, 5.623799999999999, 0.0, 0.0],
                        'A1BG-AS1': [4.695444999999999, 3.89096, 4.06879, 0.14160499999999976, 0.0, 0.0],
                        'A1CF': [0.0029543015, 0.0, 0.00144417, 0.0020792085, 0.0, 0.0],
                        'A2M': [0.7741495, 0.894067, 0.343044, 0.5593604999999999, 0.0, 0.0]}
        mean_std_log_exp = {'A1BG': [2.851239535413847, 2.829831160327372, 2.9700605567975327, 0.31944134810466673, 0.0, 0.0],
                            'A1BG-AS1': [1.7393575563162778, 1.5873886032371287, 1.6231021303424336, 0.024867975912888696, 0.0, 0.0],
                            'A1CF': [0.002947797284049545, 0.0, 0.001443128189419399, 0.002073086954696768, 0.0, 0.0],
                            'A2M': [0.5209719854833603, 0.6387263690062148, 0.2949386794764998, 0.3264015863233757, 0.0, 0.0]}
        
        self.assertEqual(mean_std, mean_std_exp)
        self.assertEqual(mean_std_log, mean_std_log_exp)
            
    def test_read_cuffnorm_counts_mean_variance(self):
        root_dir = '...'
        filename = root_dir + 'genes_mini.fpkm_table'
        mean_std = read_cuffnorm_counts_mean_variance(filename)
        mean_std_log = read_cuffnorm_counts_mean_variance(filename, True)
        mean_std_exp = {'A1BG': [17.1999, 15.9426, 18.4931, 5.623799999999999, 0.0, 0.0],
                        'A1BG-AS1': [4.695444999999999, 3.89096, 4.06879, 0.14160499999999976, 0.0, 0.0],
                        'A1CF': [0.0029543015, 0.0, 0.00144417, 0.0020792085, 0.0, 0.0],
                        'A2M': [0.7741495, 0.894067, 0.343044, 0.5593604999999999, 0.0, 0.0]}
        mean_std_log_exp = {'A1BG': [2.851239535413847, 2.829831160327372, 2.9700605567975327, 0.31944134810466673, 0.0, 0.0],
                            'A1BG-AS1': [1.7393575563162778, 1.5873886032371287, 1.6231021303424336, 0.024867975912888696, 0.0, 0.0],
                            'A1CF': [0.002947797284049545, 0.0, 0.001443128189419399, 0.002073086954696768, 0.0, 0.0],
                            'A2M': [0.5209719854833603, 0.6387263690062148, 0.2949386794764998, 0.3264015863233757, 0.0, 0.0]}
        
        self.assertEqual(mean_std, mean_std_exp)
        self.assertEqual(mean_std_log, mean_std_log_exp)
        
    def test_compare_cuffnorm_cuffdiff(self):
        root_diff = '...'
        root_norm = '...'
        
        # Two condition tests (AH, CT)
        cuffdiff_counts = root_diff + 'genes_mini.read_group_tracking'
        cuffdiff_info = root_diff + 'read_groups.info'
        cuffnorm_counts = root_norm + 'genes_mini2.fpkm_table'
        cuffnorm_info = root_norm + 'samples.table'
        inc = compare_cuffnorm_cuffdiff(cuffdiff_counts, cuffdiff_info, cuffnorm_counts, cuffnorm_info)
        self.assertAlmostEqual(inc, 2.413793103)
        
    def test_compare_cuffdiff_cuffdiff(self):
        # Three condition tests (AH, DA, CT)
        root = '...'
        file1 = root + 'genes_totalcond3.read_group_tracking'
        file3 = root + 'genes_fold3cond3.read_group_tracking'
        margin = 0.2
        
        inc = compare_cuffdiff_cuffdiff(file1, file3, margin)
        self.assertAlmostEqual(inc, 33.33333333)
        
    def test_compare_cuffnorm_sample_table_files(self):
        root = '...'
        file1 = root + 'samples_one.table'
        file2 = root + 'samples_two.table'
        file3 = root + 'samples_three.table'
        file4 = root + 'samples_four.table'
        self.assertTrue(compare_cuffdiff_read_groups_info_files([file1,file2]))
        self.assertFalse(compare_cuffdiff_read_groups_info_files([file1,file2,file3]))
        self.assertFalse(compare_cuffdiff_read_groups_info_files([file1,file2,file4]))
        self.assertFalse(compare_cuffdiff_read_groups_info_files([file3,file4]))
    
    def test_compare_cuffdiff_read_groups_info_files(self):
        root = '...'
        file1 = root + 'read_groups_one.info'
        file2 = root + 'read_groups_two.info'
        file3 = root + 'read_groups_three.info'
        file4 = root + 'read_groups_four.info'
        self.assertTrue(compare_cuffdiff_read_groups_info_files([file1,file2]))
        self.assertFalse(compare_cuffdiff_read_groups_info_files([file1,file2,file3]))
        self.assertFalse(compare_cuffdiff_read_groups_info_files([file1,file2,file4]))
        self.assertFalse(compare_cuffdiff_read_groups_info_files([file3,file4]))
        
    def test_compare_csv_files(self):
        root = '...'
        file1 = root + 'CSVFile1.csv'
        file2 = root + 'CSVFile2.csv'
        file3 = root + 'CSVFile3.csv'
        file4 = root + 'CSVfile4.txt'
        self.assertTrue(compare_csv_files(file1, file2, [0], ',', ','))
        self.assertTrue(compare_csv_files(file1, file2, [2], ',', ','))
        self.assertTrue(compare_csv_files(file1, file2, [0,2], ',', ','))
        self.assertFalse(compare_csv_files(file1, file2, [0,1,2,3], ',', ','))
        
        self.assertTrue(compare_csv_files(file2, file3, [0], ',', ','))
        self.assertTrue(compare_csv_files(file2, file3, [2], ',', ','))
        self.assertTrue(compare_csv_files(file2, file3, [0,2], ',', ','))
        self.assertFalse(compare_csv_files(file2, file3, [0,1,2], ',', ','))
        self.assertFalse(compare_csv_files(file2, file3, [1], ',', ','))
        
        self.assertTrue(compare_csv_files(file3, file4, [0,1,2], ',', '\t'))
        try:compare_csv_files(file3, file4, [0,1,2,3], ',', '\t')
        except ValueError:pass
        
    def test_validate_cuffnorm_cuffdiff_pipeline_files_one_setting(self):
        # This function is trivial, as long as compare_cuffnorm_cuffdiff 
        # and compare_cuffdiff_cuffdiff are valid it should be valid as well.
        pass
    
    def test_filenames_to_replicate_names_cuffdiff(self):
        root_dir = '...'
        read_groups = root_dir + 'read_groups.info'
        AH_PB_filenames = []
        CT_PB_filenames = []
        
        data_filenames = read_cuffdiff_group_info_filenames(read_groups)
        
        for fname in data_filenames:
            if(fname[0:2] == 'AH'):
                AH_PB_filenames.append(fname)
            elif(fname[0:2] == 'CT'):
                CT_PB_filenames.append(fname)
            else:
                raise ValueError("This .cxb filename is not AH or CT.")
                
        AH_rep_names = filenames_to_replicate_names_cuffdiff(read_groups, AH_PB_filenames)
        CT_rep_names = filenames_to_replicate_names_cuffdiff(read_groups, CT_PB_filenames)
        
        AH_rep_names_expected = ['q1_0', 'q1_1', 'q1_2', 'q1_3', 'q1_4', 'q1_5', 'q1_6', 'q1_7', 'q1_8', 'q1_9',
                                 'q1_10', 'q1_11', 'q1_12', 'q1_13', 'q1_14', 'q1_15', 'q1_16', 'q1_17', 'q1_18',
                                 'q1_19', 'q1_20', 'q1_21', 'q1_22', 'q1_23', 'q1_24', 'q1_25', 'q1_26', 'q1_27',
                                 'q1_28', 'q1_29', 'q1_30', 'q1_31', 'q1_32', 'q1_33', 'q1_34', 'q1_35', 'q1_36', 'q1_37']
        CT_rep_names_expected = ['q2_0', 'q2_1', 'q2_2', 'q2_3', 'q2_4', 'q2_5', 'q2_6', 'q2_7', 'q2_8', 'q2_9', 'q2_10',
                                 'q2_11', 'q2_12', 'q2_13', 'q2_14', 'q2_15', 'q2_16', 'q2_17', 'q2_18', 'q2_19']
        
        self.assertEqual(AH_rep_names, AH_rep_names_expected)
        self.assertEqual(CT_rep_names, CT_rep_names_expected)
    
    def test_filenames_to_replicate_names_cuffnorm(self):
        root_dir = '...'
        samples_table = root_dir + 'samples.table'
        AH_PB_filenames = []
        CT_PB_filenames = []
        
        data_filenames = read_cuffnorm_samples_table_filenames(samples_table)
        
        for fname in data_filenames:
            if(fname[0:2] == 'AH'):
                AH_PB_filenames.append(fname)
            elif(fname[0:2] == 'CT'):
                CT_PB_filenames.append(fname)
            else:
                raise ValueError("This .cxb filename is not AH or CT.")
                
        AH_rep_names = filenames_to_replicate_names_cuffnorm(samples_table, AH_PB_filenames)
        CT_rep_names = filenames_to_replicate_names_cuffnorm(samples_table, CT_PB_filenames)
        
        AH_rep_names_expected = ['q1_0', 'q1_1', 'q1_2', 'q1_3', 'q1_4', 'q1_5', 'q1_6', 'q1_7', 'q1_8', 'q1_9',
                                 'q1_10', 'q1_11', 'q1_12', 'q1_13', 'q1_14', 'q1_15', 'q1_16', 'q1_17', 'q1_18',
                                 'q1_19', 'q1_20', 'q1_21', 'q1_22', 'q1_23', 'q1_24', 'q1_25', 'q1_26', 'q1_27',
                                 'q1_28', 'q1_29', 'q1_30', 'q1_31', 'q1_32', 'q1_33', 'q1_34', 'q1_35', 'q1_36', 'q1_37']
        CT_rep_names_expected = ['q2_0', 'q2_1', 'q2_2', 'q2_3', 'q2_4', 'q2_5', 'q2_6', 'q2_7', 'q2_8', 'q2_9', 'q2_10',
                                 'q2_11', 'q2_12', 'q2_13', 'q2_14', 'q2_15', 'q2_16', 'q2_17', 'q2_18', 'q2_19']
        
        self.assertEqual(AH_rep_names, AH_rep_names_expected)
        self.assertEqual(CT_rep_names, CT_rep_names_expected)
    
    
    def test_detect_outlier_features_by_std(self):
        root_dir = '...'
        # A 3 condition test
        fname = root_dir + 'genes_cond3_mini.fpkm_table'
        out_dir = '...'
        taboo_list = detect_outlier_features_by_std(fname, 3, out_dir, 'Cuffnorm', 3.5)
        self.assertEqual(taboo_list, ['A1CF', 'AAAS', 'AACS'])
        
        root_dir = '...'
        # A 3 condition test
        fname = root_dir + 'genes_cond3_mini.read_group_tracking'
        out_dir = '...'
        taboo_list = detect_outlier_features_by_std(fname, 3, out_dir, 'Cuffdiff', 3.5)
        self.assertEqual(taboo_list, ['A1CF', 'AAAS', 'AACS'])
    
    def test_generate_filtered_csv_file_one_col(self):
        root_dir = '...'
        out_dir = '...'
        fname = root_dir + 'File1.txt'
        taboo_list = ['FOINEP3.10', 'MIR55', 'SNOR10']
        to_write = generate_filtered_csv_file_one_col(fname, '\t', taboo_list, out_dir)
        expected = ['ABC', 'DFGB', 'QL', 'WKMNF5', 'QPKLK', 'OIUYR', 'MN', 'BCO5', 'PLKAN','QPPAG' ,'QPKSMC']
        self.assertEqual(to_write, expected)
    
    def test_select_top_cuffdiff_DEs(self):
        root_dir = '...'
        filename2 = root_dir + 'gene_exp_2way.diff'
        filename3 = root_dir + 'gene_exp_3way.diff'
        filename4 = root_dir + 'gene_exp_3way_mini.diff'
        top_genes2 = select_top_cuffdiff_DEs(filename2, 1, 'ALL', ['q1', 'q2'], scheme = 'MEAN')
        top_genes3 = select_top_cuffdiff_DEs(filename3, 1, 5, ['q1', 'q2', 'q3'], scheme = 'MEAN')
        self.assertEqual(top_genes2, ['A1CF', 'A2M', 'A2M-AS1', 'A1BG-AS1', 'A1BG', '5S_rRNA',
                                      '5_8S_rRNA', '7SK'])
        self.assertEqual(top_genes3, ['A1CF', 'A2M', 'A2ML1', 'A2M-AS1', 'A1BG-AS1'])
        
        top_genes2 = select_top_cuffdiff_DEs(filename2, 1, 'ALL', ['q1', 'q2'],  scheme = 'MAX')
        top_genes3 = select_top_cuffdiff_DEs(filename3, 1, 5,  ['q1', 'q2', 'q3'],  scheme = 'MAX')
        self.assertEqual(top_genes2, ['A1CF', 'A2M', 'A2M-AS1', 'A1BG-AS1', 'A1BG', '5S_rRNA',
                                      '5_8S_rRNA', '7SK'])
        self.assertEqual(top_genes3, ['A1CF', 'A2M', 'A2ML1', 'A2M-AS1', 'A1BG-AS1'])
        
        top_genes2 = select_top_cuffdiff_DEs(filename4, 1, 'ALL', ['q1', 'q2', 'q3'],  scheme = 'W-MEAN',
                                             weights = [1,2,1])
        top_genes3 = select_top_cuffdiff_DEs(filename3, 1, 5,  ['q1', 'q3'],  scheme = 'W-MEAN',
                                             weights = [1.0])
        self.assertEqual(top_genes2, ['A1CF', 'A1BG-AS1', 'A1BG'])
        self.assertEqual(top_genes3, ['A2M', 'A2ML1', 'A1BG-AS1', 'A2M-AS1', 'A1CF'])
        
        two_way = select_top_cuffdiff_DEs(filename2, 1, 3, ['q1', 'q2'], scheme = 'Pairwise')
        three_way = select_top_cuffdiff_DEs(filename3, 1, 5, ['q1', 'q2', 'q3'], scheme = 'Pairwise')
        three_way2 = select_top_cuffdiff_DEs(filename3, 1, 5, ['q1', 'q3'], scheme = 'Pairwise')
        
        two_way_exp = ['A1CF', 'A2M', 'A2M-AS1']
        three_way_exp = ['A1CF', 'A2M', 'A2ML1', 'A1BG-AS1', 'A2M-AS1']
        three_way2_exp = ['A2M', 'A2ML1', 'A1BG-AS1', 'A2M-AS1', 'A1CF']
        
        self.assertEqual(two_way, two_way_exp)
        self.assertEqual(three_way, three_way_exp)
        self.assertEqual(three_way2, three_way2_exp)
        
        try:
            two_way = select_top_cuffdiff_DEs(filename2, 1, 3, ['q1', 'q2'], scheme = 'MIN')
            self.assertEqual(True, False)
        except ValueError:
            pass
        
        try:
            two_way = select_top_cuffdiff_DEs(filename2, 1, 'SOME', ['q1', 'q2'], scheme = 'MAX')
            self.assertEqual(True, False)
        except TypeError:
            pass
        
        try:
            two_way = select_top_cuffdiff_DEs(filename2, 1, 'ALL', ['q1', 'q2'], scheme = 'Pairwise')
            self.assertEqual(True, False)
        except TypeError:
            pass
    
    def test_read_in_cuffdiff_gene_exp_file_pairwise(self):
        root_dir = '...'
        filename2 = root_dir + 'gene_exp_2way.diff'
        filename3 = root_dir + 'gene_exp_3way.diff'
        two_way = read_in_cuffdiff_gene_exp_file_pairwise(filename2, 1, 3, 0, ['q1', 'q2'])
        three_way = read_in_cuffdiff_gene_exp_file_pairwise(filename3, 1, 5, 0, ['q1', 'q2', 'q3'])
        three_way2 = read_in_cuffdiff_gene_exp_file_pairwise(filename3, 1, 5, 0, ['q1', 'q3'])
        
        two_way_exp = ['A1CF', 'A2M', 'A2M-AS1']
        three_way_exp = ['A1CF', 'A2M', 'A2ML1', 'A1BG-AS1', 'A2M-AS1']
        three_way2_exp = ['A2M', 'A2ML1', 'A1BG-AS1', 'A2M-AS1', 'A1CF']
        
        self.assertEqual(two_way, two_way_exp)
        self.assertEqual(three_way, three_way_exp)
        self.assertEqual(three_way2, three_way2_exp)
        
        try:
            three_way = read_in_cuffdiff_gene_exp_file_pairwise(filename3, 1, 5, 0, ['q1'])
            self.assertEqual(True, False)
        except ValueError:
            pass
        
        try:
            three_way = read_in_cuffdiff_gene_exp_file_pairwise(filename3, 1, 15, 0, ['q1', 'q2', 'q3'])
            self.assertEqual(True, False)
        except StopIteration:
            pass
    
    def test_read_in_cuffdiff_gene_exp_file(self):
        root_dir = '...'
        filename2 = root_dir + 'gene_exp_2way.diff'
        filename3 = root_dir + 'gene_exp_3way.diff'
        filename4 = root_dir + 'gene_exp_3way_mini.diff'
        two_way = read_in_cuffdiff_gene_exp_file(filename2, 1, 'ALL', 0, ['q1', 'q2'], 'MEAN')
        three_way = read_in_cuffdiff_gene_exp_file(filename3, 1, 'ALL', 0, ['q1', 'q2', 'q3'], 'MEAN')
        
        two_way_exp = ['A1CF', 'A2M', 'A2M-AS1', 'A1BG-AS1', 'A1BG', '5S_rRNA', '5_8S_rRNA', '7SK']
        three_way_exp = ['A1CF', 'A2M', 'A2ML1', 'A2M-AS1', 'A1BG-AS1', 'A1BG',
                         '5S_rRNA', '5_8S_rRNA', '7SK']
        
        self.assertEqual(two_way, two_way_exp)
        self.assertEqual(three_way, three_way_exp)
        
        two_way = read_in_cuffdiff_gene_exp_file(filename2, 1, 'ALL', 0, ['q1', 'q2'], 'MAX')
        three_way = read_in_cuffdiff_gene_exp_file(filename3, 1, 'ALL', 0, ['q1', 'q2', 'q3'], 'MAX')
        
        two_way_exp = ['A1CF', 'A2M','A2M-AS1','A1BG-AS1', 'A1BG', '5S_rRNA', '5_8S_rRNA', '7SK']
        three_way_exp = ['A1CF', 'A2M', 'A2ML1', 'A2M-AS1',  'A1BG-AS1', 'A1BG', '5S_rRNA', '5_8S_rRNA', '7SK']
        
        self.assertEqual(two_way, two_way_exp)
        self.assertEqual(three_way, three_way_exp)
        
        three_way = read_in_cuffdiff_gene_exp_file(filename4, 1, 'ALL', 0, ['q1', 'q2', 'q3'], 'W-MEAN', [1, 2, 1])
        three_way2 = read_in_cuffdiff_gene_exp_file(filename3, 1, 'ALL', 0, ['q1', 'q3'], 'W-MEAN', [1.0])
        
        three_way_exp = ['A1CF', 'A1BG-AS1', 'A1BG']
        three_way_exp2 = ['A2M', 'A2ML1', 'A1BG-AS1', 'A2M-AS1', 'A1CF',  'A1BG', '5S_rRNA', '5_8S_rRNA', '7SK'] 
        
        self.assertEqual(three_way, three_way_exp)
        self.assertEqual(three_way2, three_way_exp2)
        
        try:
            three_way = read_in_cuffdiff_gene_exp_file(filename4, 1, 'ALL', 0, ['q1', 'q2', 'q3'],
                                                       'W-MEAN', [1, 2, 1, 2])
            self.assertEqual(True, False)
        except ValueError:
            pass
        
        try:
            three_way = read_in_cuffdiff_gene_exp_file(filename4, 1, 'ALL', 0, ['q1', 'q2', 'q3'],
                                                       'W-MEAN', [1, 2.0])
            self.assertEqual(True, False)
        except ValueError:
            pass
        
        try:
            two_way = read_in_cuffdiff_gene_exp_file(filename2, 1, 20, 0, ['q1', 'q2'], 'MAX')
            self.assertEqual(True, False)
        except ValueError:
            pass
        
        try:
            two_way = read_in_cuffdiff_gene_exp_file(filename2, 1, 'ALL', 0, ['q1', 'q2'], 'MIN')
            self.assertEqual(True, False)
        except ValueError:
            pass
        
        try:
            two_way = read_in_cuffdiff_gene_exp_file(filename2, 1, 'ALL', 0, ['q1'], 'MAX')
            self.assertEqual(True, False)
        except ValueError:
            pass
    
    def test_read_cuffdiff_counts2(self):
        # All condition tests
        
        root = '...'
        file = root + 'genes_mini2.read_group_tracking'
        counts = read_cuffdiff_counts2(file)
        counts_exp = {'A1BG':[('q1_25', 22.8237), ('q1_23', 11.5761), ('q2_3', 15.9426), ('q3_3', 18.4931)],
                      'A1BG-AS1':[('q1_25', 4.83705), ('q1_23',4.55384), ('q2_3', 3.89096), ('q3_3', 4.06879)],
                      'A1CF':[('q1_25', 0.000875093), ('q1_23',0.00503351), ('q2_3', 0), ('q3_3', 0.00144417)],
                      'A2M':[('q1_25', 1.33351), ('q1_23',0.214789), ('q2_3', 0.894067), ('q3_3', 0.343044)]}
        for k,v in counts.items():
            self.assertEqual(v, counts_exp[k])
            
        counts = read_cuffdiff_counts2(file, 'ALL', True)
        counts_exp = {'A1BG': [('q1_25', 3.170680883518514), ('q1_23', 2.5317981873091804), ('q2_3', 2.829831160327372),
                     ('q3_3', 2.9700605567975327)], 'A1BG-AS1': [('q1_25', 1.7642255322291664),
                     ('q1_23', 1.714489580403389), ('q2_3', 1.5873886032371287), ('q3_3', 1.6231021303424336)],
                     'A1CF': [('q1_25', 0.000874710329352777), ('q1_23', 0.005020884238746313), ('q2_3', 0.0),
                     ('q3_3', 0.001443128189419399)], 'A2M': [('q1_25', 0.847373571806736), ('q1_23', 0.19457039915998461),
                     ('q2_3', 0.6387263690062148), ('q3_3', 0.2949386794764998)]}
        
        for k,v in counts.items():
            self.assertEqual(v, counts_exp[k])
        
        # Subset of conditions tests
        root = '...'
        file = root + 'genes_cond3_mini2.read_group_tracking'
        counts = read_cuffdiff_counts2(file, ['q1', 'q2'])
        counts_exp = {'A1BG':[('q1_31', 21.1117), ('q1_15', 6.60412), ('q1_9', 18.1683), ('q2_13', 24.0633),
                              ('q2_0', 25.2086), ('q2_15', 44.2252), ('q2_8', 16.2018)],
                      'A1BG-AS1':[('q1_31', 4.28402), ('q1_15', 3.10187), ('q1_9', 4.5401), ('q2_13', 3.61982),
                                  ('q2_0', 3.23702), ('q2_15', 5.84055), ('q2_8', 4.02096)],
                      'A1CF':[('q1_31', 0.00316058), ('q1_15', 0.0158307), ('q1_9', 0), ('q2_13', 0),
                              ('q2_0', 0), ('q2_15', 0), ('q2_8', 0.000892556)]}
        
        for k,v in counts.items():
            self.assertEqual(v, counts_exp[k])
        
        
        counts = read_cuffdiff_counts2(file, ['q3'])
        counts_exp = {'A1BG':[('q3_0', 10.2617), ('q3_16', 27.4816)],
                      'A1BG-AS1':[('q3_0', 2.08565), ('q3_16', 3.75979)],
                      'A1CF':[('q3_0', 0.00132156), ('q3_16', 0.0197848)]}
        
        for k,v in counts.items():
            self.assertEqual(v, counts_exp[k])
        
    
    def test_read_cuffnorm_counts2(self):
        # All condition tests
        
        root = '...'
        file = root + 'genes_mini.fpkm_table'
        counts = read_cuffnorm_counts2(file)
        counts_exp = {'A1BG':[('q1_25', 22.8237), ('q1_23', 11.5761), ('q2_3', 15.9426), ('q3_3', 18.4931)],
                      'A1BG-AS1':[('q1_25', 4.83705), ('q1_23',4.55384), ('q2_3', 3.89096), ('q3_3', 4.06879)],
                      'A1CF':[('q1_25', 0.000875093), ('q1_23',0.00503351), ('q2_3', 0), ('q3_3', 0.00144417)],
                      'A2M':[('q1_25', 1.33351), ('q1_23',0.214789), ('q2_3', 0.894067), ('q3_3', 0.343044)]}
        for k,v in counts.items():
            self.assertEqual(v, counts_exp[k])
            
        counts = read_cuffnorm_counts2(file, 'ALL', True)
        counts_exp = {'A1BG': [('q1_25', 3.170680883518514), ('q1_23', 2.5317981873091804), ('q2_3', 2.829831160327372),
                     ('q3_3', 2.9700605567975327)], 'A1BG-AS1': [('q1_25', 1.7642255322291664),
                     ('q1_23', 1.714489580403389), ('q2_3', 1.5873886032371287), ('q3_3', 1.6231021303424336)],
                     'A1CF': [('q1_25', 0.000874710329352777), ('q1_23', 0.005020884238746313), ('q2_3', 0.0),
                     ('q3_3', 0.001443128189419399)], 'A2M': [('q1_25', 0.847373571806736), ('q1_23', 0.19457039915998461),
                     ('q2_3', 0.6387263690062148), ('q3_3', 0.2949386794764998)]}
        
        for k,v in counts.items():
            self.assertEqual(v, counts_exp[k])
            
        # Subset of conditions tests
        root = '...'
        file = root + 'genes_cond3_mini2.fpkm_table'
        counts = read_cuffnorm_counts2(file, ['q1', 'q2'])
        counts_exp = {'A1BG':[('q1_31', 21.1117), ('q1_15', 6.60412), ('q1_9', 18.1683), ('q2_13', 24.0633),
                              ('q2_0', 25.2086), ('q2_15', 44.2252), ('q2_8', 16.2018)],
                      'A1BG-AS1':[('q1_31', 4.28402), ('q1_15', 3.10187), ('q1_9', 4.5401), ('q2_13', 3.61982),
                                  ('q2_0', 3.23702), ('q2_15', 5.84055), ('q2_8', 4.02096)],
                      'A1CF':[('q1_31', 0.00316058), ('q1_15', 0.0158307), ('q1_9', 0), ('q2_13', 0),
                              ('q2_0', 0), ('q2_15', 0), ('q2_8', 0.000892556)]}
        
        for k,v in counts.items():
            self.assertEqual(v, counts_exp[k])
        
        
        counts = read_cuffnorm_counts2(file, ['q3'])
        counts_exp = {'A1BG':[('q3_0', 10.2617), ('q3_16', 27.4816)],
                      'A1BG-AS1':[('q3_0', 2.08565), ('q3_16', 3.75979)],
                      'A1CF':[('q3_0', 0.00132156), ('q3_16', 0.0197848)]}
        
        for k,v in counts.items():
            self.assertEqual(v, counts_exp[k])
    
    def test_read_cuffnorm_gene_names(self):
        root = '...'
        file = root + 'genes_mini.fpkm_table'
        gene_names = read_cuffnorm_gene_names(file)
        gene_names_exp = ['A1BG','A1BG-AS1','A1CF','A2M']
        self.assertEqual(gene_names, gene_names_exp)
        
    def test_read_cuffdiff_gene_names(self):
        root = '...'
        file = root + 'genes_mini2.read_group_tracking'
        gene_names = read_cuffdiff_gene_names(file)
        gene_names_exp = ['A1BG','A1BG-AS1','A1CF','A2M']
        self.assertEqual(gene_names, gene_names_exp)
        
    def test_read_cuffnorm_samples_table_filenames(self):
        root_dir = '...'
        samples_table = root_dir + 'samples.table'
        data_filenames = read_cuffnorm_samples_table_filenames(samples_table)
        
        data_filenames_exp = ['...']
        self.assertEqual(data_filenames, data_filenames_exp)
    
    def test_read_cuffdiff_group_info_filenames(self):
        root_dir = '...'
        filename = root_dir + 'read_groups.info'
        files = read_cuffdiff_group_info_filenames(filename)
        files_exp = ['...']
        self.assertEqual(files, files_exp)
    
    def test_generate_train_validate_data(self):
        # Add tests with Cuffdiff format count files (genes.read_group_tracking, read_groups.info).
        
        # Test1aa: 3 conditions from within 3 condition file (valid)
        
        root_dir = '...'
        out_dir = '...'
        pipeline = 'hg38_Starcq_Ensembl'
        normalization = 'GEOM'
        dispersion = 'POOL'
        num_samples = {'AH':2, 'DA':2, 'CT':2}
        v_transform = True
        taboo_list = []
        
        fpkm_table = root_dir + '/' + pipeline + '/Cuffnorm_' + normalization + '/genes.fpkm_table'
    
        #Translate condtion names (ex: AH) to replicate condition names (ex: q1, q2, etc.)
        conditions = num_samples.keys()
        temp_dir = root_dir +'/' + pipeline + '/Cuffnorm_' + normalization + '/'
        cond_to_rep_cond_map = generate_cond_name_to_rep_name_map(temp_dir)
        rep_cond_names = []
        for cond in conditions:
            if(cond_to_rep_cond_map[cond] not in rep_cond_names):
                rep_cond_names.append(cond_to_rep_cond_map[cond])
            
        counts = read_cuffnorm_counts2(fpkm_table, rep_cond_names, v_transform)
        gene_names = read_cuffnorm_gene_names(fpkm_table)
        
        X_ts, X_vs, Y_ts, Y_vs, label_map = generate_train_validate_data(2, 2, num_samples,root_dir, pipeline,
                                      normalization, dispersion, counts, gene_names, 'top_genes.txt', taboo_list, 'PB', out_dir,
                                      False, 'Cuffnorm', False, True)
        
        exp_X_ts = [np.array([[3.43770287, 3.69005376], [2.41613697, 3.40368429], [2.70935601, 3.19777067]]),
                    np.array([[0.83697909, 1.25541374], [0.23857428, 0.96382678], [0.24903038, 0.56030983]])]
        
        exp_X_vs = [np.array([[3.50116639, 3.52026256], [1.8214461 , 2.89596719], [2.14911568, 3.03518373]]),
                    np.array([[0.63601477, 1.9815751 ], [0.29030114, 1.78014394], [0.31356836, 1.28000564]])]
        exp_Y_ts = [np.array([0, 1, 2]), np.array([0, 1, 2])]
        exp_Y_vs = [np.array([0, 1, 2]), np.array([0, 1, 2])]
        exp_label_map = {'q1': 0, 'q3': 1, 'q2': 2}
        
        matrix_index = 0
        while matrix_index < len(X_ts):
            row_index = 0
            while row_index < X_ts[matrix_index].shape[0]:
                column_index = 0
                while column_index < X_ts[matrix_index].shape[1]:
                    self.assertAlmostEqual(X_ts[matrix_index][row_index][column_index],
                                           exp_X_ts[matrix_index][row_index][column_index], 5)
                    column_index += 1
                row_index += 1
            matrix_index += 1
            
        matrix_index = 0
        while matrix_index < len(X_vs):
            row_index = 0
            while row_index < X_vs[matrix_index].shape[0]:
                column_index = 0
                while column_index < X_vs[matrix_index].shape[1]:
                    self.assertAlmostEqual(X_vs[matrix_index][row_index][column_index],
                                           exp_X_vs[matrix_index][row_index][column_index], 5)
                    column_index += 1
                row_index += 1
            matrix_index += 1
        
        matrix_index = 0
        while matrix_index < len(Y_ts):
            index = 0
            while index < Y_ts[matrix_index].shape[0]:
                self.assertEqual(Y_ts[matrix_index][index], exp_Y_ts[matrix_index][index])
                index += 1
            matrix_index += 1
            
        matrix_index = 0
        while matrix_index < len(Y_vs):
            index = 0
            while index < Y_vs[matrix_index].shape[0]:
                self.assertEqual(Y_vs[matrix_index][index], exp_Y_vs[matrix_index][index])
                index += 1
            matrix_index += 1
        
        self.assertEqual(label_map, exp_label_map)
        
        # Test1ab: 3 conditions from within 3 condition file (valid) with a taboo list
        
        root_dir = '...'
        out_dir = '...'
        pipeline = 'hg38_Starcq_Ensembl'
        normalization = 'GEOM'
        dispersion = 'POOL'
        num_samples = {'AH':2, 'DA':2, 'CT':2}
        v_transform = True
        taboo_list = ['TCN2']
        
        fpkm_table = root_dir + '/' + pipeline + '/Cuffnorm_' + normalization + '/genes.fpkm_table'
    
        #Translate condtion names (ex: AH) to replicate condition names (ex: q1, q2, etc.)
        conditions = num_samples.keys()
        temp_dir = root_dir +'/' + pipeline + '/Cuffnorm_' + normalization + '/'
        cond_to_rep_cond_map = generate_cond_name_to_rep_name_map(temp_dir)
        rep_cond_names = []
        for cond in conditions:
            if(cond_to_rep_cond_map[cond] not in rep_cond_names):
                rep_cond_names.append(cond_to_rep_cond_map[cond])
            
        counts = read_cuffnorm_counts2(fpkm_table, rep_cond_names, v_transform)
        gene_names = read_cuffnorm_gene_names(fpkm_table)
        
        X_ts, X_vs, Y_ts, Y_vs, label_map = generate_train_validate_data(2, 2, num_samples,root_dir, pipeline,
                                      normalization, dispersion, counts, gene_names, 'top_genes.txt', taboo_list, 'PB', out_dir,
                                      False, 'Cuffnorm', False, True)
        
        exp_X_ts = [np.array([[3.69005376, 5.34879657], [3.40368429, 5.34761692], [3.19777067, 4.69257648]]),
                    np.array([[0.83697909, 1.25541374], [0.23857428, 0.96382678], [0.24903038, 0.56030983]])]
        
        exp_X_vs = [np.array([[3.52026256, 5.01220739], [2.89596719, 3.78809789], [3.03518373, 4.43878614]]),
                    np.array([[0.63601477, 1.9815751 ], [0.29030114, 1.78014394], [0.31356836, 1.28000564]])]
        exp_Y_ts = [np.array([0, 1, 2]), np.array([0, 1, 2])]
        exp_Y_vs = [np.array([0, 1, 2]), np.array([0, 1, 2])]
        exp_label_map = {'q1': 0, 'q3': 1, 'q2': 2}
        
        matrix_index = 0
        while matrix_index < len(X_ts):
            row_index = 0
            while row_index < X_ts[matrix_index].shape[0]:
                column_index = 0
                while column_index < X_ts[matrix_index].shape[1]:
                    self.assertAlmostEqual(X_ts[matrix_index][row_index][column_index],
                                           exp_X_ts[matrix_index][row_index][column_index], 5)
                    column_index += 1
                row_index += 1
            matrix_index += 1
            
        matrix_index = 0
        while matrix_index < len(X_vs):
            row_index = 0
            while row_index < X_vs[matrix_index].shape[0]:
                column_index = 0
                while column_index < X_vs[matrix_index].shape[1]:
                    self.assertAlmostEqual(X_vs[matrix_index][row_index][column_index],
                                           exp_X_vs[matrix_index][row_index][column_index], 5)
                    column_index += 1
                row_index += 1
            matrix_index += 1
        
        matrix_index = 0
        while matrix_index < len(Y_ts):
            index = 0
            while index < Y_ts[matrix_index].shape[0]:
                self.assertEqual(Y_ts[matrix_index][index], exp_Y_ts[matrix_index][index])
                index += 1
            matrix_index += 1
            
        matrix_index = 0
        while matrix_index < len(Y_vs):
            index = 0
            while index < Y_vs[matrix_index].shape[0]:
                self.assertEqual(Y_vs[matrix_index][index], exp_Y_vs[matrix_index][index])
                index += 1
            matrix_index += 1
        
        self.assertEqual(label_map, exp_label_map)
        
        # Test 1ba: 3 conditions from within 3 condition file (valid) with 2 conditions mapping to same replicate name
        # That is CT & DA are both q2
        
        root_dir = '...'
        pipeline = 'hg38_Starcq_Ensembl'
        normalization = 'GEOM'
        dispersion = 'POOL'
        num_samples = {'AH':2, 'DA':2, 'CT':2}
        v_transform = True
        taboo_list = []
        
        fpkm_table = root_dir + '/' + pipeline + '/Cuffnorm_' + normalization + '/genes.fpkm_table'
    
        #Translate condtion names (ex: AH) to replicate condition names (ex: q1, q2, etc.)
        conditions = num_samples.keys()
        temp_dir = root_dir +'/' + pipeline + '/Cuffnorm_' + normalization + '/'
        cond_to_rep_cond_map = generate_cond_name_to_rep_name_map(temp_dir)
        rep_cond_names = []
        for cond in conditions:
            if(cond_to_rep_cond_map[cond] not in rep_cond_names):
                rep_cond_names.append(cond_to_rep_cond_map[cond])
            
        counts = read_cuffnorm_counts2(fpkm_table, rep_cond_names, v_transform)
        gene_names = read_cuffnorm_gene_names(fpkm_table)
        
        X_ts, X_vs, Y_ts, Y_vs, label_map = generate_train_validate_data(2, 2, num_samples,root_dir, pipeline,
                                      normalization, dispersion, counts, gene_names, 'top_genes.txt', taboo_list, 'PB', out_dir,
                                      False,'Cuffnorm', False, True)
        
        exp_X_ts = [np.array([[3.43770287, 3.69005376], [2.41613697, 3.40368429], [2.70935601, 3.19777067]]),
                    np.array([[0.83697909, 1.25541374], [0.23857428, 0.96382678], [0.24903038, 0.56030983]])]
        
        exp_X_vs = [np.array([[3.50116639, 3.52026256], [1.8214461 , 2.89596719], [2.14911568, 3.03518373]]),
                    np.array([[0.63601477, 1.9815751 ], [0.29030114, 1.78014394], [0.31356836, 1.28000564]])]
        exp_Y_ts = [np.array([0, 1, 1]), np.array([0, 1, 1])]
        exp_Y_vs = [np.array([0, 1, 1]), np.array([0, 1, 1])]
        exp_label_map = {'q1': 0, 'q2': 1}
        
        matrix_index = 0
        while matrix_index < len(X_ts):
            row_index = 0
            while row_index < X_ts[matrix_index].shape[0]:
                column_index = 0
                while column_index < X_ts[matrix_index].shape[1]:
                    self.assertAlmostEqual(X_ts[matrix_index][row_index][column_index],
                                            exp_X_ts[matrix_index][row_index][column_index], 5)
                    column_index += 1
                row_index += 1
            matrix_index += 1
            
        matrix_index = 0
        while matrix_index < len(X_vs):
            row_index = 0
            while row_index < X_vs[matrix_index].shape[0]:
                column_index = 0
                while column_index < X_vs[matrix_index].shape[1]:
                    self.assertAlmostEqual(X_vs[matrix_index][row_index][column_index],
                                            exp_X_vs[matrix_index][row_index][column_index], 5)
                    column_index += 1
                row_index += 1
            matrix_index += 1
        
        matrix_index = 0
        while matrix_index < len(Y_ts):
            index = 0
            while index < Y_ts[matrix_index].shape[0]:
                self.assertEqual(Y_ts[matrix_index][index], exp_Y_ts[matrix_index][index])
                index += 1
            matrix_index += 1
            
        matrix_index = 0
        while matrix_index < len(Y_vs):
            index = 0
            while index < Y_vs[matrix_index].shape[0]:
                self.assertEqual(Y_vs[matrix_index][index], exp_Y_vs[matrix_index][index])
                index += 1
            matrix_index += 1
        
        self.assertEqual(label_map, exp_label_map)
        
        # Test 1bb: 3 conditions from within 3 condition file (valid) with 2 conditions mapping to same replicate name with taboo
        # That is CT & DA are both q2
        
        root_dir = '...'
        pipeline = 'hg38_Starcq_Ensembl'
        normalization = 'GEOM'
        dispersion = 'POOL'
        num_samples = {'AH':2, 'DA':2, 'CT':2}
        v_transform = True
        taboo_list = ['NKIRAS2', 'PTGR1', 'MERTK']
        
        fpkm_table = root_dir + '/' + pipeline + '/Cuffnorm_' + normalization + '/genes.fpkm_table'
    
        #Translate condtion names (ex: AH) to replicate condition names (ex: q1, q2, etc.)
        conditions = num_samples.keys()
        temp_dir = root_dir +'/' + pipeline + '/Cuffnorm_' + normalization + '/'
        cond_to_rep_cond_map = generate_cond_name_to_rep_name_map(temp_dir)
        rep_cond_names = []
        for cond in conditions:
            if(cond_to_rep_cond_map[cond] not in rep_cond_names):
                rep_cond_names.append(cond_to_rep_cond_map[cond])
            
        counts = read_cuffnorm_counts2(fpkm_table, rep_cond_names, v_transform)
        gene_names = read_cuffnorm_gene_names(fpkm_table)
        
        X_ts, X_vs, Y_ts, Y_vs, label_map = generate_train_validate_data(2, 2, num_samples,root_dir, pipeline,
                                      normalization, dispersion, counts, gene_names, 'top_genes.txt', taboo_list, 'PB', out_dir,
                                      False,'Cuffnorm', False, True)
        
        exp_X_ts = [np.array([[3.43770287, 5.34879657], [2.41613697, 5.34761692], [2.70935601, 4.69257648]]),
                    np.array([[3.50116638, 0.87849332], [1.82144609, 0.82794027], [2.14911568, 0.43143820]])]
        
        exp_X_vs = [np.array([[3.50116639, 5.01220739], [1.8214461, 3.78809789], [2.14911568, 4.43878614]]),
                    np.array([[3.43770287, 3.19303685], [2.41613696, 1.97857646], [2.70935601, 0.51435359]])]
        
        exp_Y_ts = [np.array([0, 1, 1]), np.array([0, 1, 1])]
        exp_Y_vs = [np.array([0, 1, 1]), np.array([0, 1, 1])]
        exp_label_map = {'q1': 0, 'q2': 1}
        
        matrix_index = 0
        while matrix_index < len(X_ts):
            row_index = 0
            while row_index < X_ts[matrix_index].shape[0]:
                column_index = 0
                while column_index < X_ts[matrix_index].shape[1]:
                    self.assertAlmostEqual(X_ts[matrix_index][row_index][column_index],
                                            exp_X_ts[matrix_index][row_index][column_index], 5)
                    column_index += 1
                row_index += 1
            matrix_index += 1
            
        matrix_index = 0
        while matrix_index < len(X_vs):
            row_index = 0
            while row_index < X_vs[matrix_index].shape[0]:
                column_index = 0
                while column_index < X_vs[matrix_index].shape[1]:
                    self.assertAlmostEqual(X_vs[matrix_index][row_index][column_index],
                                            exp_X_vs[matrix_index][row_index][column_index], 5)
                    column_index += 1
                row_index += 1
            matrix_index += 1
        
        matrix_index = 0
        while matrix_index < len(Y_ts):
            index = 0
            while index < Y_ts[matrix_index].shape[0]:
                self.assertEqual(Y_ts[matrix_index][index], exp_Y_ts[matrix_index][index])
                index += 1
            matrix_index += 1
            
        matrix_index = 0
        while matrix_index < len(Y_vs):
            index = 0
            while index < Y_vs[matrix_index].shape[0]:
                self.assertEqual(Y_vs[matrix_index][index], exp_Y_vs[matrix_index][index])
                index += 1
            matrix_index += 1
        
        self.assertEqual(label_map, exp_label_map)
        
        # Test 2: 2 conditions from within 3 condition file (valid)
        root_dir = '...'
        pipeline = 'hg38_Starcq_Ensembl'
        normalization = 'GEOM'
        dispersion = 'POOL'
        num_samples = {'AH':2, 'CT':2}
        v_transform = True
        taboo_list = []
        
        fpkm_table = root_dir + '/' + pipeline + '/Cuffnorm_' + normalization + '/genes.fpkm_table'
        
        #Translate condtion names (ex: AH) to replicate condition names (ex: q1, q2, etc.)
        conditions = num_samples.keys()
        temp_dir = root_dir +'/' + pipeline + '/Cuffnorm_' + normalization + '/'
        cond_to_rep_cond_map = generate_cond_name_to_rep_name_map(temp_dir)
        rep_cond_names = []
        for cond in conditions:
            if(cond_to_rep_cond_map[cond] not in rep_cond_names):
                rep_cond_names.append(cond_to_rep_cond_map[cond])
            
        counts = read_cuffnorm_counts2(fpkm_table, rep_cond_names, v_transform)
        gene_names = read_cuffnorm_gene_names(fpkm_table)
        
        X_ts, X_vs, Y_ts, Y_vs, label_map = generate_train_validate_data(2, 2, num_samples,root_dir, pipeline,
                                      normalization, dispersion, counts, gene_names, 'top_genes.txt', taboo_list, 'PB', out_dir,
                                      False,'Cuffnorm', False, True)
        
        exp_X_ts = [np.array([[3.43770287, 3.69005376], [2.70935601, 3.19777067]]),
                    np.array([[0.83697909, 1.25541374], [0.24903038, 0.56030983]])]
        
        exp_X_vs = [np.array([[3.50116639, 3.52026256], [2.14911568, 3.03518373]]),
                    np.array([[0.63601477, 1.9815751 ], [0.31356836, 1.28000564]])]
        exp_Y_ts = [np.array([0, 1]), np.array([0, 1])]
        exp_Y_vs = [np.array([0, 1]), np.array([0, 1])]
        exp_label_map = {'q1': 0, 'q2': 1}
        
        matrix_index = 0
        while matrix_index < len(X_ts):
            row_index = 0
            while row_index < X_ts[matrix_index].shape[0]:
                column_index = 0
                while column_index < X_ts[matrix_index].shape[1]:
                    self.assertAlmostEqual(X_ts[matrix_index][row_index][column_index],
                                            exp_X_ts[matrix_index][row_index][column_index], 5)
                    column_index += 1
                row_index += 1
            matrix_index += 1
            
        matrix_index = 0
        while matrix_index < len(X_vs):
            row_index = 0
            while row_index < X_vs[matrix_index].shape[0]:
                column_index = 0
                while column_index < X_vs[matrix_index].shape[1]:
                    self.assertAlmostEqual(X_vs[matrix_index][row_index][column_index],
                                            exp_X_vs[matrix_index][row_index][column_index], 5)
                    column_index += 1
                row_index += 1
            matrix_index += 1
        
        matrix_index = 0
        while matrix_index < len(Y_ts):
            index = 0
            while index < Y_ts[matrix_index].shape[0]:
                self.assertEqual(Y_ts[matrix_index][index], exp_Y_ts[matrix_index][index])
                index += 1
            matrix_index += 1
            
        matrix_index = 0
        while matrix_index < len(Y_vs):
            index = 0
            while index < Y_vs[matrix_index].shape[0]:
                self.assertEqual(Y_vs[matrix_index][index], exp_Y_vs[matrix_index][index])
                index += 1
            matrix_index += 1
        
        self.assertEqual(label_map, exp_label_map)
        
        # Test 3+: consider testing a few cases that are meant to raise an error.
        # This is one of the most complex functions in the codebase with many validation layers, so 
        # it would be good to double check that they work as intended.
        try:
            Xs, Xvs, Ys, Yvs, lmap = generate_train_validate_data([2], 2, num_samples,root_dir, pipeline,
                                      normalization, dispersion, counts, gene_names, 'top_genes.txt', taboo_list, 'PB', out_dir,
                                      False,'Cuffnorm', False, True)
            self.assertEqual(True, False)
        except AssertionError:
            pass
        
        try:
            Xs, Xvs, Ys, Yvs, lmap = generate_train_validate_data(2, 1, num_samples,root_dir, pipeline,
                                      normalization, dispersion, counts, gene_names, 'top_genes.txt', taboo_list, 'PB', out_dir,
                                      False,'Cuffnorm', False, True)
            self.assertEqual(True, False)
        except AssertionError:
            pass
        
        try:
            Xs, Xvs, Ys, Yvs, lmap = generate_train_validate_data(2, 10, num_samples,root_dir, pipeline,
                                      normalization, dispersion, counts, gene_names, 'top_genes.txt', taboo_list, 'PB', out_dir,
                                      False,'Cuffnorm', False, True)
            self.assertEqual(True, False)
        except AssertionError:
            pass
        
        try:
            Xs, Xvs, Ys, Yvs, lmap = generate_train_validate_data(100000, 2, num_samples,root_dir, pipeline,
                                      normalization, dispersion, counts, gene_names, 'top_genes.txt', taboo_list, 'PB', out_dir,
                                      False,'Cuffnorm', False, True)
            self.assertEqual(True, False)
        except AssertionError:
            pass
        
        try:
            Xs, Xvs, Ys, Yvs, lmap = generate_train_validate_data(2, 2, num_samples, root_dir, 'Test',
                                      normalization, dispersion, counts, gene_names, 'top_genes.txt', taboo_list, 'PB', out_dir,
                                      False,'Cuffnorm', False, False)
            self.assertEqual(True, False)
        except AssertionError:
            pass
        
        try:
            Xs, Xvs, Ys, Yvs, lmap = generate_train_validate_data(2, 2, {'AH':3, 'CT':3}, root_dir, pipeline, out_dir,
                                      normalization, dispersion, counts, gene_names, 'top_genes.txt', taboo_list, 'PB',
                                      False,'Cuffnorm', False, True)
            self.assertEqual(True, False)
        except ValueError:
            pass
        
        
        root_dir = '...'
        pipeline = 'hg38_Starcq_Ensembl'
        normalization = 'GEOM'
        dispersion = 'POOL'
        num_samples = {'AH':2, 'DA':2, 'CT':2}
        v_transform = True
        taboo_list = []
        
        fpkm_table = root_dir + '/' + pipeline + '/Cuffnorm_' + normalization + '/genes.fpkm_table'
    
        #Translate condtion names (ex: AH) to replicate condition names (ex: q1, q2, etc.)
        conditions = num_samples.keys()
        temp_dir = root_dir +'/' + pipeline + '/Cuffnorm_' + normalization + '/'
        cond_to_rep_cond_map = generate_cond_name_to_rep_name_map(temp_dir)
        rep_cond_names = []
        for cond in conditions:
            if(cond_to_rep_cond_map[cond] not in rep_cond_names):
                rep_cond_names.append(cond_to_rep_cond_map[cond])
            
        counts = read_cuffnorm_counts2(fpkm_table, rep_cond_names, v_transform)
        gene_names = read_cuffnorm_gene_names(fpkm_table)
        
        try:
            Xs, Xvs, Ys, Yvs, lmap = generate_train_validate_data(2, 2, num_samples,root_dir, pipeline,
                                      normalization, dispersion, counts, gene_names, 'top_genes.txt', taboo_list, 'PB', out_dir,
                                      False,'Cuffnorm', False, True)
            self.assertEqual(True, False)
        except ValueError as e:
            self.assertEqual(e.__str__(), "The replicate names are not unique.")
        
        root_dir = '...'
        pipeline = 'hg38_Starcq_Ensembl'
        normalization = 'GEOM'
        dispersion = 'POOL'
        num_samples = {'AH':2, 'DA':2, 'CT':2}
        v_transform = True
        taboo_list = []
        
        fpkm_table = root_dir + '/' + pipeline + '/Cuffnorm_' + normalization + '/genes.fpkm_table'
    
        #Translate condtion names (ex: AH) to replicate condition names (ex: q1, q2, etc.)
        conditions = num_samples.keys()
        temp_dir = root_dir +'/' + pipeline + '/Cuffnorm_' + normalization + '/'
        cond_to_rep_cond_map = generate_cond_name_to_rep_name_map(temp_dir)
        rep_cond_names = []
        for cond in conditions:
            if(cond_to_rep_cond_map[cond] not in rep_cond_names):
                rep_cond_names.append(cond_to_rep_cond_map[cond])
            
        counts = read_cuffnorm_counts2(fpkm_table, rep_cond_names, v_transform)
        gene_names = read_cuffnorm_gene_names(fpkm_table)
        
        try:
            Xs, Xvs, Ys, Yvs, lmap = generate_train_validate_data(2, 2, num_samples,root_dir, pipeline,
                                      normalization, dispersion, counts, gene_names, 'top_genes.txt', taboo_list, 'PB', out_dir,
                                      False,'Cuffnorm', False, True)
            self.assertEqual(True, False)
        except ValueError as e:
            self.assertEqual(e.__str__(), "Each value in counts must be compromised of all replicates (= # of samples).")
        
        root_dir = '...'
        pipeline = 'hg38_Starcq_Ensembl'
        normalization = 'GEOM'
        dispersion = 'POOL'
        num_samples = {'AH':2, 'DA':2, 'CT':2}
        v_transform = True
        taboo_list = []
        
        fpkm_table = root_dir + '/' + pipeline + '/Cuffnorm_' + normalization + '/genes.fpkm_table'
    
        #Translate condtion names (ex: AH) to replicate condition names (ex: q1, q2, etc.)
        conditions = num_samples.keys()
        temp_dir = root_dir +'/' + pipeline + '/Cuffnorm_' + normalization + '/'
        cond_to_rep_cond_map = generate_cond_name_to_rep_name_map(temp_dir)
        rep_cond_names = []
        for cond in conditions:
            if(cond_to_rep_cond_map[cond] not in rep_cond_names):
                rep_cond_names.append(cond_to_rep_cond_map[cond])
            
        counts = read_cuffnorm_counts2(fpkm_table, rep_cond_names, v_transform)
        gene_names = read_cuffnorm_gene_names(fpkm_table)
        
        try:
            Xs, Xvs, Ys, Yvs, lmap = generate_train_validate_data(2, 2, num_samples,root_dir, pipeline,
                                      normalization, dispersion, counts, gene_names, 'top_genes.txt', taboo_list, 'PB', out_dir,
                                      False,'Cuffnorm', False, True)
            self.assertEqual(True, False)
        except ValueError as e:
            self.assertEqual(e.__str__(), "The following replicate: q3_a is not valid.")
        
        root_dir = '...'
        pipeline = 'hg38_Starcq_Ensembl'
        normalization = 'GEOM'
        dispersion = 'POOL'
        num_samples = {'AH':2, 'DA':2, 'CT':2}
        v_transform = True
        taboo_list = []
        
        fpkm_table = root_dir + '/' + pipeline + '/Cuffnorm_' + normalization + '/genes.fpkm_table'
    
        #Translate condtion names (ex: AH) to replicate condition names (ex: q1, q2, etc.)
        conditions = num_samples.keys()
        temp_dir = root_dir +'/' + pipeline + '/Cuffnorm_' + normalization + '/'
        cond_to_rep_cond_map = generate_cond_name_to_rep_name_map(temp_dir)
        rep_cond_names = []
        for cond in conditions:
            if(cond_to_rep_cond_map[cond] not in rep_cond_names):
                rep_cond_names.append(cond_to_rep_cond_map[cond])
            
        counts = read_cuffnorm_counts2(fpkm_table, rep_cond_names, v_transform)
        gene_names = read_cuffnorm_gene_names(fpkm_table)
        
        try:
            Xs, Xvs, Ys, Yvs, lmap = generate_train_validate_data(2, 2, num_samples,root_dir, pipeline,
                                      normalization, dispersion, counts, gene_names, 'top_genes.txt', taboo_list, 'PB', out_dir,
                                      False,'Cuffnorm', False, True)
            self.assertEqual(True, False)
        except ValueError as e:
            self.assertEqual(e.__str__(), "More than 1% of top genes cannot be found in Cuffnorm counts.")
        
        root_dir = '...'
        pipeline = 'hg38_Starcq_Ensembl'
        normalization = 'GEOM'
        dispersion = 'POOL'
        num_samples = {'AH':2, 'DA':2, 'CT':2}
        v_transform = True
        taboo_list = []
        
        fpkm_table = root_dir + '/' + pipeline + '/Cuffnorm_' + normalization + '/genes.fpkm_table'
    
        #Translate condtion names (ex: AH) to replicate condition names (ex: q1, q2, etc.)
        conditions = num_samples.keys()
        temp_dir = root_dir +'/' + pipeline + '/Cuffnorm_' + normalization + '/'
        cond_to_rep_cond_map = generate_cond_name_to_rep_name_map(temp_dir)
        rep_cond_names = []
        for cond in conditions:
            if(cond_to_rep_cond_map[cond] not in rep_cond_names):
                rep_cond_names.append(cond_to_rep_cond_map[cond])
            
        counts = read_cuffnorm_counts2(fpkm_table, rep_cond_names, v_transform)
        gene_names = read_cuffnorm_gene_names(fpkm_table)
        
        try:
            Xs, Xvs, Ys, Yvs, lmap = generate_train_validate_data(2, 2, num_samples,root_dir, pipeline,
                                      normalization, dispersion, counts, gene_names, 'top_genes.txt', taboo_list, 'PB', out_dir,
                                      False,'Cuffnorm', False, True)
            self.assertEqual(True, False)
        except ValueError as e:
            self.assertEqual(e.__str__(), "Unrecognized condition. Only AH, CT, DA, AA, NF, and HP .cxb files are permitted.")
        
        root_dir = '...'
        pipeline = 'hg38_Starcq_Ensembl'
        normalization = 'GEOM'
        dispersion = 'POOL'
        num_samples = {'AH':2, 'DA':2, 'CT':2}
        v_transform = True
        taboo_list = []
        
        fpkm_table = root_dir + '/' + pipeline + '/Cuffnorm_' + normalization + '/genes.fpkm_table'
    
        #Translate condtion names (ex: AH) to replicate condition names (ex: q1, q2, etc.)
        conditions = num_samples.keys()
        temp_dir = root_dir +'/' + pipeline + '/Cuffnorm_' + normalization + '/'
        cond_to_rep_cond_map = generate_cond_name_to_rep_name_map(temp_dir)
        rep_cond_names = []
        for cond in conditions:
            if(cond_to_rep_cond_map[cond] not in rep_cond_names):
                rep_cond_names.append(cond_to_rep_cond_map[cond])
            
        counts = read_cuffnorm_counts2(fpkm_table, rep_cond_names, v_transform)
        gene_names = read_cuffnorm_gene_names(fpkm_table)
        
        try:
            Xs, Xvs, Ys, Yvs, lmap = generate_train_validate_data(2, 2, num_samples,root_dir, pipeline,
                                      normalization, dispersion, counts, gene_names, 'top_genes.txt', taboo_list, 'PB', out_dir,
                                      False,'Cuffnorm', False, True)
            self.assertEqual(True, False)
        except ValueError as e:
            self.assertEqual(e.__str__(), "Generated training and validation data matrices are invalid.")
        
        root_dir = '...'
        pipeline = 'hg38_Starcq_Ensembl'
        normalization = 'GEOM'
        dispersion = 'POOL'
        num_samples = {'AH':2, 'DA':2, 'CT':2, 'AA':1}
        v_transform = True
        taboo_list = []
        
        fpkm_table = root_dir + '/' + pipeline + '/Cuffnorm_' + normalization + '/genes.fpkm_table'
    
        #Translate condtion names (ex: AH) to replicate condition names (ex: q1, q2, etc.)
        conditions = num_samples.keys()
        temp_dir = root_dir +'/' + pipeline + '/Cuffnorm_' + normalization + '/'
        cond_to_rep_cond_map = generate_cond_name_to_rep_name_map(temp_dir)
        rep_cond_names = []
        for cond in conditions:
            if(cond_to_rep_cond_map[cond] not in rep_cond_names):
                rep_cond_names.append(cond_to_rep_cond_map[cond])
            
        counts = read_cuffnorm_counts2(fpkm_table, rep_cond_names, v_transform)
        gene_names = read_cuffnorm_gene_names(fpkm_table)
        
        try:
            Xs, Xvs, Ys, Yvs, lmap = generate_train_validate_data(2, 2, num_samples,root_dir, pipeline,
                                      normalization, dispersion, counts, gene_names, 'top_genes.txt', taboo_list, 'PB', out_dir,
                                      False,'Cuffnorm', False, True)
            self.assertEqual(True, False)
        except ValueError as e:
            self.assertEqual(e.__str__(), "Generated training and validation data matrices are invalid.")
    
    def test_generate_data(self):
        root_dir = '...'
        fold = 1
        X, Y, gene_names = generate_data({'AH':34, 'CT':18}, False, root_dir, 'hg38_Starcq_Ensembl', 'GEOM', 'POOL', fold)
        X_exp = [[7.88391, 3.53608, 0.0420014], [16.7383, 4.14641, 0.0], [17.7691, 4.36101, 0.0126079], [14.1845, 3.58283, 0.0],
                 [42.8423, 1.61298, 0.0757167], [11.9061, 2.06823, 0.00336543], [15.6387, 3.01016, 0.0],
                 [15.7512, 3.32179, 2.72336e-12], [39.7524, 4.28753, 0.00121871], [18.0964, 4.52213, 0.0],
                 [11.5212, 4.53223, 0.00500963], [13.0893, 4.95695, 0.00139864], [24.4613, 2.76009, 0.0],
                 [16.0292, 3.05458, 0.00962138], [5.91922, 1.11481, 0.0103098], [6.58358, 3.09222, 0.0157814],
                 [25.8321, 3.97162, 0.0269552], [39.965, 2.75576, 0.00334643], [22.6631, 4.80301, 0.000868936],
                 [22.3081, 5.74062, 0.0], [19.7136, 2.58082, 0.000863692], [29.6065, 2.70165, 0.0], [7.29916, 1.77918, 0.0240827],
                 [10.6207, 2.53904, 0.0], [15.6721, 4.0199, 0.00154084], [5.35104, 0.765172, 0.0], [19.8884, 4.51623, 0.0],
                 [11.5389, 2.75439, 0.0], [16.1968, 4.19106, 0.00762296], [16.7471, 4.33035, 0.0155117],
                 [15.7639, 5.43479, 0.000838561], [20.9872, 4.25877, 0.00314195], [13.55, 2.41929, 0.00435907],
                 [15.9346, 3.8949, 0.0], [25.1598, 3.23074, 0.0], [11.9264, 3.30717, 0.00240646], [5.65841, 0.947301, 0.00239017],
                 [10.7545, 2.98302, 0.0], [15.8664, 3.87235, 0.0], [17.5241, 4.06875, 0.00709207], [19.997, 3.33069, 0.00367158],
                 [20.8372, 2.34038, 0.00071191], [16.0931, 3.99396, 0.000886564], [23.2805, 4.55551, 0.00350407],
                 [14.6824, 3.46625, 0.00199345], [11.87, 2.03333, 0.000813736], [17.2083, 3.03788, 0.00397897],
                 [24.0494, 3.61774, 0.0], [13.7818, 2.88274, 0.00489945], [44.2072, 5.83818, 0.0], [9.76784, 3.4074, 0.0],
                 [18.0142, 3.49368, 0.0120355]]
        Y_exp = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
        gene_names_exp = ['A1BG', 'A1BG-AS1', 'A1CF']
        
        i = 0
        for row in X:
            j = 0
            for value in row:
                self.assertEqual(value, X_exp[i][j])
                j += 1
            i += 1
            
        i = 0
        for value in Y:
            self.assertEqual(value, Y_exp[i])
            i += 1
            
        self.assertEqual(gene_names, gene_names_exp)
        
        root_dir = '...'
        fold = 1
        X, Y, gene_names = generate_data({'AH':34, 'CT':18}, False, root_dir, 'hg38_Starcq_Ensembl', 'GEOM', 'POOL', fold,
                                         counts_format = 'Cuffdiff', verbose = False)
        X_exp = [[7.88391, 3.53608, 0.0420014], [16.7383, 4.14641, 0.0], [17.7691, 4.36101, 0.0126079], [14.1845, 3.58283, 0.0],
                 [42.8423, 1.61298, 0.0757167], [11.9061, 2.06823, 0.00336543], [15.6387, 3.01016, 0.0],
                 [15.7512, 3.32179, 2.72336e-12], [39.7524, 4.28753, 0.00121871], [18.0964, 4.52213, 0.0],
                 [11.5212, 4.53223, 0.00500963], [13.0893, 4.95695, 0.00139864], [24.4613, 2.76009, 0.0],
                 [16.0292, 3.05458, 0.00962138], [5.91922, 1.11481, 0.0103098], [6.58358, 3.09222, 0.0157814],
                 [25.8321, 3.97162, 0.0269552], [39.965, 2.75576, 0.00334643], [22.6631, 4.80301, 0.000868936],
                 [22.3081, 5.74062, 0.0], [19.7136, 2.58082, 0.000863692], [29.6065, 2.70165, 0.0], [7.29916, 1.77918, 0.0240827],
                 [10.6207, 2.53904, 0.0], [15.6721, 4.0199, 0.00154084], [5.35104, 0.765172, 0.0], [19.8884, 4.51623, 0.0],
                 [11.5389, 2.75439, 0.0], [16.1968, 4.19106, 0.00762296], [16.7471, 4.33035, 0.0155117],
                 [15.7639, 5.43479, 0.000838561], [20.9872, 4.25877, 0.00314195], [13.55, 2.41929, 0.00435907],
                 [15.9346, 3.8949, 0.0], [25.1598, 3.23074, 0.0], [11.9264, 3.30717, 0.00240646], [5.65841, 0.947301, 0.00239017],
                 [10.7545, 2.98302, 0.0], [15.8664, 3.87235, 0.0], [17.5241, 4.06875, 0.00709207], [19.997, 3.33069, 0.00367158],
                 [20.8372, 2.34038, 0.00071191], [16.0931, 3.99396, 0.000886564], [23.2805, 4.55551, 0.00350407],
                 [14.6824, 3.46625, 0.00199345], [11.87, 2.03333, 0.000813736], [17.2083, 3.03788, 0.00397897],
                 [24.0494, 3.61774, 0.0], [13.7818, 2.88274, 0.00489945], [44.2072, 5.83818, 0.0], [9.76784, 3.4074, 0.0],
                 [18.0142, 3.49368, 0.0120355]]
        Y_exp = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
        gene_names_exp = ['A1BG', 'A1BG-AS1', 'A1CF']
        
        i = 0
        for row in X:
            j = 0
            for value in row:
                self.assertEqual(value, X_exp[i][j])
                j += 1
            i += 1
            
        i = 0
        for value in Y:
            self.assertEqual(value, Y_exp[i])
            i += 1
            
        self.assertEqual(gene_names, gene_names_exp)
        
        fold = 0
        X, Y, gene_names = generate_data({'AH':38, 'CT':20}, False, root_dir, 'hg38_Starcq_Ensembl', 'GEOM', 'POOL', fold)
        X_exp = [[14.6789, 2.03791], [7.88078, 3.53468], [42.7616, 1.60994], [39.7925, 4.29185], [24.4567, 2.75958],
                 [25.8266, 3.97078], [19.7149, 2.581], [15.6702, 4.0194], [16.204, 4.19291], [20.9929, 4.25993],
                 [8.94342, 2.41084], [16.7339, 4.14533], [11.9138, 2.06957], [18.0819, 4.51851], [16.0253, 3.05384],
                 [39.9883, 2.75737], [29.6353, 2.70427], [5.35499, 0.765737], [16.7291, 4.32571], [13.541, 2.41768],
                 [26.1356, 3.90262], [17.7748, 4.36241], [15.6411, 3.01062], [11.5132, 4.52909], [5.91017, 1.11311],
                 [22.6779, 4.80615], [7.2918, 1.77738], [19.8688, 4.51178], [15.7617, 5.43401], [15.9127, 3.88955],
                 [25.3871, 5.55418], [14.1706, 3.57932], [15.7372, 3.31884], [13.085, 4.95533], [6.57755, 3.08939],
                 [22.3149, 5.74237], [10.6116, 2.53687], [11.5419, 2.75509], [33.6955, 3.98531], [25.114, 3.22487],
                 [5.65205, 0.946237], [15.8492, 3.86815], [19.9924, 3.32993], [16.1032, 3.99647], [14.6793, 3.46552],
                 [17.1743, 3.03188], [13.7831, 2.883], [9.77743, 3.41074], [7.804, 2.35818], [11.9133, 3.30355], [10.7348, 2.97754],
                 [17.5278, 4.0696], [20.83, 2.33957], [23.272, 4.55385], [11.8702, 2.03336], [24.0221, 3.61363],
                 [44.1985, 5.83703], [18.0178, 3.49439]]
        Y_exp = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
        gene_names_exp = ['A1BG', 'A1BG-AS1']
        
        i = 0
        for row in X:
            j = 0
            for value in row:
                self.assertEqual(value, X_exp[i][j])
                j += 1
            i += 1
            
        i = 0
        for value in Y:
            self.assertEqual(value, Y_exp[i])
            i += 1
        
        self.assertEqual(gene_names, gene_names_exp)
        
        fold = 0
        X, Y, gene_names = generate_data({'AH':38, 'CT':20}, False, root_dir, 'hg38_Starcq_Ensembl', 'GEOM', 'POOL', fold,
                                         counts_format = 'Cuffdiff')
        X_exp = [[14.6789, 2.03791], [7.88078, 3.53468], [42.7616, 1.60994], [39.7925, 4.29185], [24.4567, 2.75958],
                 [25.8266, 3.97078], [19.7149, 2.581], [15.6702, 4.0194], [16.204, 4.19291], [20.9929, 4.25993],
                 [8.94342, 2.41084], [16.7339, 4.14533], [11.9138, 2.06957], [18.0819, 4.51851], [16.0253, 3.05384],
                 [39.9883, 2.75737], [29.6353, 2.70427], [5.35499, 0.765737], [16.7291, 4.32571], [13.541, 2.41768],
                 [26.1356, 3.90262], [17.7748, 4.36241], [15.6411, 3.01062], [11.5132, 4.52909], [5.91017, 1.11311],
                 [22.6779, 4.80615], [7.2918, 1.77738], [19.8688, 4.51178], [15.7617, 5.43401], [15.9127, 3.88955],
                 [25.3871, 5.55418], [14.1706, 3.57932], [15.7372, 3.31884], [13.085, 4.95533], [6.57755, 3.08939],
                 [22.3149, 5.74237], [10.6116, 2.53687], [11.5419, 2.75509], [33.6955, 3.98531], [25.114, 3.22487],
                 [5.65205, 0.946237], [15.8492, 3.86815], [19.9924, 3.32993], [16.1032, 3.99647], [14.6793, 3.46552],
                 [17.1743, 3.03188], [13.7831, 2.883], [9.77743, 3.41074], [7.804, 2.35818], [11.9133, 3.30355], [10.7348, 2.97754],
                 [17.5278, 4.0696], [20.83, 2.33957], [23.272, 4.55385], [11.8702, 2.03336], [24.0221, 3.61363],
                 [44.1985, 5.83703], [18.0178, 3.49439]]
        Y_exp = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
        gene_names_exp = ['A1BG', 'A1BG-AS1']
        
        i = 0
        for row in X:
            j = 0
            for value in row:
                self.assertEqual(value, X_exp[i][j])
                j += 1
            i += 1
            
        i = 0
        for value in Y:
            self.assertEqual(value, Y_exp[i])
            i += 1
        
        self.assertEqual(gene_names, gene_names_exp)
        
        root_dir = '...'
        fold = 1
        X, Y, gene_names = generate_data({'AH':34, 'CT':18, 'DA':18}, False, root_dir, 'hg38_Starcq_Ensembl', 'GEOM', 'POOL', fold)
        X_exp = [[7.91264, 3.54897], [16.7939, 4.16019], [17.8641, 4.38433], [14.27, 3.60441], [42.8787, 1.61435],
                 [12.0017, 2.08484], [15.7012, 3.02218], [15.8251, 3.33739], [39.9908, 4.31324], [18.1683, 4.5401],
                 [11.5686, 4.5509], [13.1511, 4.98034], [24.548, 2.76987], [16.0686, 3.06208], [5.93588, 1.11795],
                 [6.60412, 3.10187], [25.9908, 3.99603], [40.153, 2.76872], [22.8262, 4.83759], [22.4177, 5.76881],
                 [19.7856, 2.59025], [29.7355, 2.71342], [7.35942, 1.79386], [10.6407, 2.54382], [15.71, 4.02961],
                 [5.38309, 0.769754], [20.0, 4.54157], [11.6007, 2.76914], [16.2639, 4.2084], [16.7902, 4.34149],
                 [15.8545, 5.46602], [21.1117, 4.28402], [13.5797, 2.42459], [15.9794, 3.90585], [25.2086, 3.23702],
                 [11.991, 3.32508], [5.68076, 0.951042], [10.7838, 2.99114], [15.9376, 3.88972], [17.5844, 4.08275],
                 [20.0098, 3.33281], [20.8743, 2.34454], [16.2018, 4.02096], [23.365, 4.57206], [14.7853, 3.49054],
                 [11.9182, 2.04159], [17.2668, 3.0482], [24.0633, 3.61982], [13.8103, 2.88869], [44.2252, 5.84055],
                 [9.81285, 3.4231], [18.0966, 3.50966], [10.2617, 2.08565], [18.4093, 4.61017], [28.4229, 2.84776],
                 [13.8388, 3.24302], [18.4871, 4.06747], [23.6749, 1.73114], [22.954, 4.15223], [31.5688, 4.92618],
                 [13.4121, 3.65076], [27.8794, 4.05255], [13.5963, 2.93462], [10.2348, 2.68572], [11.9774, 2.66503],
                 [16.0476, 2.57275], [18.8149, 4.52], [7.07267, 1.61202], [27.4816, 3.75979], [18.4944, 4.18715]]
        Y_exp = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
                 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2]
        gene_names_exp = ['A1BG', 'A1BG-AS1']
        
        i = 0
        for row in X:
            j = 0
            for value in row:
                self.assertEqual(value, X_exp[i][j])
                j += 1
            i += 1
            
        i = 0
        for value in Y:
            self.assertEqual(value, Y_exp[i])
            i += 1
        
        self.assertEqual(gene_names, gene_names_exp)
        
        root_dir = '...'
        fold = 1
        X, Y, gene_names = generate_data({'AH':34, 'CT':18, 'DA':18}, False, root_dir, 'hg38_Starcq_Ensembl', 'GEOM',
                                         'POOL', fold, counts_format = 'Cuffdiff')
        X_exp = [[7.91264, 3.54897], [16.7939, 4.16019], [17.8641, 4.38433], [14.27, 3.60441], [42.8787, 1.61435],
                 [12.0017, 2.08484], [15.7012, 3.02218], [15.8251, 3.33739], [39.9908, 4.31324], [18.1683, 4.5401],
                 [11.5686, 4.5509], [13.1511, 4.98034], [24.548, 2.76987], [16.0686, 3.06208], [5.93588, 1.11795],
                 [6.60412, 3.10187], [25.9908, 3.99603], [40.153, 2.76872], [22.8262, 4.83759], [22.4177, 5.76881],
                 [19.7856, 2.59025], [29.7355, 2.71342], [7.35942, 1.79386], [10.6407, 2.54382], [15.71, 4.02961],
                 [5.38309, 0.769754], [20.0, 4.54157], [11.6007, 2.76914], [16.2639, 4.2084], [16.7902, 4.34149],
                 [15.8545, 5.46602], [21.1117, 4.28402], [13.5797, 2.42459], [15.9794, 3.90585], [25.2086, 3.23702],
                 [11.991, 3.32508], [5.68076, 0.951042], [10.7838, 2.99114], [15.9376, 3.88972], [17.5844, 4.08275],
                 [20.0098, 3.33281], [20.8743, 2.34454], [16.2018, 4.02096], [23.365, 4.57206], [14.7853, 3.49054],
                 [11.9182, 2.04159], [17.2668, 3.0482], [24.0633, 3.61982], [13.8103, 2.88869], [44.2252, 5.84055],
                 [9.81285, 3.4231], [18.0966, 3.50966], [10.2617, 2.08565], [18.4093, 4.61017], [28.4229, 2.84776],
                 [13.8388, 3.24302], [18.4871, 4.06747], [23.6749, 1.73114], [22.954, 4.15223], [31.5688, 4.92618],
                 [13.4121, 3.65076], [27.8794, 4.05255], [13.5963, 2.93462], [10.2348, 2.68572], [11.9774, 2.66503],
                 [16.0476, 2.57275], [18.8149, 4.52], [7.07267, 1.61202], [27.4816, 3.75979], [18.4944, 4.18715]]
        Y_exp = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
                 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2]
        gene_names_exp = ['A1BG', 'A1BG-AS1']
        
        i = 0
        for row in X:
            j = 0
            for value in row:
                self.assertEqual(value, X_exp[i][j])
                j += 1
            i += 1
            
        i = 0
        for value in Y:
            self.assertEqual(value, Y_exp[i])
            i += 1
        
        self.assertEqual(gene_names, gene_names_exp)
        
        root_dir = '...'
        fold = 1
        X, Y, gene_names = generate_data({'AH':34, 'CT':18, 'DA':18}, False, root_dir, 'hg38_Starcq_Ensembl', 'GEOM', 'POOL', fold)
        X_exp = [[7.91264, 3.54897], [16.7939, 4.16019], [17.8641, 4.38433], [14.27, 3.60441], [42.8787, 1.61435],
                 [12.0017, 2.08484], [15.7012, 3.02218], [15.8251, 3.33739], [39.9908, 4.31324], [18.1683, 4.5401],
                 [11.5686, 4.5509], [13.1511, 4.98034], [24.548, 2.76987], [16.0686, 3.06208], [5.93588, 1.11795],
                 [6.60412, 3.10187], [25.9908, 3.99603], [40.153, 2.76872], [22.8262, 4.83759], [22.4177, 5.76881],
                 [19.7856, 2.59025], [29.7355, 2.71342], [7.35942, 1.79386], [10.6407, 2.54382], [15.71, 4.02961],
                 [5.38309, 0.769754], [20.0, 4.54157], [11.6007, 2.76914], [16.2639, 4.2084], [16.7902, 4.34149],
                 [15.8545, 5.46602], [21.1117, 4.28402], [13.5797, 2.42459], [15.9794, 3.90585], [25.2086, 3.23702],
                 [11.991, 3.32508], [5.68076, 0.951042], [10.7838, 2.99114], [15.9376, 3.88972], [17.5844, 4.08275],
                 [20.0098, 3.33281], [20.8743, 2.34454], [16.2018, 4.02096], [23.365, 4.57206], [14.7853, 3.49054],
                 [11.9182, 2.04159], [17.2668, 3.0482], [24.0633, 3.61982], [13.8103, 2.88869], [44.2252, 5.84055],
                 [9.81285, 3.4231], [18.0966, 3.50966], [10.2617, 2.08565], [18.4093, 4.61017], [28.4229, 2.84776],
                 [13.8388, 3.24302], [18.4871, 4.06747], [23.6749, 1.73114], [22.954, 4.15223], [31.5688, 4.92618],
                 [13.4121, 3.65076], [27.8794, 4.05255], [13.5963, 2.93462], [10.2348, 2.68572], [11.9774, 2.66503],
                 [16.0476, 2.57275], [18.8149, 4.52], [7.07267, 1.61202], [27.4816, 3.75979], [18.4944, 4.18715]]
        Y_exp = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
                 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
        gene_names_exp = ['A1BG', 'A1BG-AS1']
        
        i = 0
        for row in X:
            j = 0
            for value in row:
                self.assertEqual(value, X_exp[i][j])
                j += 1
            i += 1
            
        i = 0
        for value in Y:
            self.assertEqual(value, Y_exp[i])
            i += 1
        
        self.assertEqual(gene_names, gene_names_exp)
        
        root_dir = '...'
        fold = 1
        X, Y, gene_names = generate_data({'AH':34, 'CT':18}, False, root_dir, 'hg38_Starcq_Ensembl', 'GEOM', 'POOL', fold)
        X_exp = [[7.91264, 3.54897], [16.7939, 4.16019], [17.8641, 4.38433], [14.27, 3.60441], [42.8787, 1.61435],
                 [12.0017, 2.08484], [15.7012, 3.02218], [15.8251, 3.33739], [39.9908, 4.31324], [18.1683, 4.5401],
                 [11.5686, 4.5509], [13.1511, 4.98034], [24.548, 2.76987], [16.0686, 3.06208], [5.93588, 1.11795],
                 [6.60412, 3.10187], [25.9908, 3.99603], [40.153, 2.76872], [22.8262, 4.83759], [22.4177, 5.76881],
                 [19.7856, 2.59025], [29.7355, 2.71342], [7.35942, 1.79386], [10.6407, 2.54382], [15.71, 4.02961],
                 [5.38309, 0.769754], [20.0, 4.54157], [11.6007, 2.76914], [16.2639, 4.2084], [16.7902, 4.34149],
                 [15.8545, 5.46602], [21.1117, 4.28402], [13.5797, 2.42459], [15.9794, 3.90585], [25.2086, 3.23702],
                 [11.991, 3.32508], [5.68076, 0.951042], [10.7838, 2.99114], [15.9376, 3.88972], [17.5844, 4.08275],
                 [20.0098, 3.33281], [20.8743, 2.34454], [16.2018, 4.02096], [23.365, 4.57206], [14.7853, 3.49054],
                 [11.9182, 2.04159], [17.2668, 3.0482], [24.0633, 3.61982], [13.8103, 2.88869], [44.2252, 5.84055],
                 [9.81285, 3.4231], [18.0966, 3.50966]]
        Y_exp = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
                 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
        gene_names_exp = ['A1BG', 'A1BG-AS1']
        
        i = 0
        for row in X:
            j = 0
            for value in row:
                self.assertEqual(value, X_exp[i][j])
                j += 1
            i += 1
            
        i = 0
        for value in Y:
            self.assertEqual(value, Y_exp[i])
            i += 1
        
        self.assertEqual(gene_names, gene_names_exp)
        
        root_dir = '...'
        fold = 1
        X, Y, gene_names = generate_data({'AH':34, 'CT':18}, False, root_dir, 'hg38_Starcq_Ensembl', 'GEOM', 'POOL',
                                         fold, counts_format = 'Cuffdiff')
        X_exp = [[7.91264, 3.54897], [16.7939, 4.16019], [17.8641, 4.38433], [14.27, 3.60441], [42.8787, 1.61435],
                 [12.0017, 2.08484], [15.7012, 3.02218], [15.8251, 3.33739], [39.9908, 4.31324], [18.1683, 4.5401],
                 [11.5686, 4.5509], [13.1511, 4.98034], [24.548, 2.76987], [16.0686, 3.06208], [5.93588, 1.11795],
                 [6.60412, 3.10187], [25.9908, 3.99603], [40.153, 2.76872], [22.8262, 4.83759], [22.4177, 5.76881],
                 [19.7856, 2.59025], [29.7355, 2.71342], [7.35942, 1.79386], [10.6407, 2.54382], [15.71, 4.02961],
                 [5.38309, 0.769754], [20.0, 4.54157], [11.6007, 2.76914], [16.2639, 4.2084], [16.7902, 4.34149],
                 [15.8545, 5.46602], [21.1117, 4.28402], [13.5797, 2.42459], [15.9794, 3.90585], [25.2086, 3.23702],
                 [11.991, 3.32508], [5.68076, 0.951042], [10.7838, 2.99114], [15.9376, 3.88972], [17.5844, 4.08275],
                 [20.0098, 3.33281], [20.8743, 2.34454], [16.2018, 4.02096], [23.365, 4.57206], [14.7853, 3.49054],
                 [11.9182, 2.04159], [17.2668, 3.0482], [24.0633, 3.61982], [13.8103, 2.88869], [44.2252, 5.84055],
                 [9.81285, 3.4231], [18.0966, 3.50966]]
        Y_exp = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
                 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
        gene_names_exp = ['A1BG', 'A1BG-AS1']
        
        i = 0
        for row in X:
            j = 0
            for value in row:
                self.assertEqual(value, X_exp[i][j])
                j += 1
            i += 1
            
        i = 0
        for value in Y:
            self.assertEqual(value, Y_exp[i])
            i += 1
        
        self.assertEqual(gene_names, gene_names_exp)
        
        root_dir = '...'
        fold = 1
        X, Y, gene_names = generate_data({'AH':34, 'DA':18}, False, root_dir, 'hg38_Starcq_Ensembl', 'GEOM', 'POOL', fold)
        X_exp = [[7.91264, 3.54897], [16.7939, 4.16019], [17.8641, 4.38433], [14.27, 3.60441], [42.8787, 1.61435],
                 [12.0017, 2.08484], [15.7012, 3.02218], [15.8251, 3.33739], [39.9908, 4.31324], [18.1683, 4.5401],
                 [11.5686, 4.5509], [13.1511, 4.98034], [24.548, 2.76987], [16.0686, 3.06208], [5.93588, 1.11795],
                 [6.60412, 3.10187], [25.9908, 3.99603], [40.153, 2.76872], [22.8262, 4.83759], [22.4177, 5.76881],
                 [19.7856, 2.59025], [29.7355, 2.71342], [7.35942, 1.79386], [10.6407, 2.54382], [15.71, 4.02961],
                 [5.38309, 0.769754], [20.0, 4.54157], [11.6007, 2.76914], [16.2639, 4.2084], [16.7902, 4.34149],
                 [15.8545, 5.46602], [21.1117, 4.28402], [13.5797, 2.42459], [15.9794, 3.90585], [10.2617, 2.08565], [18.4093, 4.61017], [28.4229, 2.84776],
                 [13.8388, 3.24302], [18.4871, 4.06747], [23.6749, 1.73114], [22.954, 4.15223], [31.5688, 4.92618],
                 [13.4121, 3.65076], [27.8794, 4.05255], [13.5963, 2.93462], [10.2348, 2.68572], [11.9774, 2.66503],
                 [16.0476, 2.57275], [18.8149, 4.52], [7.07267, 1.61202], [27.4816, 3.75979], [18.4944, 4.18715]]
        Y_exp = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
                 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
        gene_names_exp = ['A1BG', 'A1BG-AS1']
        
        i = 0
        for row in X:
            j = 0
            for value in row:
                self.assertEqual(value, X_exp[i][j])
                j += 1
            i += 1
            
        i = 0
        for value in Y:
            self.assertEqual(value, Y_exp[i])
            i += 1
        
        self.assertEqual(gene_names, gene_names_exp)
        
        root_dir = '...'
        fold = 1
        X, Y, gene_names = generate_data({'AH':34, 'DA':18}, False, root_dir, 'hg38_Starcq_Ensembl', 'GEOM', 'POOL',
                                         fold, counts_format = 'Cuffdiff')
        X_exp = [[7.91264, 3.54897], [16.7939, 4.16019], [17.8641, 4.38433], [14.27, 3.60441], [42.8787, 1.61435],
                 [12.0017, 2.08484], [15.7012, 3.02218], [15.8251, 3.33739], [39.9908, 4.31324], [18.1683, 4.5401],
                 [11.5686, 4.5509], [13.1511, 4.98034], [24.548, 2.76987], [16.0686, 3.06208], [5.93588, 1.11795],
                 [6.60412, 3.10187], [25.9908, 3.99603], [40.153, 2.76872], [22.8262, 4.83759], [22.4177, 5.76881],
                 [19.7856, 2.59025], [29.7355, 2.71342], [7.35942, 1.79386], [10.6407, 2.54382], [15.71, 4.02961],
                 [5.38309, 0.769754], [20.0, 4.54157], [11.6007, 2.76914], [16.2639, 4.2084], [16.7902, 4.34149],
                 [15.8545, 5.46602], [21.1117, 4.28402], [13.5797, 2.42459], [15.9794, 3.90585], [10.2617, 2.08565], [18.4093, 4.61017], [28.4229, 2.84776],
                 [13.8388, 3.24302], [18.4871, 4.06747], [23.6749, 1.73114], [22.954, 4.15223], [31.5688, 4.92618],
                 [13.4121, 3.65076], [27.8794, 4.05255], [13.5963, 2.93462], [10.2348, 2.68572], [11.9774, 2.66503],
                 [16.0476, 2.57275], [18.8149, 4.52], [7.07267, 1.61202], [27.4816, 3.75979], [18.4944, 4.18715]]
        Y_exp = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
                 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
        gene_names_exp = ['A1BG', 'A1BG-AS1']
        
        i = 0
        for row in X:
            j = 0
            for value in row:
                self.assertEqual(value, X_exp[i][j])
                j += 1
            i += 1
            
        i = 0
        for value in Y:
            self.assertEqual(value, Y_exp[i])
            i += 1
        
        self.assertEqual(gene_names, gene_names_exp)
        
        root_dir = '...'
        fold = 1
        X, Y, gene_names = generate_data({'AH':34, 'DA':18}, True, root_dir, 'hg38_Starcq_Ensembl', 'GEOM','POOL', fold)
        X_exp = [[7.91264, 3.54897], [16.7939, 4.16019], [17.8641, 4.38433], [14.27, 3.60441], [42.8787, 1.61435],
                 [12.0017, 2.08484], [15.7012, 3.02218], [15.8251, 3.33739], [39.9908, 4.31324], [18.1683, 4.5401],
                 [11.5686, 4.5509], [13.1511, 4.98034], [24.548, 2.76987], [16.0686, 3.06208], [5.93588, 1.11795],
                 [6.60412, 3.10187], [25.9908, 3.99603], [40.153, 2.76872], [22.8262, 4.83759], [22.4177, 5.76881],
                 [19.7856, 2.59025], [29.7355, 2.71342], [7.35942, 1.79386], [10.6407, 2.54382], [15.71, 4.02961],
                 [5.38309, 0.769754], [20.0, 4.54157], [11.6007, 2.76914], [16.2639, 4.2084], [16.7902, 4.34149],
                 [15.8545, 5.46602], [21.1117, 4.28402], [13.5797, 2.42459], [15.9794, 3.90585], [10.2617, 2.08565], [18.4093, 4.61017], [28.4229, 2.84776],
                 [13.8388, 3.24302], [18.4871, 4.06747], [23.6749, 1.73114], [22.954, 4.15223], [31.5688, 4.92618],
                 [13.4121, 3.65076], [27.8794, 4.05255], [13.5963, 2.93462], [10.2348, 2.68572], [11.9774, 2.66503],
                 [16.0476, 2.57275], [18.8149, 4.52], [7.07267, 1.61202], [27.4816, 3.75979], [18.4944, 4.18715]]
        i = 0
        while i < len(X_exp):
            X_exp[i][0] = math.log(1 + X_exp[i][0])
            X_exp[i][1] = math.log(1 + X_exp[i][1])
            i += 1
        Y_exp = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
                 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
        gene_names_exp = ['A1BG', 'A1BG-AS1']
        
        i = 0
        for row in X:
            j = 0
            for value in row:
                self.assertAlmostEqual(value, X_exp[i][j])
                j += 1
            i += 1
            
        i = 0
        for value in Y:
            self.assertEqual(value, Y_exp[i])
            i += 1
        
        self.assertEqual(gene_names, gene_names_exp)
        
        root_dir = '...'
        fold = 1
        X, Y, gene_names = generate_data({'AH':34, 'DA':18}, True, root_dir, 'hg38_Starcq_Ensembl', 'GEOM','POOL', 
                                         fold, counts_format = 'Cuffdiff')
        X_exp = [[7.91264, 3.54897], [16.7939, 4.16019], [17.8641, 4.38433], [14.27, 3.60441], [42.8787, 1.61435],
                 [12.0017, 2.08484], [15.7012, 3.02218], [15.8251, 3.33739], [39.9908, 4.31324], [18.1683, 4.5401],
                 [11.5686, 4.5509], [13.1511, 4.98034], [24.548, 2.76987], [16.0686, 3.06208], [5.93588, 1.11795],
                 [6.60412, 3.10187], [25.9908, 3.99603], [40.153, 2.76872], [22.8262, 4.83759], [22.4177, 5.76881],
                 [19.7856, 2.59025], [29.7355, 2.71342], [7.35942, 1.79386], [10.6407, 2.54382], [15.71, 4.02961],
                 [5.38309, 0.769754], [20.0, 4.54157], [11.6007, 2.76914], [16.2639, 4.2084], [16.7902, 4.34149],
                 [15.8545, 5.46602], [21.1117, 4.28402], [13.5797, 2.42459], [15.9794, 3.90585], [10.2617, 2.08565], [18.4093, 4.61017], [28.4229, 2.84776],
                 [13.8388, 3.24302], [18.4871, 4.06747], [23.6749, 1.73114], [22.954, 4.15223], [31.5688, 4.92618],
                 [13.4121, 3.65076], [27.8794, 4.05255], [13.5963, 2.93462], [10.2348, 2.68572], [11.9774, 2.66503],
                 [16.0476, 2.57275], [18.8149, 4.52], [7.07267, 1.61202], [27.4816, 3.75979], [18.4944, 4.18715]]
        i = 0
        while i < len(X_exp):
            X_exp[i][0] = math.log(1 + X_exp[i][0])
            X_exp[i][1] = math.log(1 + X_exp[i][1])
            i += 1
        Y_exp = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
                 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
        gene_names_exp = ['A1BG', 'A1BG-AS1']
        
        i = 0
        for row in X:
            j = 0
            for value in row:
                self.assertAlmostEqual(value, X_exp[i][j])
                j += 1
            i += 1
            
        i = 0
        for value in Y:
            self.assertEqual(value, Y_exp[i])
            i += 1
        
        self.assertEqual(gene_names, gene_names_exp)
        
        root_dir = '...'
        fold = 1
        try:
            X, Y, gene_names = generate_data({'AH':40, 'DA':20}, True, root_dir, 'hg38_Starcq_Ensembl', 'GEOM', 'POOL', fold)
            self.assertEqual(True, False)
        except ValueError as e:
            self.assertEqual(e.__str__(), 'Error in data generation process.')
            
        root_dir = '...'
        fold = 1
        try:
            X, Y, gene_names = generate_data({'AH':34, 'DA':18}, True, root_dir, 'hg38_Starcq_Ensembl', 'GEOM','POOL', 
                                             fold, counts_format = 'Cuff')
            self.assertEqual(True, False)
        except ValueError as e:
            self.assertEqual(e.__str__(), 'Counts format should be either Cuffnorm or Cuffdiff.')
            
        root_dir = '...'
        fold = 0
        try:
            X, Y, gene_names = generate_data({'AH':2, 'DA':2}, True, root_dir, 'hg38_Starcq_Ensembl', 'GEOM', 'POOL', fold)
            self.assertEqual(True, False)
        except ValueError as e:
            self.assertEqual(e.__str__(), 'This filename belongs to an unkown conditions. Only AH, AC, CT, DA, AA, NF, and HP are supported.')
            
        root_dir = '...'
        fold = 0
        try:
            X, Y, gene_names = generate_data({'AH':2, 'CT':2}, True, root_dir, 'hg38_Starcq_Ensembl', 'GEOM', 'POOL', fold)
            self.assertEqual(True, False)
        except ValueError as e:
            self.assertEqual(e.__str__(), 'This filename: '...' is not valid.')
        
        root_dir = '...'
        fold = 0
        try:
            X, Y, gene_names = generate_data({'AH':2, 'CT':2, 'DA':2}, True, root_dir, 'hg38_Starcq_Ensembl', 'GEOM', 'POOL', fold)
            self.assertEqual(True, False)
        except ValueError as e:
            print(e)
    
    def test_tune_ML_model(self):
        num_folds = 10
        num_runs = 2
        num_samples = {'AH':34, 'CT':18, 'DA':18}
        hyper_param_names = ['C', 'class_weight', 'solver']
        hyper_param_values = [[0.5, 1.0], [None, 'balanced'], ['newton-cg', 'lbfgs', 'liblinear']]
        classifier = LogisticRegression
        
        # Read in data matrices
        X_trains, X_validates, Y_trains, Y_validates = [], [], [], []
        root_dir = '...'
        filename1 = root_dir + 'X_Trains.txt'
        filename2 = root_dir + 'X_Validates.txt'
        filename3 = root_dir + 'Y_Trains.txt'
        filename4 = root_dir + 'Y_Validates.txt'
        with open(filename1) as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=',')
            sub_matrix = []
            i = 0
            for row in csv_reader:
                double_row = []
                for value in row:
                    double_row.append(float(value))
                if(len(row) > 0):
                    sub_matrix.append(double_row)
                else:
                    X_trains.append(np.array(sub_matrix))
                    i += 1
                    sub_matrix = []
                    if(i == num_folds):
                        break
                    
        with open(filename2) as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=',')
            sub_matrix = []
            i = 0
            for row in csv_reader:
                double_row = []
                for value in row:
                    double_row.append(float(value))
                if(len(row) > 0):
                    sub_matrix.append(double_row)
                else:
                    X_validates.append(np.array(sub_matrix))
                    i += 1
                    sub_matrix = []
                    if(i == num_folds):
                        break
                    
        with open(filename3) as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=',')
            sub_matrix = []
            i = 0
            for row in csv_reader:
                for value in row:
                    sub_matrix.append(int(value))
                if(len(row) == 0):
                    Y_trains.append(np.array(sub_matrix).reshape(len(sub_matrix),))
                    i += 1
                    sub_matrix = []
                    if(i == num_folds):
                        break
                    
        with open(filename4) as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=',')
            sub_matrix = []
            i = 0
            for row in csv_reader:
                for value in row:
                    sub_matrix.append(int(value))
                if(len(row) == 0):
                    Y_validates.append(np.array(sub_matrix).reshape(len(sub_matrix), ))
                    i += 1
                    sub_matrix = []
                    if(i == num_folds):
                        break
        
        out_dir = '...'
        perf_metrics, y_trues, probs = tune_ML_model(X_trains, X_validates, Y_trains, Y_validates, num_folds,
                                                     num_runs, num_samples, hyper_param_names, hyper_param_values,
                                                     classifier, out_dir, 'Grid', None, None, None, False, False, True)
        self.assertAlmostEqual(perf_metrics['Accuracy'], 0.5128, 4)
        exp_precision_array = np.array([0.73076923, 0.44444444, 0.36])
        exp_recall_array = np.array([0.5, 0.6, 0.45])
        exp_conf_matrix = np.array([[19, 8, 11],[3, 12, 5],[4, 7, 9]])
        i = 0
        for v in exp_precision_array:
            self.assertAlmostEqual(v, perf_metrics['Precision'][i], 4)
            i += 1
            
        i = 0
        for v in exp_recall_array:
            self.assertAlmostEqual(v, perf_metrics['Recall'][i], 4)
            i += 1
            
        r = 0
        for row in exp_conf_matrix:
            c = 0
            for val in row:
                self.assertEqual(val, perf_metrics['Confusion_Matrix'][r][c])
                c += 1
            r += 1
    
    def test_generate_mean_ROC_curves(self):
        # Skipping unit testing this function for now. Strong correlation between accuracy and ROC curves is an implicit measure 
        # of this functions current correctness.
        pass
    
    def test_compare_cuffnorm_cuffnorm(self):
        root_dir = '...'
        file1 = root_dir + 'genes_cond3_f1_mini.fpkm_table'
        file2 = root_dir + 'genes_cond3_f2_mini.fpkm_table'
        inc = compare_cuffnorm_cuffnorm(file1, file2, 0.2)
        self.assertAlmostEqual(inc, 11.1111111)
    
    def test_generate_variance_mean_plots(self):
        root_dir = '...'
        file1 = root_dir + 'genes_cond3_mini.fpkm_table'
        out_dir = '...'
        cond_means, cond_stds = generate_variance_mean_plots(file1, out_dir, False, 'Cuffnorm')
        cond_means_exp = {'q1': [17.954848529411766, 3.468998647058824, 0.00785615085302165, 0.0, 22.250279411764712,
                                 3.4448764705882358], 'q2': [17.867533888888893, 3.3671817777777777, 0.0024749464444444444,
                                 0.0, 24.455044444444447, 3.7594588888888887], 'q3': [18.47937055555555, 3.350222777777778,
                                 0.00701328611111111, 0.0, 22.64237222222222, 3.6898633333333333]}
        cond_stds_exp = {'q1': [9.091050321175324, 1.1877061573798682, 0.015091255915282297, 0.0, 2.8126535202837126,
                                0.7352095329339488], 'q2': [8.155824288573125, 1.0004718492204654, 0.0030740117949021564,
                                0.0, 3.4138577976343916, 0.5169960536158096], 'q3': [6.9586187312221925, 0.9787478087150471,
                                0.009179822863751625, 0.0, 4.001234179295399, 0.5907526635102412]}
        self.assertEqual(cond_means, cond_means_exp)
        self.assertEqual(cond_stds, cond_stds_exp)
    
    def test_count_nonmRNAs(self):
        a = ['ABC', 'DEFG', 'LARK', 'DARK', 'SNOR', 'SNORA', 'MIR', 'LARK.K']
        self.assertEqual(count_nonmRNAs(a), 4)
    
    def test_read_in_csv_file_one_column(self):
        root_dir = '...'
        filename = root_dir + 'hg38_Starcq_Ensembl.csv'
        data = read_in_csv_file_one_column(filename, 0, ',')
        self.assertEqual(data, ['Features', '10', '30', '50', '100', '150', '200', '250'])
        data2 = read_in_csv_file_one_column(filename, 1, ',')
        self.assertEqual(data2, ['log reg', '82.0', '96.33333333333334', '100.0', '98.00000000000001', '100.0', '100.0', '100.0'])
    
    def test_compare_classification_accuracy_all_pipelines(self):
        # Looks like a one time script, will not test for now.
        pass
    
    def test_calculate_pipeline_accuracy_metrics(self):
        root_dir = '...'
        filename = root_dir + 'hg38_Starcq_Ensembl.csv'
        pipeline_avg, point_totals, model_avgs_alt = calculate_pipeline_accuracy_metrics(filename)
        self.assertEqual(pipeline_avg, 96.5)
        self.assertEqual(point_totals, [4, 4, 0, 3, 7])
        self.assertEqual(model_avgs_alt, [96.61904761904762, 97.0, 94.47619047619048, 96.23809523809526, 98.19047619047619])
    
    def test_generate_cuffnorm_or_cuffdiff_batch_file_HPC(self):
        # First 3 tests examine returned output of the function, while the 4th test examines the output files.
        
        # Verify that cxbs listed in each file correctly represent their perspective fold.
        conditions = ['AH', 'CT']
        Cond_Folds_Dict = {}
        for condition in conditions:
            Cond_Folds = generate_kfolds(cond_samples[condition], 10)
            Cond_Folds_Dict[condition] = Cond_Folds
        
        for k,v in Cond_Folds_Dict.items():
            for sublist in v:
                i = 0
                while i < len(sublist):
                    sublist[i] = sublist[i] + '.hg19_tuxedo_refflat.geneexp.cxb'
                    i += 1
        
        fold = 0
        while fold < 10:
            cond_train_fold_filenames = {'AH':[], 'CT':[], 'DA':[], 'AA':[], 'NF':[], 'HP':[]}     
            for key, folded_filenames in Cond_Folds_Dict.items():
                i = 0
                for sublist in folded_filenames:
                    if(i == fold):
                        pass
                    else:
                        for filename in sublist:
                            cond_train_fold_filenames[key].append(filename)
                    i += 1
                    
            for cond in conditions:
                cond_train_fold_filenames[cond].sort()
                
            root_dir = '...'
            file = root_dir + 'FOLD' +str(fold+1) + '.sh'
            with open(file) as reader:
                line = reader.readlines()[13]
                line = line.split(' ')
                AH_train_filenames_fold = line[10].split(',')
                AH_train_filenames_fold.sort()
                CT_train_filenames_fold = line[11].split(',')
                CT_train_filenames_fold.sort()
            
            self.assertEqual(cond_train_fold_filenames['AH'], AH_train_filenames_fold)
            self.assertEqual(cond_train_fold_filenames['CT'], CT_train_filenames_fold)
            
            fold += 1
            
        Cond_Folds_Dict = generate_cuffnorm_or_cuffdiff_batch_file_HPC('hg38', 'starcq', 'ensembl', 'GEOM',
                                                                        ['AH', 'CT', 'DA'],
                                                                        'Cuffnorm', 2, '', 'TEST')
        
        conditions = ['AH', 'CT', 'DA']
        cond_train_filenames_fold = {'AH': [], 'CT': [], 'DA': []}
        
        fold = 0
        while fold < 2:
            root_dir = '...'
            file = root_dir + 'FOLD' + str(fold+1) + 'Test.sh'
            with open(file) as reader:
                line = reader.readlines()[13]
                line = line.split(' ')
                cond_train_filenames_fold['AH'] = line[8].split(',')
                cond_train_filenames_fold['AH'].sort()
                cond_train_filenames_fold['CT'] = line[9].split(',')
                cond_train_filenames_fold['CT'].sort()
                cond_train_filenames_fold['DA'] = line[10].split(',')
                cond_train_filenames_fold['DA'].sort()
                
            for cond in conditions:
                temp = []
                i = 0
                while i < 2:
                    if(i != fold):
                        temp2 = Cond_Folds_Dict[cond][i]
                        for sample_name in temp2:
                            temp.append(sample_name + '.hg38_starcq_ensembl.geneexp.cxb')
                    i += 1
                temp.sort()
                # print('temp:', temp)
                # print('read in: ', cond_train_filenames_fold[cond])
                self.assertEqual(cond_train_filenames_fold[cond], temp)
            
            fold += 1
        # A test that directly inspects the written file.
        
        out_files = ['SL_4AH_vs_3CT_vs_2DA_LV_hg38_Starcq_Ensembl_GEOM_FOLD1.sh',
                     'SL_4AH_vs_3CT_vs_2DA_LV_hg38_Starcq_Ensembl_GEOM_FOLD2.sh']
        
        Cond_Folds_Dict = generate_cuffnorm_or_cuffdiff_batch_file_HPC('hg38', 'starcq', 'ensembl', 'GEOM',
                                                                        ['AH', 'CT', 'DA'],
                                                                        'Cuffnorm', 2, '', 'TEST')
        
        conditions = ['AH', 'CT', 'DA']
        cond_train_filenames_fold_exp = {'AH': [], 'CT': [], 'DA': []}
        cond_train_filenames_fold_read = {'AH': [], 'CT': [], 'DA': []}
        
        fold = 0
        root_dir = '...'
        while fold < 2:
            file = root_dir + 'FOLD' + str(fold+1) + 'Test.sh'
            with open(file) as reader:
                line = reader.readlines()[13]
                line = line.split(' ')
                cond_train_filenames_fold_exp['AH'] = line[8].split(',')
                cond_train_filenames_fold_exp['AH'].sort()
                cond_train_filenames_fold_exp['CT'] = line[9].split(',')
                cond_train_filenames_fold_exp['CT'].sort()
                cond_train_filenames_fold_exp['DA'] = line[10].split(',')
                cond_train_filenames_fold_exp['DA'].sort()
            
            fold += 1
        
        fold = 0
        root_dir = '...'
        while fold < 2:
            file = root_dir + out_files[fold]
            with open(file) as reader:
                line = reader.readlines()[17]
                line = line.split(' ')
                cond_train_filenames_fold_read['AH'] = line[8].split(',')
                cond_train_filenames_fold_read['AH'].sort()
                cond_train_filenames_fold_read['CT'] = line[9].split(',')
                cond_train_filenames_fold_read['CT'].sort()
                cond_train_filenames_fold_read['DA'] = line[10].split(',')
                cond_train_filenames_fold_read['DA'].sort()
            
            fold += 1
            
        self.assertEqual(cond_train_filenames_fold_read['AH'], cond_train_filenames_fold_exp['AH'])
        self.assertEqual(cond_train_filenames_fold_read['CT'], cond_train_filenames_fold_exp['CT'])
        self.assertEqual(cond_train_filenames_fold_read['DA'], cond_train_filenames_fold_exp['DA'])
    
    def test_filter_cuffnorm_counts(self):
        root_dir = '...'
        file1 = root_dir + 'genes_cond3_f1_mini.fpkm_table'
        file2 = root_dir + 'genes_cond3_f1_mini2.fpkm_table'
        filtered_counts, eliminated_genes = filter_cuffnorm_counts(file1)
        self.assertEqual(eliminated_genes, ['5S_rRNA', '5_8S_rRNA', '7SK', 'A2ML1-AS1', 'A2ML1-AS2', 'AA06'])
        filtered_counts, eliminated_genes = filter_cuffnorm_counts(file2)
        self.assertEqual(eliminated_genes, ['5S_rRNA', '5_8S_rRNA', '7SK', 'A2ML1-AS2', 'AA06'])
        
    def test_generate_cond_name_to_rep_name_map(self):
        file_dir = '...'
        file_option = 0
        result = generate_cond_name_to_rep_name_map(file_dir, file_option)
        self.assertEqual(result, {'AH':'q1', 'CT':'q2', 'DA':'q3'})
        
        file_option = 1
        result = generate_cond_name_to_rep_name_map(file_dir, file_option)
        self.assertEqual(result, {'AH':'q1', 'CT':'q2', 'DA':'q3'})
        
    def test_identify_misclassified_samples(self):
        exp_dir = '...'
        perf_fname = exp_dir + 'Performance_PB2Way.txt'
        out_dir = '...'
        identify_misclassified_samples(perf_fname, out_dir)
        exp_file = exp_dir + 'miss_samples_PB2Way.txt'
        out_file = out_dir + 'miss_samples_TestOutput.txt'
        with open(exp_file) as reader:
            lines_exp = reader.readlines()
        with open(out_file) as reader:
            lines_out = reader.readlines()
        self.assertEqual(len(lines_out), len(lines_exp))
        z = 0
        for line in lines_out:
            self.assertEqual(line, lines_exp[z])
            z += 1
        
    # Full System Black Box Tests    
    
    def test_rnaseq_classification_with_feature_selection_binary(self):
        
        # Test grid vs random search
        root_dir = '...'
        out_dir = '...'
        rnaseq_classification_with_feature_selection({'AH':38, 'CT':20}, ['AH', 'CT'], 'hg38', 'starcq', 'ensembl',
                                                     'GEOM', 'POOL', root_dir, 'top_IG_genes.txt', out_dir, 
                                                     [10], ['log reg'], num_runs = 1, search_heuristic = 'Grid')
        
        log_file = out_dir + 'Performance_PB_Grid_Log_Reg.txt'
        lines_exp = ["Config:  (0.5, None, 'newton-cg')\n",
                     "Config:  (0.5, None, 'lbfgs')\n",
                     "Config:  (0.5, None, 'liblinear')\n",
                     "Config:  (0.5, None, 'saga')\n",
                     "Config:  (0.5, 'balanced', 'newton-cg')\n",
                     "Config:  (0.5, 'balanced', 'lbfgs')\n",
                     "Config:  (0.5, 'balanced', 'liblinear')\n",
                     "Config:  (0.5, 'balanced', 'saga')\n",
                     "Config:  (1.0, None, 'newton-cg')\n",
                     "Config:  (1.0, None, 'lbfgs')\n",
                     "Config:  (1.0, None, 'liblinear')\n",
                     "Config:  (1.0, None, 'saga')\n",
                     "Config:  (1.0, 'balanced', 'newton-cg')\n",
                     "Config:  (1.0, 'balanced', 'lbfgs')\n",
                     "Config:  (1.0, 'balanced', 'liblinear')\n",
                     "Config:  (1.0, 'balanced', 'saga')\n",
                     "Config:  (2.0, None, 'newton-cg')\n",
                     "Config:  (2.0, None, 'lbfgs')\n",
                     "Config:  (2.0, None, 'liblinear')\n",
                     "Config:  (2.0, None, 'saga')\n",
                     "Config:  (2.0, 'balanced', 'newton-cg')\n",
                     "Config:  (2.0, 'balanced', 'lbfgs')\n",
                     "Config:  (2.0, 'balanced', 'liblinear')\n",
                     "Config:  (2.0, 'balanced', 'saga')\n",
                     "Config:  (3.0, None, 'newton-cg')\n",
                     "Config:  (3.0, None, 'lbfgs')\n",
                     "Config:  (3.0, None, 'liblinear')\n",
                     "Config:  (3.0, None, 'saga')\n",
                     "Config:  (3.0, 'balanced', 'newton-cg')\n",
                     "Config:  (3.0, 'balanced', 'lbfgs')\n",
                     "Config:  (3.0, 'balanced', 'liblinear')\n",
                     "Config:  (3.0, 'balanced', 'saga')\n",
                     "Config:  (4.0, None, 'newton-cg')\n",
                     "Config:  (4.0, None, 'lbfgs')\n",
                     "Config:  (4.0, None, 'liblinear')\n",
                     "Config:  (4.0, None, 'saga')\n",
                     "Config:  (4.0, 'balanced', 'newton-cg')\n",
                     "Config:  (4.0, 'balanced', 'lbfgs')\n",
                     "Config:  (4.0, 'balanced', 'liblinear')\n",
                     "Config:  (4.0, 'balanced', 'saga')\n",
                     "Config:  (5.0, None, 'newton-cg')\n",
                     "Config:  (5.0, None, 'lbfgs')\n",
                     "Config:  (5.0, None, 'liblinear')\n",
                     "Config:  (5.0, None, 'saga')\n",
                     "Config:  (5.0, 'balanced', 'newton-cg')\n",
                     "Config:  (5.0, 'balanced', 'lbfgs')\n",
                     "Config:  (5.0, 'balanced', 'liblinear')\n",
                     "Config:  (5.0, 'balanced', 'saga')\n"]
        
        lines_out = []
        with open(log_file) as reader:
            lines = reader.readlines()
        for line in lines:
            if ('Config:' in line):
                lines_out.append(line)
        self.assertEqual(lines_exp, lines_out)
        
        out_dir = '...'
        rnaseq_classification_with_feature_selection({'AH':38, 'CT':20}, ['AH', 'CT'], 'hg38', 'starcq', 'ensembl',
                                                     'GEOM', 'POOL', root_dir, 'top_IG_genes.txt', out_dir, 
                                                     [10], ['log reg'], num_runs = 1, search_heuristic = 'Random')
        
        log_file = out_dir + 'Performance_PB_Random_Log_Reg.txt'
        lines_out = []
        with open(log_file) as reader:
            lines = reader.readlines()
        for line in lines:
            if ('Config:' in line):
                lines_out.append(line)
        i = 0
        tuples_out = []
        while i < len(lines_out):
            temp1 = lines_out[i].index('(')
            temp2 = lines_out[i][temp1:]
            temp3 = temp2.replace('(', '')
            temp3 = temp3.replace(')', '')
            temp3 = temp3.replace('\n', '')
            temp3 = temp3.replace("'", '')
            temp4 = temp3.split(',')
            temp = []
            for l in temp4:
                l = l.replace(' ', '')
                try:
                    l = float(l)
                except:
                    pass
                temp.append(l)
            tuples_out.append(temp)
            i += 1
        
        for t in tuples_out:
            self.assertTrue(t[0] in [0.1, 0.5, 1.0, 2.0, 3.0, 4.0, 5.0, 10.0])
            self.assertTrue(t[1] in ['None', 'balanced'])
            self.assertTrue(t[2] in ['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga'])
            self.assertTrue(t[3] in [0.0001, 0.00001, 0.001])
            self.assertTrue(t[4] in [75, 100, 150])
        self.assertTrue(len(tuples_out) < (8*2*5*3*3))
        
        out_dir = '...'
        rnaseq_classification_with_feature_selection({'AH':38, 'CT':20}, ['AH', 'CT'], 'hg38', 'starcq', 'ensembl',
                                                     'GEOM', 'POOL', root_dir, 'top_IG_genes.txt', out_dir, 
                                                     [10], ['kNN'], num_runs = 1, search_heuristic = 'Grid')
        log_file = out_dir + 'Performance_PB_Grid_kNN.txt'
        
        lines_exp = ["Config:  (3, 'uniform', 1)\n",
                     "Config:  (3, 'uniform', 1.5)\n",
                     "Config:  (3, 'uniform', 2.0)\n",
                     "Config:  (3, 'distance', 1)\n",
                     "Config:  (3, 'distance', 1.5)\n",
                     "Config:  (3, 'distance', 2.0)\n",
                     "Config:  (5, 'uniform', 1)\n",
                     "Config:  (5, 'uniform', 1.5)\n",
                     "Config:  (5, 'uniform', 2.0)\n",
                     "Config:  (5, 'distance', 1)\n",
                     "Config:  (5, 'distance', 1.5)\n",
                     "Config:  (5, 'distance', 2.0)\n",
                     "Config:  (7, 'uniform', 1)\n",
                     "Config:  (7, 'uniform', 1.5)\n",
                     "Config:  (7, 'uniform', 2.0)\n",
                     "Config:  (7, 'distance', 1)\n",
                     "Config:  (7, 'distance', 1.5)\n",
                     "Config:  (7, 'distance', 2.0)\n",
                     "Config:  (9, 'uniform', 1)\n",
                     "Config:  (9, 'uniform', 1.5)\n",
                     "Config:  (9, 'uniform', 2.0)\n",
                     "Config:  (9, 'distance', 1)\n",
                     "Config:  (9, 'distance', 1.5)\n",
                     "Config:  (9, 'distance', 2.0)\n",
                     "Config:  (15, 'uniform', 1)\n",
                     "Config:  (15, 'uniform', 1.5)\n",
                     "Config:  (15, 'uniform', 2.0)\n",
                     "Config:  (15, 'distance', 1)\n",
                     "Config:  (15, 'distance', 1.5)\n",
                     "Config:  (15, 'distance', 2.0)\n"]
        
        lines_out = []
        with open(log_file) as reader:
            lines = reader.readlines()
        for line in lines:
            if ('Config:' in line):
                lines_out.append(line)
        self.assertEqual(lines_exp, lines_out)
        
        out_dir = '...'
        rnaseq_classification_with_feature_selection({'AH':38, 'CT':20}, ['AH', 'CT'], 'hg38', 'starcq', 'ensembl',
                                                     'GEOM', 'POOL', root_dir, 'top_IG_genes.txt', out_dir, 
                                                     [10], ['kNN'], num_runs = 1, search_heuristic = 'Random')
        
        log_file = out_dir + 'Performance_PB_Random_kNN.txt'
        lines_out = []
        with open(log_file) as reader:
            lines = reader.readlines()
        for line in lines:
            if ('Config:' in line):
                lines_out.append(line)
        i = 0
        tuples_out = []
        while i < len(lines_out):
            temp1 = lines_out[i].index('(')
            temp2 = lines_out[i][temp1:]
            temp3 = temp2.replace('(', '')
            temp3 = temp3.replace(')', '')
            temp3 = temp3.replace('\n', '')
            temp3 = temp3.replace("'", '')
            temp4 = temp3.split(',')
            temp = []
            for l in temp4:
                l = l.replace(' ', '')
                try:
                    l = float(l)
                except:
                    pass
                temp.append(l)
            tuples_out.append(temp)
            i += 1
        
        for t in tuples_out:
            self.assertTrue(t[0] in [3,5,7,9,15,21])
            self.assertTrue(t[1] in ['uniform', 'distance'])
            self.assertTrue(t[2] in ['ball_tree', 'kd_tree', 'brute'])
            self.assertTrue(t[3] in [1, 1.25, 1.5, 1.75, 2.0])
        self.assertTrue(len(tuples_out) < (6*2*3*5))
        
        # Test all of the pipeline outputs
        out_dir = '...'
        exp_dir = '...'
        root_dir = '...'
        reference = 'hg38'
        annotation = 'ensembl'
        aligner = 'starcq'
        normalization = 'GEOM'
        dispersion = 'POOL'
        features_file = 'top_DE_W-MEAN_genes.txt'
        run_name = 'PB2Way'
        num_folds = 10
        num_runs = 1
        perf_metrics = rnaseq_classification_with_feature_selection({'AH':38, 'CT':20}, ['AH', 'CT'], reference,
                                                               aligner, annotation, normalization, dispersion, 
                                                               root_dir, features_file, out_dir, [10, 30, 50, 100],
                                                               ["log reg", "kNN", "GNB", "SVM"], num_folds, num_runs, 
                                                               gene_lists = True, validate = False)
        
        # Ideally we'd also want to verify confusion matrix plots, counts heatmaps, and ROC curves.
        
        # Verify All The Gene Lists
        i = 0 
        while i < num_folds:
            dir_fold = root_dir + reference + '_' + aligner.capitalize() + '_' + annotation.capitalize() + '/'
            dir_fold += 'FOLD' + str(num_folds) + '/Cuffnorm_' + normalization + '_FOLD' + str(i+1) + '/'
            features_filepath = dir_fold + features_file
            
            gene_list_filepath = out_dir + 'GeneLists/' + 'gene_list_fold' + str(i+1) + '_' + run_name + '.txt'
            
            with open(gene_list_filepath) as reader:
                lines_out = reader.readlines()
            
            with open(features_filepath) as reader:
                lines_exp = reader.readlines()
            
            j = 0
            while j < len(lines_out):
                self.assertEqual(lines_out[j], lines_exp[j])
                j += 1
            
            i += 1
        
        # Verify Config File
        config_file_out = out_dir + 'config_PB2Way.txt'
        with open(config_file_out) as reader:
            lines_out = reader.readlines()
        self.assertEqual(lines_out[0], 'Configuration Name: PB2Way\n')
        self.assertEqual(lines_out[1], 'Pipeline: hg38_Starcq_Ensembl\n')
        self.assertEqual(lines_out[2], 'Number of folds: 10\n')
        self.assertEqual(lines_out[3], "Samples Used: {'AH': 38, 'CT': 20}\n")
        self.assertEqual(lines_out[4], "Conditions: ['AH', 'CT']\n")
        self.assertEqual(lines_out[5], 'Variance Transform (log(1+n)): True\n')
        self.assertEqual(lines_out[6], 'Root Directory: '...'\n')
        self.assertEqual(lines_out[7], 'Normalization: GEOM\n')
        self.assertEqual(lines_out[8], 'Dispersion: POOL\n')
        self.assertEqual(lines_out[9], "Classifiers: ['log reg', 'kNN', 'GNB', 'SVM']\n")
        self.assertEqual(lines_out[10], 'Feature Sizes (Filter): [10, 30, 50, 100]\n')
        self.assertEqual(lines_out[11], 'Number of runs: 1\n')
        self.assertEqual(lines_out[12], 'Feature Selection File: top_DE_W-MEAN_genes.txt\n')
        self.assertEqual(lines_out[13], 'Tissue: PB\n')
        self.assertEqual(lines_out[14], 'Note: all performance metrics are reported as average of all the runs. \n')
        self.assertEqual(lines_out[15], 'Decision boundaries use only the two top genes. \n')
        
        # Verify Missclassified Samples File
        
        miss_file_out = out_dir + 'miss_samples_PB2Way.txt'
        with open(miss_file_out) as reader:
            lines_out = reader.readlines()
        miss_file_exp = exp_dir + 'miss_samples_PB2Way.txt'
        with open(miss_file_exp) as reader:
            lines_exp = reader.readlines()
            
        self.assertEqual(len(lines_out), len(lines_exp))
        z = 0
        for line in lines_out:
            self.assertEqual(line, lines_exp[z])
            z += 1
        
        # Verify All The Performance Metrics
        out_acc_filename = 'Accuracy_PB2Way.csv'
        out_acc_table = []
        with open(out_dir + 'Metrics/' + out_acc_filename) as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=',')
            for row in csv_reader:
                out_acc_table.append(row)
        
        acc_table_exp = [['Features', 'log reg', 'kNN', 'GNB', 'SVM'],
                         ['10', '84.48%', '87.93%',  '82.76%', '86.21%'],
                         ['30', '96.55%', '94.83%',  '91.38%', '94.83%'],
                         ['50', '100.0%', '96.55%', '96.55%', '98.28%'],
                         ['100', '98.28%', '100.0%', '98.28%', '98.28%']]
        row_counter = 0
        for row in out_acc_table:
            value_counter = 0
            for value in row:
                if (row_counter == 0):
                    pass
                else:
                    self.assertEqual(value, acc_table_exp[row_counter][value_counter])
                value_counter += 1
            row_counter += 1
        
        out_prec_filename = 'Precision_PB2Way.csv'
        out_prec_table = []
        with open(out_dir + 'Metrics/' + out_prec_filename) as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=',')
            for row in csv_reader:
                out_prec_table.append(row)
        
        prec_table_exp = [['Features', 'log reg', 'kNN', 'GNB', 'SVM'],
                         ['10', '72.0%', '74.07%', '69.23%', '73.08%'],
                         ['30', '95.0%', '86.96%', '82.61%', '94.74%'],
                         ['50', '100.0%', '90.91%', '90.91%', '95.24%'],
                         ['100', '100.0%', '100.0%', '100.0%','100.0%']]
        
        row_counter = 0
        for row in out_prec_table:
            value_counter = 0
            for value in row:
                if (row_counter == 0):
                    pass
                else:
                    self.assertEqual(value, prec_table_exp[row_counter][value_counter])
                value_counter += 1
            row_counter += 1
            
        out_rec_filename = 'Recall_PB2Way.csv'
        out_rec_table = []
        with open(out_dir + 'Metrics/' + out_rec_filename) as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=',')
            for row in csv_reader:
                out_rec_table.append(row)
        
        rec_table_exp = [['Features', 'log reg', 'kNN', 'GNB', 'SVM'],
                         ['10', '90.0%', '100.0%', '90.0%', '95.0%'],
                         ['30', '95.0%', '100.0%', '95.0%', '90.0%'],
                         ['50', '100.0%', '100.0%', '100.0%', '100.0%'],
                         ['100', '95.0%', '100.0%', '95.0%','95.0%']]
        
        row_counter = 0
        for row in out_rec_table:
            value_counter = 0
            for value in row:
                if (row_counter == 0):
                    pass
                else:
                    self.assertEqual(value, rec_table_exp[row_counter][value_counter])
                value_counter += 1
            row_counter += 1
            
        out_bacc_filename = 'Balanced_Accuracy_PB2Way.csv'
        out_bacc_table = []
        with open(out_dir + 'Metrics/' + out_bacc_filename) as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=',')
            for row in csv_reader:
                out_bacc_table.append(row)
        
        bacc_table_exp = [['Features', 'log reg', 'kNN', 'GNB', 'SVM'],
                         ['10', '85.79%', '90.79%', '84.47%', '88.29%'],
                         ['30', '96.18%', '96.05%', '92.24%', '93.68%'],
                         ['50', '100.0%', '97.37%', '97.37%', '98.68%'],
                         ['100', '97.5%', '100.0%', '97.5%','97.5%']]
        
        row_counter = 0
        for row in out_bacc_table:
            value_counter = 0
            for value in row:
                if (row_counter == 0):
                    pass
                else:
                    self.assertEqual(value, bacc_table_exp[row_counter][value_counter])
                value_counter += 1
            row_counter += 1
            
        out_f1_filename = 'F1_PB2Way.csv'
        out_f1_table = []
        with open(out_dir + 'Metrics/' + out_f1_filename) as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=',')
            for row in csv_reader:
                out_f1_table.append(row)
        
        f1_table_exp = [['Features', 'log reg', 'kNN', 'GNB', 'SVM'],
                         ['10', '80.0%', '85.11%', '78.26%', '82.61%'],
                         ['30', '95.0%', '93.02%', '88.37%', '92.31%'],
                         ['50', '100.0%', '95.24%', '95.24%', '97.56%'],
                         ['100', '97.44%', '100.0%', '97.44%', '97.44%']]
        
        row_counter = 0
        for row in out_f1_table:
            value_counter = 0
            for value in row:
                if (row_counter == 0):
                    pass
                else:
                    self.assertEqual(value, f1_table_exp[row_counter][value_counter])
                value_counter += 1
            row_counter += 1
            
        out_fms_filename = 'Fold_Mean_Stability_PB2Way.csv'
        out_fms_table = []
        with open(out_dir + 'Metrics/' + out_fms_filename) as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=',')
            for row in csv_reader:
                out_fms_table.append(row)
        
        fms_table_exp = [['Features', 'log reg', 'kNN', 'GNB', 'SVM'],
                         ['10', '0.2095', '0.1212', '0.1727', '0.1732'],
                         ['30', '0.0667', '0.0819', '0.0907', '0.0764'],
                         ['50', '0.0', '0.0667', '0.0737', '0.05'],
                         ['100', '0.06', '0.0', '0.06', '0.06']]
        
        row_counter = 0
        for row in out_fms_table:
            value_counter = 0
            for value in row:
                if (row_counter == 0):
                    pass
                else:
                    self.assertEqual(value, fms_table_exp[row_counter][value_counter])
                value_counter += 1
            row_counter += 1
            
        out_rs_filename = 'Run_Stability_PB2Way.csv'
        out_rs_table = []
        with open(out_dir + 'Metrics/' + out_rs_filename) as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=',')
            for row in csv_reader:
                out_rs_table.append(row)
        
        rs_table_exp = [['Features', 'log reg', 'kNN', 'GNB', 'SVM'],
                         ['10', '0.0', '0.0', '0.0', '0.0'],
                         ['30', '0.0', '0.0', '0.0', '0.0'],
                         ['50', '0.0', '0.0', '0.0', '0.0'],
                         ['100', '0.0', '0.0', '0.0', '0.0']]
        
        row_counter = 0
        for row in out_rs_table:
            value_counter = 0
            for value in row:
                if (row_counter == 0):
                    pass
                else:
                    self.assertEqual(value, rs_table_exp[row_counter][value_counter])
                value_counter += 1
            row_counter += 1
            
        out_ra_filename = 'ROC_AUC_PB2Way.csv'
        out_ra_table = []
        with open(out_dir + 'Metrics/' + out_ra_filename) as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=',')
            for row in csv_reader:
                out_ra_table.append(row)
        
        ra_table_exp = [['Features', 'log reg', 'kNN', 'GNB', 'SVM'],
                         ['10', '[]', '[]', '[]', '[]'],
                         ['30', '[]', '[]', '[]', '[]'],
                         ['50', '[]', '[]', '[]', '[]'],
                         ['100', '[]', '[]', '[]', '[]']]
        
        row_counter = 0
        for row in out_ra_table:
            value_counter = 0
            for value in row:
                if (row_counter == 0):
                    pass
                else:
                    self.assertEqual(value, ra_table_exp[row_counter][value_counter])
                value_counter += 1
            row_counter += 1


        # Same test as above, but use Cuffdiff counts instead of Cuffnorm counts.
        out_dir = '...'
        exp_dir = '...'
        root_dir = '...'
        reference = 'hg38'
        annotation = 'ensembl'
        aligner = 'starcq'
        normalization = 'GEOM'
        dispersion = 'POOL'
        features_file = 'top_DE_W-MEAN_genes.txt'
        run_name = 'PB2Way'
        num_folds = 10
        num_runs = 1
        perf_metrics = rnaseq_classification_with_feature_selection({'AH':38, 'CT':20}, ['AH', 'CT'], reference,
                                                               aligner, annotation, normalization, dispersion, 
                                                               root_dir, features_file, out_dir, [10, 30, 50, 100],
                                                               ["log reg", "kNN", "GNB", "SVM"], num_folds, num_runs, 
                                                               gene_lists = True, validate = False, 
                                                               counts_format = 'Cuffdiff')
        
        # Ideally we'd also want to verify confusion matrix plots, counts heatmaps, and ROC curves.
        
        # Verify All The Gene Lists
        i = 0 
        while i < num_folds:
            dir_fold = root_dir + reference + '_' + aligner.capitalize() + '_' + annotation.capitalize() + '/'
            dir_fold += 'FOLD' + str(num_folds) + '/Cuffdiff_' + normalization + '_' + dispersion + '_FOLD' + str(i+1) + '/'
            features_filepath = dir_fold + features_file
            
            gene_list_filepath = out_dir + 'GeneLists/' + 'gene_list_fold' + str(i+1) + '_' + run_name + '.txt'
            
            with open(gene_list_filepath) as reader:
                lines_out = reader.readlines()
            
            with open(features_filepath) as reader:
                lines_exp = reader.readlines()
            
            j = 0
            while j < len(lines_out):
                self.assertEqual(lines_out[j], lines_exp[j])
                j += 1
            
            i += 1
        
        # Verify Config File
        config_file_out = out_dir + 'config_PB2Way.txt'
        with open(config_file_out) as reader:
            lines_out = reader.readlines()
        self.assertEqual(lines_out[0], 'Configuration Name: PB2Way\n')
        self.assertEqual(lines_out[1], 'Pipeline: hg38_Starcq_Ensembl\n')
        self.assertEqual(lines_out[2], 'Number of folds: 10\n')
        self.assertEqual(lines_out[3], "Samples Used: {'AH': 38, 'CT': 20}\n")
        self.assertEqual(lines_out[4], "Conditions: ['AH', 'CT']\n")
        self.assertEqual(lines_out[5], 'Variance Transform (log(1+n)): True\n')
        self.assertEqual(lines_out[6], 'Root Directory: '...'n')
        self.assertEqual(lines_out[7], 'Normalization: GEOM\n')
        self.assertEqual(lines_out[8], 'Dispersion: POOL\n')
        self.assertEqual(lines_out[9], "Classifiers: ['log reg', 'kNN', 'GNB', 'SVM']\n")
        self.assertEqual(lines_out[10], 'Feature Sizes (Filter): [10, 30, 50, 100]\n')
        self.assertEqual(lines_out[11], 'Number of runs: 1\n')
        self.assertEqual(lines_out[12], 'Feature Selection File: top_DE_W-MEAN_genes.txt\n')
        self.assertEqual(lines_out[13], 'Tissue: PB\n')
        self.assertEqual(lines_out[14], 'Note: all performance metrics are reported as average of all the runs. \n')
        self.assertEqual(lines_out[15], 'Decision boundaries use only the two top genes. \n')
        
        # Verify Missclassified Samples File
        
        miss_file_out = out_dir + 'miss_samples_PB2Way.txt'
        with open(miss_file_out) as reader:
            lines_out = reader.readlines()
        miss_file_exp = exp_dir + 'miss_samples_PB2Way.txt'
        with open(miss_file_exp) as reader:
            lines_exp = reader.readlines()
            
        self.assertEqual(len(lines_out), len(lines_exp))
        z = 0
        for line in lines_out:
            self.assertEqual(line, lines_exp[z])
            z += 1
        
        # Verify All The Performance Metrics
        out_acc_filename = 'Accuracy_PB2Way.csv'
        out_acc_table = []
        with open(out_dir + 'Metrics/' + out_acc_filename) as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=',')
            for row in csv_reader:
                out_acc_table.append(row)
        
        acc_table_exp = [['Features', 'log reg', 'kNN', 'GNB', 'SVM'],
                         ['10', '84.48%', '87.93%',  '82.76%', '86.21%'],
                         ['30', '96.55%', '94.83%',  '91.38%', '94.83%'],
                         ['50', '100.0%', '96.55%', '96.55%', '98.28%'],
                         ['100', '98.28%', '100.0%', '98.28%', '98.28%']]
        row_counter = 0
        for row in out_acc_table:
            value_counter = 0
            for value in row:
                if (row_counter == 0):
                    pass
                else:
                    self.assertEqual(value, acc_table_exp[row_counter][value_counter])
                value_counter += 1
            row_counter += 1
        
        out_prec_filename = 'Precision_PB2Way.csv'
        out_prec_table = []
        with open(out_dir + 'Metrics/' + out_prec_filename) as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=',')
            for row in csv_reader:
                out_prec_table.append(row)
        
        prec_table_exp = [['Features', 'log reg', 'kNN', 'GNB', 'SVM'],
                         ['10', '72.0%', '74.07%', '69.23%', '73.08%'],
                         ['30', '95.0%', '86.96%', '82.61%', '94.74%'],
                         ['50', '100.0%', '90.91%', '90.91%', '95.24%'],
                         ['100', '100.0%', '100.0%', '100.0%','100.0%']]
        
        row_counter = 0
        for row in out_prec_table:
            value_counter = 0
            for value in row:
                if (row_counter == 0):
                    pass
                else:
                    self.assertEqual(value, prec_table_exp[row_counter][value_counter])
                value_counter += 1
            row_counter += 1
            
        out_rec_filename = 'Recall_PB2Way.csv'
        out_rec_table = []
        with open(out_dir + 'Metrics/' + out_rec_filename) as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=',')
            for row in csv_reader:
                out_rec_table.append(row)
        
        rec_table_exp = [['Features', 'log reg', 'kNN', 'GNB', 'SVM'],
                         ['10', '90.0%', '100.0%', '90.0%', '95.0%'],
                         ['30', '95.0%', '100.0%', '95.0%', '90.0%'],
                         ['50', '100.0%', '100.0%', '100.0%', '100.0%'],
                         ['100', '95.0%', '100.0%', '95.0%','95.0%']]
        
        row_counter = 0
        for row in out_rec_table:
            value_counter = 0
            for value in row:
                if (row_counter == 0):
                    pass
                else:
                    self.assertEqual(value, rec_table_exp[row_counter][value_counter])
                value_counter += 1
            row_counter += 1
            
        out_bacc_filename = 'Balanced_Accuracy_PB2Way.csv'
        out_bacc_table = []
        with open(out_dir + 'Metrics/' + out_bacc_filename) as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=',')
            for row in csv_reader:
                out_bacc_table.append(row)
        
        bacc_table_exp = [['Features', 'log reg', 'kNN', 'GNB', 'SVM'],
                         ['10', '85.79%', '90.79%', '84.47%', '88.29%'],
                         ['30', '96.18%', '96.05%', '92.24%', '93.68%'],
                         ['50', '100.0%', '97.37%', '97.37%', '98.68%'],
                         ['100', '97.5%', '100.0%', '97.5%','97.5%']]
        
        row_counter = 0
        for row in out_bacc_table:
            value_counter = 0
            for value in row:
                if (row_counter == 0):
                    pass
                else:
                    self.assertEqual(value, bacc_table_exp[row_counter][value_counter])
                value_counter += 1
            row_counter += 1
            
        out_f1_filename = 'F1_PB2Way.csv'
        out_f1_table = []
        with open(out_dir + 'Metrics/' + out_f1_filename) as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=',')
            for row in csv_reader:
                out_f1_table.append(row)
        
        f1_table_exp = [['Features', 'log reg', 'kNN', 'GNB', 'SVM'],
                         ['10', '80.0%', '85.11%', '78.26%', '82.61%'],
                         ['30', '95.0%', '93.02%', '88.37%', '92.31%'],
                         ['50', '100.0%', '95.24%', '95.24%', '97.56%'],
                         ['100', '97.44%', '100.0%', '97.44%', '97.44%']]
        
        row_counter = 0
        for row in out_f1_table:
            value_counter = 0
            for value in row:
                if (row_counter == 0):
                    pass
                else:
                    self.assertEqual(value, f1_table_exp[row_counter][value_counter])
                value_counter += 1
            row_counter += 1
            
        out_fms_filename = 'Fold_Mean_Stability_PB2Way.csv'
        out_fms_table = []
        with open(out_dir + 'Metrics/' + out_fms_filename) as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=',')
            for row in csv_reader:
                out_fms_table.append(row)
        
        fms_table_exp = [['Features', 'log reg', 'kNN', 'GNB', 'SVM'],
                         ['10', '0.2095', '0.1212', '0.1727', '0.1732'],
                         ['30', '0.0667', '0.0819', '0.0907', '0.0764'],
                         ['50', '0.0', '0.0667', '0.0737', '0.05'],
                         ['100', '0.06', '0.0', '0.06', '0.06']]
        
        row_counter = 0
        for row in out_fms_table:
            value_counter = 0
            for value in row:
                if (row_counter == 0):
                    pass
                else:
                    self.assertEqual(value, fms_table_exp[row_counter][value_counter])
                value_counter += 1
            row_counter += 1
            
        out_rs_filename = 'Run_Stability_PB2Way.csv'
        out_rs_table = []
        with open(out_dir + 'Metrics/' + out_rs_filename) as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=',')
            for row in csv_reader:
                out_rs_table.append(row)
        
        rs_table_exp = [['Features', 'log reg', 'kNN', 'GNB', 'SVM'],
                         ['10', '0.0', '0.0', '0.0', '0.0'],
                         ['30', '0.0', '0.0', '0.0', '0.0'],
                         ['50', '0.0', '0.0', '0.0', '0.0'],
                         ['100', '0.0', '0.0', '0.0', '0.0']]
        
        row_counter = 0
        for row in out_rs_table:
            value_counter = 0
            for value in row:
                if (row_counter == 0):
                    pass
                else:
                    self.assertEqual(value, rs_table_exp[row_counter][value_counter])
                value_counter += 1
            row_counter += 1
            
        out_ra_filename = 'ROC_AUC_PB2Way.csv'
        out_ra_table = []
        with open(out_dir + 'Metrics/' + out_ra_filename) as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=',')
            for row in csv_reader:
                out_ra_table.append(row)
        
        ra_table_exp = [['Features', 'log reg', 'kNN', 'GNB', 'SVM'],
                         ['10', '[]', '[]', '[]', '[]'],
                         ['30', '[]', '[]', '[]', '[]'],
                         ['50', '[]', '[]', '[]', '[]'],
                         ['100', '[]', '[]', '[]', '[]']]
        
        row_counter = 0
        for row in out_ra_table:
            value_counter = 0
            for value in row:
                if (row_counter == 0):
                    pass
                else:
                    self.assertEqual(value, ra_table_exp[row_counter][value_counter])
                value_counter += 1
            row_counter += 1
    
            
    def test_rnaseq_classification_with_feature_selection_multiclass(self):
        # Brief RFE Test (15 target size, step_size = (num_degs - target_size) // 5)
        
        out_dir = '...'
        root_dir = '...'
        reference = 'hg38'
        annotation = 'ensembl'
        aligner = 'starcq'
        normalization = 'GEOM'
        dispersion = 'POOL'
        features_file = 'top_IG_genes.txt'
        num_folds = 10
        num_runs = 1
        perf_metrics = rnaseq_classification_with_feature_selection({'AH':38, 'CT':20, 'DA':20, 'AA':20, 'NF':20, 'HP':20},
                                                               ['AH', 'CT', 'DAAA', 'NF', 'HP'], reference, aligner, 
                                                               annotation, normalization, dispersion, root_dir, 
                                                               features_file, out_dir, [20, 30], 
                                                               ["log reg"], num_folds, num_runs, 
                                                               wrapper = 'RFE', target_size = 15, validate = False)
        
        out_acc_filename = 'Accuracy_PB5Way_RFE.csv'
        out_acc_table = []
        with open(out_dir + 'Metrics/' + out_acc_filename) as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=',')
            for row in csv_reader:
                out_acc_table.append(row)
        
        acc_table_exp = [['Features', 'log reg'],
                          ['20', '67.39%'],
                          ['30', '68.84%']]
        row_counter = 0
        for row in out_acc_table:
            value_counter = 0
            for value in row:
                if (row_counter == 0):
                    pass
                else:
                    self.assertEqual(value, acc_table_exp[row_counter][value_counter])
                value_counter += 1
            row_counter += 1
            
        # Brief RFE Test (15 target size, step_size = (num_degs - target_size) // 5) with Cuffdiff Counts
        
        out_dir = '...'
        root_dir = '...'
        reference = 'hg38'
        annotation = 'ensembl'
        aligner = 'starcq'
        normalization = 'GEOM'
        dispersion = 'POOL'
        features_file = 'top_IG_genes.txt'
        num_folds = 10
        num_runs = 1
        perf_metrics = rnaseq_classification_with_feature_selection({'AH':38, 'CT':20, 'DA':20, 'AA':20, 'NF':20, 'HP':20},
                                                               ['AH', 'CT', 'DAAA', 'NF', 'HP'], reference, aligner, 
                                                               annotation, normalization, dispersion, root_dir, 
                                                               features_file, out_dir, [20, 30], 
                                                               ["log reg"], num_folds, num_runs, 
                                                               wrapper = 'RFE', target_size = 15, validate = False,
                                                               counts_format = 'Cuffdiff')
        
        out_acc_filename = 'Accuracy_PB5Way_RFE.csv'
        out_acc_table = []
        with open(out_dir + 'Metrics/' + out_acc_filename) as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=',')
            for row in csv_reader:
                out_acc_table.append(row)
        
        acc_table_exp = [['Features', 'log reg'],
                          ['20', '67.39%'],
                          ['30', '68.84%']]
        row_counter = 0
        for row in out_acc_table:
            value_counter = 0
            for value in row:
                if (row_counter == 0):
                    pass
                else:
                    self.assertEqual(value, acc_table_exp[row_counter][value_counter])
                value_counter += 1
            row_counter += 1
        
        
        # Brief SFS Test
        out_dir = '...'
        exp_dir = '...'
        root_dir = '...'
        reference = 'hg38'
        annotation = 'ensembl'
        aligner = 'starcq'
        normalization = 'GEOM'
        dispersion = 'POOL'
        features_file = 'top_IG_genes.txt'
        num_folds = 10
        num_runs = 1
        perf_metrics = rnaseq_classification_with_feature_selection({'AH':38, 'CT':20, 'DA':20, 'AA':20, 'NF':20, 'HP':20},
                                                                ['AH', 'CT', 'DAAA', 'NF', 'HP'], reference, aligner, 
                                                                annotation, normalization, dispersion, root_dir, 
                                                                features_file, out_dir, [20,30], 
                                                                ["log reg"], num_folds, num_runs, 
                                                                wrapper = 'SFS', target_size = 3, 
                                                                gene_lists = True, validate = False)
        
        
        # Verify All The SFS Gene Lists
        f_pool_sizes = [20,30]
        for f_pool_size in f_pool_sizes:
            i = 1 
            while i <= num_folds:
                gene_list_out = out_dir + 'GeneLists/SFS_LogisticRegression/gene_list_SFS_fold' + str(i)
                gene_list_out += '_fpool_' + str(f_pool_size) + '_tsize_3_run_0_PBMC5Way_SFS.txt'
                gene_list_exp = exp_dir + 'GeneLists/SFS_LogisticRegression/gene_list_SFS_fold' + str(i)
                gene_list_exp += '_fpool_' + str(f_pool_size) + '_tsize_3_run_0_PBMC5Way_SFS_Test.txt'
                
                with open(gene_list_out) as reader:
                    lines_out = reader.readlines()
                
                with open(gene_list_exp) as reader:
                    lines_exp = reader.readlines()
                
                j = 0
                while j < len(lines_out):
                    self.assertEqual(lines_out[j], lines_exp[j])
                    j += 1
                
                i += 1
        
        out_acc_filename = 'Accuracy_PBMC5Way_SFS.csv'
        out_acc_table = []
        with open(out_dir + 'Metrics/' + out_acc_filename) as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=',')
            for row in csv_reader:
                out_acc_table.append(row)
        
        acc_table_exp = [['Features', 'log reg'],
                          ['20', '71.74%'],
                          ['30', '74.64%']]
        row_counter = 0
        for row in out_acc_table:
            value_counter = 0
            for value in row:
                if (row_counter == 0):
                    pass
                else:
                    self.assertEqual(value, acc_table_exp[row_counter][value_counter])
                value_counter += 1
            row_counter += 1
            
        out_prec_filename = 'Precision_PBMC5Way_SFS.csv'
        out_prec_table = []
        with open(out_dir + 'Metrics/' + out_prec_filename) as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=',')
            for row in csv_reader:
                out_prec_table.append(row)
        
        prec_table_exp = [['Features', 'log reg'],
                         ['20', 'AH:86.05% CT:88.24% DAAA:54.69% NF:84.62% HP:100.0% '],
                         ['30', 'AH:75.51% CT:70.83% DAAA:71.05% NF:75.0% HP:100.0% ']]
        
        row_counter = 0
        for row in out_prec_table:
            value_counter = 0
            for value in row:
                if (row_counter == 0):
                    pass
                else:
                    self.assertEqual(value, prec_table_exp[row_counter][value_counter])
                value_counter += 1
            row_counter += 1
            
        out_rec_filename = 'Recall_PBMC5Way_SFS.csv'
        out_rec_table = []
        with open(out_dir + 'Metrics/' + out_rec_filename) as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=',')
            for row in csv_reader:
                out_rec_table.append(row)
        
        rec_table_exp = [['Features', 'log reg'],
                         ['20', 'AH:97.37% CT:75.0% DAAA:87.5% NF:55.0% HP:5.0% '],
                         ['30', 'AH:97.37% CT:85.0% DAAA:67.5% NF:75.0% HP:35.0% ']]
        
        row_counter = 0
        for row in out_rec_table:
            value_counter = 0
            for value in row:
                if (row_counter == 0):
                    pass
                else:
                    self.assertEqual(value, rec_table_exp[row_counter][value_counter])
                value_counter += 1
            row_counter += 1
            
        # Brief SFS Test - Same as above but with Cuffdiff counts
        out_dir = '...'
        exp_dir = '...'
        root_dir = '...'
        reference = 'hg38'
        annotation = 'ensembl'
        aligner = 'starcq'
        normalization = 'GEOM'
        dispersion = 'POOL'
        features_file = 'top_IG_genes.txt'
        num_folds = 10
        num_runs = 1
        perf_metrics = rnaseq_classification_with_feature_selection({'AH':38, 'CT':20, 'DA':20, 'AA':20, 'NF':20, 'HP':20},
                                                                ['AH', 'CT', 'DAAA', 'NF', 'HP'], reference, aligner, 
                                                                annotation, normalization, dispersion, root_dir, 
                                                                features_file, out_dir, [20,30], 
                                                                ["log reg"], num_folds, num_runs, 
                                                                wrapper = 'SFS', target_size = 3, 
                                                                gene_lists = True, validate = False,
                                                                counts_format = 'Cuffdiff')
        
        
        # Verify All The SFS Gene Lists
        f_pool_sizes = [20,30]
        for f_pool_size in f_pool_sizes:
            i = 1 
            while i <= num_folds:
                gene_list_out = out_dir + 'GeneLists/SFS_LogisticRegression/gene_list_SFS_fold' + str(i)
                gene_list_out += '_fpool_' + str(f_pool_size) + '_tsize_3_run_0_PBMC5Way_SFS.txt'
                gene_list_exp = exp_dir + 'GeneLists/SFS_LogisticRegression/gene_list_SFS_fold' + str(i)
                gene_list_exp += '_fpool_' + str(f_pool_size) + '_tsize_3_run_0_PBMC5Way_SFS_Test.txt'
                
                with open(gene_list_out) as reader:
                    lines_out = reader.readlines()
                
                with open(gene_list_exp) as reader:
                    lines_exp = reader.readlines()
                
                j = 0
                while j < len(lines_out):
                    self.assertEqual(lines_out[j], lines_exp[j])
                    j += 1
                
                i += 1
        
        out_acc_filename = 'Accuracy_PBMC5Way_SFS.csv'
        out_acc_table = []
        with open(out_dir + 'Metrics/' + out_acc_filename) as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=',')
            for row in csv_reader:
                out_acc_table.append(row)
        
        acc_table_exp = [['Features', 'log reg'],
                          ['20', '71.74%'],
                          ['30', '74.64%']]
        row_counter = 0
        for row in out_acc_table:
            value_counter = 0
            for value in row:
                if (row_counter == 0):
                    pass
                else:
                    self.assertEqual(value, acc_table_exp[row_counter][value_counter])
                value_counter += 1
            row_counter += 1
            
        out_prec_filename = 'Precision_PBMC5Way_SFS.csv'
        out_prec_table = []
        with open(out_dir + 'Metrics/' + out_prec_filename) as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=',')
            for row in csv_reader:
                out_prec_table.append(row)
        
        prec_table_exp = [['Features', 'log reg'],
                         ['20', 'AH:86.05% CT:88.24% DAAA:54.69% NF:84.62% HP:100.0% '],
                         ['30', 'AH:75.51% CT:70.83% DAAA:71.05% NF:75.0% HP:100.0% ']]
        
        row_counter = 0
        for row in out_prec_table:
            value_counter = 0
            for value in row:
                if (row_counter == 0):
                    pass
                else:
                    self.assertEqual(value, prec_table_exp[row_counter][value_counter])
                value_counter += 1
            row_counter += 1
            
        out_rec_filename = 'Recall_PBMC5Way_SFS.csv'
        out_rec_table = []
        with open(out_dir + 'Metrics/' + out_rec_filename) as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=',')
            for row in csv_reader:
                out_rec_table.append(row)
        
        rec_table_exp = [['Features', 'log reg'],
                         ['20', 'AH:97.37% CT:75.0% DAAA:87.5% NF:55.0% HP:5.0% '],
                         ['30', 'AH:97.37% CT:85.0% DAAA:67.5% NF:75.0% HP:35.0% ']]
        
        row_counter = 0
        for row in out_rec_table:
            value_counter = 0
            for value in row:
                if (row_counter == 0):
                    pass
                else:
                    self.assertEqual(value, rec_table_exp[row_counter][value_counter])
                value_counter += 1
            row_counter += 1
        
        # Brief SFS Test2
        out_dir = '...'
        exp_dir = '...'
        root_dir = '...'
        reference = 'hg38'
        annotation = 'ensembl'
        aligner = 'starcq'
        normalization = 'GEOM'
        dispersion = 'POOL'
        features_file = 'top_IG_genes.txt'
        num_folds = 10
        num_runs = 1
        perf_metrics = rnaseq_classification_with_feature_selection({'AH':38, 'CT':20, 'DA':20, 'AA':20, 'NF':20, 'HP':20},
                                                                ['AH', 'CT', 'DAAA', 'NF', 'HP'], reference, aligner, 
                                                                annotation, normalization, dispersion, root_dir, 
                                                                features_file, out_dir, [20,30], 
                                                                ["log reg"], num_folds, num_runs, 
                                                                wrapper = 'SFS', target_size = 3, gene_lists = True,
                                                                sfs_replace = False, validate = False)
        
        # Verify All The SFS Gene Lists
        f_pool_sizes = [20,30]
        for f_pool_size in f_pool_sizes:
            i = 1 
            while i <= num_folds:
                gene_list_out = out_dir + 'GeneLists/SFS_LogisticRegression/gene_list_SFS_fold' + str(i)
                gene_list_out += '_fpool_' + str(f_pool_size) + '_tsize_3_run_0_PBMC5Way_SFS_No_Replacement.txt'
                gene_list_exp = exp_dir + 'GeneLists/SFS_LogisticRegression/gene_list_SFS_fold' + str(i)
                gene_list_exp += '_fpool_' + str(f_pool_size) + '_tsize_3_run_0_PBMC5Way_SFS_Exp_No_Replacement.txt'
                
                with open(gene_list_out) as reader:
                    lines_out = reader.readlines()
                
                with open(gene_list_exp) as reader:
                    lines_exp = reader.readlines()
                
                j = 0
                while j < len(lines_out):
                    self.assertEqual(lines_out[j], lines_exp[j])
                    j += 1
                
                i += 1
        
        
        out_acc_filename = 'Accuracy_PBMC5Way_SFS_No_Replacement.csv'
        out_acc_table = []
        with open(out_dir + 'Metrics/' + out_acc_filename) as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=',')
            for row in csv_reader:
                out_acc_table.append(row)
        
        acc_table_exp = [['Features', 'log reg'],
                          ['20', '71.74%'],
                          ['30', '73.91%']]
        row_counter = 0
        for row in out_acc_table:
            value_counter = 0
            for value in row:
                if (row_counter == 0):
                    pass
                else:
                    self.assertEqual(value, acc_table_exp[row_counter][value_counter])
                value_counter += 1
            row_counter += 1
        
        
        # Full Output Test
        out_dir = '...'
        exp_dir = '...'
        root_dir = '...'
        reference = 'hg38'
        annotation = 'ensembl'
        aligner = 'starcq'
        normalization = 'GEOM'
        dispersion = 'POOL'
        features_file = 'top_IG_genes.txt'
        run_name = 'PB5Way'
        num_folds = 10
        num_runs = 2
        perf_metrics = rnaseq_classification_with_feature_selection({'AH':38, 'CT':20, 'DA':20, 'AA':20, 'NF':20, 'HP':20},
                                                               ['AH', 'CT', 'DAAA', 'NF', 'HP'], reference, aligner, 
                                                               annotation, normalization, dispersion, root_dir, 
                                                               features_file, out_dir, [10, 30], 
                                                               ["log reg", "kNN", "GNB", "SVM"], num_folds, num_runs, 
                                                               gene_lists = True, validate = False)
        
        # Verify All The Gene Lists
        i = 0 
        while i < num_folds:
            dir_fold = root_dir + reference + '_' + aligner.capitalize() + '_' + annotation.capitalize() + '/'
            dir_fold += 'FOLD' + str(num_folds) + '/Cuffnorm_' + normalization + '_FOLD' + str(i+1) + '/'
            features_filepath = dir_fold + features_file
            
            gene_list_filepath = out_dir + 'GeneLists/' + 'gene_list_fold' + str(i+1) + '_' + run_name + '.txt'
            
            with open(gene_list_filepath) as reader:
                lines_out = reader.readlines()
            
            with open(features_filepath) as reader:
                lines_exp = reader.readlines()
            
            j = 0
            while j < len(lines_out):
                self.assertEqual(lines_out[j], lines_exp[j])
                j += 1
            
            i += 1
        
        # Verify Config File
        config_file_out = out_dir + 'config_PB5Way.txt'
        with open(config_file_out) as reader:
            lines_out = reader.readlines()
        self.assertEqual(lines_out[0], 'Configuration Name: PB5Way\n')
        self.assertEqual(lines_out[1], 'Pipeline: hg38_Starcq_Ensembl\n')
        self.assertEqual(lines_out[2], 'Number of folds: 10\n')
        self.assertEqual(lines_out[3], "Samples Used: {'AH': 38, 'CT': 20, 'DA': 20, 'AA': 20, 'NF': 20, 'HP': 20}\n")
        self.assertEqual(lines_out[4], "Conditions: ['AH', 'CT', 'DAAA', 'NF', 'HP']\n")
        self.assertEqual(lines_out[5], 'Variance Transform (log(1+n)): True\n')
        self.assertEqual(lines_out[6], 'Root Directory: '...'\n')
        self.assertEqual(lines_out[7], 'Normalization: GEOM\n')
        self.assertEqual(lines_out[8], 'Dispersion: POOL\n')
        self.assertEqual(lines_out[9], "Classifiers: ['log reg', 'kNN', 'GNB', 'SVM']\n")
        self.assertEqual(lines_out[10], 'Feature Sizes (Filter): [10, 30]\n')
        self.assertEqual(lines_out[11], 'Number of runs: 2\n')
        self.assertEqual(lines_out[12], 'Feature Selection File: top_IG_genes.txt\n')
        self.assertEqual(lines_out[13], 'Tissue: PB\n')
        self.assertEqual(lines_out[14], 'Note: all performance metrics are reported as average of all the runs. \n')
        self.assertEqual(lines_out[15], 'Decision boundaries use only the two top genes. \n')
        
        # Verify Missclassified Samples File
        
        miss_file_out = out_dir + 'miss_samples_PB5Way.txt'
        with open(miss_file_out) as reader:
            lines_out = reader.readlines()
        miss_file_exp = exp_dir + 'miss_samples_PB5Way.txt'
        with open(miss_file_exp) as reader:
            lines_exp = reader.readlines()
            
        self.assertEqual(len(lines_out), len(lines_exp))
        z = 0
        for line in lines_out:
            self.assertEqual(line, lines_exp[z])
            z += 1
        
        # Verify All The Performance Metrics
        out_acc_filename = 'Accuracy_PB5Way.csv'
        out_acc_table = []
        with open(out_dir + 'Metrics/' + out_acc_filename) as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=',')
            for row in csv_reader:
                out_acc_table.append(row)
        
        acc_table_exp = [['Features', 'log reg', 'kNN', 'GNB', 'SVM'],
                          ['10', '65.22%', '62.32%', '61.59%', '65.22%'],
                          ['30', '69.57%', '65.94%', '63.77%', '68.84%']]
        row_counter = 0
        for row in out_acc_table:
            value_counter = 0
            for value in row:
                if (row_counter == 0):
                    pass
                else:
                    self.assertEqual(value, acc_table_exp[row_counter][value_counter])
                value_counter += 1
            row_counter += 1
            
        out_prec_filename = 'Precision_PB5Way.csv'
        out_prec_table = []
        with open(out_dir + 'Metrics/' + out_prec_filename) as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=',')
            for row in csv_reader:
                out_prec_table.append(row)
        
        prec_table_exp = [['Features', 'log reg', 'kNN', 'GNB', 'SVM'],
                         ['10', 'AH:76.19% CT:59.26% DAAA:56.25% NF:60.0% HP:81.82% ',
                          'AH:86.49% CT:51.43% DAAA:58.7% NF:37.5% HP:50.0% ',
                          'AH:86.49% CT:46.88% DAAA:61.54% NF:40.0% HP:53.33% ',
                          'AH:78.05% CT:50.0% DAAA:67.74% NF:57.14% HP:66.67% '],
                         ['30', 'AH:85.37% CT:64.29% DAAA:71.43% NF:53.85% HP:52.38% ',
                          'AH:85.71% CT:52.94% DAAA:56.25% NF:77.78% HP:75.0% ',
                          'AH:85.0% CT:50.0% DAAA:63.89% NF:42.86% HP:55.0% ',
                          'AH:86.11% CT:57.14% DAAA:67.44% NF:57.14% HP:64.71% ']]
        
        row_counter = 0
        for row in out_prec_table:
            value_counter = 0
            for value in row:
                if (row_counter == 0):
                    pass
                else:
                    self.assertEqual(value, prec_table_exp[row_counter][value_counter])
                value_counter += 1
            row_counter += 1
            
        out_rec_filename = 'Recall_PB5Way.csv'
        out_rec_table = []
        with open(out_dir + 'Metrics/' + out_rec_filename) as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=',')
            for row in csv_reader:
                out_rec_table.append(row)
        
        rec_table_exp = [['Features', 'log reg', 'kNN', 'GNB', 'SVM'],
                         ['10', 'AH:84.21% CT:80.0% DAAA:67.5% NF:30.0% HP:45.0% ',
                          'AH:84.21% CT:90.0% DAAA:67.5% NF:15.0% HP:30.0% ',
                          'AH:84.21% CT:75.0% DAAA:60.0% NF:30.0% HP:40.0% '
                          ,'AH:84.21% CT:85.0% DAAA:52.5% NF:40.0% HP:60.0% '],
                         ['30', 'AH:92.11% CT:90.0% DAAA:62.5% NF:35.0% HP:55.0% ',
                          'AH:78.95% CT:90.0% DAAA:67.5% NF:35.0% HP:45.0% ',
                          'AH:89.47% CT:70.0% DAAA:57.5% NF:30.0% HP:55.0% ',
                          'AH:81.58% CT:80.0% DAAA:72.5% NF:40.0% HP:55.0% ']]
        
        row_counter = 0
        for row in out_rec_table:
            value_counter = 0
            for value in row:
                if (row_counter == 0):
                    pass
                else:
                    self.assertEqual(value, rec_table_exp[row_counter][value_counter])
                value_counter += 1
            row_counter += 1
            
        out_bacc_filename = 'Balanced_Accuracy_PB5Way.csv'
        out_bacc_table = []
        with open(out_dir + 'Metrics/' + out_bacc_filename) as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=',')
            for row in csv_reader:
                out_bacc_table.append(row)
        
        bacc_table_exp = [['Features', 'log reg', 'kNN', 'GNB', 'SVM'],
                         ['10', '61.34%', '57.34%', '57.84%', '64.34%'],
                         ['30', '66.92%', '63.29%', '60.39%', '65.82%']]
        
        row_counter = 0
        for row in out_bacc_table:
            value_counter = 0
            for value in row:
                if (row_counter == 0):
                    pass
                else:
                    self.assertEqual(value, bacc_table_exp[row_counter][value_counter])
                value_counter += 1
            row_counter += 1
            
        out_f1_filename = 'F1_PB5Way.csv'
        out_f1_table = []
        with open(out_dir + 'Metrics/' + out_f1_filename) as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=',')
            for row in csv_reader:
                out_f1_table.append(row)
        
        f1_table_exp = [['Features', 'log reg', 'kNN', 'GNB', 'SVM'],
                         ['10', 'AH:80.0% CT:68.09% DAAA:61.36% NF:40.0% HP:58.06% ',
                          'AH:85.33% CT:65.45% DAAA:62.79% NF:21.43% HP:37.5% ',
                          'AH:85.33% CT:57.69% DAAA:60.76% NF:34.29% HP:45.71% ',
                          'AH:81.01% CT:62.96% DAAA:59.15% NF:47.06% HP:63.16% '],
                         ['30', 'AH:88.61% CT:75.0% DAAA:66.67% NF:42.42% HP:53.66% ',
                          'AH:82.19% CT:66.67% DAAA:61.36% NF:48.28% HP:56.25% ',
                          'AH:87.18% CT:58.33% DAAA:60.53% NF:35.29% HP:55.0% ',
                          'AH:83.78% CT:66.67% DAAA:69.88% NF:47.06% HP:59.46% ']]
        
        row_counter = 0
        for row in out_f1_table:
            value_counter = 0
            for value in row:
                if (row_counter == 0):
                    pass
                else:
                    self.assertEqual(value, f1_table_exp[row_counter][value_counter])
                value_counter += 1
            row_counter += 1
            
        out_fms_filename = 'Fold_Mean_Stability_PB5Way.csv'
        out_fms_table = []
        with open(out_dir + 'Metrics/' + out_fms_filename) as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=',')
            for row in csv_reader:
                out_fms_table.append(row)
        
        fms_table_exp = [['Features', 'log reg', 'kNN', 'GNB', 'SVM'],
                         ['10', '0.1199', '0.0995', '0.1257', '0.124'],
                         ['30', '0.132', '0.0777', '0.1568', '0.1435']]
        
        row_counter = 0
        for row in out_fms_table:
            value_counter = 0
            for value in row:
                if (row_counter == 0):
                    pass
                else:
                    self.assertEqual(value, fms_table_exp[row_counter][value_counter])
                value_counter += 1
            row_counter += 1
            
        out_rs_filename = 'Run_Stability_PB5Way.csv'
        out_rs_table = []
        with open(out_dir + 'Metrics/' + out_rs_filename) as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=',')
            for row in csv_reader:
                out_rs_table.append(row)
        
        rs_table_exp = [['Features', 'log reg', 'kNN', 'GNB', 'SVM'],
                         ['10', '0.0', '0.0', '0.0', '0.0'],
                         ['30', '0.0', '0.0', '0.0', '0.0']]
        
        row_counter = 0
        for row in out_rs_table:
            value_counter = 0
            for value in row:
                if (row_counter == 0):
                    pass
                else:
                    self.assertEqual(value, rs_table_exp[row_counter][value_counter])
                value_counter += 1
            row_counter += 1
            
        out_ra_filename = 'ROC_AUC_PB5Way.csv'
        out_ra_table = []
        with open(out_dir + 'Metrics/' + out_ra_filename) as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=',')
            for row in csv_reader:
                out_ra_table.append(row)
        
        ra_table_exp = [['Features', 'log reg', 'kNN', 'GNB', 'SVM'],
                         ['10', '84.1%', '77.94%', '81.32%', '84.08%'],
                         ['30', '88.69%', '83.62%', '84.49%', '87.04%']]
        
        row_counter = 0
        for row in out_ra_table:
            value_counter = 0
            for value in row:
                if (row_counter == 0):
                    pass
                else:
                    self.assertAlmostEqual(float(value.replace('%', '')),
                                           float(ra_table_exp[row_counter][value_counter].replace('%', '')), delta = 1.0)
                value_counter += 1
            row_counter += 1    
            
        # Full Output Test - Same as above, except use Cuffdiff counts
        out_dir = '...'
        exp_dir = '...'
        root_dir = '...'
        reference = 'hg38'
        annotation = 'ensembl'
        aligner = 'starcq'
        normalization = 'GEOM'
        dispersion = 'POOL'
        features_file = 'top_IG_genes.txt'
        run_name = 'PB5Way'
        num_folds = 10
        num_runs = 2
        perf_metrics = rnaseq_classification_with_feature_selection({'AH':38, 'CT':20, 'DA':20, 'AA':20, 'NF':20, 'HP':20},
                                                                   ['AH', 'CT', 'DAAA', 'NF', 'HP'], reference, aligner, 
                                                                   annotation, normalization, dispersion, root_dir, 
                                                                   features_file, out_dir, [10, 30], 
                                                                   ["log reg", "kNN", "GNB", "SVM"], num_folds, num_runs, 
                                                                   gene_lists = True, validate = False,
                                                                   counts_format = 'Cuffdiff')
        
        # Verify All The Gene Lists
        i = 0 
        while i < num_folds:
            dir_fold = root_dir + reference + '_' + aligner.capitalize() + '_' + annotation.capitalize() + '/'
            dir_fold += 'FOLD' + str(num_folds) + '/Cuffdiff_' + normalization + '_' + dispersion
            dir_fold += '_FOLD' + str(i+1) + '/'
            features_filepath = dir_fold + features_file
            
            gene_list_filepath = out_dir + 'GeneLists/' + 'gene_list_fold' + str(i+1) + '_' + run_name + '.txt'
            
            with open(gene_list_filepath) as reader:
                lines_out = reader.readlines()
            
            with open(features_filepath) as reader:
                lines_exp = reader.readlines()
            
            j = 0
            while j < len(lines_out):
                self.assertEqual(lines_out[j], lines_exp[j])
                j += 1
            
            i += 1
        
        # Verify Config File
        config_file_out = out_dir + 'config_PB5Way.txt'
        with open(config_file_out) as reader:
            lines_out = reader.readlines()
        self.assertEqual(lines_out[0], 'Configuration Name: PB5Way\n')
        self.assertEqual(lines_out[1], 'Pipeline: hg38_Starcq_Ensembl\n')
        self.assertEqual(lines_out[2], 'Number of folds: 10\n')
        self.assertEqual(lines_out[3], "Samples Used: {'AH': 38, 'CT': 20, 'DA': 20, 'AA': 20, 'NF': 20, 'HP': 20}\n")
        self.assertEqual(lines_out[4], "Conditions: ['AH', 'CT', 'DAAA', 'NF', 'HP']\n")
        self.assertEqual(lines_out[5], 'Variance Transform (log(1+n)): True\n')
        self.assertEqual(lines_out[6], 'Root Directory: '...'\n')
        self.assertEqual(lines_out[7], 'Normalization: GEOM\n')
        self.assertEqual(lines_out[8], 'Dispersion: POOL\n')
        self.assertEqual(lines_out[9], "Classifiers: ['log reg', 'kNN', 'GNB', 'SVM']\n")
        self.assertEqual(lines_out[10], 'Feature Sizes (Filter): [10, 30]\n')
        self.assertEqual(lines_out[11], 'Number of runs: 2\n')
        self.assertEqual(lines_out[12], 'Feature Selection File: top_IG_genes.txt\n')
        self.assertEqual(lines_out[13], 'Tissue: PB\n')
        self.assertEqual(lines_out[14], 'Note: all performance metrics are reported as average of all the runs. \n')
        self.assertEqual(lines_out[15], 'Decision boundaries use only the two top genes. \n')
        
        # Verify Missclassified Samples File
        
        miss_file_out = out_dir + 'miss_samples_PB5Way.txt'
        with open(miss_file_out) as reader:
            lines_out = reader.readlines()
        miss_file_exp = exp_dir + 'miss_samples_PB5Way.txt'
        with open(miss_file_exp) as reader:
            lines_exp = reader.readlines()
            
        self.assertEqual(len(lines_out), len(lines_exp))
        z = 0
        for line in lines_out:
            self.assertEqual(line, lines_exp[z])
            z += 1
        
        # Verify All The Performance Metrics
        out_acc_filename = 'Accuracy_PB5Way.csv'
        out_acc_table = []
        with open(out_dir + 'Metrics/' + out_acc_filename) as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=',')
            for row in csv_reader:
                out_acc_table.append(row)
        
        acc_table_exp = [['Features', 'log reg', 'kNN', 'GNB', 'SVM'],
                          ['10', '65.22%', '62.32%', '61.59%', '65.22%'],
                          ['30', '69.57%', '65.94%', '63.77%', '68.84%']]
        row_counter = 0
        for row in out_acc_table:
            value_counter = 0
            for value in row:
                if (row_counter == 0):
                    pass
                else:
                    self.assertEqual(value, acc_table_exp[row_counter][value_counter])
                value_counter += 1
            row_counter += 1
            
        out_prec_filename = 'Precision_PB5Way.csv'
        out_prec_table = []
        with open(out_dir + 'Metrics/' + out_prec_filename) as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=',')
            for row in csv_reader:
                out_prec_table.append(row)
        
        prec_table_exp = [['Features', 'log reg', 'kNN', 'GNB', 'SVM'],
                         ['10', 'AH:76.19% CT:59.26% DAAA:56.25% NF:60.0% HP:81.82% ',
                          'AH:86.49% CT:51.43% DAAA:58.7% NF:37.5% HP:50.0% ',
                          'AH:86.49% CT:46.88% DAAA:61.54% NF:40.0% HP:53.33% ',
                          'AH:78.05% CT:50.0% DAAA:67.74% NF:57.14% HP:66.67% '],
                         ['30', 'AH:85.37% CT:64.29% DAAA:71.43% NF:53.85% HP:52.38% ',
                          'AH:85.71% CT:52.94% DAAA:56.25% NF:77.78% HP:75.0% ',
                          'AH:85.0% CT:50.0% DAAA:63.89% NF:42.86% HP:55.0% ',
                          'AH:86.11% CT:57.14% DAAA:67.44% NF:57.14% HP:64.71% ']]
        
        row_counter = 0
        for row in out_prec_table:
            value_counter = 0
            for value in row:
                if (row_counter == 0):
                    pass
                else:
                    self.assertEqual(value, prec_table_exp[row_counter][value_counter])
                value_counter += 1
            row_counter += 1
            
        out_rec_filename = 'Recall_PB5Way.csv'
        out_rec_table = []
        with open(out_dir + 'Metrics/' + out_rec_filename) as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=',')
            for row in csv_reader:
                out_rec_table.append(row)
        
        rec_table_exp = [['Features', 'log reg', 'kNN', 'GNB', 'SVM'],
                         ['10', 'AH:84.21% CT:80.0% DAAA:67.5% NF:30.0% HP:45.0% ',
                          'AH:84.21% CT:90.0% DAAA:67.5% NF:15.0% HP:30.0% ',
                          'AH:84.21% CT:75.0% DAAA:60.0% NF:30.0% HP:40.0% '
                          ,'AH:84.21% CT:85.0% DAAA:52.5% NF:40.0% HP:60.0% '],
                         ['30', 'AH:92.11% CT:90.0% DAAA:62.5% NF:35.0% HP:55.0% ',
                          'AH:78.95% CT:90.0% DAAA:67.5% NF:35.0% HP:45.0% ',
                          'AH:89.47% CT:70.0% DAAA:57.5% NF:30.0% HP:55.0% ',
                          'AH:81.58% CT:80.0% DAAA:72.5% NF:40.0% HP:55.0% ']]
        
        row_counter = 0
        for row in out_rec_table:
            value_counter = 0
            for value in row:
                if (row_counter == 0):
                    pass
                else:
                    self.assertEqual(value, rec_table_exp[row_counter][value_counter])
                value_counter += 1
            row_counter += 1
            
        out_bacc_filename = 'Balanced_Accuracy_PB5Way.csv'
        out_bacc_table = []
        with open(out_dir + 'Metrics/' + out_bacc_filename) as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=',')
            for row in csv_reader:
                out_bacc_table.append(row)
        
        bacc_table_exp = [['Features', 'log reg', 'kNN', 'GNB', 'SVM'],
                         ['10', '61.34%', '57.34%', '57.84%', '64.34%'],
                         ['30', '66.92%', '63.29%', '60.39%', '65.82%']]
        
        row_counter = 0
        for row in out_bacc_table:
            value_counter = 0
            for value in row:
                if (row_counter == 0):
                    pass
                else:
                    self.assertEqual(value, bacc_table_exp[row_counter][value_counter])
                value_counter += 1
            row_counter += 1
            
        out_f1_filename = 'F1_PB5Way.csv'
        out_f1_table = []
        with open(out_dir + 'Metrics/' + out_f1_filename) as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=',')
            for row in csv_reader:
                out_f1_table.append(row)
        
        f1_table_exp = [['Features', 'log reg', 'kNN', 'GNB', 'SVM'],
                         ['10', 'AH:80.0% CT:68.09% DAAA:61.36% NF:40.0% HP:58.06% ',
                          'AH:85.33% CT:65.45% DAAA:62.79% NF:21.43% HP:37.5% ',
                          'AH:85.33% CT:57.69% DAAA:60.76% NF:34.29% HP:45.71% ',
                          'AH:81.01% CT:62.96% DAAA:59.15% NF:47.06% HP:63.16% '],
                         ['30', 'AH:88.61% CT:75.0% DAAA:66.67% NF:42.42% HP:53.66% ',
                          'AH:82.19% CT:66.67% DAAA:61.36% NF:48.28% HP:56.25% ',
                          'AH:87.18% CT:58.33% DAAA:60.53% NF:35.29% HP:55.0% ',
                          'AH:83.78% CT:66.67% DAAA:69.88% NF:47.06% HP:59.46% ']]
        
        row_counter = 0
        for row in out_f1_table:
            value_counter = 0
            for value in row:
                if (row_counter == 0):
                    pass
                else:
                    self.assertEqual(value, f1_table_exp[row_counter][value_counter])
                value_counter += 1
            row_counter += 1
            
        out_fms_filename = 'Fold_Mean_Stability_PB5Way.csv'
        out_fms_table = []
        with open(out_dir + 'Metrics/' + out_fms_filename) as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=',')
            for row in csv_reader:
                out_fms_table.append(row)
        
        fms_table_exp = [['Features', 'log reg', 'kNN', 'GNB', 'SVM'],
                         ['10', '0.1199', '0.0995', '0.1257', '0.124'],
                         ['30', '0.132', '0.0777', '0.1568', '0.1435']]
        
        row_counter = 0
        for row in out_fms_table:
            value_counter = 0
            for value in row:
                if (row_counter == 0):
                    pass
                else:
                    self.assertEqual(value, fms_table_exp[row_counter][value_counter])
                value_counter += 1
            row_counter += 1
            
        out_rs_filename = 'Run_Stability_PB5Way.csv'
        out_rs_table = []
        with open(out_dir + 'Metrics/' + out_rs_filename) as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=',')
            for row in csv_reader:
                out_rs_table.append(row)
        
        rs_table_exp = [['Features', 'log reg', 'kNN', 'GNB', 'SVM'],
                         ['10', '0.0', '0.0', '0.0', '0.0'],
                         ['30', '0.0', '0.0', '0.0', '0.0']]
        
        row_counter = 0
        for row in out_rs_table:
            value_counter = 0
            for value in row:
                if (row_counter == 0):
                    pass
                else:
                    self.assertEqual(value, rs_table_exp[row_counter][value_counter])
                value_counter += 1
            row_counter += 1
            
        out_ra_filename = 'ROC_AUC_PB5Way.csv'
        out_ra_table = []
        with open(out_dir + 'Metrics/' + out_ra_filename) as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=',')
            for row in csv_reader:
                out_ra_table.append(row)
        
        ra_table_exp = [['Features', 'log reg', 'kNN', 'GNB', 'SVM'],
                         ['10', '84.1%', '77.94%', '81.32%', '84.08%'],
                         ['30', '88.69%', '83.62%', '84.49%', '87.04%']]
        
        row_counter = 0
        for row in out_ra_table:
            value_counter = 0
            for value in row:
                if (row_counter == 0):
                    pass
                else:
                    self.assertAlmostEqual(float(value.replace('%', '')),
                                           float(ra_table_exp[row_counter][value_counter].replace('%', '')), delta = 1.0)
                value_counter += 1
            row_counter += 1    
    
    def test_generate_top_DE_genes(self):
        for pipeline in pipelines_ALL:
            curr_result_gen = generate_top_genes_DE('...', pipeline,
                                                    'GEOM', 'POOL', 100, {'AH':38, 'CT':20}, 'MEAN', [],
                                                    False, True, True, 0)
            filename = "top_100_genes_" + pipeline + ".txt"
            root = os.getcwd()
            root = root.replace("\\", '/')
            full_filename = root + "/TestInput/TopGenesDE/" + filename
            curr_result_record = read_in_csv_file_one_column(full_filename, 0, '\t')
            v0 = set(curr_result_record)
            v1 = set(curr_result_gen)
            intersection = v0 & v1
            self.assertGreater(len(intersection), 95)
            
    def test_generate_top_IG_genes(self):
        curr_result_gen = generate_top_genes_IG('...', 'hg38_Starcq_Ensembl',
                                                'GEOM', 'POOL', 10, {'AH':38, 'CT':20}, True, False, True, 0)
        exp_genes = ['G6PD', 'MMP14', 'MS4A4A', 'HADHB', 'C1QC', 'IQCE', 'PGD', 'SERPINB1', 'RAB31', 'H2AFY']
        self.assertEqual(curr_result_gen, exp_genes)
        
        curr_result_gen = generate_top_genes_IG('...', 'hg38_Starcq_Ensembl',
                                                'GEOM', 'POOL', 10, {'AH':38, 'CT':20}, True, False, True, 0,
                                                counts_format = 'Cuffdiff')
        exp_genes = ['G6PD', 'MMP14', 'MS4A4A', 'HADHB', 'C1QC', 'IQCE', 'PGD', 'SERPINB1', 'RAB31', 'H2AFY']
        self.assertEqual(curr_result_gen, exp_genes)
        
        # curr_result_gen = generate_top_genes_IG('...', 'hg38_Starcq_Ensembl',
        #                                         'GEOM', 10, {'AH':38, 'CT':20, 'DA':20}, True, False, True, 0)
        # exp_genes = ['TCN2', 'PTGR1', 'FAM20A', 'MS4A4A', 'MERTK', 'SLC11A1', 'GAS6', 'SERPINB1', 'LPCAT3', 'FLVCR2']
        # self.assertEqual(curr_result_gen, exp_genes)
        
    def test_generate_top_RF_genes(self):
        curr_result_gen = generate_top_genes_RF('...', 'hg38_Starcq_Ensembl',
                                                'GEOM', 'POOL', 200, {'AH':38, 'CT':20}, 5000, 5, True, False, 0)
        curr_result_gen2 = generate_top_genes_RF('...', 'hg38_Starcq_Ensembl',
                                                'GEOM', 'POOL', 200, {'AH':38, 'CT':20}, 5000, 5, True, False, 0,
                                                counts_format = 'Cuffdiff')
        filename = "top_200_genes_hg38_Starcq_Ensembl_RF_vtrans_avg(5000).txt"
        
        root = os.getcwd()
        root = root.replace("\\", '/')
        full_filename = root + "/TestInput/TopGenesRF/" + filename
        
        curr_result_record = read_in_csv_file_one_column(full_filename, 0, '\t')
        
        v0 = set(curr_result_record)
        v1 = set(curr_result_gen)
        v2 = set(curr_result_gen2)
        #print(v0)
        #print(v1)
        #print(v2)
        intersection = v0 & v1
        intersection2 = v0 & v2
        
        #print("Test generate top RF binary; intersection length out of 200: ", len(intersection))
        self.assertGreater(len(intersection), 160)
        self.assertGreater(len(intersection2), 160)
        
def rnaseq_classification_with_feature_selection(num_samples:dict, class_labels:list, reference:str, aligner:str,
                                                 annotation:str, normalization:str, dispersion:str,
                                                 root_dir:str, features_file:str, out_dir:str, num_features:list,
                                                 classifiers:list, num_folds:int = 10, num_runs:int = 10,
                                                 tissue:str = 'PB', perf_metrics:list = ['Accuracy', 'Run_Stability', 
                                                 'Fold_Mean_Stability', 'Precision', 'Recall', 'ROC_AUC',
                                                 'Balanced_Accuracy','F1'], taboo_list:list = [], wrapper:str = None,
                                                 target_size:int = None, sfs_replace:bool = True, search_heuristic:str = 'Grid',
                                                 gene_lists:bool = False, conf_matrices:bool = False, dec_boundaries:bool = False,
                                                 mean_variance_plots:bool = False, ROC_plot:bool = False,
                                                 var_transform:bool = True, validate:bool = False,
                                                 counts_format:str = 'Cuffnorm', verbose:bool = True):    
    ''' This function performs multi-class (>= 2 classes) classification of the RNA-seq counts attained by sequencing of blood 
    samples taken from liver disease patients and healthy controls. 
    num_samples: A dictionary that maps each class name to the nubmer of samples within that class. 
                 Such as AH, CT, etc. Ex: {'AH':38, 'CT':20, 'DA':20, 'AA':20, 'NF':20, 'HP':20}.
                 These corresponds to actual classes as defined by experiment.
    class_labels: These are names of the classes that will be used in confusion matrix annotation. 
                  These corresponds to classes as defined in Cuffnorm. Ex: ['AH', 'CT', 'DAAA', 'NF', 'HP'].
    reference: The genome reference used in calculation of RNA-seq counts.
    aligner: The genome aligner used in calculation of RNA-seq counts.
    annotation: The genome annotation used in calculation of RNA-seq counts.
    normalization: The normalization algorithm used within Cuffnorm (RNA-seq Counts software) 
                   and Cuffdiff (Differential Expression Software).
    dispersion: The dispersion algorithm used within Cuffdiff.
    root_dir: The directory containing the RNA-seq files. Specifically, the Cuffdiff and Cuffnorm files.
    features_file: List of genes provided for each fold individually. 
    out_dir: Directory into which all output files will be placed.
    num_features: The total number of features to be used within the Machine Learning models. Most relevant features are selected 
    according to feature selection method.
    classifiers: The machine learning models used to classify the RNA-seq counts.
    num_folds: Number of folds to be used for cross-validation.
    num_runs: The total number of times that each Machine Learning model is ran before being averaged. This helps to address 
    unstable models whose performance varies highly between runs.
    tissue: 'PB', 'PB_Excluded', 'LV', 'LV_Excluded', 'TEST'
    perf_metrics: All the metrics to report within individual csv files.
    taboo_list: All the genes (prefixes) that should not be used for training/validation purposes.
    wrapper: The choice of a wrapper model. Can be 'RFE', 'SFS', or None.
    target_size: Target size to use in the wrapper model. Only used if wrapper is not None.
    sfs_replace: Whether to use replacement or not within SFS model.
    search_heuristic: 'Random' or 'Grid'.
    gene_lists: Boolean specifying whether to produce per-fold gene lists.
    conf_matrices: Boolean specifying whether to produce confusion matrices.
    dec_boundaries: Boolean specifying whether to produce decision boundaries.
    mean_variance_plots: Whether to generate plots of mean gene expression against variance of gene expression in the console.
    ROC_plot: Whether to generate ROC plots. This will only work for binary classification.
    var_transform: Feature transformation that reduces within feature variance. Default is a log(1 + x) function.
    validate: Whether to perform some simple validation/verification of the Cuffnorm and Cuffdiff files ensuring that the 
    correct samples were analyzed.
    counts_format: either 'Cuffnorm' (fpkm talbe) or 'Cuffdiff' (read_group_tracking).
    verbose: Whether to print additional output, primarily in the performance log file.
    '''
    
    # Turn off matplot lib interactive mode.
    plt.ioff()
    
    # The relevant cuffnorm (rna-seq count files) are genes.fpkm_table and samples.table.
    # The relevant cuffdiff (rna-seq differential expression files) are genes.read_group_tracking, 
    # read_groups.info, gene_exp.diff, and gene_exp.text (filtered gene_exp.diff).
    aligner = aligner.capitalize()
    annotation = annotation.capitalize()
    
    pipeline = reference + '_' + aligner + '_' + annotation
    
    # Output a text file with configuration/pipeline information.
    temp = out_dir.split('/')
    run_name = temp[-2]
    if(not os.path.isdir(out_dir)):
        os.mkdir(out_dir)
    f = open(out_dir + 'config_' + run_name + '.txt', "w")
    f.write('Configuration Name: ' + run_name + '\n')
    f.write('Pipeline: ' + pipeline + '\n')
    f.write('Number of folds: ' + str(num_folds) + '\n')
    f.write('Samples Used: ' + str(num_samples) + '\n')
    f.write('Conditions: ' + str(class_labels) + '\n')
    f.write('Variance Transform (log(1+n)): ' + str(var_transform) + '\n')
    f.write('Root Directory: ' + root_dir + '\n')
    f.write('Normalization: ' + normalization + '\n')
    f.write('Dispersion: ' + dispersion + '\n')
    f.write('Classifiers: ' + str(classifiers) + '\n')
    f.write('Feature Sizes (Filter): ' + str(num_features) + '\n')
    f.write('Number of runs: ' + str(num_runs) + '\n')
    f.write('Feature Selection File: ' + features_file + '\n')
    f.write('Tissue: ' + tissue + '\n')
    #f.write('Taboo List Present: ' + bool(taboo_list) + '\n')
    f.write('Note: all performance metrics are reported as average of all the runs. \n')
    f.write('Decision boundaries use only the two top genes. \n')
    f.close()
    
    # VALIDATE CUFFNORM AND CUFFDIFF FILES AGAINST EACH OTHER.
    if(validate):
        validate_cuffnorm_cuffdiff_pipeline_files_one_setting(root_dir, pipeline, num_folds, normalization, dispersion)
        print("Passed cuffnorm and cuffdiff file validation checks.")
    if(mean_variance_plots):
        # EXPLORE CUFFNORM COUNTS
        fname = ''
        if(counts_format == 'Cuffdiff'):
            fname = root_dir + '/' + pipeline + '/Cuffdiff_' + normalization + '/genes.fpkm_table'
        elif(counts_format == 'Cuffnorm'):
            fname = root_dir + '/' + pipeline + '/Cuffnorm_' + normalization + '/genes.fpkm_table'
        else:
            raise ValueError('Counts format must be Cuffnorm or Cuffdiff.')
            
        out_name = 'Feature_Variance_' + pipeline + '_' + normalization + '_' + dispersion
        
        generate_variance_mean_plots(fname, out_name + '_False', False, counts_format)
        # EXPLORE LOG TRANSFORMED CUFFNORM COUNTS
        generate_variance_mean_plots(fname, out_name + '_True', True, counts_format)
    
    to_return = train_validate_ML_models(num_samples, class_labels, pipeline, normalization, dispersion, root_dir, 
                                         features_file, out_dir, num_features, classifiers, num_folds, num_runs, 
                                         tissue, perf_metrics, taboo_list, wrapper, target_size, sfs_replace,
                                         search_heuristic, gene_lists, conf_matrices, dec_boundaries, ROC_plot,
                                         var_transform, counts_format, verbose)
    # plt.ion()
    return to_return
    
    
def test_suite():
    # Unit Tests
    unit_test_suite = unittest.TestSuite()
    
    # Validation Tests:
    unit_test_suite.addTest(TestAHProjectCodeBase("test_compare_cuffdiff_cuffdiff"))
    unit_test_suite.addTest(TestAHProjectCodeBase("test_compare_cuffnorm_cuffdiff"))
    unit_test_suite.addTest(TestAHProjectCodeBase("test_compare_cuffnorm_cuffnorm"))
    unit_test_suite.addTest(TestAHProjectCodeBase("test_compare_cuffdiff_read_groups_info_files"))
    unit_test_suite.addTest(TestAHProjectCodeBase("test_compare_cuffnorm_sample_table_files"))
    unit_test_suite.addTest(TestAHProjectCodeBase("test_compare_cuffdiff_mappings_to_cuffnorm_mappings"))
    
    # Simple/Helper Function Tests:
    unit_test_suite.addTest(TestAHProjectCodeBase("test_generate_kfolds"))
    unit_test_suite.addTest(TestAHProjectCodeBase("test_two_dim_list_len"))
    unit_test_suite.addTest(TestAHProjectCodeBase("test_calculate_pipeline_accuracy_metrics"))
    unit_test_suite.addTest(TestAHProjectCodeBase("test_filenames_to_replicate_names_cuffnorm"))
    unit_test_suite.addTest(TestAHProjectCodeBase("test_filenames_to_replicate_names_cuffdiff"))
    
    # CSV File Functions:
    unit_test_suite.addTest(TestAHProjectCodeBase("test_compare_one_column_csv_file_contents"))
    unit_test_suite.addTest(TestAHProjectCodeBase("test_compare_csv_file_contents"))
    unit_test_suite.addTest(TestAHProjectCodeBase("test_compare_csv_files"))
    unit_test_suite.addTest(TestAHProjectCodeBase("test_read_in_csv_file_one_column"))
    
    # TXT File Functions:
    unit_test_suite.addTest(TestAHProjectCodeBase("test_compare_one_column_txt_file_contents"))
    
    # Read & parse rna-seq data:
    unit_test_suite.addTest(TestAHProjectCodeBase("test_filter_cuffdiff_file_by_gene_list"))
    unit_test_suite.addTest(TestAHProjectCodeBase("test_read_cuffdiff_counts_mean"))
    unit_test_suite.addTest(TestAHProjectCodeBase("test_read_cuffdiff_counts"))
    unit_test_suite.addTest(TestAHProjectCodeBase("test_read_cuffnorm_counts"))
    unit_test_suite.addTest(TestAHProjectCodeBase("test_read_cuffnorm_samples_table_filenames"))
    unit_test_suite.addTest(TestAHProjectCodeBase("test_read_in_cuffdiff_gene_exp_file"))
    unit_test_suite.addTest(TestAHProjectCodeBase("test_read_in_cuffdiff_gene_exp_file_pairwise"))
    unit_test_suite.addTest(TestAHProjectCodeBase("test_read_cuffnorm_counts2"))
    unit_test_suite.addTest(TestAHProjectCodeBase("test_read_cuffdiff_counts2"))
    unit_test_suite.addTest(TestAHProjectCodeBase("test_read_cuffnorm_gene_names"))
    unit_test_suite.addTest(TestAHProjectCodeBase("test_read_cuffdiff_gene_names"))
    unit_test_suite.addTest(TestAHProjectCodeBase("test_read_cuffdiff_group_info_filenames"))
    unit_test_suite.addTest(TestAHProjectCodeBase("test_read_cuffnorm_counts_mean_variance"))
    unit_test_suite.addTest(TestAHProjectCodeBase("test_read_cuffdiff_counts_mean_variance"))
    unit_test_suite.addTest(TestAHProjectCodeBase("test_detect_outlier_features_by_std"))
    unit_test_suite.addTest(TestAHProjectCodeBase("test_generate_filtered_csv_file_one_col"))
    unit_test_suite.addTest(TestAHProjectCodeBase("test_select_top_cuffdiff_DEs"))
    unit_test_suite.addTest(TestAHProjectCodeBase("test_count_nonmRNAs"))
    unit_test_suite.addTest(TestAHProjectCodeBase("test_filter_cuffnorm_counts"))
    
    #Generate data:
    unit_test_suite.addTest(TestAHProjectCodeBase("test_generate_train_validate_data"))
    unit_test_suite.addTest(TestAHProjectCodeBase("test_generate_data"))
    unit_test_suite.addTest(TestAHProjectCodeBase("test_generate_variance_mean_plots"))
    unit_test_suite.addTest(TestAHProjectCodeBase("test_generate_cuffnorm_or_cuffdiff_batch_file_HPC"))
    unit_test_suite.addTest(TestAHProjectCodeBase("test_generate_cond_name_to_rep_name_map"))
    unit_test_suite.addTest(TestAHProjectCodeBase("test_identify_misclassified_samples"))
    
    # Machine Learning:
    unit_test_suite.addTest(TestAHProjectCodeBase("test_tune_ML_model"))
    
    # Black Box Tests
    black_box_suite = unittest.TestSuite()
    
    black_box_suite.addTest(TestAHProjectCodeBase("test_rnaseq_classification_with_feature_selection_binary"))
    black_box_suite.addTest(TestAHProjectCodeBase("test_rnaseq_classification_with_feature_selection_multiclass"))
    black_box_suite.addTest(TestAHProjectCodeBase("test_generate_top_RF_genes"))
    black_box_suite.addTest(TestAHProjectCodeBase("test_generate_top_IG_genes"))
    black_box_suite.addTest(TestAHProjectCodeBase("test_generate_top_DE_genes"))
    
    runner = unittest.TextTestRunner()
    runner.run(unit_test_suite)
    runner.run(black_box_suite)

if __name__ == "__main__":
    # MUST BE PYTHON 3.7+ since, DICTIONARIES are ASSUMED to be ORDERED throughout the codebase.
    assert sys.version_info.major == 3
    assert sys.version_info.minor >= 7   
    
    plt.rcParams['figure.dpi'] = 300
    plt.rcParams['savefig.dpi'] = 300
    
    # test_suite()
    
    # ---------------------------------------------------------------------------------------------------------------- 
    # ***************************************************SCRATCH SPACE************************************************
    # ---------------------------------------------------------------------------------------------------------------- 
    
    
    # ---------------------------------------------------------------------------------------------------------------- 
    # ***************************************************GENERATING TOP FEATURES**************************************
    # ----------------------------------------------------------------------------------------------------------------
    
    '''
    i = 1
    while i < 9:
        generate_top_genes_DE('...', 'hg38_Starcq_Ensembl',
                              'GEOM', 'POOL', 1000, {'AH':34, 'CT':18, 'DA':18, 'AA':18, 'NF':18, 'HP':17}, 'Pairwise', [],
                              True, True, True, i)
        i += 1
    generate_top_genes_DE('...', 'hg38_Starcq_Ensembl',
                              'GEOM', 'POOL', 1000, {'AH':35, 'CT':18, 'DA':18, 'AA':18, 'NF':18, 'HP':17}, 'Pairwise', [],
                              True, True, True, 9)
    generate_top_genes_DE('...', 'hg38_Starcq_Ensembl',
                              'GEOM', 'POOL', 1000, {'AH':35, 'CT':18, 'DA':18, 'AA':18, 'NF':18, 'HP':18}, 'Pairwise', [],
                              True, True, True, 10)
    
    '''
    
    '''
    i = 1
    while i < 9:
        generate_top_genes_RF('...', 'hg38_Starcq_Ensembl',
                              'GEOM', 'POOL', 10000, {'AH':34, 'CT':18, 'DA':18, 'AA':18, 'NF':18, 'HP':17},
                              5000, 5, True, True, i, num_folds = 10, tissue ='PB_Excluded', counts_format = 'Cuffdiff')
        i += 1
       
    generate_top_genes_RF('...', 'hg38_Starcq_Ensembl',
                              'GEOM', 'POOL', 10000, {'AH':35, 'CT':18, 'DA':18, 'AA':18, 'NF':18, 'HP':17},
                              5000, 5, True, True, 9, num_folds = 10, tissue ='PB_Excluded', counts_format = 'Cuffdiff')
    
    generate_top_genes_RF('...', 'hg38_Starcq_Ensembl',
                              'GEOM', 'POOL', 10000, {'AH':35, 'CT':18, 'DA':18, 'AA':18, 'NF':18, 'HP':18},
                              5000, 5, True, True, 10, num_folds = 10, tissue ='PB_Excluded', counts_format = 'Cuffdiff')
    
    '''
    # ---------------------------------------------------------------------------------------------------------------- 
    # ************************************GENERATE CUFFDIFF / CUFFNORM FILES******************************************
    # ---------------------------------------------------------------------------------------------------------------- 
    '''
    generate_cuffnorm_or_cuffdiff_batch_file_HPC('hg38', 'starcq', 'ensembl', 'GEOM',['AH', 'CT', 'DA', 'AA', 'NF', 'HP'],
                                                 'Cuffdiff', 1, 'POOL', 'PB_Excluded')
    '''
    # ---------------------------------------------------------------------------------------------------------------- 
    # ***********************************COMPARING CUFFDIFF / CUFFNORM FILES******************************************
    # ---------------------------------------------------------------------------------------------------------------- 
    
    root1 = '...'
    root2 = '...'
    '''
    # Compare read group info files.
    print(compare_cuffdiff_read_groups_info_files([root1+'read_groups.info', root2+'read_groups.info']))
    
    # Compare read group tracking files.
    
    # Comparing exact counts for each sample.
    
    counts1 = read_cuffdiff_counts(root1 + 'genes.read_group_tracking')
    counts2 = read_cuffdiff_counts(root2 + 'genes.read_group_tracking')
    
    inc_entries = 0
    i = 0
    margin = 1e-6
    for key,value in counts1.items():
        try:
            if(math.isclose(counts2[key], value, rel_tol = margin)):
                pass
            else:
                inc_entries = inc_entries + 1
                # print("GENE_REPLICATE: ", key, "; File1 value: ", value, "; File2 value: ", counts2[key])
        except KeyError:
            #print("Warning! Gene: ", key, "found in file1, but not file2.")
            inc_entries = inc_entries + 1
            pass
        i = i + 1
        
    print(inc_entries/i * 100)
    
    # Comparing mean of counts for each condition.
    compare_cuffdiff_cuffdiff(root1+'genes.read_group_tracking', root2 + 'genes.read_group_tracking', 0.001)
    
    # Compare gene_exp.diff files.
    
    fcs1 = get_cuffdiff_gene_exp_file_fcs(root1 + 'gene_exp.diff', 1, 0, ['q1', 'q2', 'q3', 'q4', 'q5'])
    fcs2 = get_cuffdiff_gene_exp_file_fcs(root2 + 'gene_exp.diff', 1, 0, ['q1', 'q2', 'q3', 'q4', 'q5'])
    
    print(len(fcs1))
    print(len(fcs2))
    
    for k,v in fcs1.items():
        try:
            if (not compare_float_lists(v, fcs2[k])):
                print("Gene: ", k, " has different recorded fold changes across two files.")
        except KeyError:
            print("Warning! Gene: ", k, "found in file1, but not file2.")
    '''
    #print(compare_cuffnorm_cuffnorm(root2+'genes.fpkm_table', root1+'genes.fpkm_table', 0.01))
    
    
    # ---------------------------------------------------------------------------------------------------------------- 
    # ***********************************FEATURE SELECTION AND CLASSIFICATION BLOCK***********************************
    # ---------------------------------------------------------------------------------------------------------------- 
    
    # Code block to execute feature selection and classification pipeline.
    
    '''
    # Code block to generate the non protein-coding gene list using ensembl database file.
    
    filename = '...'
    
    pc_genes = []
    with open(filename) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        for row in csv_reader:
            if(row[6] in ['protein_coding', 'IG_C_gene', 'IG_D_gene', 'IG_J_gene', 'IG_V_gene', 'TR_C_gene',
                          'TR_D_gene', 'TR_V_gene', 'TR_J_gene', 'polymorphic_pseudogene']):
                pc_genes.append(row[3])
    pc_genes = set(pc_genes)
    print(len(pc_genes))
    
    filename = '...'
    all_genes_data = read_in_csv_file_one_column(filename, 0, '\t')
    
    
    non_pc_genes = []
    for gene in all_genes_data:
        if gene not in pc_genes:
            non_pc_genes.append(gene)
    
    non_pc_genes = set(non_pc_genes)
    
    write_list_to_file('...',
                       list(non_pc_genes))
    '''
    
    '''
    # Code block to identify highly varying genes.
    fname = '...'
    out_dir = '...'
    
    detect_outlier_features_by_std(fname, 6, out_dir, 'Cuffdiff', 3.5)
    '''
    
    '''
    fname = '...'
    fname += '...'
    #fname_HPC ='...'
    #fname_HPC += '...'
    std_list = read_in_csv_file_one_column(fname, 0, ',')
    
    fname = '...'
    fname += '...'
    #fname_HPC = '...'
    #fname_HPC += '...'
    non_pc_genes = read_in_csv_file_one_column(fname, 0, ',')
    
    outlier = list(set(non_pc_genes) & set(std_list))
    
    
    #root_dir = '...'
    root_dir_HPC = '...'
    root_dir_HPC += '...'
    reference = 'hg38'
    annotation = 'ensembl'
    aligner = 'starcq'
    normalization = 'GEOM'
    dispersion = 'POOL'
    features_files1 = {'DE':'top_1000_DE_Pairwise.txt'}
    features_files2 = {'IG':'top_IG_genes.txt'}
    features_files3 = {'RF': 'top_RF_genes.txt'}
    taboo_lists1 = {'None': []}
    taboo_lists4 = {'Outlier': outlier}
    num_folds = 10
    num_runs = 5
    filter_feature_sizes = [10,30,50,100,250,500]
    #filter_feature_sizes = [10]
    feature_pool_sizes = [30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45]
    #target_sizes = [12,15,18,21,25]
    target_sizes = [12,15]
    '''
    
    # Filter Runs
    '''
    for filter_mode,features_file in features_files1.items():
        for taboo_name, taboo_list in taboo_lists4.items():
            out_dir_HPC = '...'
            out_dir_HPC += '...'
            # out_dir = '...'
            # out_dir += '...'
            perf_metrics = rnaseq_classification_with_feature_selection({'AH':38, 'CT':20, 'DA':20, 'AA':20, 'NF':20, 'HP':19},
                                                                        ['AH', 'CT', 'DA', 'AA', 'NF', 'HP'], reference,
                                                                        aligner, annotation, normalization, dispersion,
                                                                        root_dir_HPC, features_file, out_dir_HPC,
                                                                        filter_feature_sizes, 
                                                                        ['log reg', 'kNN', 'GNB', 'SVM', 'NN'], 
                                                                        num_folds, num_runs, tissue = 'PB_Excluded',
                                                                        taboo_list = taboo_list, wrapper = None,
                                                                        target_size = None, sfs_replace = False,
                                                                        gene_lists = True, validate = False,
                                                                        counts_format = 'Cuffdiff')
    '''
    # Filter + Wrapper Runs
    '''
    for filter_mode,features_file in features_files1.items():
        for taboo_name, taboo_list in taboo_lists4.items():
            for ts in target_sizes:
                out_dir_HPC = '...'
                out_dir_HPC += '...'
                # out_dir = '...'
                # out_dir += '...'
                perf_metrics = rnaseq_classification_with_feature_selection({'AH':38, 'CT':20, 'DA':20, 'AA':20, 'NF':20, 'HP':19},
                                                                            ['AH', 'CT', 'DA','AA', 'NF', 'HP'], reference, aligner, 
                                                                            annotation, normalization, dispersion, root_dir_HPC, 
                                                                            features_file, out_dir_HPC, feature_pool_sizes, 
                                                                            ['log reg', 'kNN'], num_folds, num_runs,
                                                                            tissue = 'PB_Excluded', taboo_list = taboo_list,
                                                                            wrapper = 'SFS', target_size = ts,
                                                                            sfs_replace = False, gene_lists = True, 
                                                                            validate = False, counts_format = 'Cuffdiff')
    '''
    '''
    # Wrapper + Embedded Runs
    feature_sizes = [500, 1000, 4000, 8000]
    target_sizes = [10,20,30]
    #feature_sizes = [100]
    #target_sizes = [10]
    for filter_mode,features_file in features_files3.items():
        for taboo_name, taboo_list in taboo_lists4.items():
            for ts in target_sizes:
                out_dir_HPC = '...'
                out_dir_HPC += '...'
                #out_dir = '...'
                #out_dir += '...'
                perf_metrics = rnaseq_classification_with_feature_selection({'AH':38, 'CT':20, 'DA':20, 'AA':20, 'NF':20, 'HP':19},
                                                                            ['AH', 'CT', 'DAAA', 'NF', 'HP'], reference, aligner, 
                                                                            annotation, normalization, dispersion, root_dir_HPC, 
                                                                            features_file, out_dir_HPC, feature_sizes, 
                                                                            ['RF'], num_folds, num_runs,  
                                                                            tissue = 'PB_Excluded', taboo_list = taboo_list,
                                                                            wrapper = 'RFE', target_size = ts,
                                                                            sfs_replace = False, gene_lists = True,
                                                                            validate = False, counts_format = 'Cuffdiff')
    '''
    
    # ---------------------------------------------------------------------------------------------------------------- 
    # ***********************************BIOLOGICAL VALIDATION BLOCK**************************************************
    # ---------------------------------------------------------------------------------------------------------------- 
    
    
    
    filter_methods = ['DE', 'IG']
    #filter_methods = ['IG']
    taboo_strats = ['None', 'Outlier']
    # taboo_strats = ['None']
    target_sizes = [15]
    feature_pool_sizes = [30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45]
    # feature_pool_sizes = [30,31,32,33,34,35]
    feature_size = 30
    ML_Models = ['kNN', 'LR']
    # #ML_Models = ['kNN']
    # datasets = ['PBMC_AH_CT_DAAA_NF_HP', 'PBMC_AH_CT_DA_AA_NF_HP', 'PBMC_AH_CT', 'LV_AC_AH_CT_HP_NF']
    datasets = ['PBMC_AH_CT_DAAA_NF_HP_Excluded']
    
    out_dir = '...'
    filename = out_dir + 'PB5Way_Excluded_RF_RFE30.txt'
    writer = open(filename, 'w')
    
    # Manual run of biological validation
    '''
    out_dir = '...'
    
    total = ['DEFA4', 'SDC2', 'AHSP', 'AC005912.1', 'IGLV3-10', 'IFIT3', 'SLC4A1', 'SELENBP1', 'TGM2', 'CD177', 'IGHG1', 'USP32P1', 'PI3', 'HES4', 'KRT1', 'GYPB', 'USP18', 'IFITM1', 'LRRN3', 'ALAS2', 'CRISP3', 'CA1', 'SEMA6B', 'CCL3L3', 'IGKV2-29', 'HBG2', 'MYL4', 'RNU4ATAC', 'DEFA3', 'CYP4F29P', 'RSAD2', 'HBM', 'ALPL', 'CXCL10', 'CCL7', 'LAMB3', 'SEC14L2', 'RNU6-574P', 'FAM20A', 'TNS1', 'SERPINB2', 'HBD', 'OLFM4', 'SERPING1', 'IFI27', 'MMP8', 'SIGLEC1', 'IGLV2-23', 'LY75-CD302', 'C1QB']
    r1,r2,r7 = biological_validation(total, 'test', out_dir, 'Pathway')
    r3,r4,r8 = biological_validation(total, 'test', out_dir, 'Tissue')
    r5,r6,r9 = biological_validation(total, 'test', out_dir, 'Disease')
    '''
    # Biological Validation of Filter (IG, DE)
    
    '''
    for dataset in datasets:
        for filter_method in filter_methods:
            for taboo_strat in taboo_strats:
                folder_name = dataset + '_' + filter_method + '_Filter_' + taboo_strat
    
                file_dir = '...'
                file_dir += dataset + '/FinalSetup/' + folder_name + '/GeneLists/'
                
                file1 = file_dir + 'gene_list_fold1_'+ folder_name + '.txt'
                file2 = file_dir + 'gene_list_fold2_'+ folder_name + '.txt'
                file3 = file_dir + 'gene_list_fold3_'+ folder_name + '.txt'
                file4 = file_dir + 'gene_list_fold4_'+ folder_name + '.txt'
                file5 = file_dir + 'gene_list_fold5_'+ folder_name + '.txt'
                if(dataset[0:4] == 'PBMC'):
                    file6 = file_dir + 'gene_list_fold6_'+ folder_name + '.txt'
                    file7 = file_dir + 'gene_list_fold7_'+ folder_name + '.txt'
                    file8 = file_dir + 'gene_list_fold8_'+ folder_name + '.txt'
                    file9 = file_dir + 'gene_list_fold9_'+ folder_name + '.txt'
                    file10 = file_dir + 'gene_list_fold10_'+ folder_name + '.txt'
                inter, diff, union1 = compare_one_column_txt_file_contents(file1, file2, feature_size)
                inter, diff, union2 = compare_one_column_txt_file_contents(file3, file4, feature_size)
                if(dataset[0:4] == 'PBMC'):
                    inter, diff, union3 = compare_one_column_txt_file_contents(file5, file6, feature_size)
                    inter, diff, union4 = compare_one_column_txt_file_contents(file7, file8, feature_size)
                    inter, diff, union5 = compare_one_column_txt_file_contents(file9, file10, feature_size)
                    total = union1 | union2 | union3 | union4 | union5
                else:
                    inter, diff, union3 = compare_one_column_txt_file_contents(file4, file5, feature_size)
                    total = union1 | union2 | union3
                print(len(total))
                print(total)
                total = list(total)
                unique_genes = len(total)
                
                #fname7 = out_dir + folder_name + 'pathway_sel.csv'
                #fname8 = out_dir + folder_name + 'tissue_sel.csv'
                #fname9 = out_dir + folder_name + 'disease_sel.csv'
                
                r1,r2,r7 = biological_validation(total, dataset, out_dir, 'Pathway')
                r3,r4,r8 = biological_validation(total, dataset, out_dir, 'Tissue')
                r5,r6,r9 = biological_validation(total, dataset, out_dir, 'Disease')
                
                # # Output all the hits information into individual csv files.
                # r7.to_csv(fname7)
                # r8.to_csv(fname8)
                # r9.to_csv(fname9)
                
                # Output number of hits & unique genes per dataset.
                
                writer.write(folder_name)
                writer.write(':' + str(r7.size/10) + '\\' + str(r8.size/10) + '\\' + str(r9.size/10) + '\n')
                writer.write('num_uniqiue_genes: ' + str(unique_genes) + '\n')
                writer.write('genes: ' + str(total) + '\n')
                writer.write('\n\n')
                # Divide by 10 to account for number of columns in the dataframe
    '''
    # Biological Validation of Embedded (RF + RFE)
    '''
    for dataset in datasets:
        for taboo_strat in taboo_strats:
            run = 0
            path_accum, tissue_accum, disease_accum = 0,0,0
            unique_genes_accum = 0
            
            folder_name = dataset + '_RF_RFE_30_' + taboo_strat
    
            file_dir = '...'
            file_dir += dataset + '/FinalSetup/' + folder_name + '/GeneLists/RFE_RandomForestClassifier/'
            writer.write(folder_name + '\n')
            while run < 5:
                file1 = file_dir + 'gene_list_RFE'+ '_fold1_fpool_500_tsize_30_' + 'run_' + str(run) + '_' + folder_name + '.txt'  
                file2 = file_dir + 'gene_list_RFE'+ '_fold2_fpool_500_tsize_30_' + 'run_' + str(run) + '_' + folder_name + '.txt' 
                file3 = file_dir + 'gene_list_RFE'+ '_fold3_fpool_500_tsize_30_' + 'run_' + str(run) + '_' + folder_name + '.txt' 
                file4 = file_dir + 'gene_list_RFE'+ '_fold4_fpool_500_tsize_30_' + 'run_' + str(run) + '_' + folder_name + '.txt' 
                file5 = file_dir + 'gene_list_RFE'+ '_fold5_fpool_500_tsize_30_' + 'run_' + str(run) + '_' + folder_name + '.txt' 
                if(dataset[0:4] == 'PBMC'):
                    file6 = file_dir + 'gene_list_RFE'+ '_fold6_fpool_500_tsize_30_' + 'run_' + str(run) + '_' + folder_name + '.txt' 
                    file7 = file_dir + 'gene_list_RFE'+ '_fold7_fpool_500_tsize_30_' + 'run_' + str(run) + '_' + folder_name + '.txt' 
                    file8 = file_dir + 'gene_list_RFE'+ '_fold8_fpool_500_tsize_30_' + 'run_' + str(run) + '_' + folder_name + '.txt' 
                    file9 = file_dir + 'gene_list_RFE'+ '_fold9_fpool_500_tsize_30_' + 'run_' + str(run) + '_' + folder_name + '.txt' 
                    file10 = file_dir + 'gene_list_RFE'+ '_fold10_fpool_500_tsize_30_' + 'run_' + str(run) + '_' + folder_name + '.txt' 
                inter, diff, union1 = compare_one_column_txt_file_contents(file1, file2, feature_size)
                inter, diff, union2 = compare_one_column_txt_file_contents(file3, file4, feature_size)
                if(dataset[0:4] == 'PBMC'):
                    inter, diff, union3 = compare_one_column_txt_file_contents(file5, file6, feature_size)
                    inter, diff, union4 = compare_one_column_txt_file_contents(file7, file8, feature_size)
                    inter, diff, union5 = compare_one_column_txt_file_contents(file9, file10, feature_size)
                    total = union1 | union2 | union3 | union4 | union5
                else:
                    inter, diff, union3 = compare_one_column_txt_file_contents(file4, file5, feature_size)
                    total = union1 | union2 | union3
                print(len(total))
                print(total)
                total = list(total)
                unique_genes_accum += len(total)
                
                writer.write('genes: ' + str(total) + '\n')
                
                r1,r2,r7 = biological_validation(total, dataset, out_dir, 'Pathway')
                r3,r4,r8 = biological_validation(total, dataset, out_dir, 'Tissue')
                r5,r6,r9 = biological_validation(total, dataset, out_dir, 'Disease')
                
                # Output all the hits information into individual csv files.
                #fname7 = out_dir + folder_name + '_run_' + str(run) + 'pathway_sel.csv'
                #fname8 = out_dir + folder_name + '_run_' + str(run) + 'tissue_sel.csv'
                #fname9 = out_dir + folder_name + '_run_' + str(run) + 'disease_sel.csv'
                
                #r7.to_csv(fname7)
                #r8.to_csv(fname8)
                #r9.to_csv(fname9)
                
                run += 1
                # Divide by 10 to account for number of columns in the dataframe
                path_accum += r7.size / 10
                tissue_accum += r8.size / 10
                disease_accum += r9.size / 10
            path_accum = path_accum / 5
            tissue_accum = tissue_accum / 5
            disease_accum = disease_accum / 5
            unique_genes_accum = unique_genes_accum / 5
            writer.write(':' + str(path_accum) + '\\' + str(tissue_accum) + '\\' + str(disease_accum) + '\n')
            writer.write('num_uniqiue_genes: ' + str(unique_genes_accum) + '\n')
            writer.write('\n\n')
    '''
    # Biological Validation of Filter (IG, DE) + Wrapper (SFS)
    '''
    for dataset in datasets:
        for filter_method in filter_methods:
            for taboo_strat in taboo_strats:
                for ts in target_sizes:
                    for ML_Model in ML_Models:
                        path_accum, tissue_accum, disease_accum = 0,0,0
                        unique_genes_accum = 0
                        
                        folder_name = dataset + '_' + filter_method + '_SFS' + str(ts) + '_NR_' + taboo_strat
            
                        file_dir = '...'
                        file_dir += dataset + '/FinalSetup/' + folder_name + '/GeneLists/'
                        if(ML_Model == 'kNN'):
                            file_dir += 'SFS_KNeighborsClassifier/'
                        elif(ML_Model == 'LR'):
                            file_dir += 'SFS_LogisticRegression/'
                        else:
                            raise ValueError('ML Model must be kNN or LR.')
                            
                        writer.write(folder_name + '_' + ML_Model + '\n')
                        
                        for fpool_size in feature_pool_sizes:
                            file1 = file_dir + 'gene_list_SFS_fold1_fpool_'+str(fpool_size)+'_tsize_'+str(ts)+'_run_0_' + folder_name + '.txt'
                            file2 = file_dir + 'gene_list_SFS_fold2_fpool_'+str(fpool_size)+'_tsize_'+str(ts)+'_run_0_' + folder_name + '.txt'
                            file3 = file_dir + 'gene_list_SFS_fold3_fpool_'+str(fpool_size)+'_tsize_'+str(ts)+'_run_0_' + folder_name + '.txt'
                            file4 = file_dir + 'gene_list_SFS_fold4_fpool_'+str(fpool_size)+'_tsize_'+str(ts)+'_run_0_' + folder_name + '.txt'
                            file5 = file_dir + 'gene_list_SFS_fold5_fpool_'+str(fpool_size)+'_tsize_'+str(ts)+'_run_0_' + folder_name + '.txt'
                            if(dataset[0:4] == 'PBMC'):
                                file6 = file_dir + 'gene_list_SFS_fold6_fpool_'+str(fpool_size)+'_tsize_'+str(ts)+'_run_0_' + folder_name + '.txt'
                                file7 = file_dir + 'gene_list_SFS_fold7_fpool_'+str(fpool_size)+'_tsize_'+str(ts)+'_run_0_' + folder_name + '.txt'
                                file8 = file_dir + 'gene_list_SFS_fold8_fpool_'+str(fpool_size)+'_tsize_'+str(ts)+'_run_0_' + folder_name + '.txt'
                                file9 = file_dir + 'gene_list_SFS_fold9_fpool_'+str(fpool_size)+'_tsize_'+str(ts)+'_run_0_' + folder_name + '.txt'
                                file10 = file_dir + 'gene_list_SFS_fold10_fpool_'+str(fpool_size)+'_tsize_'+str(ts)+'_run_0_' + folder_name + '.txt'
                            inter, diff, union1 = compare_one_column_txt_file_contents(file1, file2)
                            inter, diff, union2 = compare_one_column_txt_file_contents(file3, file4)
                            if(dataset[0:4] == 'PBMC'):
                                inter, diff, union3 = compare_one_column_txt_file_contents(file5, file6)
                                inter, diff, union4 = compare_one_column_txt_file_contents(file7, file8)
                                inter, diff, union5 = compare_one_column_txt_file_contents(file9, file10)
                                total = union1 | union2 | union3 | union4 | union5
                            else:
                                inter, diff, union3 = compare_one_column_txt_file_contents(file4, file5)
                                total = union1 | union2 | union3
                            print(len(total))
                            # print(total)
                            total = list(total)
                            unique_genes_accum += len(total)
                            writer.write('genes: ' + str(total) + '\n')
                            
                            r1,r2,r7 = biological_validation(total, dataset, out_dir, 'Pathway')
                            r3,r4,r8 = biological_validation(total, dataset, out_dir, 'Tissue')
                            r5,r6,r9 = biological_validation(total, dataset, out_dir, 'Disease')
                            
                            # Output all the hits information into individual csv files.
                            #fname7 = out_dir + folder_name + '_fpool_size_' + str(fpool_size) + 'pathway_sel.csv'
                            #fname8 = out_dir + folder_name + '_fpool_size_' + str(fpool_size) + 'tissue_sel.csv'
                            #fname9 = out_dir + folder_name + '_fpool_size_' + str(fpool_size) + 'disease_sel.csv'
                            
                            # r7.to_csv(fname7)
                            # r8.to_csv(fname8)
                            # r9.to_csv(fname9)
                            
                            #writer.write(folder_name + '_fpool_' + str(fpool_size))
                            writer.write(':' + str(r7.size/10) + '\\' + str(r8.size/10) + '\\' + str(r9.size/10) + '\n')
                            # Divide by 10 to account for number of columns in the dataframe
                            path_accum += r7.size / 10
                            tissue_accum += r8.size / 10
                            disease_accum += r9.size / 10
                        path_accum = path_accum / len(feature_pool_sizes)
                        tissue_accum = tissue_accum / len(feature_pool_sizes)
                        disease_accum = disease_accum / len(feature_pool_sizes)
                        unique_genes_accum = unique_genes_accum / len(feature_pool_sizes)
                        writer.write(':' + str(path_accum) + '\\' + str(tissue_accum) + '\\' + str(disease_accum) + '\n')
                        writer.write('num_uniqiue_genes: ' + str(unique_genes_accum) + '\n')
    '''
    writer.close()
    
    # ---------------------------------------------------------------------------------------------------------------- 
    # ***********************************GENERATING FIGURES BLOCK*****************************************************
    # ---------------------------------------------------------------------------------------------------------------- 
    
    
    '''
    # Generating custom RNA-seq (cuffnorm counts) heatmaps
    genes = ['DEFA4', 'SDC2', 'AHSP', 'AC005912.1', 'IGLV3-10', 'IFIT3', 'SLC4A1', 'SELENBP1', 'TGM2', 'CD177', 'IGHG1', 'USP32P1', 'PI3', 'HES4', 'KRT1', 'GYPB', 'USP18', 'IFITM1', 'LRRN3', 'ALAS2', 'CRISP3', 'CA1', 'SEMA6B', 'CCL3L3', 'IGKV2-29', 'HBG2', 'MYL4', 'RNU4ATAC', 'DEFA3', 'CYP4F29P', 'RSAD2', 'HBM', 'ALPL', 'CXCL10', 'CCL7', 'LAMB3', 'SEC14L2', 'RNU6-574P', 'FAM20A', 'TNS1', 'SERPINB2', 'HBD', 'OLFM4', 'SERPING1', 'IFI27', 'MMP8', 'SIGLEC1', 'IGLV2-23', 'LY75-CD302', 'C1QB']
    
    # genes.fpkm_table
    out_dir = '...'
    fname = '...'
    
    fname2 = '...'
    #counts = read_cuffnorm_counts2(fname, 'ALL', True)
    counts = read_cuffdiff_counts2(fname, 'ALL', True)
    
    filtered_counts_temp = {}
    for gene in genes:
        if(gene in counts.keys()):
            filtered_counts_temp[gene] = counts[gene]
    
    # Its crucial that genes counts are ordered identically for each replicate.
    p = 0 
    filtered_counts = {}            
    for gene, gene_feature in filtered_counts_temp.items():
        if(p == 0):
            for rep_count_tuple in gene_feature:
                rep_name = rep_count_tuple[0]
                filtered_counts[rep_name] = []
        for rep_count_tuple in gene_feature:
            rep_name = rep_count_tuple[0]
            gene_count = rep_count_tuple[1]
            filtered_counts[rep_name].append(gene_count)
        p += 1
    
    plot_per_sample_counts_heatmap(filtered_counts, genes, fname2, 0, out_dir, 'Cuffdiff', 100)
    
    # Alternative process to generate mean counts per condition
    
    #counts_mean_variance = read_cuffnorm_counts_mean_variance(fname, True)
    counts_mean_variance = read_cuffdiff_counts_mean_variance(fname, True)
    
    mean_counts = {}
    num_conditions = 5
    conditions = ['AH', 'CT', 'DAAA', 'NF', 'HP']
    for gene in genes:
        if gene in counts_mean_variance.keys():
            mean_counts[gene] = counts_mean_variance[gene][0:num_conditions]
    
    plot_per_condition_counts_heatmap(mean_counts, genes, conditions, 0, out_dir, 100)
    '''
    
    
    # Generating custom confusion matrices
    out_dir = '...'
    array = np.array([[35,  0,  2,  0,  1],[ 1, 16,  1,  2,  0],[ 3,  3,  32,  0,  2],
                     [ 0,  2,  0,  18,  0],[ 0,  1,  1,  1, 16]])
    cm_labels = ['AH', 'CT', 'DAAA', 'NF', 'HP']
    accuracy = 0.85
    
    acc_text = 'Total Accuracy: ' + str(int(accuracy*100)) + '%'
    
    
    annotation = array.tolist()
    # Add per class accuracy on diagonal entries.
    i = 0
    while i < array.shape[0]:
        j = 0
        while j < array.shape[1]:
            annotation[i][j] = str(array[i][j])
            if(i == j):
                annotation[i][j] += '\n' + str( int((array[i][j]/np.sum(array[i,:]))*100) ) + '%'
            j += 1
        i += 1
    
    df_cm = pandas.DataFrame(array, index = cm_labels,
                             columns = cm_labels)
    df_cm = df_cm.div(df_cm.sum(axis=1), axis=0)
    plt.figure()
    
    seaborn.heatmap(df_cm, cmap = 'Blues', annot=annotation, fmt='', annot_kws={"fontsize":12}, vmin = 0, vmax = 1.0)
    plt.xlabel('Predicted Classes', fontsize = 14)
    plt.ylabel('Actual Classes', fontsize = 14)
    plt.title('PB5Way_Excluded_DE_SFS18_NR_Outlier' , fontsize = 16)

    plt.figtext(0.1, -0.05, acc_text, horizontalalignment='left', fontsize = 12) 
    
    out_file = out_dir + 'PB5Way_Excluded_DE_SFS18_NR_Outlier.png'
    plt.savefig(out_file, bbox_inches="tight")
    
    
    # ---------------------------------------------------------------------------------------------------------------- 
    # ***********************************************IPA ANALYSIS PREPARATION*****************************************
    # ----------------------------------------------------------------------------------------------------------------
    '''
    gene_list_file = '...'
    cuffdiff_file = '...'
    output_dir = '...'
    
    filter_cuffdiff_file_by_gene_list(gene_list_file, cuffdiff_file, output_dir)
    '''
    # ---------------------------------------------------------------------------------------------------------------- 
    # ***********************************MISCLASSIFIED SAMPLES ANALYSIS BLOCK*****************************************
    # ----------------------------------------------------------------------------------------------------------------
    '''
    
    fname = '...'
    fname += '...'
    dict1,set1 = process_misclassified_samples(fname, 120)
    
    fname = '...'
    fname += '...'
    dict2,set2 = process_misclassified_samples(fname, 120)
    

    xxx = set1 & set2
    print(xxx)
    print(len(xxx))
    '''
