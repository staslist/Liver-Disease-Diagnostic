# -*- coding: utf-8 -*-
"""
Created on Mon Jun  3 17:06:52 2019

@author: Stanislav Listopad
The code below was used to read-in and parse RNAseq counts from Cuffnorm/Cuffdiff files. To perform classification of these counts 
in a supervised (labeled) manner. Both liver tissue and PBMC datasets were evaluated using nested cross validation.
Liver tissue data was validated in an independent dataset.
"""

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
Try to keep function length below 30 lines (great), 50 lines should be max
(should refactor). Around 100 lines is absolute max (must refactor).
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
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.feature_selection import RFE
from sklearn.feature_selection import SequentialFeatureSelector

from scipy import interp
from sklearn.utils import shuffle
from sklearn.metrics import roc_curve, auc

import matplotlib.pyplot as plt

import time

'''PBMC (Blood) Sample IDS from the AH-Project.'''

samples_AH_PB = ['Redacted.']
        
samples_CT_PB = ['Redacted.']

samples_DA_PB = ['Redacted.']

samples_AA_PB = ['Redacted.']

sample_DAAA_PB = samples_DA_PB + samples_AA_PB

samples_NF_PB = ['Redacted.']

samples_HP_PB = ['Redacted.']

samples_HP_PB_Excluded = ['Redacted.']

cond_samples_PB = {"AH":samples_AH_PB, "CT": samples_CT_PB, "DA" : samples_DA_PB, "AA" : samples_AA_PB, 
                   "NF": samples_NF_PB, "HP": samples_HP_PB}

cond_samples_PB_Excluded_Alt = {"AH":samples_AH_PB, "CT": samples_CT_PB, "DAAA" : sample_DAAA_PB, 
                                "NF": samples_NF_PB, "HP": samples_HP_PB_Excluded}

cond_samples_PB2 = {"AH":samples_AH_PB, "CT": samples_CT_PB}

cond_samples_PB_Excluded = {"AH":samples_AH_PB, "CT": samples_CT_PB, "DA" : samples_DA_PB, "AA" : samples_AA_PB, 
                            "NF": samples_NF_PB, "HP": samples_HP_PB_Excluded}

'''Liver tissue samples from the AH-Project '''

samples_AH_LV = ['Redacted.']

samples_AH_LV_Excluded = ['Redacted.']

samples_CT_LV = ['Redacted.']

samples_AC_LV = ['Redacted.']

samples_NF_LV = ['Redacted.']

samples_HP_LV = ['Redacted.']

cond_samples_LV = {'AH': samples_AH_LV, 'CT': samples_CT_LV, 'AC': samples_AC_LV,
                   'NF': samples_NF_LV, 'HP': samples_HP_LV}

cond_samples_LV2_Excluded = {'AH': samples_AH_LV_Excluded, 'CT': samples_CT_LV}

cond_samples_LV2 = {'AC': samples_AC_LV, 'NF': samples_NF_LV}

cond_samples_LV3_Excluded = {'AH': samples_AH_LV_Excluded, 'CT': samples_CT_LV, 'AC': samples_AC_LV}

cond_samples_LV_Excluded= {'AH': samples_AH_LV_Excluded, 'CT': samples_CT_LV, 'AC': samples_AC_LV,
                            'NF': samples_NF_LV, 'HP': samples_HP_LV}

tissue_dict = {'PB':cond_samples_PB, 'PB_Excluded':cond_samples_PB_Excluded, 'PB2': cond_samples_PB2, 
               'PB_Excluded_Alt':cond_samples_PB_Excluded_Alt, 'LV':cond_samples_LV,
               'LV2': cond_samples_LV2, 'LV2_Excluded':cond_samples_LV2_Excluded, 'LV3_Excluded':cond_samples_LV3_Excluded, 
               'LV_Excluded':cond_samples_LV_Excluded}

pipelines_ALL = ['hg19_Hisat2_Curated', 'hg19_Hisat2_Ensembl', 'hg19_Hisat2_Gencode', 'hg19_Hisat2_Refflat', 'hg19_Starcq_Curated',
                 'hg19_Starcq_Ensembl', 'hg19_Starcq_Gencode', 'hg19_Starcq_Refflat', 'hg19_Tuxedo_Curated', 'hg19_Tuxedo_Ensembl',
                 'hg19_Tuxedo_Gencode', 'hg19_Tuxedo_Refflat', 'hg38_Hisat2_Curated', 'hg38_Hisat2_Ensembl', 'hg38_Hisat2_Gencode',
                 'hg38_Hisat2_Refflat', 'hg38_Starcq_Curated', 'hg38_Starcq_Ensembl', 'hg38_Starcq_Gencode', 'hg38_Starcq_Refflat',
                 'hg38_Tuxedo_Curated', 'hg38_Tuxedo_Ensembl', 'hg38_Tuxedo_Gencode', 'hg38_Tuxedo_Refflat']

models_ALL = ['LR', 'kNN', 'GNB', 'SVM', 'DT', 'RF', 'Ada']

p_grid_Ada = {"n_estimators": {5, 10, 15, 20, 25, 50}, "learning_rate": {0.5, 0.75, 0.9, 1, 1.15, 1.3}}
p_grid_RF = {"n_estimators": [10,25,50,100,200], "criterion": ['gini', 'entropy'], 
             "max_features": ["auto", "sqrt", "log2", None]}
p_grid_DT = {"criterion": ["gini", "entropy"], "max_depth": {5,10,None}, "min_samples_split": {2,3,4},
             "min_samples_leaf": [1,2,3],"max_features": ["auto", "sqrt", "log2", None], 
             "min_impurity_decrease": [0, 0.1], "ccp_alpha": [0, 0.1]}
p_grid_SVM = {"kernel":['linear', 'poly', 'rbf'], "C": [0.5, 1.0, 2.0, 3.0, 4.0, 5.0],
          "class_weight": [None, 'balanced'], "probability": [True], "degree": [1, 3, 5],
          "gamma": ['scale', 'auto']}
p_grid_kNN = {"n_neighbors" : [3,5,7,9], "weights": ['uniform', 'distance'], 
         "metric": ['euclidean', 'manhattan', 'chebyshev', 'minkowski']}
p_grid_LR = {"C":[0.5, 1.0, 2.0, 3.0, 4.0, 5.0], "class_weight": [None, 'balanced'],
          "solver": ['newton-cg', 'lbfgs', 'liblinear', 'saga']}

p_grid_dict = {'LR': p_grid_LR, 'kNN': p_grid_kNN, 'SVM': p_grid_SVM, 'DT': p_grid_DT,
               'RF': p_grid_RF, 'Ada': p_grid_Ada, 'GNB': {}}

def count_elements_in_2dlist(a:list):
    # Count number of elements in a 2-dimensional list. Return result as a dictionary.
    result = {}
    for collection in a:
        for ele in collection:
            try:
                result[ele] += 1
            except KeyError:
                result[ele] = 1
                
    return result

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

def read_in_csv_file_one_column(filename:str, column:int, delim:str, skip_n = 0, limit = None)->list:
    # Read in the selected column froma csv file.
    temp = []
    with open(filename) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=delim)
        i = 0
        for row in csv_reader:
            if(limit != None):
                if(i >= limit):
                    break
            if (i < skip_n):
                i += 1
                continue
            try:
                temp.append(row[column])
            except ValueError:
                pass
            i += 1
    return temp

def select_features_from_matrix(X, feature_names:list, feature_selection:list):
    '''Given a matrix, a corresponding list of feature names, and a list of features to 
    select, return a matrix with only the selected features included.'''
    feature_name_indeces = []
    for feature in feature_selection:
        if (feature_names.count(feature) != 1):
            raise ValueError("Top gene was either not read in or read in multiple times.")
        feature_name_indeces.append(feature_names.index(feature))

    result = X[:, feature_name_indeces]
    return result

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

def read_cuffdiff_counts_mean_std2(filename: str, log=False, fpkm=True)->dict:
    '''Returns a dictionary that maps each gene name to a 
    list of means and standard deviations. There is one mean and one standard deviation for each condition. 
    The means are listed first and the standard deviations are listed second.
    Assume that conditions are listed in order within the Cuffnorm file (q1, q2, ...). The individual replicates 
    within conditions are not listed in order. Assume that there maybe up to 9 conditions (q1 - q9).'''
    counts = read_cuffdiff_counts2(filename, 'ALL', log, fpkm)
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

def read_cuffdiff_counts_mean_std(filename: str, log=False)->dict:
    '''Returns a dictionary that maps each gene name to a 
    list of means and standard deviations. There is one mean and one standard deviation for each condition. 
    The means are listed first and the standard deviations are listed second.
    Assume that conditions are listed in order within the Cuffnorm file (q1, q2, ...). The individual replicates 
    within conditions are not listed in order. Assume that there maybe up to 9 conditions (q1 - q9).'''
    counts = read_cuffdiff_counts2(filename, 'ALL', log)
    counts_out = {}
    for k,v in counts.items():
        cond_counts, means, stds = {}, {}, {}
        means_variance = []
        for cond_count_tuple in v:
            temp = cond_count_tuple[0].split('_')
            cond = temp[0]
            count = cond_count_tuple[1]
            try:
                cond_counts[cond].append(count)
            except KeyError:
                cond_counts[cond] = [count]
                means[cond], stds[cond] = 0, 0
                
        for k2 in means.keys():
            means[k2] = sum(cond_counts[k2]) / len(cond_counts[k2])
            
        for k3,v3 in cond_counts.items():
            stds[k3] = np.std(np.array(cond_counts[k3]))
        
        counts_out[k] = (means, stds)
    return counts_out

def detect_outlier_features_by_std2(X, Y, feature_names:list, treshold:float = 3.5, flag_zeros:bool = True):
    '''Given X and Y matrices and a correspondingly ordered list of feature names identify 
    highly variant features. Return the list of feature names that are highly variant.
    A feature is highly variant if its counts for at least one sample are > than (mean + std * treshold) or < 
    than (mean - std * treshold). Also, optionally, detect features that are mostly 0.
    Rows are samples. Columns are features.
    Recommend using lower tresholds with smaller sample sizes. 5-10 samples: treshold = 2. 10+ samples: treshold 2.5+.
    '''
    
    outlier_features = set()
    #print("X:", X)
    #print("Y:", Y)
    
    class_labels = set(Y)
    for class_label in class_labels:
        #print('Class label: ', class_label)
        class_indeces_tuple = np.where(Y == class_label)
        class_indeces_array = class_indeces_tuple[0]
        sub_X = X[class_indeces_array, :]
        #print("sub_X:", sub_X)
        sub_X_t = sub_X.transpose()
        i = 0
        for feature in sub_X_t:
            # print('The value of i: ', i)
            # print("Feature in sub_X:", feature)
            if(flag_zeros):
                if(np.count_nonzero(feature) < (len(feature)/2)):
                    outlier_features.add(feature_names[i])
                    i += 1
                    continue
            fea_mean = np.mean(feature)
            fea_std = np.std(feature)
            # print("Feature mean:", fea_mean)
            # print("Feature std:", fea_std)
            for count_value in feature:
                if (count_value > (fea_mean + fea_std*treshold)) or (count_value < (fea_mean - fea_std*treshold)):
                    outlier_features.add(feature_names[i])
            
            i += 1
    return outlier_features

def generate_HPC_batch_job_header(jobname:str, queue:str = 'highmem'):
    to_return = ''
    to_return += "#!/bin/bash" + "\n"
    # Specify job name.
    to_return += "#SBATCH --job-name=" + jobname + "\n"
    # Select the compute notes.
    to_return += "#SBATCH -A REDACTED" + "\n"
    to_return += "#SBATCH -p " + queue + "\n"
    to_return += "#SBATCH --nodes=1 " + "\n"
    to_return += "#SBATCH --ntasks=1" + "\n"
    to_return += "#SBATCH --cpus-per-task=32" + "\n"
    to_return += "#SBATCH --mail-type=begin,end,fail" + "\n"
    to_return += "#SBATCH --mail-user=REDACTED" + "\n"
    to_return += "\n"
    to_return += "\n"
    to_return += "module load openmpi/4.0.3/gcc.8.4.0" + "\n"
    to_return += "export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK" + "\n"
    to_return += "module load cufflinks/2.2.1" + "\n"
    to_return += "\n"
    
    return to_return

def generate_confusion_matrix(out_dir:str, cm_array, labels:list, acc:float, title:str):
    # Generating custom confusion matrices
    acc_text = 'Total Accuracy: ' + str(int(acc*100)) + '%'
    
    annotation = cm_array.tolist()
    # Add per class accuracy on diagonal entries.
    i = 0
    while i < cm_array.shape[0]:
        j = 0
        while j < cm_array.shape[1]:
            annotation[i][j] = str(cm_array[i][j])
            if(i == j):
                if(cm_array[i][j] == 0 and np.sum(cm_array[i,:]) == 0):
                    annotation[i][j] += '\n0%'
                else:
                    annotation[i][j] += '\n' + str( int((cm_array[i][j]/np.sum(cm_array[i,:]))*100) ) + '%'
            j += 1
        i += 1
        
    df_cm = pandas.DataFrame(cm_array, index = labels,
                             columns = labels)
    df_cm = df_cm.div(df_cm.sum(axis=1), axis=0)
    df_cm = df_cm.replace(np.NaN, 0)
    plt.figure()
    
    seaborn.heatmap(df_cm, cmap = 'Blues', annot=annotation, fmt='', annot_kws={"fontsize":12}, vmin = 0, vmax = 1.0)
    plt.xlabel('Predicted Classes', fontsize = 14)
    plt.ylabel('Actual Classes', fontsize = 14)
    plt.title(title , fontsize = 16)

    plt.figtext(0.1, -0.05, acc_text, horizontalalignment='left', fontsize = 12) 
    
    out_file = out_dir + title + '.png'
    plt.savefig(out_file, bbox_inches="tight")

def divide_folds_in_training_and_validation(folds:list):
    '''Folds is a list of lists. Each inner list is a collection of samples (fold). We want to training and validation sets using 
    these folds. The number of training/validation sets must equal to number of folds. 
    Training and validation sets order must correspond. Training and validation 'sets' are 1-d lists. '''
    assert(len(folds)>1)
    training_sets = []
    validation_sets = []
    fold_index = 0
    while fold_index < len(folds):
        training_set = []
        validation_set = []
        i = 0
        while i < len(folds):
            if(i != fold_index):
                training_set += folds[i]
            else:
                validation_st = folds[i]
            i += 1
        fold_index += 1
        training_sets.append(training_set)
        validation_sets.append(validation_set)
    return training_sets, validation_sets

def generate_cuffnorm_or_cuffdiff_batch_file_HPC(reference_genome:str, aligner:str, annotation: str, normalization_method:str,
                                                 conditions:list, in_folder:str, out_folder:str, out_dir:str, mode:str = "Cuffnorm",
                                                 folds = 1, dispersion_method:str = "", tissue:str = 'PB'):
    ''' Generate the batch files for running cuffnorm or cuffdiff over AH project data on HPC. 
    For nested folds, put in the inner k first, and outer k second. Example: folds = [5, 10] wherein inner k is 5 and 
    outer k is 10.'''
    # Mode can be Cuffnorm or Cuffdiff
    # reference genome, aligner, and annotation must be lower case
    # normalization_method must be UQ or GEOM
    # dispersion_method must be POOL or COND
    cond_samples_tissue = tissue_dict[tissue]
    
    Cond_Folds_Dict = {}
    for condition in conditions:
        if(condition not in ["AH", "AC", "CT", "AA", "DA", "DAAA", "NF", "HP"]):
            raise ValueError("Unknown condition.")
        if(type(folds) == int):
            Cond_Folds = generate_kfolds(cond_samples_tissue[condition], folds)
        elif(type(folds) == list):
            assert(len(folds) == 2)
            Cond_Folds = generate_nested_kfolds(cond_samples_tissue[condition], folds[0], folds[1])
        else:
            raise ValueError("Folds should be an int or a list of length 2.")
        Cond_Folds_Dict[condition] = Cond_Folds
        #print('Cond_Folds_Dict[', condition, ']: ', Cond_Folds_Dict)
        #print()
    
    fold_index = 0
    folds_total = 0
    if(type(folds) == list):
        folds_total = folds[0] * folds[1]
    else:
        folds_total = folds
    
    Training_Sets = {}
    if(type(folds) == list):
        for condition in conditions:
            temp = []
            for outer_train_set in Cond_Folds_Dict[condition]:
                train_set, _ = divide_folds_in_training_and_validation(outer_train_set)
                temp += train_set
            Training_Sets[condition] = temp
    elif(type(folds) == int and folds > 1):
        for condition in conditions:
            Training_Sets[condition], _ = divide_folds_in_training_and_validation(Cond_Folds_Dict[condition])
    else:
        for condition in conditions:
            Training_Sets[condition] = Cond_Folds_Dict[condition]
    
    while fold_index < folds_total:
        if(type(folds) == list):
            # nested cross-validation
            outer_train_set_index = fold_index // folds[0]
            inner_train_set_index = fold_index % folds[0]
        
        part1 = reference_genome + "_" + aligner + "_" + annotation
        part1_capitalized = reference_genome + "_" + aligner.capitalize() + "_" + annotation.capitalize()
        
        part2 = normalization_method
        if(mode == "Cuffdiff"):
            part2 += "_" + dispersion_method
        if(type(folds) == int):
            if(folds > 1):
                part2 += "_FOLD" + str(fold_index + 1)
        else:
            part2 += "_FOLD" + str(outer_train_set_index + 1) + '_' + str(inner_train_set_index + 1)
            
        temp = "SL_"
        r = 0
        num_conds = len(conditions)
        for condition in conditions:
            temp += str(len(cond_samples_tissue[condition])) + condition
            if(r < (num_conds - 1)):
                temp += "_vs_"
            else:
                temp += '_' + tissue[0:2] + '_'

            r += 1
            
        filename = temp + part1 + "_" + part2 + '.sh'
        jobname = temp + part1 + "_" + part2
    
        with open(out_dir + filename, 'w') as writer:
            writer.write(generate_HPC_batch_job_header(jobname))
                
            writer.write("cd REDACTED" + 
                         "REDACTED" + in_folder + "/" + part1_capitalized + "/cxbs/" + "\n")
            
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
            
            writer.write("-o REDACTED" 
                         + "REDACTED" + out_folder + "/")
            
            writer.write(part1_capitalized + "/" + mode +"_" + part2 + "/ " + reference_genome + "_" + annotation + ".gtf ")
            
            for k,Cond_Training_Sets in Training_Sets.items():
                num_Samples = len(Cond_Training_Sets[fold_index])
                #print(num_Samples)
                k = 0
                for Sample in Cond_Training_Sets[fold_index]:
                    writer.write(Sample + "." + part1 + ".geneexp.cxb")
                    k = k + 1
                    if(k < num_Samples):
                        writer.write(",")
                    else:
                        writer.write(" ")
                        
            writer.write('\n')
                            
        fold_index = fold_index + 1
    return Cond_Folds_Dict

def generate_kfolds(samples:list, k:int):
    ''' Separates samples into k-folds (lists). 
    Returns a list of lists.'''
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

def generate_nested_kfolds(samples:list, k_inner:int, k_outer:int):
    ''' A 3-d list. Each inner list represents folds formed using the outer loop's training sets. 
    The way in which outer training sets are formed mimics the generate_cuffnorm_or_cuffdiff_batch_file_HPC function. '''
    if(k_inner > len(samples) or k_outer > len(samples)):
        raise ValueError("The number of folds must be <= than the number of samples.")
    
    assert(k_outer > 1 and k_inner > 1)
    
    outer_folds = generate_kfolds(samples, k_outer)
    nested_kfolds = []
    i = 0
    training_sets, validation_sets = divide_folds_in_training_and_validation(outer_folds)
    #print("training_sets:", training_sets)
    while i < k_outer:
        training_set = training_sets[i]
        #print('training_set:', training_set)
        nested_folds = generate_kfolds(training_set, k_inner)
        nested_kfolds.append(nested_folds)
        i += 1
        
    return nested_kfolds

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

def generate_sample_to_replicate_map_cuffdiff(fname:str, reverse:bool = False):
    result = {}
    with open(fname) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter='\t')
        line_count = 0
        for row in csv_reader:
            if(line_count > 0):
                if (not reverse):
                    result[row[0]] = row[1] + '_' + row[2]
                else:
                    result[row[1] + '_' + row[2]] = row[0]
            line_count += 1
    
    return result

def read_cuffdiff_sample_names(filename:str):
    result = []
    with open(filename) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter='\t')
        line_count = 0
        for row in csv_reader:
            if(line_count > 0):
                result.append(row[0])
            line_count += 1
                
    return result

def read_cuffdiff_counts2(filename: str, conditions:"list or 'ALL'" = 'ALL', log=False, fpkm=True)->dict:
    '''Reads in counts from cuffdiff genes.read_group_tracking. Only reads the counts for specified conditions.
    If conditions = ALL, read in all counts. Stores the cuffnorm RNA-seq counts in a dictionary.
    Genes are keys. Values are lists of two valued tuples. Each tuple contains the replicate name 
    (ex: q1_0) and the count value.
    Important: assuming Python >= 3.7 the ordering of genes is preserved and is consistent with the ordering within 
    the input Cuffdiff file.'''
    counts = {}
    if(fpkm):
        col_to_read = 6
    else:
        col_to_read = 5
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
                            counts[gene_name].append( (cond_name + '_' + rep_name, math.log(1 + float(row[col_to_read]))) )
                        else:
                            counts[gene_name].append( (cond_name + '_' + rep_name, float(row[col_to_read])) )
                    except KeyError:
                        if(log):
                            counts[gene_name] = [ (cond_name + '_' + rep_name, math.log(1 + float(row[col_to_read]))) ]
                        else:
                            counts[gene_name] = [ (cond_name + '_' + rep_name, float(row[col_to_read])) ]
            line_count += 1
    return counts

def generate_CV_split_cuffdiff(cv_k:int, fnames:list, sample_order:list):
    # Splits the data for cross-validation using existing Cuffdiff files.
    index_outer = 0
    outer_cv_split = []
    while index_outer < cv_k:
        fname = fnames[index_outer]
        #print(fname)
        # Translate sample names within the corresponding training set into indices of sample_order. 
        # These indeces are the training indeces. The remaining indeces are validation indeces. 
        train_set_sample_names = read_cuffdiff_sample_names(fname)
        #print(train_set_sample_names)
        train_set_indeces = []
        validation_set_indeces = []
        for train_set_sample in train_set_sample_names:
            if(train_set_sample not in sample_order):
                # print(fname)
                # print(train_set_sample)
                raise ValueError("Could not find a training sample within sample_order.")
            sample_index = sample_order.index(train_set_sample)
            train_set_indeces.append(sample_index)
            
        j = 0 
        while j < len(sample_order):
            if j not in train_set_indeces:
                validation_set_indeces.append(j)
            j += 1
        #print(validation_set_indeces)   
        outer_cv_split.append( (train_set_indeces, validation_set_indeces) )
        index_outer += 1
        
    return outer_cv_split

def generate_top_DE_features(fname_in:str, num_features:int, feature_names:list, fname_out:str, num_conditions:int,
                             taboo_features:list = [], q_value:float = 0.05, fpkm:float = 1.0):
    # Generates top differential expression features using Cuffdiff's gene_exp.diff file.
    '''Assume there is <= 9 conditions.'''
    i = 1 
    DE_dati = {}
    while i < num_conditions:
        j = i + 1
        while j <= num_conditions:
            name = str(i) + '_' + str(j)
            DE_dati[name] = {}
            j += 1
        i += 1
    
    with open(fname_in) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter='\t')
        i = 0
        for row in csv_reader:
            if (i > 0):
                # 11 = p_value, 12 = q_value
                if((float(row[7]) > fpkm) and (float(row[8]) > fpkm) and (float(row[12]) < q_value) and (row[0] in feature_names) 
                   and (row[0] not in taboo_features)):
                    assert(row[4] in ['q1', 'q2', 'q3', 'q4', 'q5', 'q6', 'q7', 'q8', 'q9'])
                    assert(row[5] in ['q1', 'q2', 'q3', 'q4', 'q5', 'q6', 'q7', 'q8', 'q9'])
                    cond_label1 = row[4]
                    cond_label2 = row[5]
                    DE_dati[cond_label1[1] + '_' + cond_label2[1]][row[0]] = abs(float(row[9]))
            i += 1
                    
    iters = []
    for v in DE_dati.values():
        s_DE_data = sorted(v.items(), key = lambda x: abs(x[1]), reverse = True)
        #print("s_DE_data:", s_DE_data)
        iters.append(iter(s_DE_data))
    
    i = 0
    iter_flags = []
    while i < len(iters):
        iter_flags.append(False)
        i += 1
    
    result = []
    j = 0
    while len(result) < num_features:
        #print('j:', j)
        #print('result:', result)
        #print('iter_flags:', iter_flags)
        if(all(iter_flags)):
            break
        
        if(not iter_flags[j]):
            try:
                gene_count = next(iters[j])
                if(gene_count[0] not in result):
                    result.append(gene_count[0])
            except StopIteration:
                iter_flags[j] = True
            
        j += 1
        j = j % (len(iter_flags))
    
    with open(fname_out, 'w') as writer:
        for gene_feature in result:
            writer.write(gene_feature + '\n')
            
    return result
    # DE Feature Ranking Over

def generate_top_SFS_features(X, Y, feature_names:list, num_features:int, fname:str, model_name:str,
                              hyper_params:dict, taboo_features:list = []):
    # Generate top sequential feature selection features.
    print("In generate top SFS features.")
    if(model_name == 'Ada'):model = AdaBoostClassifier(**hyper_params)
    elif(model_name == 'RF'):model = RandomForestClassifier(**hyper_params)
    elif(model_name == 'DT'):model = DecisionTreeClassifier(**hyper_params)
    elif(model_name == 'SVM'):model = SVC(**hyper_params)
    elif(model_name == 'kNN'):model = KNeighborsClassifier(**hyper_params)
    elif(model_name == 'LR'):model = LogisticRegression(**hyper_params)
    elif(model_name == 'GNB'):model = GaussianNB(**hyper_params)
    print("Current Features: ", feature_names)
    
    sfs = SequentialFeatureSelector(model, n_features_to_select = num_features)
    sfs.fit(X, Y)
    mask = sfs.get_support()
    print("SFS Feature Mask: ", mask)
    
    with open(fname, 'w') as writer:
        i = 0
        for gene_feature in feature_names:
            if(mask[i] and (gene_feature not in taboo_features)):
                writer.write(gene_feature + '\n')
            i += 1

def generate_top_RF_features(X, Y, feature_names, num_features, fname, taboo_features = []):
    # Generate top random forest features.
    model = RandomForestClassifier()
    clf = model.fit(X, Y)
    RF = clf.feature_importances_
    
    rnaseq_rf = {}
    i = 0
    while i < len(RF):
        rnaseq_rf[feature_names[i]] = RF[i]
        i += 1
    s_rnaseq_rf = sorted(rnaseq_rf.items(), key = lambda x: abs(x[1]), reverse = True)
    
    top_gene_features = []
    
    i = 0
    for kv_tup in s_rnaseq_rf:
        if(str(kv_tup[0]) not in taboo_features):
            top_gene_features.append(str(kv_tup[0]))
            i += 1
            if(i == num_features):
                break
            
    with open(fname, 'w') as writer:
        for gene_feature in top_gene_features:
            writer.write(gene_feature + '\n')
            
    return top_gene_features

def generate_top_IG_features(X, Y, feature_names, num_features, fname, taboo_features = []):
    # Generate top information gain features.
    '''For fname provide full directory.'''
    # IG Feature Ranking Block
    IG = mutual_info_classif(X, Y)
    #print(IG)
    
    rnaseq_ig = {}
    i = 0
    while i < len(IG):
        rnaseq_ig[feature_names[i]] = IG[i]
        i += 1
    s_rnaseq_ig = sorted(rnaseq_ig.items(), key = lambda x: abs(x[1]), reverse = True)
    #print(s_rnaseq_ig)
    top_gene_features = []
    
    i = 0
    for kv_tup in s_rnaseq_ig:
        if(str(kv_tup[0]) not in taboo_features):
            top_gene_features.append(str(kv_tup[0]))
            i += 1
            if(i == num_features):
                break
    
    #print(top_gene_features)
    
    with open(fname, 'w') as writer:
        for gene_feature in top_gene_features:
            writer.write(gene_feature + '\n')
            
    return top_gene_features

def verify_nested_cross_val_cuffdiff(root_dir:str, samples_expected:list):
    # Verify that the nested cross validation setup is valid as demonstrated within cuffdiff files.
    sample_names = []
    fdir = root_dir + 'Cuffdiff_GEOM_POOL/'
    fname = fdir + 'read_groups.info'
    with open(fname) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter='\t')
        i = 0
        for row in csv_reader:
            if (i > 0):
                sample_name = row[0].split('.')[0]
                sample_names.append(sample_name)
            i += 1
    
    # Verify sample composition
    assert(len(set(samples_expected) - set(sample_names)) == 0)
    assert(len(set(sample_names) - set(samples_expected)) == 0)
    
    sample_count = {}
    samples_union = []
    for sample in samples_expected:
        sample_count[sample] = 0
    # Verify that each sample is in k-1 training sets (outer cv split)
    k = 5
    i = 1
    while i <= k:
        samples = []
        fname = root_dir + 'Cuffdiff_GEOM_POOL_FOLD' + str(i) + '/read_groups.info'
        with open(fname) as csv_file:
            csv_reader = csv.reader(csv_file, delimiter='\t')
            j = 0
            for row in csv_reader:
                if (j > 0):
                    sample_name = row[0].split('.')[0]
                    samples.append(sample_name)
                    sample_count[sample_name] += 1
                j += 1
        i += 1
        samples_union += samples
    
    samples_union = set(samples_union)
    for sample,count in sample_count.items():
        #print("sample:", sample)
        #print(count)
        assert(count == (k-1))
    
    # Also verify that the union of all outer train sets equals to the expected samples
    assert(len(set(samples_expected) - set(samples_union)) == 0)
    assert(len(set(samples_union) - set(samples_expected)) == 0)
    
    # Verify that each inner CV loops is valid
    # Each inner CV loop must only possess samples from the corresponding outer CV training set
    # Each sample should appear in k-1 training sets within inner CV loop
    
    z = 1
    outer_train_sets = []
    while z <= k:
        
        outer_train_set = []
        fname = root_dir + 'Cuffdiff_GEOM_POOL_FOLD' + str(z) + '/read_groups.info'
        with open(fname) as csv_file:
            csv_reader = csv.reader(csv_file, delimiter='\t')
            j = 0
            for row in csv_reader:
                if(j > 0):
                    sample_name = row[0].split('.')[0]
                    outer_train_set.append(sample_name)
                j += 1
        outer_train_set = set(outer_train_set)
        
        i = 1
        samples_inner = {}
        while i <= k:
            fname = root_dir + 'Cuffdiff_GEOM_POOL_FOLD' + str(z) + '_' + str(i) + '/read_groups.info'
            with open(fname) as csv_file:
                csv_reader = csv.reader(csv_file, delimiter='\t')
                j = 0
                for row in csv_reader:
                    if (j > 0):
                        sample_name = row[0].split('.')[0]
                        try:
                            samples_inner[sample_name] += 1
                        except KeyError:
                            samples_inner[sample_name] = 1
                    j += 1
            i = i + 1
        for sample,count in samples_inner.items():
            # print(count)
            assert(count == (k-1))
            
        # The union of all inner samples should equal to the corresponding outer training set.
        samples_inner_union = []
        for sample in samples_inner.keys():
            samples_inner_union.append(sample)
        
        samples_inner_union = set(samples_inner_union)
        assert(len(outer_train_set - samples_inner_union) == 0)
        assert(len(samples_inner_union - outer_train_set) == 0)
        
        z += 1
    
    # Assume that all training sets are of similar length ((k-1)/k) * N. When combined with the above performed assertions, 
    # this should be sufficient to establish that the outer and inner splits were performed correctly. 
    
def generate_X_Y_from_cuffdiff(input_dir:str, conditions, log_transform:bool, fpkm_norm:bool, feature_superset:list = None):
    # Generate X (counts data) and Y (labels) matrices.
    
    # Read in X, Y.
    fname = input_dir + 'genes.read_group_tracking'
    counts = read_cuffdiff_counts2(fname, conditions, log_transform, fpkm_norm)
    gene_names = []
    
    rep_to_cond_map = generate_cond_name_to_rep_name_map(input_dir, 1)
    # AH, CT.
    rep_to_label_map = {'q1': 0, 'q2': 1, 'q3':2, 'q4':3, 'q5':4}
    
    rep_to_sample_map = generate_sample_to_replicate_map_cuffdiff(input_dir + 'read_groups.info', True)
    
    X = {}
    for k,v in counts.items():
        if(feature_superset != None):
            if(k not in feature_superset):
                continue
        for rep_count_tuple in v:
            try:
                X[rep_count_tuple[0]].append(rep_count_tuple[1])
            except KeyError:
                X[rep_count_tuple[0]] = [rep_count_tuple[1]]
        gene_names.append(k)
                    
    X_list = []
    Y_list = []
    sample_order = []
    
    for k,v in X.items():
        sample_order.append(rep_to_sample_map[k])
        X_list.append(v)
        label = rep_to_label_map[k[0:2]]
        Y_list.append(label)
            
    X = np.array(X_list)
    Y = np.array(Y_list)
    
    return X, Y, rep_to_cond_map, gene_names, sample_order
    
def select_features_in_CV(X, Y, sample_order:list, gene_names:list, root_dir:str, work_dir:str, cv_k:int, 
                          feature_size:int, num_conditions:int, FS_Mode = 'DE', Filter:bool = True, Filter_Mode = 'Variance',
                          std_treshold:float = 3.5):
    assert(Filter_Mode in ['Variance', 'Intersect', 'Union'])
    assert(FS_Mode in ['DE', 'IG', 'RF'])
    index = 0
    
    fname = os.getcwd() + '/non_pc_taboo_genes.csv'
    non_pc_genes = read_in_csv_file_one_column(fname, 0, ',')
    
    fnames = []
    z = 0
    while z < cv_k:
        fname_in = root_dir + 'Cuffdiff_GEOM_POOL_FOLD' + str(z+1) + '/read_groups.info'
        fnames.append(fname_in)
        z += 1
    
    cv_split = generate_CV_split_cuffdiff(cv_k, fnames, sample_order)
    
    for tr_va_indeces in cv_split:
        
        X_train = X[tr_va_indeces[0], :]
        Y_train = Y[tr_va_indeces[0]]

        if(Filter):
            outlier_features = detect_outlier_features_by_std2(X_train, Y_train, gene_names, std_treshold, True)
            if(Filter_Mode == 'Intersect'):
                outlier_features = set(outlier_features) & set(non_pc_genes)
            elif(Filter_Mode == 'Union'):
                outlier_features = set(outlier_features) | set(non_pc_genes)
        else:
            outlier_features = []

        filename = work_dir + 'top_rnaseq_features_set' + str(index)
        filename += '.txt'
        if(FS_Mode == 'IG'):
            generate_top_IG_features(X_train, Y_train, gene_names, feature_size, filename, outlier_features)
        else:
            fname_in = root_dir + 'Cuffdiff_GEOM_POOL_FOLD' + str(index+1) + '/gene_exp.diff'
            generate_top_DE_features(fname_in, feature_size, gene_names, filename, num_conditions, outlier_features)
        
        index += 1

def select_features_in_nested_CV(X, Y, sample_order:list, gene_names:list, root_dir:str, work_dir:str, num_conditions:int, 
                                 cv_k_outer:int = 5, cv_k_inner:int = 5, num_DEGs:int = 500,
                                 FS_Mode:str = 'DE', Filter:bool = True, Filter_Mode = 'Variance', std_treshold:float = 3.5):
    assert(Filter_Mode in ['Variance', 'Intersect', 'Union'])
    assert(FS_Mode in ['DE', 'IG', 'RF'])
    
    fname = os.getcwd() + '/non_pc_taboo_genes.csv'
    non_pc_genes = read_in_csv_file_one_column(fname, 0, ',')
    
    index_outer = 0
    fnames = []
    while index_outer < cv_k_outer:
        fdir = root_dir + 'Cuffdiff_GEOM_POOL_FOLD' + str(index_outer + 1) + '/'
        fname = fdir + 'read_groups.info'
        
        fnames.append(fname)
        index_outer += 1
        
    outer_cv_split = generate_CV_split_cuffdiff(cv_k_outer, fnames, sample_order)
    #print("Outer_CV_SPLIT: ", outer_cv_split)
    
    index_outer = 0
    for tr_va_te_indeces in outer_cv_split:
        
        X_train_validate = X[tr_va_te_indeces[0], :]
        Y_train_validate = Y[tr_va_te_indeces[0]]
        
        #Flag highly variant or mostly 0 features.
        if(Filter):
            outlier_features = detect_outlier_features_by_std2(X_train_validate, Y_train_validate, gene_names, std_treshold, True)
            if(Filter_Mode == 'Intersect'):
                outlier_features = set(outlier_features) & set(non_pc_genes)
            elif(Filter_Mode == 'Union'):
                outlier_features = set(outlier_features) | set(non_pc_genes)
        else:
            outlier_features = []
        
        fname = work_dir + 'top_rnaseq_features_set' + str(index_outer) + '.txt'
        fname_in = root_dir + 'Cuffdiff_GEOM_POOL_FOLD' + str(index_outer+1) + '/' + 'gene_exp.diff'
        if(FS_Mode == 'IG'):
            generate_top_IG_features(X_train_validate, Y_train_validate, gene_names, num_DEGs, fname, outlier_features)
        else:
            generate_top_DE_features(fname_in, num_DEGs, gene_names, fname, num_conditions, outlier_features)

        index_inner = 0
        fnames = []
        while index_inner < cv_k_inner:
            test_accuracies = []
            fdir = root_dir + 'Cuffdiff_GEOM_POOL_FOLD' 
            fdir += str(index_outer+ 1) + '_' + str(index_inner + 1) + '/'
            fname = fdir + 'read_groups.info'
            
            fnames.append(fname)
            
            index_inner += 1
            
        fdir = root_dir + 'Cuffdiff_GEOM_POOL_FOLD' + str(index_outer + 1) + '/'
        fname = fdir + 'read_groups.info'
        
        outer_sample_order = read_cuffdiff_sample_names(fname)
        
        inner_cv_split = generate_CV_split_cuffdiff(cv_k_inner, fnames, outer_sample_order)
        #print("INNER_CV_SPLIT: ", inner_cv_split)
            
        index_inner = 0
        for tr_va_indeces in inner_cv_split:
            
            X_train = X_train_validate[tr_va_indeces[0], :]
            Y_train = Y_train_validate[tr_va_indeces[0]]
            
            #Flag highly variant or mostly 0 features.
            if(Filter):
                outlier_features = detect_outlier_features_by_std2(X_train, Y_train, gene_names, std_treshold, True)
                if(Filter_Mode == 'Intersect'):
                    outlier_features = set(outlier_features) & set(non_pc_genes)
                elif(Filter_Mode == 'Union'):
                    outlier_features = set(outlier_features) | set(non_pc_genes)
            else:
                outlier_features = []
            
            fname = work_dir + 'top_rnaseq_features_set' + str(index_outer) + '_' + str(index_inner) + '.txt'
            fname_in = root_dir + 'Cuffdiff_GEOM_POOL_FOLD' + str(index_outer+1) + '_' 
            fname_in += str(index_inner+1) + '/' + 'gene_exp.diff'
            if(FS_Mode == 'IG'):
                generate_top_IG_features(X_train, Y_train, gene_names, num_DEGs, fname, outlier_features)
            else:
                generate_top_DE_features(fname_in, num_DEGs, gene_names, fname, num_conditions, outlier_features)
            index_inner += 1
            
        index_outer += 1
    
def perform_CV(X, Y, sample_order:list, gene_names:list, root_dir:str, root_dir2:str, feature_size:int, cv_k:int, 
               model_name:str, sfs:bool = False, balanced_acc:bool = False):
    '''If SFS is selected it will further half the feature size using the IG/DE filtered features as the starting point.'''
    assert(model_name in models_ALL)
    assert(not(model_name == 'RF' and sfs))
    
    p_grid = p_grid_dict[model_name]
    
    hyper_param_names = []
    hyper_param_values = []
    for k,v in p_grid.items():
        hyper_param_names.append(k)
        hyper_param_values.append(v)
    
    hyper_param_grid = list(itertools.product(*hyper_param_values))
    
    max_acc = 0
    best_hyper_param_dict = {}
    
    for hyper_param_tup in hyper_param_grid:
        hyper_param_dict = {}
        i = 0
        for hyper_param_name in hyper_param_names:
            hyper_param_dict[hyper_param_name] = hyper_param_tup[i]
            i += 1
           
        index = 0
        validate_accuracies = []
        
        fnames = []
        z = 0
        while z < cv_k:
            fname_in = root_dir + str(z+1) + '/read_groups.info'
            fnames.append(fname_in)
            z += 1
        
        cv_split = generate_CV_split_cuffdiff(cv_k, fnames, sample_order)
        
        for tr_va_indeces in cv_split:
            
            X_train = X[tr_va_indeces[0], :]
            Y_train = Y[tr_va_indeces[0]]
            X_validate = X[tr_va_indeces[1], :]
            Y_validate = Y[tr_va_indeces[1]]

            # Read in top features (except in case of Random Forest).
            if(model_name != 'RF'):
                feature_file = root_dir2 + str(index) + '.txt'
                genes_to_read = read_in_csv_file_one_column(feature_file, 0, '\t', 0, feature_size)
        
                X_train = select_features_from_matrix(X_train, gene_names, genes_to_read)
                X_validate = select_features_from_matrix(X_validate, gene_names, genes_to_read)
            
            if(model_name == 'Ada'):model = AdaBoostClassifier(**hyper_param_dict)
            elif(model_name == 'RF'):model = RandomForestClassifier(**hyper_param_dict)
            elif(model_name == 'DT'):model = DecisionTreeClassifier(**hyper_param_dict)
            elif(model_name == 'SVM'):model = SVC(**hyper_param_dict)
            elif(model_name == 'kNN'):model = KNeighborsClassifier(**hyper_param_dict)
            elif(model_name == 'LR'):model = LogisticRegression(**hyper_param_dict)
            elif(model_name == 'GNB'):model = GaussianNB(**hyper_param_dict)
            
            if(sfs):
                sfs = SequentialFeatureSelector(model, n_features_to_select = feature_size//2)
                sfs.fit(X_train, Y_train)
                mask = sfs.get_support()
                X_train = X_train[:, mask]
                X_validate = X_validate[:, mask]
                
            clf = model.fit(X_train, Y_train)
            Y_hat = clf.predict(X_validate)
            if(not balanced_acc):
                validate_accuracy = metrics.accuracy_score(Y_validate, Y_hat)
            else:
                validate_accuracy = metrics.balanced_accuracy_score(Y_validate, Y_hat)
            validate_accuracies.append(validate_accuracy)
            
            index += 1
                
        validate_accuracy = np.mean(np.array(validate_accuracies))
        if(validate_accuracy > max_acc):
            max_acc = validate_accuracy
            best_hyper_param_dict = hyper_param_dict
        
    # print("Max Accuracy: ", max_acc)
    # print("Hyper-parameter tuning is over.") 
    # print("Best hyper-parameters are: ", best_hyper_param_dict)
        
    return best_hyper_param_dict
    
def perform_nested_CV(X, Y, sample_order:list, gene_names:list, root_dir:str, work_dir:str, feature_size:int,
                      cv_k_outer:int, cv_k_inner:int, model_name:str, sfs:bool = False, balanced_acc:bool = False):  
    assert(model_name in models_ALL)
    test_accuracies = []
    conf_matrices = []
    
    index_outer = 0
    outer_cv_split = []
    fnames = []
    while index_outer < cv_k_outer:
        fdir = root_dir + 'Cuffdiff_GEOM_POOL_FOLD' + str(index_outer + 1) + '/'
        fname = fdir + 'read_groups.info'
        fnames.append(fname)
        
        index_outer += 1
    
    outer_cv_split = generate_CV_split_cuffdiff(cv_k_outer, fnames, sample_order)
    print("Sample order in nested_CV: ", sample_order)
    
    index_outer = 0
    for tr_va_te_indeces in outer_cv_split:
        
        X_train_validate = X[tr_va_te_indeces[0], :]
        Y_train_validate = Y[tr_va_te_indeces[0]]
        X_test = X[tr_va_te_indeces[1], :]
        Y_test = Y[tr_va_te_indeces[1]]
        print("Test indeces: ", tr_va_te_indeces[1])
        
        # Do inner loop of CV once to find best hyper-parameter configuration.
        index_inner = 0
        fnames = []
        while index_inner < cv_k_inner:
            fdir = root_dir + 'Cuffdiff_GEOM_POOL_FOLD' + str(index_outer+ 1) + '_' + str(index_inner + 1) + '/'
            fname = fdir + 'read_groups.info'
            fnames.append(fname)
            
            index_inner += 1
            
        fdir = root_dir + 'Cuffdiff_GEOM_POOL_FOLD' + str(index_outer + 1) + '/'
        fname = fdir + 'read_groups.info'
        
        outer_sample_order = read_cuffdiff_sample_names(fname)
        
        temp = work_dir + '/top_rnaseq_features_set'
        root_dir_inner = root_dir + 'Cuffdiff_GEOM_POOL_FOLD' + str(index_outer + 1) + '_'
        root_dir_inner2 = temp + str(index_outer) + '_'
        best_hyper_param_dict = perform_CV(X_train_validate, Y_train_validate, outer_sample_order,
                                           gene_names, root_dir_inner, root_dir_inner2, feature_size, cv_k_inner, model_name,
                                           sfs, balanced_acc)
        
        print("NESTED CROSS VALIDATION BEST HYPER-PARAM DICT.")
        print(best_hyper_param_dict)
        
        if(model_name != 'RF'):
            feature_file = temp + str(index_outer) + '.txt'
            genes_to_read = read_in_csv_file_one_column(feature_file, 0, '\t', 0, feature_size)
                    
            X_train_validate = select_features_from_matrix(X_train_validate, gene_names, genes_to_read)
            X_test = select_features_from_matrix(X_test, gene_names, genes_to_read)
        
        if(model_name == 'Ada'):model = AdaBoostClassifier(**best_hyper_param_dict)
        elif(model_name == 'RF'):model = RandomForestClassifier(**best_hyper_param_dict)
        elif(model_name == 'DT'):model = DecisionTreeClassifier(**best_hyper_param_dict)
        elif(model_name == 'SVM'):model = SVC(**best_hyper_param_dict)
        elif(model_name == 'kNN'):model = KNeighborsClassifier(**best_hyper_param_dict)
        elif(model_name == 'LR'):model = LogisticRegression(**best_hyper_param_dict)
        elif(model_name == 'GNB'):model = GaussianNB(**best_hyper_param_dict)
        
        if(sfs):
            sfs = SequentialFeatureSelector(model, n_features_to_select = feature_size//2)
            sfs.fit(X_train_validate, Y_train_validate)
            mask = sfs.get_support()
            X_train_validate = X_train_validate[:, mask]
            X_test = X_test[:, mask]
        
        clf = model.fit(X_train_validate, Y_train_validate)
        Y_hat = clf.predict(X_test)
        print("Y_hat: ", Y_hat)
        print("Y_test: ", Y_test)
        if(not balanced_acc):
            accuracy = metrics.accuracy_score(Y_test, Y_hat)
        else:
            accuracy = metrics.balanced_accuracy_score(Y_test, Y_hat)
        #print("Accuracy: ", accuracy)
        test_accuracies.append(accuracy)
        conf_matrices.append(metrics.confusion_matrix(Y_test, Y_hat))
        
        index_outer += 1
            
    test_accuracies = np.array(test_accuracies)
    mean_test_accuracy = np.mean(test_accuracies)
    print("**************************************")
    print("FEATURE SIZE:", feature_size)
    print("MEAN TEST ACCURACY: ", mean_test_accuracy)       
    i = 0
    mean_conf_matrix = 0
    while i < len(conf_matrices):
        if(i == 0):
            mean_conf_matrix = conf_matrices[0]
        else:
            mean_conf_matrix = mean_conf_matrix + conf_matrices[i]
        i += 1
    print("CONFUSION MATRIX: ", mean_conf_matrix)
    print("**************************************")
    # return mean_test_accuracy, mean_conf_matrix

def classify_with_nested_CV(root_dir:str, work_dir:str, model_name:str, num_conditions:int, cv_k_outer:int, cv_k_inner:int,
                            features_to_gen:int, feature_sizes:list, FS_Mode:str, Filter:bool, Filter_Mode:str, sfs:bool,
                            balanced_acc:bool, tissue:str = 'LV', fpkm_fs:bool = True, fpkm_ml:bool = False, 
                            std_treshold:float = 3.5):
    # ---------------------------------------------------------------------------------------------------------------- 
    # ******************************************Verify Inner / Outer CV Splits****************************************
    # ---------------------------------------------------------------------------------------------------------------- 
    print("Nested cross-validation within our project data.")
    if(FS_Mode != 'DE' and (fpkm_fs != fpkm_ml)):
        raise ValueError("The fpkm policy must be the same for FS and ML portions unless the feature selection is done via DE.")
    assert(num_conditions in [2,3,5])
    assert(FS_Mode in ['DE', 'IG', 'RF'])
    assert(Filter_Mode in ['Variance', 'Intersect', 'Union'])
    assert(model_name in models_ALL)
    assert(tissue in ['PB', 'LV'])
    if(tissue == 'LV'):
        if(num_conditions == 5):
            Expected_Samples = samples_AH_LV_Excluded + samples_CT_LV + samples_AC_LV + samples_NF_LV + samples_HP_LV
        elif(num_conditions == 3):
            Expected_Samples = samples_AH_LV_Excluded + samples_CT_LV + samples_AC_LV
        elif(num_conditions == 2):
            Expected_Samples = samples_AH_LV_Excluded + samples_CT_LV
    elif(tissue == 'PB'):
        if(num_conditions == 2):
            Expected_Samples = samples_AH_PB + samples_CT_PB
        elif(num_conditions == 5):
            Expected_Samples = samples_AH_PB + samples_CT_PB + samples_DA_PB + samples_AA_PB + samples_NF_PB + samples_HP_PB_Excluded
    verify_nested_cross_val_cuffdiff(root_dir, Expected_Samples)
    
    # ---------------------------------------------------------------------------------------------------------------- 
    # *****************************************************FS + Nested CV*********************************************
    # ---------------------------------------------------------------------------------------------------------------- 
    
    input_dir = root_dir + '/Cuffdiff_GEOM_POOL/'
    X, Y, rep_to_cond_map, gene_names, sample_order = generate_X_Y_from_cuffdiff(input_dir, 'ALL', True, fpkm_fs)
    
    # FEATURE SELECTION SECTION
    select_features_in_nested_CV(X, Y, sample_order, gene_names, root_dir, work_dir, num_conditions, cv_k_outer, cv_k_inner,
                                 features_to_gen, FS_Mode, Filter, Filter_Mode, std_treshold)
    
    X, Y, rep_to_cond_map, gene_names, sample_order = generate_X_Y_from_cuffdiff(input_dir, 'ALL', True, fpkm_ml)
    
    # ML PORTION
    for feature_size in feature_sizes:
        fs = feature_size
        if(sfs):
            fs = feature_size*2
        perform_nested_CV(X, Y, sample_order, gene_names, root_dir, work_dir, feature_size, cv_k_outer,
                          cv_k_inner, model_name, sfs, balanced_acc)

def validate_in_independent_data(root_dir:str, work_dir:str, model_name:str, FS_Mode:str, num_conditions:int, cv_k:int, features_to_generate:int,
                            Filter:bool, Filter_Mode:str, sfs:bool, balanced_accuracy:bool, feature_sizes:list, fpkm_fs:bool = True,
                            fpkm_ml:bool = False, std_treshold:float = 3.5, FS_Test:str = 'Total'):
    assert(FS_Test <= cv_k)
    assert(FS_Test >= 0)
    assert(Filter_Mode  in ['Variance', 'Intersect', 'Union'])
    assert(FS_Mode in ['DE', 'IG', 'RF'])
    assert(model_name in models_ALL)
    
    if(FS_Mode != 'DE' and (fpkm_fs != fpkm_ml)):
        raise ValueError("The fpkm policy must be the same for FS and ML portions unless the feature selection is done via DE.")
    
    # AH vs CT Liver Tissue.
    # Step 1: Find top IG/DE features for each training set. 
    # Step 2: Find best hyper-parameters from a pre-determined grid for a given model using pre-computed top IG/DE features in 
    # each training set.
    # Step 3: Find top features for the entire training dataset. ALTERNATIVE: use top features identified in outer CV. 
    # Step 4: Train a model with hyper-parameters from step 2 and top IG features from step 3.
    # Step 5: Test in the independent test set. 
    
    fname = os.getcwd() + '/non_pc_taboo_genes.csv'
    non_pc_genes = read_in_csv_file_one_column(fname, 0, ',')
    
    input_dir = root_dir + 'Cuffdiff_GEOM_POOL/'
    fname = input_dir + 'genes.read_group_tracking'
    
    hg38_starcq_ensembl_genes = set(read_in_csv_file_one_column(fname, 0, '\t', 1))
            
    independent_dataset = os.getcwd() + '/GSE142530_Annoted-RNAseq-with-SampleIDs.csv'
            
    independent_gene_names = set(read_in_csv_file_one_column(independent_dataset, 1, ','))
            
    intersection = hg38_starcq_ensembl_genes & independent_gene_names
    
    # Read in X, Y.
    X, Y, rep_to_cond_map, gene_names, sample_order = generate_X_Y_from_cuffdiff(input_dir, 'ALL', True, fpkm_fs, intersection)
    X2, Y2, rep_to_cond_map2, gene_names2, sample_order2 = generate_X_Y_from_cuffdiff(input_dir, 'ALL', True, fpkm_ml, intersection)
    
    print("Validation in independent's dataset.")
    # FEATURE SELECTION PORTION
    
    if(model_name == 'RF'):
        runs = 5
    else:
        runs = 1
    run_index = 0
    
    while run_index < runs:
        print("RUN: ", run_index)
        if(FS_Mode in ['DE', 'IG']):
            ftg = features_to_generate
            if(sfs):ftg = features_to_generate*2
            select_features_in_CV(X, Y, sample_order, gene_names, root_dir, work_dir, cv_k, ftg,
                                  num_conditions, FS_Mode, Filter, Filter_Mode, std_treshold)
        
        # Step 3:
        if(FS_Test == 0):
            outlier_features = []
            if(Filter):
                outlier_features = detect_outlier_features_by_std2(X, Y, gene_names, std_treshold, True)
                if(Filter_Mode == 'Intersect'):
                    outlier_features = set(outlier_features) & set(non_pc_genes)
                elif(Filter_Mode == 'Union'):
                    outlier_features = set(outlier_features) | set(non_pc_genes)
            
            filename = work_dir + 'top_rnaseq_features.txt'
            
            if(FS_Mode == 'IG' and not sfs):
                generate_top_IG_features(X, Y, gene_names, features_to_generate, filename, outlier_features)
            elif(FS_Mode == 'DE' and not sfs):
                fname_in = root_dir + '/Cuffdiff_GEOM_POOL/gene_exp.diff'
                generate_top_DE_features(fname_in, features_to_generate, gene_names, filename, num_conditions, outlier_features)
            elif(FS_Mode == 'RF'):
                generate_top_RF_features(X, Y, gene_names, features_to_generate, filename, outlier_features)
            elif(FS_Mode == 'DE' and sfs):
                print("DE and SFS.")
                fname_in = root_dir + '/Cuffdiff_GEOM_POOL/gene_exp.diff'
                top_features = generate_top_DE_features(fname_in, features_to_generate*2, gene_names, filename, num_conditions, outlier_features)
                X_filtered = select_features_from_matrix(X, gene_names, top_features)
                generate_top_SFS_features(X_filtered, Y, top_features, features_to_generate, filename, 'LR', {}, outlier_features)
            elif(FS_Mode == 'IG' and sfs):
                top_features = generate_top_IG_features(X, Y, gene_names, features_to_generate*2, filename, outlier_features)
                X_filtered = select_features_from_matrix(X, gene_names, top_features)
                generate_top_SFS_features(X_filtered, Y, top_features, features_to_generate, filename, 'LR', {}, outlier_features)
        
        # ML PORTION
        for feature_size in feature_sizes:
            dir1 = root_dir + 'Cuffdiff_GEOM_POOL_FOLD'
            dir2 = work_dir + 'top_rnaseq_features_set'
            fs = feature_size
            if(sfs):fs = feature_size*2
            best_hyper_param_dict = perform_CV(X2, Y2, sample_order2, gene_names2, dir1, dir2, fs, cv_k, model_name, 
                                                sfs, balanced_accuracy)
            # best_hyper_param_dict = {'kernel': 'poly', 'C': 0.5, 'class_weight': None, 'probability': True, 'degree': 5, 'gamma': 'scale'} 
            print("Best hyper-param dict in validation code: ", best_hyper_param_dict)
            # Step 4:
            # Read in top features.
            
            if(FS_Test == 0):
                feature_file = work_dir + 'top_rnaseq_features.txt'
                genes_to_read = read_in_csv_file_one_column(feature_file, 0, '\t', 0, feature_size)
            else:
                genes_to_read_2d = []
                ii = 0
                while ii < cv_k:
                    feature_file = work_dir + 'top_rnaseq_features_set' + str(ii) + '.txt'
                    genes_to_read = read_in_csv_file_one_column(feature_file, 0, '\t', 0, feature_size)
                    genes_to_read_2d.append(set(genes_to_read))
                    ii += 1
                genes_to_read_2d_eval = count_elements_in_2dlist(genes_to_read_2d)
                genes_to_read = []
                for k,v in genes_to_read_2d_eval.items():
                    if v >= FS_Test:
                        genes_to_read.append(k)
                print("Features selected for independent dataset testing: ", genes_to_read)
                print("Number of features selected: ", len(genes_to_read))
                if(len(genes_to_read) == 0):
                    continue
                    
            X_top = select_features_from_matrix(X2, gene_names, genes_to_read)
                
            if(model_name == 'Ada'):model = AdaBoostClassifier(**best_hyper_param_dict)
            elif(model_name == 'RF'):model = RandomForestClassifier(**best_hyper_param_dict)
            elif(model_name == 'DT'):model = DecisionTreeClassifier(**best_hyper_param_dict)
            elif(model_name == 'SVM'):model = SVC(**best_hyper_param_dict)
            elif(model_name == 'kNN'):model = KNeighborsClassifier(**best_hyper_param_dict)
            elif(model_name == 'LR'):model = LogisticRegression(**best_hyper_param_dict)
            elif(model_name == 'GNB'):model = GaussianNB(**best_hyper_param_dict)
            clf = model.fit(X_top, Y2)
            #print(clf.coef_)
            
            # Step 5:
            # Test in independent dataset.
            counts_independent = {}
            headers = []
            if(num_conditions == 2):
                conditions = ['RB_A', 'RB_N']
            else:
                conditions = ['RB_A', 'RB_E', 'RB_N']
            with open(independent_dataset) as csv_file:
                csv_reader = csv.reader(csv_file, delimiter=',')
                line_count = 0
                for row in csv_reader:
                    if line_count == 0:
                        headers = row
                    else:
                        i = 1
                        gene_name = row[1]
                        if(gene_name not in genes_to_read):
                            continue
                        counts_independent[gene_name] = []
                        while i < len(row):
                            if(headers[i][0:4] in conditions):
                                counts_independent[gene_name].append((headers[i], math.log(1 + float(row[i]))))
                            i = i + 1
                    line_count += 1 
               
            # Have to arrange the features (genes) in exactly the same order for both training and testing data. 
            # This follows the ordering of genes_to_read AKA the ordering of the top_features file.
            
            counts_independent2 = {}
            for gene in genes_to_read:
                v = counts_independent[gene]
                for rep_count_tuple in v:
                    rep = rep_count_tuple[0]
                    count = rep_count_tuple[1]
                    try:
                        counts_independent2[rep].append( count)
                    except KeyError:
                        counts_independent2[rep] = [count]
                
            X_independent = []
            Y_independent = []
            
            if(num_conditions == 2):
                cond_to_label_map = {'RB_A':0, 'RB_N': 1}
            else:
                cond_to_label_map = {'RB_A': 0, 'RB_N': 1, 'RB_E': 2}
            
            rep_order = []
            
            for k,v in counts_independent2.items():
                rep_order.append(k)
                X_independent.append(v)
                Y_independent.append(cond_to_label_map[k[0:4]])
                
            X_ind_test = np.array(X_independent)
            Y_ind_test = np.array(Y_independent)
            
            print("independent's dataset sample order: ", rep_order)
            
            Y_hat = clf.predict(X_ind_test)
            print("Y_hat: ", Y_hat)
            print("Y_ind_test: ", Y_ind_test)
            acc = metrics.accuracy_score(Y_ind_test, Y_hat)
            bacc = metrics.balanced_accuracy_score(Y_ind_test, Y_hat)
            conf_matrix = metrics.confusion_matrix(Y_ind_test, Y_hat)
            print("***********************************")
            print("Feature Size: ", feature_size)
            if(not balanced_accuracy):
                print("independent Dataset Accuracy: ", acc)
            else:
                print("independent Dataset Balanced Accuracy: ", bacc)
            print("independent Dataset Confusion Matrix: ", conf_matrix)
            print("***********************************")
            
        run_index += 1

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

def parse_execution_log(fname:str, out_fname:str, tissue = 'LV'):
    assert(tissue in ['LV', 'LV_Test', 'PB'])
    with open(fname) as reader:
        i = 0
        lines = reader.readlines()
        
    feature_sizes = []
    nested_cv_accuracies = []
    independent_accuracies = []
    for line in lines:
        if(line.startswith('Feature Size:')):
            f_size = line[15:].replace('\n', '')
            feature_sizes.append(f_size)
        if(line.startswith('MEAN TEST ACCURACY: ') and tissue != 'LV_Test'):
            accuracy = (line.replace('MEAN TEST ACCURACY:  ', ''))[:4]
            nested_cv_accuracies.append(accuracy)
        if(line.startswith('independent Dataset Accuracy:  ') and tissue in ['LV', 'LV_Test']):
            accuracy = (line.replace('independent Dataset Accuracy:  ', ''))[:4]
            independent_accuracies.append(accuracy)
        elif(line.startswith('independent Dataset Balanced Accuracy:  ') and tissue in ['LV', 'LV_Test']):
            accuracy = (line.replace('independent Dataset Balanced Accuracy:  ', ''))[:4]
            independent_accuracies.append(accuracy)
    
    feature_sizes[-1] = feature_sizes[-1].replace('\n', '')
    with open(os.getcwd() + '/' + out_fname, 'w') as writer:
        i = 0
        if(tissue != 'LV_Test'):
            limit = len(feature_sizes)
        else:
            limit = len(independent_accuracies)
        while i < len(feature_sizes):
            writer.write(feature_sizes[i])
            if(tissue != 'LV_Test'):
                writer.write(',')
                writer.write(str(nested_cv_accuracies[i]))
            if(tissue in ['LV', 'LV_Test']):
                writer.write(',')
                writer.write(str(independent_accuracies[i]))
            writer.write('\n')
            i += 1

def identify_misclassified_samples(filename:str):
    '''Parse through the performance log. Identify which sample IDs were classified correctly 
    and which were misclassified. Also identify what the samples are being misclassified as.'''
    sample_order = []
    test_indices_all = []
    yhats = []
    ytests = []
    fsizes = 0
    flag1 = False
    flag2 = 0
    flag3 = 0
    index3 = 0
    result = {}
    result2 = {}
    with open(filename) as reader:
        lines = reader.readlines()
        for line in lines:
            if(line.startswith('Feature Sizes:')):
                fsizes = line[16:]
                fsizes = fsizes.replace('[', '')
                fsizes = fsizes.replace(']', '')
                fsizes = fsizes.replace(' ', '')
                fsizes = fsizes.split(',')
                fsizes = len(fsizes)
                flag3 = fsizes*5
            if(line.startswith('Sample order in nested_CV:') and (not flag1)):
                sample_order = line[28:]
                sample_order = sample_order.replace('[', '')
                sample_order = sample_order.replace(']', '')
                sample_order = sample_order.replace("'", "")
                sample_order = sample_order.replace(" ", "")
                sample_order = sample_order.split(',')
                flag1 = True
            if(line.startswith('Test indeces:  ') and (flag2 < 5)):
                test_indices = line[15:]
                test_indices = test_indices.replace('[', '')
                test_indices = test_indices.replace(']', '')
                test_indices = test_indices.replace(' ', '')
                test_indices = test_indices.split(',')
                test_indices_int = []
                for test_index in test_indices:
                    test_indices_int.append(int(test_index))
                test_indices_all.append(test_indices_int)
                flag2 += 1
            if(line.startswith('Y_hat:  ')):
                yhat = line[8:]
                yhat = yhat.replace('[', '')
                yhat = yhat.replace(']', '')
                yhat = yhat.split(' ')
                yhats.append(yhat)
            if(line.startswith('Y_test:  ')):
                ytest = line[9:]
                ytest = ytest.replace('[', '')
                ytest = ytest.replace(']', '')
                ytest = ytest.split(' ')
                ytests.append(ytest)
    
    #print(sample_order)
    #print(test_indices_all)
    while index3 < flag3:
        cur_yhat = yhats[index3]
        cur_ytest = ytests[index3]
        #print(cur_yhat)
        #print(cur_ytest)
        
        i = 0
        fold = index3 % 5
        assert(len(cur_yhat) == len(cur_ytest) == len(test_indices_all[fold]))
        while i < len(cur_yhat):
            if(cur_yhat[i] != cur_ytest[i]):
                test_indices = test_indices_all[fold]
                test_index = test_indices[i]
                sample_name = sample_order[test_index]
                #print(sample_name)
                try:
                    result[sample_name] = result[sample_name] + 1
                except KeyError:
                    result[sample_name] = 1
                    
                try:
                    result2[sample_name]
                except KeyError:
                    result2[sample_name] = {}
                    
                try:
                    result2[sample_name][cur_yhat[i]] = result2[sample_name][cur_yhat[i]] + 1
                except KeyError:
                    result2[sample_name].update({cur_yhat[i]:1})
            i += 1
        
        index3 +=1
        
    #print(result)
    return result, result2
                

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
    plt.figure(figsize=(width,height), dpi=600)
    # 'BrBG'
    seaborn.heatmap(df_cm, cmap = 'Blues', vmin=0, vmax=7)
    plt.xlabel('Conditions', fontsize = 14)
    plt.ylabel('Genes', fontsize = 14)
    if(fold > 0):
        plt.title('FOLD' + str(fold), fontsize = 16)
    else:
        plt.title('Total Cuffnorm Counts', fontsize = 16)
    out_file = out_dir + 'FOLD' + str(fold) + '_' + run_name + '.png'
    plt.savefig(out_file, dpi=600, bbox_inches="tight")
    #plt.savefig(out_file, dpi=300)
    # plt.show()
    
def plot_per_sample_counts_heatmap(counts:dict, top_genes:list, samples_file:str, fold:int, out_dir:str, num_genes:int = 30):
    # Draw a per-replicate heatmap
    temp = out_dir.split('/')
    run_name = temp[-3]
    
    array = []
    rep_names = []
    for rep_name,values in counts.items():
        rep_names.append(rep_name)
        array.append(values)
    if(samples_file != None):
        filenames = filenames_to_replicate_names_cuffdiff(samples_file, rep_names, True)
        sample_names = []
        for filename in filenames:
            fname = filename.split('.')
            sample_names.append(fname[0])
    else:
        sample_names = rep_names
    array = np.array(array)
    
    # We limit number of top genes to num_genes for the heatmap.
    if(array.shape[1] > num_genes):
        array = array[:, 0:num_genes]
        top_genes = top_genes[0:num_genes]
        
    df_cm = pandas.DataFrame(array, index = sample_names,
                             columns = top_genes)
    
    width = len(top_genes) // 3
    height = len(rep_names) // 3
    print(width)
    print(height)
    plt.figure(figsize=(width,height), dpi=600)
    # 'BrBG'
    seaborn.heatmap(df_cm, cmap = 'Blues', vmin=0, vmax=7)
    plt.xlabel('Genes', fontsize = 14)
    plt.ylabel('Replicates', fontsize = 14)
    if(fold > 0):
        plt.title('FOLD' + str(fold), fontsize = 16)
    else:
        plt.title('Total Counts', fontsize = 16)
    out_file = out_dir + 'FOLD' + str(fold) + '_' + run_name + '.png'
    plt.savefig(out_file, dpi=600, bbox_inches="tight")
    #plt.savefig(out_file, dpi=300)
    # plt.show()

class TestAHProjectCodeBase(unittest.TestCase):  
    def test_read_in_csv_file_one_column(self):
        root_dir = 'REDACTED'
        filename = root_dir + 'hg38_Starcq_Ensembl.csv'
        data = read_in_csv_file_one_column(filename, 0, ',')
        self.assertEqual(data, ['Features', '10', '30', '50', '100', '150', '200', '250'])
        data2 = read_in_csv_file_one_column(filename, 1, ',')
        self.assertEqual(data2, ['log reg', '82.0', '96.33333333333334', '100.0', '98.00000000000001', '100.0', '100.0', '100.0'])
        
    def test_perform_CV(self):
        self.assertEqual(False, True)
        
    def test_perform_nested_CV(self):
        self.assertEqual(False, True)
    
    def test_select_features_in_CV(self):
        self.assertEqual(False, True)
        
    def test_select_features_in_nested_CV(self):
        self.assertEqual(False, True)
    
    def test_generate_sample_to_replicate_map_cuffdiff(self):
        self.assertEqual(False, True)
    
    def test_generate_cond_name_to_rep_name_map(self):
        file_dir = 'REDACTED'
        file_option = 0
        result = generate_cond_name_to_rep_name_map(file_dir, file_option)
        self.assertEqual(result, {'AH':'q1', 'CT':'q2', 'DA':'q3'})
        
        file_option = 1
        result = generate_cond_name_to_rep_name_map(file_dir, file_option)
        self.assertEqual(result, {'AH':'q1', 'CT':'q2', 'DA':'q3'})
    
    def test_verify_nested_cross_val_cuffdiff(self):
        self.assertEqual(False, True)
    
    def test_generate_top_DE_features(self):
        
        fname = 'REDACTED'
        feature_names = ['5S_rRNA', '5_8S_rRNA', '7SK', 'A1BG', 'A1BG-AS1', 'A1CF', 'A2M', 'A2M-AS1', 'A2ML1']
        fname_out = 'REDACTED'
        fname_out += 'TestOutput/test_generate_top_DE_features.txt'
        generate_top_DE_features(fname, 10, feature_names, fname_out, 3, taboo_features = [], q_value = 0.97, fpkm = 0.3)
        genes_exp = ['5_8S_rRNA', 'A2M', '7SK', 'A1BG-AS1', '5S_rRNA', 'A2M-AS1']
        genes_out = read_in_csv_file_one_column(fname_out, 0, ',')
        
        self.assertEqual(genes_exp, genes_out)
        
    def test_generate_top_IG_features(self):
        X = np.array([[1,64,33,1],[2,20,40,1],[2,90,30,1],[17,18,35,2],[18,24,42,2]])
        Y = np.array([1,1,1,2,2])
        feature_names = ['gene1', 'gene2', 'gene3', 'gene4']
        out_dir = 'REDACTED'
        fname_out = out_dir + 'test_generate_top_IG_features.txt'
        generate_top_IG_features(X, Y, feature_names, 4, fname_out)
        
        genes_out = read_in_csv_file_one_column(fname_out, 0, ',')
        genes_exp = ['gene4', 'gene1', 'gene2', 'gene3']
        genes_exp_alt = ['gene1', 'gene4', 'gene2', 'gene3']
        
        self.assertIn(genes_out, [genes_exp, genes_exp_alt])
    
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
    
    def test_generate_nested_kfolds(self):
        samples = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', 
                   '13', '14', '15', '16', '17', '18']
        
        ['1', '3', '5', '7', '9', '11', '13', '15', '17']
        ['2', '4', '6', '8', '10', '12', '14', '16', '18']
        
        folds = generate_nested_kfolds(samples, 2, 2)
        self.assertEqual(folds[1][0], ['1','5','9','13','17'])
        self.assertEqual(folds[1][1], ['3','7','11','15'])
        self.assertEqual(folds[0][0], ['2', '6', '10', '14', '18'])
        self.assertEqual(folds[0][1], ['4', '8', '12', '16'])
        
        ['1', '4', '7', '10', '13', '16']
        ['2', '5', '8', '11', '14', '17']
        ['3', '6', '9', '12', '15', '18']
        folds = generate_nested_kfolds(samples, 3, 3)
        self.assertEqual(folds[0][0], ['2', '11', '3', '12'])
        self.assertEqual(folds[0][1], ['5', '14', '6', '15'])
        self.assertEqual(folds[0][2], ['8', '17', '9', '18'])
        self.assertEqual(folds[1][0], ['1', '10', '3', '12'])
        self.assertEqual(folds[1][1], ['4', '13', '6', '15'])
        self.assertEqual(folds[1][2], ['7', '16', '9', '18'])
        self.assertEqual(folds[2][0], ['1', '10', '2', '11'])
        self.assertEqual(folds[2][1], ['4', '13', '5', '14'])
        self.assertEqual(folds[2][2], ['7', '16', '8', '17'])
        
        ['1', '4', '7', '10', '13', '16']
        ['2', '5', '8', '11', '14', '17']
        ['3', '6', '9', '12', '15', '18']
        folds = generate_nested_kfolds(samples, 5, 3)
        self.assertEqual(folds[0][0], ['2', '17', '15'])
        self.assertEqual(folds[0][1], ['5', '3', '18'])
        self.assertEqual(folds[0][2], ['8', '6'])
        self.assertEqual(folds[0][3], ['11', '9'])
        self.assertEqual(folds[0][4], ['14', '12'])
        self.assertEqual(folds[1][0], ['1', '16', '15'])
        self.assertEqual(folds[1][1], ['4', '3', '18'])
        self.assertEqual(folds[1][2], ['7', '6'])
        self.assertEqual(folds[1][3], ['10', '9'])
        self.assertEqual(folds[1][4], ['13', '12'])
        self.assertEqual(folds[2][0], ['1', '16', '14'])
        self.assertEqual(folds[2][1], ['4', '2', '17'])
        self.assertEqual(folds[2][2], ['7', '5'])
        self.assertEqual(folds[2][3], ['10', '8'])
        self.assertEqual(folds[2][4], ['13', '11'])
        
        ['1', '6', '11', '16']
        ['2', '7', '12', '17']
        ['3', '8', '13', '18']
        ['4', '9', '14']
        ['5', '10', '15']
        folds = generate_nested_kfolds(samples, 3, 5)
        self.assertEqual(folds[0][0], ['2', '17', '13', '9', '10'])
        self.assertEqual(folds[0][1], ['7', '3', '18', '14', '15'])
        self.assertEqual(folds[0][2], ['12', '8', '4', '5'])
        self.assertEqual(folds[1][0], ['1', '16', '13', '9', '10'])
        self.assertEqual(folds[1][1], ['6', '3', '18', '14', '15'])
        self.assertEqual(folds[1][2], ['11', '8', '4', '5'])
        self.assertEqual(folds[2][0], ['1', '16', '12', '9', '10'])
        self.assertEqual(folds[2][1], ['6', '2', '17', '14', '15'])
        self.assertEqual(folds[2][2], ['11', '7', '4', '5'])
        self.assertEqual(folds[3][0], ['1', '16', '12', '8', '5'])
        self.assertEqual(folds[3][1], ['6', '2', '17', '13', '10'])
        self.assertEqual(folds[3][2], ['11', '7', '3', '18', '15'])
        self.assertEqual(folds[4][0], ['1', '16', '12', '8', '4'])
        self.assertEqual(folds[4][1], ['6', '2', '17', '13', '9'])
        self.assertEqual(folds[4][2], ['11', '7', '3', '18', '14'])
    
    def test_generate_cuffnorm_or_cuffdiff_batch_file_HPC(self):
        # We only care about the samples being correctly distributed within correctly named files.
        # Ignore testing the header and other lines. 
        
        # CV Test
        root_dir = 'REDACTED'
        generate_cuffnorm_or_cuffdiff_batch_file_HPC('hg38', 'starcq', 'ensembl', 'GEOM', ['AC', 'NF'],
                                                     'AC_NF', 'AC_NF', root_dir, 'Cuffdiff', 5, 'POOL', 'LV2')
        
        AC_LV_Folds = generate_kfolds(samples_AC_LV, 5)
        NF_LV_Folds = generate_kfolds(samples_NF_LV, 5)
        j = 0
        AC_LV_Train_Sets, _ = divide_folds_in_training_and_validation(AC_LV_Folds)
        NF_LV_Train_Sets, _ = divide_folds_in_training_and_validation(NF_LV_Folds)
        while j < 5:
            fname = 'SL_8AC_vs_10NF_LV_hg38_starcq_ensembl_GEOM_POOL_FOLD' + str(j+1) + '.sh'
            full_path = root_dir + fname
            
            with open(full_path) as f:
                lines = f.readlines()
                sample_line = lines[17]
                AC_samples = sample_line.split(' ')[10]
                NF_samples = sample_line.split(' ')[11]
                AC_samples = AC_samples.split(',')
                NF_samples = NF_samples.split(',')
                AC_samples_trimmed = []
                NF_samples_trimmed = []
                for AC_sample in AC_samples:
                    AC_samples_trimmed.append(AC_sample.split('.')[0])
                for NF_sample in NF_samples:
                    NF_samples_trimmed.append(NF_sample.split('.')[0])
                self.assertEqual(AC_samples_trimmed, AC_LV_Train_Sets[j])
                self.assertEqual(NF_samples_trimmed, NF_LV_Train_Sets[j])
            
            j += 1
        
        
        # Nested CV Test
        root_dir = 'REDACTED'
        generate_cuffnorm_or_cuffdiff_batch_file_HPC('hg38', 'starcq', 'ensembl', 'GEOM', ['AC', 'NF'],
                                                     'AC_NF', 'AC_NF', root_dir, 'Cuffdiff', [5,5], 'POOL', 'LV2')
        
        AC_LV_Folds = generate_nested_kfolds(samples_AC_LV, 5, 5)
        NF_LV_Folds = generate_nested_kfolds(samples_NF_LV, 5, 5)
        i = 0
        while i < 5:
            j = 0
            AC_LV_Train_Sets, _ = divide_folds_in_training_and_validation(AC_LV_Folds[i])
            NF_LV_Train_Sets, _ = divide_folds_in_training_and_validation(NF_LV_Folds[i])
            while j < 5:
                fname = 'SL_8AC_vs_10NF_LV_hg38_starcq_ensembl_GEOM_POOL_FOLD' + str(i+1) + '_' + str(j+1) + '.sh'
                full_path = root_dir + fname
                
                with open(full_path) as f:
                    lines = f.readlines()
                    sample_line = lines[17]
                    AC_samples = sample_line.split(' ')[10]
                    NF_samples = sample_line.split(' ')[11]
                    AC_samples = AC_samples.split(',')
                    NF_samples = NF_samples.split(',')
                    AC_samples_trimmed = []
                    NF_samples_trimmed = []
                    for AC_sample in AC_samples:
                        AC_samples_trimmed.append(AC_sample.split('.')[0])
                    for NF_sample in NF_samples:
                        NF_samples_trimmed.append(NF_sample.split('.')[0])
                    self.assertEqual(AC_samples_trimmed, AC_LV_Train_Sets[j])
                    self.assertEqual(NF_samples_trimmed, NF_LV_Train_Sets[j])
                
                j += 1
            i += 1
    
    def test_detect_outlier_features_by_std2(self):
        X = np.array([[1,4,1313], [2.1, 3.6, 1210], [987266, 2.4, 981], [2, 5, 1000], [1, 6, 1600],
                      [3,3,999], [1, 11, 16], [2, 13, 17], [3, 1063, 17], [2, 21, 22]])
        Y = np.array([1,1,1,1,1,1,3,3,3,3])
        feature_names = ['gene1', 'gene2', 'gene3']
        outlier_feature_names = detect_outlier_features_by_std2(X, Y, feature_names, treshold = 2)
        self.assertEqual(outlier_feature_names, {'gene1'})
    
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
    
    def test_generate_CV_split_cuffdiff(self):
        fname = 'REDACTED'
        fname += 'CV_Cuffdiff/Cuffdiff_GEOM_POOL/read_groups.info'
        sample_order = read_cuffdiff_sample_names(fname)
        
        i = 0
        fnames = []
        while i < 5:
            fname = 'REDACTED'
            fname += 'CV_Cuffdiff/Cuffdiff_GEOM_POOL_FOLD' + str(i+1) + '/read_groups.info'
            i += 1
            fnames.append(fname)
        
        cv_split = generate_CV_split_cuffdiff(5, fnames, sample_order)
        cv_split_exp_0 = ([1,6,11,16,21,26,31,2,7,12,17,22,27,3,8,13,18,23,28,4,9,14,19,24,29,33,38,34,39,35,36,41,46,42,47,
                           43,44], [0,5,10,15,20,25,30,32,37,40,45])
        cv_split_0 = cv_split[0]
        assert(cv_split_exp_0[0] == cv_split_0[0])
        assert(cv_split_exp_0[1] == cv_split_0[1])
        
    def test_read_cuffdiff_counts_mean_std(self):
        root_dir = 'REDACTED'
        filename = root_dir + 'genes_mini2.read_group_tracking'
        means_stds = read_cuffdiff_counts_mean_std(filename)
        means_stds_exp = {'A1BG': ({'q1':17.1999, 'q2':15.9426, 'q3':18.4931}, {'q1':5.623799999999999, 'q2':0.0, 'q3':0.0}),
                        'A1BG-AS1':({'q1':4.695444999999999, 'q2':3.89096, 'q3':4.06879}, {'q1':0.14160499999999976, 'q2':0.0, 'q3':0.0}),
                        'A1CF':({'q1':0.0029543015, 'q2':0.0, 'q3':0.00144417}, {'q1':0.0020792085, 'q2':0.0, 'q3':0.0}),
                        'A2M':({'q1':0.7741495, 'q2':0.894067, 'q3':0.343044}, {'q1':0.5593604999999999, 'q2':0.0, 'q3':0.0})}

        self.assertEqual(means_stds, means_stds_exp)
    
    def test_read_cuffdiff_counts2(self):
        # All condition tests
        
        root = 'REDACTED'
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
        root = 'REDACTED'
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
            
    def test_filter_cuffdiff_file_by_gene_list(self):
        self.assertEqual(True, False)

    def test_generate_X_Y_from_cuffdiff(self):
        root_dir = 'REDACTED'
        root_dir += 'REDACTED'
        X, Y, rep_to_cond_map, gene_names, sample_order = generate_X_Y_from_cuffdiff(root_dir, 'ALL', False, True)
        X_exp = [[10.6207, 2.53904, 0.0], [16.7383, 4.14641, 0.0], [17.7691, 4.36101, 0.0126079], [7.88391, 3.53608, 0.0420014],
                 [16.1968, 4.19106, 0.00762296], [39.7524, 4.28753, 0.00121871], [5.35104, 0.765172, 0.0], [14.1845, 3.58283, 0.0],
                 [16.7471, 4.33035, 0.0155117], [11.9061, 2.06823, 0.00336543], [24.4613, 2.76009, 0.0], [15.7512, 3.32179, 2.72336e-12],
                 [11.5389, 2.75439, 0.0], [42.8423, 1.61298, 0.0757167], [15.9346, 3.8949, 0.0], [15.6387, 3.01016, 0.0],
                 [15.7639, 5.43479, 0.000838561], [19.7136, 2.58082, 0.000863692], [18.0964, 4.52213, 0.0],
                 [6.58358, 3.09222, 0.0157814], [29.6065, 2.70165, 0.0], [16.0292, 3.05458, 0.00962138], 
                 [11.5212, 4.53223, 0.00500963], [20.9872, 4.25877, 0.00314195], [22.3081, 5.74062, 0.0],
                 [13.0893, 4.95695, 0.00139864], [22.6631, 4.80301, 0.000868936], [25.8321, 3.97162, 0.0269552],
                 [5.91922, 1.11481, 0.0103098], [13.55, 2.41929, 0.00435907], [39.965, 2.75576, 0.00334643], 
                 [19.8884, 4.51623, 0.0], [7.29916, 1.77918, 0.0240827], [15.6721, 4.0199, 0.00154084],
                 [17.2083, 3.03788, 0.00397897], [14.6824, 3.46625, 0.00199345], [23.2805, 4.55551, 0.00350407],
                 [25.1598, 3.23074, 0.0], [20.8372, 2.34038, 0.00071191], [18.0142, 3.49368, 0.0120355], 
                 [13.7818, 2.88274, 0.00489945], [9.76784, 3.4074, 0.0], [16.0931, 3.99396, 0.000886564],
                 [24.0494, 3.61774, 0.0], [44.2072, 5.83818, 0.0], [11.9264, 3.30717, 0.00240646], [10.7545, 2.98302, 0.0],
                 [5.65841, 0.947301, 0.00239017], [11.87, 2.03333, 0.000813736], [17.5241, 4.06875, 0.00709207], 
                 [15.8664, 3.87235, 0.0], [19.997, 3.33069, 0.00367158]]
        Y_exp = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
        rep_to_cond_map_exp = {'AH':'q1', 'CT':'q2'}
        gene_names_exp = ['A1BG', 'A1BG-AS1', 'A1CF']
        sample_order_exp = ['REDACTED']
        
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
            
        for k,v in rep_to_cond_map_exp.items():
            assert(rep_to_cond_map_exp[k] == rep_to_cond_map[k])
            
        self.assertEqual(gene_names, gene_names_exp)
        self.assertEqual(sample_order, sample_order_exp)

def test_suite():
    # Unit Tests
    unit_test_suite = unittest.TestSuite()
    unit_test_suite.addTest(TestAHProjectCodeBase("test_read_in_csv_file_one_column"))
    unit_test_suite.addTest(TestAHProjectCodeBase("test_read_cuffdiff_counts2"))
    unit_test_suite.addTest(TestAHProjectCodeBase("test_generate_CV_split_cuffdiff"))
    unit_test_suite.addTest(TestAHProjectCodeBase("test_generate_X_Y_from_cuffdiff"))
    unit_test_suite.addTest(TestAHProjectCodeBase('test_read_cuffdiff_counts_mean_std'))
    unit_test_suite.addTest(TestAHProjectCodeBase('test_two_dim_list_len'))
    unit_test_suite.addTest(TestAHProjectCodeBase('test_detect_outlier_features_by_std2'))
    unit_test_suite.addTest(TestAHProjectCodeBase('test_generate_cuffnorm_or_cuffdiff_batch_file_HPC'))
    unit_test_suite.addTest(TestAHProjectCodeBase('test_generate_nested_kfolds'))
    unit_test_suite.addTest(TestAHProjectCodeBase('test_generate_kfolds'))
    unit_test_suite.addTest(TestAHProjectCodeBase('test_generate_top_IG_features'))
    unit_test_suite.addTest(TestAHProjectCodeBase('test_generate_top_DE_features'))
    # unit_test_suite.addTest(TestAHProjectCodeBase('test_perform_CV'))
    # unit_test_suite.addTest(TestAHProjectCodeBase('test_perform_nested_CV'))
    # unit_test_suite.addTest(TestAHProjectCodeBase('test_select_features_in_CV'))
    # unit_test_suite.addTest(TestAHProjectCodeBase('test_select_features_in_nested_CV'))
    # unit_test_suite.addTest(TestAHProjectCodeBase('test_generate_sample_to_replicate_map_cuffdiff'))
    unit_test_suite.addTest(TestAHProjectCodeBase('test_generate_cond_name_to_rep_name_map'))
    unit_test_suite.addTest(TestAHProjectCodeBase('test_filter_cuffdiff_file_by_gene_list'))
    # unit_test_suite.addTest(TestAHProjectCodeBase('test_verify_nested_cross_val_cuffdiff'))
    
    # MANUALLY TEST: select_features_in_nested_CV (tested the correctness of X, X_train_validate, X_train matrix generation)
    
    runner = unittest.TextTestRunner()
    runner.run(unit_test_suite)

    
