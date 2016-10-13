import pandas as pd
import numpy as np
import gc
import matplotlib.pyplot as plt
from Common import *

def FindUniqueCols(filename, writeFilename):
    path = 'train/' + filename
    df = pd.read_csv(path, nrows=1)
    cols=df.columns
    colLen=len(cols)
    #number of slices that you want to cut the columnset into so that each slice can fit into memory
    n = 20
    col_slice = int(colLen/n)
    print (colLen,col_slice)
    # dictionary to store hash,columnlist
    col_hash={}
    
    # process each column slice of the input file
    for i in range(n):
        left = i*(col_slice)
        right = (i+1)*(col_slice)+1
        print (i,left,right)
        df = pd.read_csv(path, dtype = str, skipinitialspace=True, usecols=cols[left:right])
        for c in cols[left:right]:
            hash_val=hash(tuple(df[c]))
            if hash_val in col_hash:
                col_hash[hash_val].append(c)
            else:
                col_hash[hash_val]=[c]
        print (len(col_hash))
    
    uniqCat = None
    # print all unique columns
    for key in col_hash:
        if (uniqCat == None):
            uniqCat = col_hash[key][0]
        else:
            uniqCat = np.append(uniqCat, col_hash[key][0])
    WriteFile(writeFilename, uniqCat)

def AnalyzeDateData():
    trainDateData = OpenFile('Date_Data.pk') 
    q = trainDateData.iloc[:,:].values
    q=q[~np.isnan(q)]
    plt.figure(0)
    plt.subplot(211)
    z = plt.hist(q, bins=20, hold=True)
    
    trainData = OpenFile('Response_1s.pk')    
    dateData = trainData[trainData.columns[-1156:]].copy()
    dateData = dateData.dropna(axis=1, how='all')
    dateData = dateData.T.drop_duplicates().T
    p = dateData.iloc[:,:].values
    p=p[~np.isnan(p)]
    plt.subplot(212)
    y = plt.hist(p, bins=20, hold=True)
    
    z=z[0]
    y=y[0]
    zz = [a_i - b_i for a_i, b_i in zip(z, y)]
    xx = [a_i / b_i for a_i, b_i in zip(y, zz)]
    plt.figure(1)
    plt.plot(xx)

def UniqueDateCols(trainData):    
    dateData = trainData[trainData.columns[-1156:]].copy()
    dateData = dateData.dropna(axis=1, how='all')
    dateData = dateData.T.drop_duplicates().T 
    dateCols = np.array(dateData.columns)
    dateCols = np.append(['Id'], dateCols)  
    dateCols = np.append(dateCols, ['Response']) 
    return dateCols
    
def ReadTD_Resp1():
    ''' Reads all columns for every row which has 'Response' columns == 1 '''
    Directory = 'train/'
    Files = ['train_numeric.csv', \
                  'train_categorical.csv', \
                  'train_date.csv']

    trainData = None
    maskArr = None
    for j,f in enumerate(Files):
        print("file: " + f)
        subset = None
        for i, chunk in enumerate(pd.read_csv(Directory + f, \
                                                chunksize = 50000, \
                                                low_memory = False)):
            print("chunk " + str(i))
            #First file is train_numeric, so filter rows by 'Response' value
            if j == 0:
                chunk = chunk[chunk['Response'] == 1]
            if subset is None:
                subset = chunk.copy()
            else:
                subset = pd.concat([subset, chunk])
            del chunk
            gc.collect()
            #Create array of Id's that have a Response == 1
            if j == 0:
                if subset.size > 0:
                    if maskArr is None:
                        maskArr = np.array(subset['Id'])
                    else:
                        maskArr = np.append(maskArr, subset['Id'])
            #If not reading from train_numeric file, filter on rows found in maskArr
            else:
                subset = subset[subset['Id'].isin(maskArr)]
        #Merge columns from all 3 files
        if trainData is None:
            trainData = subset.copy()
        else:
            trainData = pd.merge(trainData, subset.copy(), on="Id")
        del subset
        gc.collect()
    #Write to Pickle file
    WriteFile('TD_Resp1.pk', trainData)
        
def ReadTD_FeatureCnt():
    ''' Creates a new feature: For every file, the count of how many features are in each row'''
    Directory = 'train/'
    Files = ['train_numeric.csv', \
                  'train_categorical.csv', \
                  'train_date.csv']

    trainData = None
    for j,f in enumerate(Files):
        print("file: " + f)
        subset = None
        for i, chunk in enumerate(pd.read_csv(Directory + f, \
                                                chunksize = 50000, \
                                                low_memory = False)):
            print("chunk " + str(i))
            #Sum up the feature count in each row. Subtract 1 for the 'Id' column
            sumCnt = chunk.notnull().sum(axis=1)
            sumCnt[:] = [x - 1 for x in sumCnt]
            #Create a new column 'NumericCnt if reading from train_numeric
            if j == 0:
                #Subtract an additional 1 from the train_numeric file for the 'Response' column
                sumCnt[:] = [x - 1 for x in sumCnt]
                chunkMod = pd.DataFrame({"Id": chunk.Id.values, \
                               "NumericCnt": sumCnt, \
                               "Response": chunk.Response.values}) 
            #Create a new column 'CategoricalCnt if reading from train_categorical
            elif j == 1:
                chunkMod = pd.DataFrame({"Id": chunk.Id.values, \
                               "CategoricalCnt": sumCnt})
            #Create a new column 'DateCnt if reading from train_date
            elif j == 2:
                chunkMod = pd.DataFrame({"Id": chunk.Id.values, \
                               "DateCnt": sumCnt})
            if subset is None:
                subset = chunkMod.copy()
            else:
                subset = pd.concat([subset, chunkMod])
            del chunk
            gc.collect()
            
        if trainData is None:
            trainData = subset.copy()
        else:
            trainData = pd.merge(trainData, subset.copy(), on="Id")
        del subset
        gc.collect()
    #Write to Pickle file
    WriteFile('TD_FeatureCnt.pk', trainData)

def ReadTD_StartTime(cols):
    ''' Creates a new feature: Earliest start time for each feature'''
    trainData = None
    subset = None
    #Only use unique date columns as defined by cols
    for i, chunk in enumerate(pd.read_csv('train/train_date.csv', \
                                                chunksize = 50000, \
                                                usecols=cols, \
                                                low_memory = False)):
        print("chunk " + str(i))
        #Get only date columns, which is every column except 'Id'
        dateCols = np.setdiff1d(chunk.columns, ['Id'])
        #Find smallest value for each row
        chunk['Start_Time'] = chunk[dateCols].min(axis=1).values
        if subset is None:
            subset = chunk.copy()
        else:
            subset = pd.concat([subset, chunk])
        del chunk
        gc.collect()
    if trainData is None:
        trainData = subset.copy()
    else:
        trainData = pd.merge(trainData, subset.copy(), on="Id")
    del subset
    gc.collect()
    #Get 'Response' for all Id's from train_numeric file                                                    
    subset = None                                                
    for i, chunk in enumerate(pd.read_csv('train/train_numeric.csv', \
                                                chunksize = 50000, \
                                                usecols=['Response', 'Id'], \
                                                low_memory = False)):
        print("chunk " + str(i))
        chunk = chunk[['Response', 'Id']]
        if subset is None:
            subset = chunk.copy()
        else:
            subset = pd.concat([subset, chunk])
        del chunk
        gc.collect()
    if trainData is None:
        trainData = subset.copy()
    else:
        trainData = pd.merge(trainData, subset.copy(), on="Id")
    del subset
    gc.collect()
    #Write to Pickle file
    WriteFile('TD_StartTime.pk', trainData)

def ReadCatTrainData():
    Directory = 'train/'
    Files = ['train_categorical.csv']
    trainData = None
    for j,f in enumerate(Files):
        print("file: " + f)
        subset = None
        for i, chunk in enumerate(pd.read_csv(Directory + f, \
                                                chunksize = 50000, \
                                                low_memory = False)):
            print("chunk " + str(i))
            if subset is None:
                subset = chunk.copy()
            else:
                subset = pd.concat([subset, chunk])
            del chunk
            gc.collect()
        if trainData is None:
            trainData = subset.copy()
        else:
            trainData = pd.merge(trainData, subset.copy(), on="Id")
        del subset
        gc.collect()
    WriteFile('Numeric_Data.pk', trainData)
    return trainData