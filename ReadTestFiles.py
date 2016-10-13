import pandas as pd
import gc
from Common import *

def ReadTestData(files, cols):
    directory = 'test/'
    testData = None
    for j,f in enumerate(files):
        print("file: " + f)
        subset = None
        for i, chunk in enumerate(pd.read_csv(directory + f, \
                                                chunksize = 50000, \
                                                usecols=cols, \
                                                low_memory = False)):
            print("chunk " + str(i))
            if subset is None:
                subset = chunk.copy()
            else:
                subset = pd.concat([subset, chunk])
            del chunk
            gc.collect()
        if testData is None:
            testData = subset.copy()
        else:
            testData = pd.merge(testData, subset.copy(), on="Id")
        del subset
        gc.collect()
    WriteFile('testData_datesOnly.pk', testData)
