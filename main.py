import pandas as pd
import numpy as np
import xgboost as xgb
import matplotlib.pyplot as plt
from MCC import mcc_eval
from Preprocess import *
from ReadTestFiles import *
from Common import *
   
   
def trainModel(trainData):
    ytrain = trainData.pop('Response')

    prior = np.sum(ytrain) / (1.*len(ytrain))
    xgb_params = {
    'seed': 0,
    'colsample_bytree': 0.7,
    'silent': 1,
    'subsample': 0.7,
    'learning_rate': 0.1,
    'objective': 'binary:logistic',
    'max_depth': 4,
    'num_parallel_tree': 1,
    'min_child_weight': 2,
    'eval_metric': 'auc',
    'base_score': prior }   
    
    xgdmat = xgb.DMatrix(trainData, label=ytrain)
    cv_xgb = xgb.cv(params = xgb_params, dtrain = xgdmat, \
                    num_boost_round=10, \
                    nfold = 4, \
                    seed = 0, \
                    stratified = True, \
                    feval=mcc_eval, \
                    maximize=True,  \
                early_stopping_rounds = 1, \
                verbose_eval=1, show_stdv=True )
    return cv_xgb
#    
#    train_xgb = xgb.train(params = our_params, dtrain = xgdmat, \
#                    feval=mcc_eval, \
#                    maximize=True)
                
#    testFiles = ['test_date.csv']
#    testCols = dateCols
#    ReadTestData(testFiles, testCols)
                
#    testData = OpenFile('testData_datesOnly.pk')
#
#    testdmat = xgb.DMatrix(testData)
#    y_pred = train_xgb.predict(testdmat)
#    thresholdVal = .3
#    low_val = y_pred < thresholdVal
#    high_val = ~low_val
#    y_pred[low_val] = 0
#    y_pred[high_val] = 1
#    y_pred = y_pred.astype(int)
#    submission = pd.DataFrame({"Id": testData.Id.values, \
#                               "Response": y_pred})
#    submission[['Id', 'Response']].to_csv('xgbsubmission.csv', \
#                                            index=False) 

    
if __name__ == "__main__":
    print('Started')
    FindUniqueCols('train_categorical.csv', 'UniqueCatCols.pk')
    cols = OpenFile('UniqueCatCols.pk')
#    file = 'TD_StartTime.pk'
#    if (file == 'TD_StartTime.pk'):
#        trainData = OpenFile(file)
#        trainData = trainData.sort_values(by=['Start_Time', 'Id'], ascending=True)
#        trainData['3'] = trainData['Id'].diff().fillna(9999999).astype(int)
#        trainData['4'] = trainData['Id'].iloc[::-1].diff().fillna(9999999).astype(int)
#        trainData.pop('Id')
#        
#    else if (file == 'TD_FeatureCnt.pk'):
#        trainData = OpenFile('TD_FeatureCnt.pk')
#        
#    print(trainData.shape)
#    xgb = trainModel(trainData)
    
 
    print('Finished')