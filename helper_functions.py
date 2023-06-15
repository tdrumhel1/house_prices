import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import missingno as msno
from sklearn.model_selection import cross_val_score, KFold
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix,mean_squared_error
from helper_functions import *
from sklearn.neighbors import KNeighborsRegressor

def mean_saleprice_map(df,column):
    
    mapping_dict = dict(zip([i for i in df.groupby(column).agg({'SalePrice':'mean'}).sort_values(by='SalePrice').index],

    list(range(len([i for i in df.groupby(column).agg({'SalePrice':'mean'}).sort_values(by='SalePrice').index])))
         
        ))
    
    return mapping_dict
    
def comb_encoded_columns(df,column1,column2):
    
    encoded_df = pd.get_dummies(df[column1]).add(pd.get_dummies(df[column2]),fill_value=0).replace({2:1})
    
    return df.join(pd.get_dummies(df[column1]).add(pd.get_dummies(df[column2]),fill_value=0)\
           .replace({2:1})).drop([column1,column2],axis=1)

# Some basic fillling
def basic_filling(df):

    df.Alley.fillna('None',inplace=True)
    
    df.FireplaceQu.fillna('None',inplace=True)

    df.PoolQC.fillna('None',inplace=True)

    df.Fence.fillna('None',inplace=True)

    # Should maybe drop MiscFeature due to mostly null, but should get dropped in feature selection
    df.MiscFeature.fillna('None',inplace=True)

    df.GarageType.fillna('None',inplace=True)
    
    df.GarageYrBlt.fillna(0,inplace=True)

    df.GarageFinish.fillna('None',inplace=True)

    df.GarageQual.fillna('None',inplace=True)

    df.GarageCond.fillna('None',inplace=True)

    df.BsmtQual.fillna('None',inplace=True)

    df.BsmtCond.fillna('None',inplace=True)

    df.BsmtExposure.fillna('None',inplace=True)

    df.BsmtFinType1.fillna('None',inplace=True)

    df.BsmtFinType2.fillna('None',inplace=True)

    df.MasVnrType.fillna('None',inplace=True)

    df.MasVnrArea.fillna(0,inplace=True)
    
    # Filling with mode
    df['MSSubClass'] = df['MSSubClass'].fillna(df.MSSubClass.mode().values[0])
    df['MSZoning'] = df['MSZoning'].fillna(df.MSZoning.mode().values[0])
    df['RoofMatl'] = df['RoofMatl'].fillna(df.RoofMatl.mode().values[0])
    df['Foundation'] = df['Foundation'].fillna(df.Foundation.mode().values[0])
    df['Heating'] = df['Heating'].fillna(df.Heating.mode().values[0])
    
    # Adding a few additional fields
    df['total_bath'] = df['BsmtFullBath']+df['BsmtHalfBath']*.5+df['FullBath']+df['HalfBath']*.5
    
    return df

def mapped_values(df,train_df):

    # Mapped values based on average sale price
    mapping_dict = mean_saleprice_map(train_df,'MSSubClass')
    df.MSSubClass = df.MSSubClass.map(mapping_dict)

    # Mapped values based on average sale price
    df.MSZoning = df.MSZoning.map(mean_saleprice_map(train_df,'MSZoning'))

    df.Street = df.Street.map({'Pave':1,'Grvl':0})

    df.Alley = df.Alley.map({'None':0,'Grvl':1,'Pave':2})

    df.LotShape = df.LotShape.map({'IR3':0,'IR2':1,'IR1':2,'Reg':3})

    df.LandContour = df.LandContour.map({'Low':0,'HLS':1,'Bnk':2,'Lvl':3})

    df.Utilities = df.Utilities.map({'ELO':0,'NoSeWa':1,'NoSewr':2,'AllPub':3})

    # Mapped values based on average sale price - Shouldn't have nulls
    df.LotConfig = df.LotConfig.map(mean_saleprice_map(train_df,'LotConfig'))

    df.LandSlope = df.LandSlope.map({'Sev':0,'Mod':1,'Gtl':2})

    # Condition1 and Condition2 need to be combined as One Hot Encoded
    df = comb_encoded_columns(df,'Condition1','Condition2')

    df.BldgType = df.BldgType.map({'Twnhs':0,'TwnhsI':0,'TwnhsE':1,'Duplex':2,'2fmCon':3,'1Fam':4})

    df.HouseStyle = df.HouseStyle.map({'1Story':0,'1.5Unf':1,'1.5Fin':2,'SFoyer':3,'SLvl':3,'2Story':4,\
                                           '2.5Unf':5,'2.5Fin':6})

    # RoofStyle One Hot Encoded
    df = pd.get_dummies(data=df,columns=['RoofStyle'])

    df.RoofMatl = df.RoofMatl.map(mean_saleprice_map(train_df,'RoofMatl'))

    # Exterior1st and Exterior2nd need to be combined as One Hot Encoded
    df = comb_encoded_columns(df,'Exterior1st','Exterior2nd')

    # MasVnrType One Hot Encoded
    df.MasVnrType = df.MasVnrType.map({'None':0,'BrkCmn':1,'BrkFace':2,'Stone':3})

    df.ExterQual = df.ExterQual.map({'Po':0,'Fa':1,'TA':2,'Gd':3,'Ex':4})

    df.ExterCond = df.ExterCond.map({'Po':0,'Fa':1,'TA':2,'Gd':3,'Ex':4})

    # Mapped values based on average sale price
    df.Foundation = df.Foundation.map(mean_saleprice_map(train_df,'Foundation'))

    df.BsmtQual = df.BsmtQual.map({'None':0,'Po':1,'Fa':2,'TA':3,'Gd':4,'Ex':5})

    df.BsmtCond = df.BsmtCond.map({'None':0,'Po':1,'Fa':2,'TA':3,'Gd':4,'Ex':5})

    df.BsmtExposure = df.BsmtExposure.map({'None':0,'No':1,'Mn':2,'Av':3,'Gd':4})

    df.BsmtFinType1 = df.BsmtFinType1.map({'None':0,'Unf':1,'LwQ':2,'Rec':3,'BLQ':4,'ALQ':5,'GLQ':6})

    df.BsmtFinType2 = df.BsmtFinType2.map({'None':0,'Unf':1,'LwQ':2,'Rec':3,'BLQ':4,'ALQ':5,'GLQ':6})

    # Mapped values based on average sale price
    df.Heating = df.Heating.map(mean_saleprice_map(train_df,'Heating'))

    df.HeatingQC = df.HeatingQC.map({'Po':0,'Fa':1,'TA':2,'Gd':3,'Ex':4})

    df.CentralAir = df.CentralAir.map({'N':0,'Y':1})

    df.Electrical = df.Electrical.map({'FuseP':0,'FuseF':1,'Mixed':2,'FuseA':3,'SBrkr':4})
    df.Electrical.fillna(2,inplace=True) # Filling one missing value with average

    df.KitchenQual = df.KitchenQual.map({'Po':0,'Fa':1,'TA':2,'Gd':3,'Ex':4})

    df.Functional = df.Functional.map({'Sal':0,'Sev':1,'Maj2':2,'Maj1':3,'Mod':4,'Min2':5,'Min1':6,'Typ':7})

    df.FireplaceQu = df.FireplaceQu.map({'None':0,'Po':1,'Fa':2,'TA':3,'Gd':4,'Ex':5})

    df.GarageType = df.GarageType.map({'None':0,'Detchd':1,'CarPort':2,'BuiltIn':3,'Basment':4,'Attchd':5,
                                            '2Types':6})

    df.GarageFinish = df.GarageFinish.map({'None':0,'Unf':1,'RFn':2,'Fin':3})

    df.GarageQual = df.GarageQual.map({'None':0,'Po':1,'Fa':2,'TA':3,'Gd':4,'Ex':5})

    df.GarageCond = df.GarageCond.map({'None':0,'Po':1,'Fa':2,'TA':3,'Gd':4,'Ex':5})

    df.PavedDrive = df.PavedDrive.map({'N':0,'P':1,'Y':2})

    df.PoolQC = df.PoolQC.map({'None':0,'Fa':1,'TA':2,'Gd':3,'Ex':4})

    df.Fence = df.Fence.map({'None':0,'MnWw':1,'GdWo':2,'MnPrv':3,'GdPrv':4})

    # MiscFeature pretty low, so exclude or one hot encode
    df = pd.get_dummies(data=df,columns=['MiscFeature'])

    # MiscVal probably drop

    # SaleType one hot encode
    df = pd.get_dummies(data=df,columns=['SaleType'])

    # SaleCondition one hot encode
    df = pd.get_dummies(data=df,columns=['SaleCondition'])
    
    return df, mapping_dict

## Since LotFrontage can be determined by multiple other factors, using KNN Imputer to finish filling values
def impute_lot_frontage(df,df_type):
    df_no_nans = df.dropna().drop('Neighborhood',axis=1)

    model = KNeighborsRegressor(n_neighbors=5)
    
    if df_type == 'train':
        drop_cols = ['LotFrontage','SalePrice']
    else:
        drop_cols = ['LotFrontage']

    knr = model.fit(df_no_nans.drop(drop_cols,axis=1), df_no_nans.LotFrontage)
    
    drop_cols.append('Neighborhood')

    knn_preds = knr.predict(df[df.LotFrontage.isna()].drop(drop_cols,axis=1))

    df.iloc[df[df.LotFrontage.isna()].index-1,2] = knn_preds
    
    return df

