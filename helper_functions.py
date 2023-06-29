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
    
    mapping_dict = dict(zip([i for i in df.groupby(column).agg({'SalePrice':'mean'}).sort_values(by='SalePrice',ascending=False).index],

    list(range(len([i for i in df.groupby(column).agg({'SalePrice':'mean'}).sort_values(by='SalePrice',ascending=False).index])))
         
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
    
    # Filling with mode/mean
    df['MSSubClass'] = df['MSSubClass'].fillna(df.MSSubClass.mode().values[0])
    df['MSZoning'] = df['MSZoning'].fillna(df.MSZoning.mode().values[0])
    df['RoofMatl'] = df['RoofMatl'].fillna(df.RoofMatl.mode().values[0])
    df['Foundation'] = df['Foundation'].fillna(df.Foundation.mode().values[0])
    df['Heating'] = df['Heating'].fillna(df.Heating.mode().values[0])
    df['Electrical'] = df['Electrical'].fillna(df.Electrical.mode().values[0])
    df['LotFrontage'] = df['LotFrontage'].fillna(df.LotFrontage.mean())
    
    # Adding a few additional fields
    df['total_bath'] = df['BsmtFullBath']+df['BsmtHalfBath']*.5+df['FullBath']+df['HalfBath']*.5
    df['total_sq'] = df['TotalBsmtSF']+df['1stFlrSF']+df['2ndFlrSF']
    df['total_finished_sq'] = df['total_sq'] - df['BsmtUnfSF'] - df['LowQualFinSF']
    df['sold_date'] = (df.YrSold.astype('str') + df['MoSold'].astype('str').str.zfill(2)).astype(int)
    
    # Blanket filling the rest (1-3 missing values on minority of features) with 0 or mode
    float_cols = list(df.dtypes[df.dtypes==float].index)
    df[float_cols] = df[float_cols].fillna(df[float_cols].mean())
    
    object_cols = list(df.dtypes[df.dtypes==object].index)
    object_dict = {k:v[0] for k, v in df[object_cols].mode().to_dict().items()}
    df[object_cols] = df[object_cols].fillna(object_dict)
    
    return df

def mapped_values(df,get_dums='N'):

    # Mapped values based on average sale price
    df.MSSubClass = df.MSSubClass.map({60: 0, 120: 1, 150: 2, 75: 3, 20: 4, 80: 5, 70: 6, 40: 7, 85: 8, 50: 9, 160: 10, 90: 11,190: 12, 45: 13, 180: 14, 30: 15})

    # Mapped values based on average sale price
    df.MSZoning = df.MSZoning.map({'FV': 0, 'RL': 1, 'RH': 2, 'RM': 3, 'C (all)': 4})

    df.Street = df.Street.map({'Pave':0,'Grvl':1})

    df.Alley = df.Alley.map({'None':2,'Grvl':1,'Pave':0})

    df.LotShape = df.LotShape.map({'IR2': 0, 'IR3': 1, 'IR1': 2, 'Reg': 3})

    df.LandContour = df.LandContour.map({'HLS': 0, 'Low': 1, 'Lvl': 2, 'Bnk': 3})

    df.Utilities = df.Utilities.map({'ELO':3,'NoSeWa':2,'NoSewr':1,'AllPub':0})

    # Mapped values based on average sale price - Shouldn't have nulls
    df.LotConfig = df.LotConfig.map({'CulDSac': 0, 'FR3': 1, 'Corner': 2, 'FR2': 3, 'Inside': 4})

    df.LandSlope = df.LandSlope.map({'Sev': 0, 'Mod': 1, 'Gtl': 2})
    
    df.Neighborhood = df.Neighborhood.map({'NoRidge': 0, 'NridgHt': 1, 'StoneBr': 2, 'Timber': 3, 'Veenker': 4, 'Somerst': 5, 'ClearCr': 6, 'Crawfor': 7, 'CollgCr': 8, 'Blmngtn': 9, 'Gilbert': 10, 'NWAmes': 11, 'SawyerW': 12, 'Mitchel': 13, 'NAmes': 14, 'NPkVill': 15, 'SWISU': 16, 'Blueste': 17, 'Sawyer': 18, 'OldTown': 19, 'Edwards': 20, 'BrkSide': 21, 'BrDale': 22, 'IDOTRR': 23, 'MeadowV': 24})

    # Condition1 and Condition2 need to be combined as One Hot Encoded
#     df = comb_encoded_columns(df,'Condition1','Condition2')
    # Dropping for now, since not consistent across train and test
    df = df.drop(columns=['Condition1','Condition2'])

    df.BldgType = df.BldgType.map({'1Fam': 0, 'TwnhsE': 1, 'Twnhs': 2, 'Duplex': 3, '2fmCon': 4})

    df.HouseStyle = df.HouseStyle.map({'2.5Fin': 0,'2Story': 1,'1Story': 2,'SLvl': 3,'2.5Unf': 4,
                                         '1.5Fin': 5,'SFoyer': 6,'1.5Unf': 7})

    df.RoofStyle = df.RoofStyle.map({'Shed': 0, 'Hip': 1, 'Flat': 2, 'Mansard': 3, 'Gable': 4, 'Gambrel': 5})

    df.RoofMatl = df.RoofMatl.map({'WdShngl': 0, 'Membran': 1, 'WdShake': 2, 'Tar&Grv': 3, 'Metal': 4, 'CompShg': 5, 'ClyTile': 6, 'Roll': 7})

    # Exterior1st and Exterior2nd need to be combined as One Hot Encoded
#     df = comb_encoded_columns(df,'Exterior1st','Exterior2nd')
    # Dropping for now, since not consistent across train and test
    df = df.drop(columns=['Exterior1st','Exterior2nd'])

    # MasVnrType One Hot Encoded
    df.MasVnrType = df.MasVnrType.map({'Stone': 0, 'BrkFace': 1, 'None': 2, 'BrkCmn': 3})

    df.ExterQual = df.ExterQual.map({'Po':4,'Fa':3,'TA':2,'Gd':1,'Ex':0})

    df.ExterCond = df.ExterCond.map({'Po':4,'Fa':3,'TA':2,'Gd':1,'Ex':0})

    # Mapped values based on average sale price
    df.Foundation = df.Foundation.map({'PConc': 0, 'Wood': 1, 'Stone': 2, 'CBlock': 3, 'BrkTil': 4, 'Slab': 5})

    df.BsmtQual = df.BsmtQual.map({'None':5,'Po':4,'Fa':3,'TA':2,'Gd':1,'Ex':0})

    df.BsmtCond = df.BsmtCond.map({'None':5,'Po':4,'Fa':3,'TA':2,'Gd':1,'Ex':0})

    df.BsmtExposure = df.BsmtExposure.map({'None':4,'No':3,'Mn':2,'Av':1,'Gd':0})

    df.BsmtFinType1 = df.BsmtFinType1.map({'GLQ': 0, 'Unf': 1, 'ALQ': 2, 'LwQ': 3, 'BLQ': 4, 'Rec': 5, 'None': 6})

    # Doesn't match sales price averages, but more intuitive
    df.BsmtFinType2 = df.BsmtFinType2.map({'GLQ': 0, 'Unf': 1, 'ALQ': 2, 'LwQ': 3, 'BLQ': 4, 'Rec': 5, 'None': 6})

    # Mapped values based on average sale price
    df.Heating = df.Heating.map({'GasA': 0, 'GasW': 1, 'OthW': 2, 'Wall': 3, 'Grav': 4, 'Floor': 5})

    df.HeatingQC = df.HeatingQC.map({'Po':4,'Fa':3,'TA':2,'Gd':1,'Ex':0})

    df.CentralAir = df.CentralAir.map({'N':1,'Y':0})

    df.Electrical = df.Electrical.map({'SBrkr': 0, 'FuseA': 1, 'FuseF': 2, 'FuseP': 3, 'Mix': 4})
    df.Electrical.fillna(2,inplace=True) # Filling one missing value with average

    df.KitchenQual = df.KitchenQual.map({'Po':4,'Fa':3,'TA':2,'Gd':1,'Ex':0})

    # Doesn't match sales price averages, but more intuitive
    df.Functional = df.Functional.map({'Sal':7,'Sev':6,'Maj2':5,'Maj1':4,'Mod':3,'Min2':2,'Min1':1,'Typ':0})

    df.FireplaceQu = df.FireplaceQu.map({'None':4,'Po':5,'Fa':3,'TA':2,'Gd':1,'Ex':0})

    df.GarageType = df.GarageType.map({'BuiltIn': 0,'Attchd': 1,'Basment': 2,'2Types': 3,'Detchd': 4,
                                         'CarPort': 5,'None': 6})

    df.GarageFinish = df.GarageFinish.map({'None':3,'Unf':2,'RFn':1,'Fin':0})

    df.GarageQual = df.GarageQual.map({'None':4,'Po':5,'Fa':3,'TA':2,'Gd':1,'Ex':0})

    # Doesn't match sales price averages, but more intuitive
    df.GarageCond = df.GarageCond.map({'None':5,'Po':4,'Fa':3,'TA':2,'Gd':1,'Ex':0})

    df.PavedDrive = df.PavedDrive.map({'N':2,'P':1,'Y':0})

    df.PoolQC = df.PoolQC.map({'None':4,'Fa':3,'TA':2,'Gd':1,'Ex':0})

    df.Fence = df.Fence.map({'None': 0, 'GdPrv': 1, 'MnPrv': 2, 'GdWo': 3, 'MnWw': 4})
    
    df.SaleCondition = df.SaleCondition.map({'Partial': 0, 'Normal': 1, 'Alloca': 2, 'Family': 3, 'Abnorml': 4, 'AdjLand': 5})

    df.SaleType = df.SaleType.map({'New': 0, 'Con': 1, 'CWD': 2, 'ConLI': 3, 'WD': 4, 'COD': 5, 'ConLw': 6, 'ConLD': 7, 'Oth': 8})

    # Removing this feature with minimal data
    df = df.drop(columns=['MiscFeature'])
    
    return df

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

