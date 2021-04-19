import numpy as np
import pandas as pd
from sklearn import linear_model
from sklearn.impute import SimpleImputer

from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
import warnings; warnings.filterwarnings('ignore')


def perm_imputer(dataframe):

    """[Calculate missing permeability values using linear regression
    for Sandstone and Shaly Sandstone, and mean values for Shale and
    Sandy Shale facies]

    Args:
        dataframe ([DataFrame]): [DataFrame with porosity ['por'] and facies ['facies']]

    Returns:
        dataframe ([DataFrame]): [DataFrame with calculate permeability ['perm'] and log permeability ['log_perm']]
    """


    df =  dataframe.copy(deep=False)
    df['log_perm'] = np.log10(df['perm'])

    # Get data
    x_SS = df[df.facies =='SS'].dropna()['por'].values[:,np.newaxis]
    y_SS = df[df.facies =='SS'].dropna()['log_perm'].values[:,np.newaxis]

    x_shS = df[df.facies =='Sh-SS'].dropna()['por'].values[:,np.newaxis]
    y_shS = df[df.facies =='Sh-SS'].dropna()['log_perm'].values[:,np.newaxis]

    # Instantiate models
    lin_reg_SS = linear_model.LinearRegression()
    lin_reg_shS = linear_model.LinearRegression()

    # Train models
    lin_reg_SS.fit(x_SS, y_SS)
    lin_reg_shS.fit(x_shS, y_shS);

    # Calculate missing permeabilities
    # Sandstone
    filter = (df['facies']=='SS') & (df.por.notnull()) & (df.perm.isnull())
    df.loc[filter, 'log_perm'] = lin_reg_SS.predict(df.loc[filter, 'por'].values[:,np.newaxis])
    df.loc[filter, 'perm'] = 10 ** df.loc[filter, 'log_perm']

    # Shaly Sandstone
    filter = (df['facies']=='Sh-SS') & (df.por.notnull()) & (df.perm.isnull())
    df.loc[filter, 'log_perm'] = lin_reg_shS.predict(df.loc[filter, 'por'].values[:,np.newaxis])
    df.loc[filter, 'perm'] = 10 ** df.loc[filter, 'log_perm']

    #? For Shale and Sandy Shale we used a mean imputer
    # Instatiate imputer
    mean_imputer = SimpleImputer(strategy='mean')

    # Shale
    filter = df.facies == 'Sh','perm'
    df.loc[filter] = mean_imputer.fit_transform(df.loc[filter].values[:, np.newaxis] )

    # Sandy shale
    filter = df.facies == 'SS-Sh','perm'
    df.loc[filter] = mean_imputer.fit_transform(df.loc[filter].values[:, np.newaxis] )

    # Re-calculate log perm
    df['log_perm'] = np.log10(df['perm'])

    print('---------------------------------')
    print('Permeability initial missing values = ' + str(dataframe['perm'].isna().sum()))
    print('Permeability final missing values = ' + str(df['perm'].isna().sum()))
    print('---------------------------------')

    return df


def por_facies_imputer(dataframe):
    """
    Imputes missing porosity and facie labels using KNN 

    Args:
        df ([DataFrame]): The dataframe should includes the 
        following columns: ['X', 'Y', 'depth', 'por', 'rho','facies']
    Returns:
        df ([DataFrame])
    """
    df_original = dataframe.copy(deep=False)

    df = df_original.loc[:,['X', 'Y', 'depth', 'por', 'rho','facies']]
    categorical = ['facies']
    numerical = ['X', 'Y', 'depth', 'por', 'rho']
   
    df['Imputed'] = (df.isnull().sum(axis=1)) > 0

    df[categorical] = df[categorical].apply(lambda series: pd.Series(
                                            LabelEncoder().fit_transform(series[series.notnull()]),
                                            index=series[series.notnull()].index))

    # Instatiate imputers
    imp_num = IterativeImputer(estimator=RandomForestRegressor(),
                            initial_strategy='mean',
                            max_iter=20, random_state=0)

    imp_cat = IterativeImputer(estimator=RandomForestClassifier(), 
                            initial_strategy='most_frequent',
                            max_iter=20, random_state=0)

    # Fit
    df[numerical] = imp_num.fit_transform(df[numerical])
    df[categorical] = imp_cat.fit_transform(df[categorical])     

    #Perform corrections to facies information with density and porosity values
    df['facies'] = np.where((df.por < 0.1) & (df.rho > 2.40), 1,df.facies)
    df['facies'] = np.where((df.por < 0.08) & (df.rho < 2.25) , 2,df.facies)
    df['facies'] = np.where((df.por < 0.13) & (df.por > 0.08) & (df.rho < 2.40),3,df.facies)

    #Update por, rho and facies with the predicted values for missing data
    df_original["por"] = df["por"]
    df_original["rho"] = df["rho"]
    df_original["facies"] = df["facies"]


    facies_map = {
        0 : 'SS',
        1 : 'SS-Sh',
        2 : 'Sh',
        3 : 'Sh-SS'
    }

    df_original["facies"] = df_original["facies"].map(facies_map)

    print('---------------------------------')
    print('Porosity initial missing values = ' + str(dataframe['por'].isna().sum()))
    print('Porosity final missing values = ' + str(df_original['por'].isna().sum()))
    print('Facies initial missing values = ' + str(dataframe['facies'].isna().sum()))
    print('Facies final missing values = ' + str(df_original['facies'].isna().sum()))
    print('---------------------------------')

    return df_original


def average_per_well(dataframe):
    df = dataframe.copy(deep=False)
    facies = df['facies'].unique()                          # Get the facies names
    df_ave = df.groupby(by='ID', as_index=False).mean()
    df_all_grouped = df.groupby(by=['ID', 'facies'])        # Group dataframe by well ID and facie
    
    for ID in range(1,len(df_ave)+1): # Loop wells
        for facie in facies: # Loop facies
            try:
                df_ave.loc[df_ave.ID==ID, facie +'_perm'] = df_all_grouped.get_group((ID, facie))['perm'].mean()
                df_ave.loc[df_ave.ID==ID, facie +'_por'] = df_all_grouped.get_group((ID, facie))['por'].mean()
                df_ave.loc[df_ave.ID==ID, facie +'_fraction'] = df_all_grouped.get_group((ID, facie)).shape[0] / 20
            except KeyError:
                pass
    df_ave.fillna(value=df_ave.mean(), inplace=True)
    return df_ave


def average_per_well(dataframe):
    df = dataframe.copy(deep=False)
    facies = df['facies'].unique()                          # Get the facies names
    df_ave = df.groupby(by='ID', as_index=False).mean()
    df_all_grouped = df.groupby(by=['ID', 'facies'])        # Group dataframe by well ID and facie
    
    for ID in range(1,len(df_ave)+1): # Loop wells
        for facie in facies: # Loop facies
            try:
                df_ave.loc[df_ave.ID==ID, facie +'_perm'] = df_all_grouped.get_group((ID, facie))['perm'].mean()
                df_ave.loc[df_ave.ID==ID, facie +'_por'] = df_all_grouped.get_group((ID, facie))['por'].mean()
                df_ave.loc[df_ave.ID==ID, facie +'_fraction'] = df_all_grouped.get_group((ID, facie)).shape[0] / 20
            except KeyError:
                pass
    df_ave.fillna(value=df_ave.mean(), inplace=True)
    return df_ave


def average_per_well2(dataframe):
    df = dataframe.copy(deep=False)
    facies = df['facies'].unique()                          # Get the facies names
    df_ave = df.groupby(by='ID', as_index=False).mean()
    df_grouped = df.groupby(by=['ID', 'facies'])        # Group dataframe by well ID and facie
    
    for ID in range(1,len(df_ave)+1): # Loop wells
        # Per facies
        for facie in facies:
            try:
                df_ave.loc[df_ave.ID==ID, facie +'_perm'] = df_grouped.get_group((ID, facie))['perm'].mean()
                df_ave.loc[df_ave.ID==ID, facie +'_por'] = df_grouped.get_group((ID, facie))['por'].mean()    
                df_ave.loc[df_ave.ID==ID, facie +'_fraction'] = df_grouped.get_group((ID, facie)).shape[0] / 20
            except KeyError:
                pass
        # Per well
        df_ave.loc[df_ave.ID==ID,'perm_w_ave'] = df[df.ID == ID]['perm'].sum() * 0.05 / 9.5
    
    df_ave['pay_to_thickness'] = df_ave['SS_fraction'] + df_ave['Sh-SS_fraction']

    df_ave.fillna(value=df_ave.mean(), inplace=True)
    return df_ave



def __calc_distance_from_fracture__(X, Y):
    A = np.array([1e4-1750, 1e4-1750])         # Fault vector origin at Y= 10,000 and X = 1,750
    B = np.array([X - 1750, 1e4 - Y])          # Position vector for well
    
    norm = np.linalg.norm
    θ = np.arccos(A @ B / (norm(A) * norm(B))) # Angle between fault and position vector

    return norm(B) * np.sin(θ)


def add_dist_from_frac(dataframe):
    """[Add a column to dataframe with distance from fracture for each well]

    Args:
        dataframe[DataFrame]: [Dataframe with X and Y coordinates]
    Returns:
        dataframe[DataFrame]: [Dataframe with 'dist_frac' column]
    """
    df = dataframe.copy(deep=False)
    df.loc[:,'dist_frac'] = df.apply(lambda row: __calc_distance_from_fracture__(row.X, row.Y), axis=1 )

    print("Added 'dist_frac' column to DataFrame")
    return df


def interwell_distances(well,other_well):
    if well['ID'] == other_well['ID']:
        return np.inf # We don't want to compare a well to itself

    # x and well are pd series
    delta_x = well['X'] - other_well['X']
    delta_y = well['Y'] - other_well['Y']
    return np.sqrt(delta_x**2 + delta_y**2) # euclidean distance



def closest_well(well):
    # Find the well in df_dist_table that is closed to well
    # make sure we don't compare well to itself
    
    # Find the distance between well and every other well
    # use a deep copy of df_dist_table
    df_dist_table2 = df_dist_table.copy(deep = True)
    label = 'Dist_to_'+ str(well['ID'])
    df_dist_table2.loc[:,label] = df_dist_table2.apply(lambda row: dist(well,row), axis = 1)
    
    # return the well ID which is closest to well
    min_dist = df_dist_table2[label].min()
    #print(min_dist)
    closest_row = df_dist_table2[df_dist_table2[label] == min_dist]
    closest_ID = (int)(closest_row['ID'])
    #print(closest_ID)
    #del df_dist_table2 # to save memory and ensure that df_dist_table2 isn't used in another call 
    # to the function
    
    return closest_ID