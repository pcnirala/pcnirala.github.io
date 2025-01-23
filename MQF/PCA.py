###SOURCE DATA FOR THIS CODE IS THE SAME AS FROM QF603 - YOU CAN REUSE THE SAME SOURCE DATA CSVs IF YOU ALRREADY HAVE THEM INSTALLED

import pandas as pd;
import numpy as np;
import statsmodels.api as sm;
import statsmodels.stats.api as sms;
import statsmodels.discrete.discrete_model as smdiscrete


pd.set_option('use_inf_as_na', True)

meta_df = pd.read_csv("stockmetadata.csv")
fdata_df = pd.read_csv("corpfund.csv")
fdata_df = fdata_df[fdata_df['dimension']=='ARQ']
fdata_df['datekey'] = pd.to_datetime(fdata_df['datekey'])
df_left = pd.merge(fdata_df, meta_df, on='ticker', how='left')
df_left = df_left.set_index('datekey')

###PCA#####
from sklearn.decomposition import PCA
data = data_w_dummies  #renaming the variable for easier typing
numerator = ['cashneq', 'debt', 'ebit', 'ebt', 'eps', 'equity', 'fcf', 'gp', 'inventory', 'liabilities', 'payables', 'receivables', 'tangibles', 'workingcapital']
denominator = ['assets', 'revenue']
featureslist = []
for n in numerator:
    for d in denominator:
        tag = n+'_'+d
        data[tag] = np.log(data[n]/data[d])
        featureslist.append(tag)

data.dropna(subset=featureslist, inplace=True)
features = data.loc[:, featureslist].values
from sklearn.preprocessing import StandardScaler
features = StandardScaler().fit_transform(features)
from sklearn.decomposition import PCA
pca = PCA(n_components = 4)
principal_components = pca.fit_transform(features)
principal_components
pca.explained_variance_ratio_
pc_df = pd.DataFrame(principal_components)
pc_df.corr()
pc_df.columns = ['PC1', 'PC2', 'PC3', 'PC4']
pc_df.index = data.index
data_merge = pd.concat([data, pc_df], axis=1)
result = sm.OLS(data_merge['lnepratio'], sm.add_constant(data_merge[featureslist]), missing='drop').fit(cov_type='cluster', cov_kwds={'groups': data_w_dummies['siccode']})
print(result.summary())
result = sm.OLS(data_merge['lnepratio'], sm.add_constant(data_merge[['PC1', 'PC2', 'PC3', 'PC4']]), missing='drop').fit(cov_type='cluster', cov_kwds={'groups': data_w_dummies['siccode']})
print(result.summary())

#lets figure out what 'PC1' means
featureslist.append('PC1')
data_merge[featureslist].corr()

