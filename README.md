Kaggle_HiggsBoson_RandomForest
==============================

A Randomforest algorithm for the higgs boson challenge of kaggle with Sci-Kit Learn

#!/usr/bin/env python
# Author Jens K
# Version 1.00 for Kaggle Higgs Boson Challenge using Sklearn Python
import numpy as np
import pandas as pd
#import pylab as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.cross_validation import cross_val_score
from sklearn.ensemble import GradientBoostingClassifier

def main():
    print('Reading Training Data')
    train_df = pd.read_csv('data/training.csv', index_col=0, converters={32: lambda x:int(x=='s'.encode('utf-8')) })
    #uncomment to use different kind of classifiers
    clf = RandomForestClassifier(n_estimators=10, max_depth=None,min_samples_split=2, random_state=0)
    #clf = GradientBoostingClassifier(loss='deviance', learning_rate=0.1, n_estimators=100, subsample=1.0, min_samples_split=2, min_samples_leaf=1, max_depth=6, init=None, random_state=None, max_features=None, verbose=1)
    #clf = GradientBoostingClassifier(n_estimators=100, learning_rate=1.0,max_depth=3, random_state=0)
    all_columns = ['DER_mass_MMC','DER_mass_transverse_met_lep','DER_mass_vis','DER_pt_h','DER_deltaeta_jet_jet','DER_mass_jet_jet','DER_prodeta_jet_jet','DER_deltar_tau_lep','DER_pt_tot','DER_sum_pt','DER_pt_ratio_lep_tau','DER_met_phi_centrality','DER_lep_eta_centrality','PRI_tau_pt','PRI_tau_eta','PRI_tau_phi','PRI_lep_pt','PRI_lep_eta','PRI_lep_phi','PRI_met','PRI_met_phi','PRI_met_sumet','PRI_jet_num','PRI_jet_leading_pt','PRI_jet_leading_eta','PRI_jet_leading_phi','PRI_jet_subleading_pt','PRI_jet_subleading_eta','PRI_jet_subleading_phi','PRI_jet_all_pt']
    
    
    #If you like to use a loop and check how the CV Score evolves using 1..2..3 columns
    """   
    for x in all_columns:
        columns = []
        columns.append(x)
        labels = train_df['Label'].values
        features = train_df[list(columns)].values
    
        cv_score = cross_val_score(clf, features, labels, n_jobs=-1).mean()
        print("{0} -> CV: {1})".format(columns, cv_score))
    """
    
    #uncomment and comment out columns = all_columns, if you want to use specific rows
    #columns = ['DER_mass_MMC','DER_deltaeta_jet_jet','DER_mass_jet_jet','DER_prodeta_jet_jet','DER_pt_ratio_lep_tau','DER_pt_ratio_lep_tau']
    #columns = all_columns
    columns = ['DER_mass_MMC','DER_deltaeta_jet_jet'] # Test with 2 columns
    labels = train_df['Label'].values
    features = train_df[list(columns)].values
    print('Calculating CV Score please wait...')
    cv_score = cross_val_score(clf, features, labels, n_jobs=-1,verbose=1).mean()
    print("{0} -> CV: {1})".format(columns, cv_score))
    
    
if __name__ == '__main__':
    main()
