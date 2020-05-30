
(datetime.strptime('2016-10-05 12:22:30', '%Y-%m-%d %H:%M:%S') - datetime.strptime('2016-10-04 12:18:42', '%Y-%m-%d %H:%M:%S')).total_seconds()//60




pidparam_cols = [col for col in train_feat.columns if 'pidparam' in col and '_150' in col]

train_pid_param_pca = pca1.fit_transform(train_feat[pidparam_cols].fillna(0))
test_pid_param_pca = pca1.transform(test_feat[pidparam_cols].fillna(0))

train_feat = pd.concat([train_feat.drop(pidparam_cols, axis=1), pd.DataFrame(train_pid_param_pca,columns=['pid_param_pca1_%s'%i for i in range(pca_num)])], axis=1)
test_feat = pd.concat([test_feat.drop(pidparam_cols, axis=1), pd.DataFrame(test_pid_param_pca,columns=['pid_param_pca1_%s'%i for i in range(pca_num)])], axis=1)










