import sklearn.model_selection
import sklearn.feature_extraction
from sklearn import preprocessing
import pandas as pd
import random
import numpy as np

random.seed(0)

train_data = pd.read_csv('train.csv')
survived = train_data['Survived']
train_data = train_data.drop('Survived', axis=1)

def names(train):
    train['Name_Len'] = train['Name'].apply(lambda x: len(x))
    train['Name_Title'] = train['Name'].apply(lambda x: x.split(',')[1]).apply(lambda x: x.split()[0])
    return train

def ticket_grouped(train):
    train['Ticket_Lett'] = train['Ticket'].apply(lambda x: str(x)[0])
    train['Ticket_Lett'] = train['Ticket_Lett'].apply(lambda x: str(x))
    train['Ticket_Lett'] = np.where((train['Ticket_Lett']).isin(['1', '2', '3', 'S', 'P', 'C', 'A']), train['Ticket_Lett'],
                                   np.where((train['Ticket_Lett']).isin(['W', '4', '7', '6', 'L', '5', '8']),
                                            'Low_ticket', 'Other_ticket'))
    train['Ticket_Len'] = train['Ticket'].apply(lambda x: len(x))
    del train['Ticket']
    return train

def cabin(train):
    train['Cabin_Letter'] = train['Cabin'].apply(lambda x: str(x)[0])
    del train['Cabin']
    return train

def cabin_num(train):
    train['Cabin_num1'] = train['Cabin'].apply(lambda x: str(x).split(' ')[-1][1:])
    train['Cabin_num1'].replace('an', np.NaN, inplace = True)
    train['Cabin_num1'] = train['Cabin_num1'].apply(lambda x: int(x) if not pd.isnull(x) and x != '' else np.NaN)
    train['Cabin_num'] = pd.qcut(train['Cabin_num1'],3)
    train = pd.concat((train, pd.get_dummies(train['Cabin_num'], prefix = 'Cabin_num')), axis = 1)
    del train['Cabin_num']
    del train['Cabin_num1']
    return train

def dummies(train, columns = ['Ticket_Lett', 'Cabin_Letter', 'Name_Title']):
    for column in columns:
        train[column] = train[column].apply(lambda x: str(x))
        good_cols = [column+'_'+i for i in train[column].unique()]
        train = pd.concat((train, pd.get_dummies(train[column], prefix = column)[good_cols]), axis = 1)
        del train[column]
    return train

def age_impute(train):
    train['Age_Null_Flag'] = train['Age'].apply(lambda x: 1 if pd.isnull(x) else 0)
    data = train.groupby(['Name_Title', 'Pclass'])['Age']
    train['Age'] = data.transform(lambda x: x.fillna(x.mean()))
    return train

print(train_data['Embarked'].mode())

train_data = names(train_data)
train_data = ticket_grouped(train_data)
train_data = cabin_num(train_data)
train_data = cabin(train_data)
train_data = age_impute(train_data)
train_data = dummies(train_data)

# label encoding
label_encoder = preprocessing.LabelEncoder()

# Encode labels in column 'Sex'. 
train_data['Sex']= label_encoder.fit_transform(train_data['Sex']) 

train_data['Sex'].unique() 
# There is no need for one hot encoding

train_nan_map = {'Age': train_data['Age'].mean(),'Embarked': train_data['Embarked'].mode()[0]}
train_data.fillna(value=train_nan_map, inplace=True)
# [1,2,3,null,4] null --> 1+2+3+4/4 = 2.5

label_encoder = preprocessing.LabelEncoder() 
train_data['Embarked']= label_encoder.fit_transform(train_data['Embarked']) 
train_data['Embarked'].unique()

print(train_data.isnull().sum().sort_values(ascending=False))
vectorizer = sklearn.feature_extraction.DictVectorizer(sparse=False)
#print([elem for elem in vectorizer.fit_transform(train_data.to_dict(orient='records'))[0, :]])
train_data =  np.hstack( (vectorizer.fit_transform(train_data.to_dict(orient='records')), survived.values.reshape((-1,1 ) ) ) )
#train_data['Survived'] = survived
 
# #train_data = train_data[[c for c in train_data if c not in ['Survived']] + ['Survived']]
kf = sklearn.model_selection.KFold(n_splits=5)

for i, (train_index, test_index) in enumerate(kf.split(train_data)):
    #train_data.iloc[train_index, :].to_csv('train_' + str(i) + '.csv.gz', compression='gzip', index=False)
    #train_data.iloc[test_index, :].to_csv('test_' + str(i) + '.csv.gz', compression='gzip', index=False)
    
    np.savetxt('train_' + str(i) + '.csv.gz', train_data[train_index], delimiter=',')
    np.savetxt('test_' + str(i) + '.csv.gz', train_data[test_index], delimiter=',')