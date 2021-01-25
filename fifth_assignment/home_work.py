import operator
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from catboost import CatBoostClassifier
from xgboost import XGBClassifier


def train_models():
    # Read dataset
    happiness_dataset_path = 'dataset/covtype.data'
    df = pd.read_csv(happiness_dataset_path)
    print(df)

    # Print sum of Nan values in each column
    print('Sum of NaN values in each column:')
    print(df.isnull().sum())

    # don't need to do any inprinting, the is no NaN values

    # drop target class
    df_norm = df.drop(['5'], axis=1)

    # Normalize dataset
    # create a scaler object
    scaler = MinMaxScaler()
    # fit and transform the data
    df_norm = pd.DataFrame(scaler.fit_transform(df_norm), columns=df_norm.columns)
    print(df_norm)

    # make inputs and outputs
    x = df_norm
    y = df['5']

    # data split train 75% - 25%
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, shuffle=True, random_state=555)
    print('Entries in train: ' + str(x_train.shape[0]))
    print('Entries in test: ' + str(x_test.shape[0]))

    # collect each classifier results to dictionary for the best classifier selection
    classifiers_results = {}
    print('\n')
    print('Lets begin training and testing!!!')
    print(2 * '\n')


    print('Random forest classifier')
    # random forest
    print('Training...')
    rf = RandomForestClassifier()
    rf.fit(x_train, y_train)
    print('Training done!')
    print('Testing!')
    y_pred = rf.predict(x_test)
    # compute accuracy
    accuracy = accuracy_score(y_test, y_pred)
    print('Random forest classifier accuracy: ' + str(accuracy))
    print('\n')  # space
    name = 'Random forest classifier'
    classifiers_results.update({name: accuracy})

    print('XGBClassifier classifier')
    # xgbClassifier
    print('Training...')
    xgboost = XGBClassifier()
    xgboost.fit(x_train, y_train)
    print('Training done!')
    print('Testing!')
    y_pred = xgboost.predict(x_test)
    # compute accuracy
    accuracy = accuracy_score(y_test, y_pred)
    print('XGBClassifier accuracy: ' + str(accuracy))
    print('\n')  # space
    name = 'XGBClassifier'
    classifiers_results.update({name: accuracy})

    print('Check the results')
    sorted_classifiers_results = dict(sorted(classifiers_results.items(), key=operator.itemgetter(1), reverse=True))
    for i, (name, accuracy) in enumerate(sorted_classifiers_results.items()):
        print(str(i + 1) + ' place: ' + str(accuracy) + ' '+ name.ljust(30, ' '))


def main():
    print('Fifth assignment!')
    # HOMEWORK
    train_models()


if __name__ == '__main__':
    main()
