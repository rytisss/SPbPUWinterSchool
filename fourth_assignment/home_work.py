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


def train_models():
    """
    # You are hired by NASA as a Data Scientist, Congratulations!
    # your first project;  you have to classify stars based on their astornomical measurements,  0 non-pulsar, 1-pulsar
    # luckily astronouts gave  you a dataset after making observations through our Galaxy
    # your job is to use ML techniques as SVM, Decision_tree, Random_forest and Kneighbours
    # which ML model gives the best?
    # Because you are working in NASA, they have very clean dataset(there is no preprocessing!)
    # to learn more about pulsars
    # you will find your dataset inside the rar file named as "pulsar_stars.csv"
    """

    # Read dataset
    happiness_dataset_path = 'dataset/pulsar_stars.csv'
    df = pd.read_csv(happiness_dataset_path)
    print(df)

    # Print sum of Nan values in each column
    print('Sum of NaN values in each column:')
    print(df.isnull().sum())

    # don't need to do any inprinting, the is no NaN values

    # drop target class
    df_norm = df.drop(['target_class'], axis=1)

    # Normalize dataset
    # create a scaler object
    scaler = MinMaxScaler()
    # fit and transform the data
    df_norm = pd.DataFrame(scaler.fit_transform(df_norm), columns=df_norm.columns)
    print(df_norm)

    # make inputs and outputs
    x = df_norm
    y = df['target_class']

    # data split train 75% - 25%
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, shuffle=True, random_state=555)
    print('Entries in train: ' + str(x_train.shape[0]))
    print('Entries in test: ' + str(x_test.shape[0]))

    # collect each classifier results to dictionary for the best classifier selection
    classifiers_results = {}
    print('\n')
    print('Lets begin training and testing!!!')
    print(2 * '\n')

    # try with 3 different kernels
    kernels = ['linear', 'poly', 'rbf']
    for kernel in kernels:
        print('Support Vector Classifier with ' + kernel + ' kernel')
        print('Training...')
        svc_model = SVC(kernel=kernel)
        svc_model.fit(x_train, y_train)
        print('Training done!')
        print('Testing!')
        # predict the test dataset
        y_pred = svc_model.predict(x_test)
        # compute accuracy
        accuracy = accuracy_score(y_test, y_pred)
        print('Model with kernel \'' + kernel + '\' accuracy: ' + str(accuracy))
        print('\n')  # space
        name = 'SVC_with_kernel_' + kernel
        classifiers_results.update({name: accuracy})

    # K-nearest neighbors classifier
    print('K-nearest neighbors classifier')
    print('Training...')
    kn_classifier = KNeighborsClassifier()
    kn_classifier.fit(x_train, y_train)
    print('Training done!')
    print('Testing!')
    y_pred = kn_classifier.predict(x_test)
    # compute accuracy
    accuracy = accuracy_score(y_test, y_pred)
    print('K-nearest neighbors classifier accuracy: ' + str(accuracy))
    print('\n')  # space
    name = 'k-nearest neighbors'
    classifiers_results.update({name: accuracy})

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

    print('Decision trees classifier')
    # decision trees
    print('Training...')
    dtc = DecisionTreeClassifier()
    dtc.fit(x_train, y_train)
    print('Training done!')
    print('Testing!')
    y_pred = dtc.predict(x_test)
    # compute accuracy
    accuracy = accuracy_score(y_test, y_pred)
    print('Decision trees classifier accuracy: ' + str(accuracy))
    print('\n')  # space
    name = 'Decision trees classifier'
    classifiers_results.update({name: accuracy})

    print('Catboost classifier')
    # catboost
    print('Training...')
    catboost = CatBoostClassifier(iterations=20)
    catboost.fit(x_train, y_train)
    print('Training done!')
    print('Testing!')
    y_pred = catboost.predict(x_test)
    # compute accuracy
    accuracy = accuracy_score(y_test, y_pred)
    print('Catboost accuracy: ' + str(accuracy))
    print('\n')  # space
    name = 'Catboost'
    classifiers_results.update({name: accuracy})


    print('Check the results')
    sorted_classifiers_results = dict(sorted(classifiers_results.items(), key=operator.itemgetter(1), reverse=True))
    for i, (name, accuracy) in enumerate(sorted_classifiers_results.items()):
        print(str(i + 1) + ' place: ' + name.ljust(30, ' ') + ' ' + str(accuracy))


def main():
    print('Fourth assignment!')
    # HOMEWORK
    train_models()


if __name__ == '__main__':
    main()
