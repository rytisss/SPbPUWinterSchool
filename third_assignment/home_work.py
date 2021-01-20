import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVR
from sklearn import metrics


def analyze():
    """
    use the dataset you've found yesterday and see what you can do with any algorithm
    you are welcome to use any algorithm!

    NOTE: dataset changed
    """

    # Read dataset
    happiness_dataset_path = 'dataset/world-happiness-report-2019.csv'
    df = pd.read_csv(happiness_dataset_path)
    print(df)

    # Print sum of Nan values in each column
    print(df.isnull().sum())

    # Fill empty spaces with mean value
    df.fillna(df.mean(), inplace=True)

    # drop country name before scaling
    df_norm = df.drop(['Country (region)'], axis=1)

    # Normalize dataset
    # create a scaler object
    scaler = MinMaxScaler()
    # fit and transform the data
    df_norm = pd.DataFrame(scaler.fit_transform(df_norm), columns=df_norm.columns)
    print(df_norm)

    # Try to predict 'healthy life expectancy' for following parameters:
    # Positive affect, Negative affect, Social support, Freedom,	Corruption,	Generosity,	Log of GDP per capita
    x = df_norm.drop(['Ladder', 'SD of Ladder', 'Healthy life\nexpectancy'], axis=1)
    y = df_norm['Healthy life\nexpectancy']

    # data split train 70% - 30%
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, shuffle=True)
    print('Entries in train: ' + str(x_train.shape[0]))
    print('Entries in test: ' + str(x_test.shape[0]))

    # try regressor with 3 different kernels
    kernels = ['linear', 'poly', 'rbf', 'sigmoid']
    for kernel in kernels:
        print(5 * '\n') # space
        print('Kernel: ' + kernel)
        print('Training...')
        svr_model = SVR(kernel=kernel)
        svr_model.fit(x_train, y_train)
        print('Training done!')
        print('Testing!')
        # predict the test dataset
        y_pred = svr_model.predict(x_test)
        # compute square root error
        error = np.sqrt(metrics.mean_squared_error(y_test, y_pred))
        print('Model with kernel \'' + kernel + '\' square error: ' + str(error))


def main():
    print('Third assignment!')
    # HOMEWORK
    analyze()


if __name__ == '__main__':
    main()
