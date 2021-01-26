import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import AdaBoostClassifier


def analyze():
    dataset_path = 'dataset/car.data'
    df = pd.read_csv(dataset_path)
    df.head(15)
    # Number of missing values in each column of training data
    # no missing values you are lucky but alot of work here with
    # categorical values!
    missing_val_count_by_column = (df.isnull().sum())
    print(missing_val_count_by_column[missing_val_count_by_column > 0])
    df.vhigh.unique()
    df.select_dtypes(include=np.number).columns.tolist()

    s = (df.dtypes == 'object')  # just another way to see all categorical columns in your data
    object_cols = list(s[s].index)

    print("Categorical variables:")
    print(object_cols)

    # Apply some label fixes
    label_encoder = LabelEncoder()
    df.vhigh = label_encoder.fit_transform(df.vhigh)
    df.vhigh.unique()
    df["vhigh.1"].unique()
    label_encoder = LabelEncoder()
    df["vhigh.1"] = label_encoder.fit_transform(df["vhigh.1"])
    df["vhigh.1"].unique()

    def label_fix_2(x):
        if x == '2':
            return 2
        elif x == '3':
            return 3
        elif x == '4':
            return 4
        else:
            return 5

    df['2'] = df['2'].apply(label_fix_2)
    df['2'].unique()

    def label_fix_2_1(x_):
        if x_ == '2':
            return 2
        elif x_ == '4':
            return 4
        else:
            return 6

    df['2.1'] = df['2.1'].apply(label_fix_2_1)
    df['2.1'].unique()
    df["small"].unique()

    label_encoder = LabelEncoder()
    df.small = label_encoder.fit_transform(df.small)
    df["small"].unique()
    df["low"].unique()

    label_encoder = LabelEncoder()
    df.low = label_encoder.fit_transform(df.low)
    df["low"].unique()
    df.unacc.unique()

    def label_fix_y(x_):
        if x_ == 'unacc':
            return 0
        elif x_ == 'acc':
            return 1
        elif x_ == 'vgood':
            return 3
        elif x_ == "good":
            return 2

    df['unacc'] = df['unacc'].apply(label_fix_y)
    df.unacc.unique()
    # lets see your final dataset
    df.head(20)

    x = df.drop('unacc', axis=1)
    y = df['unacc']
    number_of_class = len(y.unique())

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.20, shuffle=True)

    print('Training...')
    model = AdaBoostClassifier()
    print('Trained!\n')
    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)

    # compute accuracy
    accuracy = accuracy_score(y_test, y_pred)
    print('Accuracy: ' + str(accuracy))


def main():
    print('Sixth assignment!')
    # HOMEWORK
    analyze()


if __name__ == '__main__':
    main()
