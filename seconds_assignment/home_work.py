import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def load_uk_covid_dataset():
    """
    Find a dataset from internet. in csv format. and show me the LAST 20 rows in your dataset using pandas-dataframe
    Load UK COVID-19 database and show last 20 rows
    """
    covid_dataset_path = 'datasets/UK_National_Total_COVID_Dataset.csv'
    df = pd.read_csv(covid_dataset_path)
    print(df.tail(n=20))


def load_and_process_candy_dataset():
    """
    # use candy.csv dataset
    # get your data with pandas and show me the first 5 rows
    # show me sugarpercent and winpercent columns use seaborn(OPTIONAL)
    # can you plot it with a regression line? use seaborn (OPTIONAL)
    # find the median of winpercent column
    # find the standard deviation of sugarpercent column
    # plot the winpercent column with histograms use matplotlib
    # show me sugarpercent and winpercent columns use matplotlib
    """
    candy_dataset_path = 'datasets/candy.csv'
    df = pd.read_csv(candy_dataset_path)
    # get your data with pandas and show me the first 5 rows
    print(df.head(n=5))

    # show me sugarpercent and winpercent columns use seaborn(OPTIONAL)
    sugar_win = df[['sugarpercent', 'winpercent']]
    sns.set_style('whitegrid')
    sns.scatterplot(data=sugar_win, x="sugarpercent", y="winpercent")

    # regression line
    sns.regplot(data=sugar_win, x="sugarpercent", y="winpercent")
    print('Press \'q\' to continue')
    plt.show()

    # find the median of winpercent column
    winpercent_median = df['winpercent'].median()
    print('winpercent median: ' + str(winpercent_median))

    # find the standard deviation of sugarpercent column
    sugarpercent_std = df['sugarpercent'].std()
    print('sugarpercent standard deviation: ' + str(sugarpercent_std))

    # plot the winpercent column with histograms use matplotlib
    winpercent_df = df['winpercent']
    plt.hist(winpercent_df, bins=10)
    plt.ylabel('winpercent')
    print('Press \'q\' to continue')
    plt.show()

    # show me sugarpercent and winpercent columns use matplotlib
    plt.scatter(data=sugar_win, x="sugarpercent", y="winpercent")
    plt.xlabel('sugarpercent')
    plt.ylabel('winpercent')
    print('Press \'q\' to continue')
    plt.show()


def main():
    print('Second assignment!')
    # HOMEWORK- 1
    load_uk_covid_dataset()
    # HOMEWORK -2
    load_and_process_candy_dataset()


if __name__ == '__main__':
    main()
