from . import pd


def validate_contents(file_1, file_2):

    df1 = pd.read_csv(file_1)
    df2 = pd.read_csv(file_2)

    print(df1.equals(df2))

