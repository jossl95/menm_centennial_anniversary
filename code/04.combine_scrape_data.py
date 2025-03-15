import os
import re
import pandas as pd

def validate_issue_format(series):
    """Creates a boolean array asserting if each entry in the series matches the expected issue format."""
    pattern = re.compile(r'Vol\s*(\d+)\s*(?:No|Nr)\s*([\d-]+)\s*\((\d{4})\)')
    return series.apply(lambda x: bool(pattern.fullmatch(x)))

def fix_year(df):
    df['valid_issue'] = df['issue'].pipe(validate_issue_format)
    
    df['year'] = (
        df
        .loc[df['valid_issue'], 'issue']
        .str.split('(').str[-1]
        .str.replace(')', '')
    )

    return(df.drop('valid_issue', axis=1))

def fix_id(df):
    df = (
        df.assign(
            id = lambda df_: df_
                .groupby('database')
                .cumcount()
                .add(1)
                .pipe(lambda x: x + df_['database']*10000)
                .astype(str)
                .pipe(lambda x: 'article-' + x)
        )
    )

    return(df)


df = (
    pd.concat(
        [
            pd.read_csv("scraped_data1.csv"),
            pd.read_csv("scraped_data2.csv"),
            pd.read_csv("scraped_data3.csv")
        ], 
        ignore_index=True
    )
    .pipe(fix_year)
    .assign(
        year = lambda df_: df_['year']
            .astype(str)\
            .mask(
                lambda x: x == 'nan',
                df_['date'].str[0:4]
            )
    )
    .sort_values(['year', 'id'], ignore_index=True)
    .pipe(fix_id)
)

df.to_csv(os.path.join('data', 'scrape_data_combined.csv'), index=False)
