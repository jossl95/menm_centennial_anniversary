"""Module for combining and harmonizing scraped data from multiple sources.

This module combines scraped article data from three different sources,
fixes year formatting, and harmonizes article IDs across databases.
"""

from typing import Any

import os
import re
import pandas as pd

from pathlib import Path


def validate_issue_format(series: pd.Series) -> pd.Series:
    """Validate if issue entries match the expected format.
    
    Args:
        series: pandas Series containing issue strings.
        
    Returns:
        Boolean Series indicating if each entry matches the format.
    """
    pattern = re.compile(r'Vol\s*(\d+)\s*(?:No|Nr)\s*([\d-]+)\s*\((\d{4})\)')
    return series.apply(lambda x: bool(pattern.fullmatch(x)))


def fix_year(df: pd.DataFrame) -> pd.DataFrame:
    """Extract and fix year information from issue column.
    
    Args:
        df: DataFrame containing article information.
        
    Returns:
        DataFrame with corrected year column.
    """
    df['valid_issue'] = df['issue'].pipe(validate_issue_format)
    
    df['year'] = (
        df
        .loc[df['valid_issue'], 'issue']
        .str.split('(').str[-1]
        .str.replace(')', '')
    )

    return df.drop('valid_issue', axis=1)


def fix_id(df: pd.DataFrame) -> pd.DataFrame:
    """Generate unique article IDs based on database source.
    
    Args:
        df: DataFrame containing article information.
        
    Returns:
        DataFrame with harmonized article IDs.
    """
    return df.assign(
        id=lambda df_: (
            df_
            .groupby('database')
            .cumcount()
            .add(1)
            .pipe(lambda x: x + df_['database']*10000)
            .astype(str)
            .pipe(lambda x: 'article-' + x)
        )
    )


def main() -> None:
    """Combine and process scraped data from multiple sources."""
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
            year=lambda df_: (
                df_['year']
                .astype(str)
                .mask(
                    lambda x: x == 'nan',
                    df_['date'].str[0:4]
                )
            )
        )
        .sort_values(['year', 'id'], ignore_index=True)
        .pipe(fix_id)
    )

    file_path = Path('data') / 'scrape_data_combined.csv'
    df.to_csv(file_path, index=False)


if __name__ == "__main__":
    main()