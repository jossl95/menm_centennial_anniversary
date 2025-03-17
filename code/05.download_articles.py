"""Module for downloading PDF articles from M&M journal archives.

This module provides functionality to download PDF articles from multiple sources
and save them in an organized directory structure based on publication year.
"""

from typing import Optional

import os
import re
import httpx
import pandas as pd
import requests
from multiprocess import Pool, cpu_count
from tqdm.notebook import tqdm


def create_article_path(year: str, article_id: str) -> str:
    """Create directory path for storing an article PDF.

    Args:
        year: Publication year of the article.
        article_id: Unique identifier for the article.

    Returns:
        Path where the PDF file should be stored.
    """
    article_path = os.path.join('data', 'pdf', year)
    os.makedirs(article_path, exist_ok=True)
    return os.path.join(article_path, f"{article_id}.pdf")


def download_article(row: pd.Series) -> None:
    """Download PDF article and save it in the appropriate directory.

    Args:
        row: Series containing article metadata including year, id, and pdf_url.
    """
    legacy = False if int(row['database']) == 2 else True
    file_path = create_article_path(row['year'], row['id'])
    
    if not os.path.isfile(file_path):
        response = requests.get(row['pdf_url']) if legacy else httpx.get(row['pdf_url'])
        
        if response.status_code == 200:
            with open(file_path, 'wb') as file:
                file.write(response.content)


def validate_issue_format(series: pd.Series) -> pd.Series:
    """Validate if issue entries match the expected format.
    
    Args:
        series: Series containing issue strings.
        
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


def task(row: pd.Series) -> None:
    """Wrapper function for downloading a single article.
    
    Args:
        row: Series containing article metadata.
    """
    download_article(row)


def process_with_workers(df: pd.DataFrame, num_workers: Optional[int] = None) -> None:
    """Run downloads concurrently using multiple CPU cores.

    Args:
        df: DataFrame containing article information.
        num_workers: Number of worker processes to use. Defaults to CPU count.
    """
    if num_workers is None:
        num_workers = cpu_count()
    
    with Pool(processes=num_workers) as pool:
        list(tqdm(
            pool.imap(task, [row for _, row in df.iterrows()]), 
            total=df.shape[0]
        ))


def main() -> None:
    """Execute the main download workflow."""
    df = (
        pd.read_csv(os.path.join('data', 'scrape_data_combined.csv'))
        .loc[lambda df_: ~df_['pdf_url'].isna(), :]
        .assign(year=lambda df_: df_['year'].astype(str))
    )

    process_with_workers(df, num_workers=8)


if __name__ == "__main__":
    main()