"""Web scraper for M&M journal articles from AUP archive.

This module provides functionality to scrape article metadata and PDF links
from the Amsterdam University Press journal archive for Mens en Maatschappij.
"""

from typing import List

import pandas as pd
import requests
from bs4 import BeautifulSoup


class TrialContextManager:
    """Custom context manager to handle exceptions without interrupting execution."""

    def __enter__(self) -> None:
        """Enter the context manager."""
        pass

    def __exit__(self, exc_type, exc_val, exc_tb) -> bool:
        """Exit the context manager, suppressing any exceptions.

        Args:
            exc_type: Type of the exception.
            exc_val: Exception instance.
            exc_tb: Exception traceback.

        Returns:
            bool: True to suppress exceptions.
        """
        return True


# Global constants
BASE_URL: str = 'https://www.aup.nl/en/journal/mens-en-maatschappij/back-issues'
ARCHIVE_URL: str = 'https://journal-archive.aup.nl'
trial = TrialContextManager()


def get_article_urls(base_url: str) -> List[BeautifulSoup]:
    """Fetch article URLs from the base page.

    Args:
        base_url: URL of the journal's back issues page.

    Returns:
        List of BeautifulSoup elements containing article links.
    """
    response = requests.get(base_url)
    soup = BeautifulSoup(response.content, 'lxml')
    
    return [a for a in soup.find_all("a") 
            if 'journal-downloads' in a.get('href', '')]


def urls_to_df(article_urls: List[BeautifulSoup]) -> pd.DataFrame:
    """Convert extracted article URLs into a DataFrame.

    Args:
        article_urls: List of BeautifulSoup elements containing article links.

    Returns:
        DataFrame containing article URLs, issues, and titles.
    """
    data = []
    
    for article in article_urls:
        href = ARCHIVE_URL + article.get('href').replace('/journal-downloads', '')
        try:
            issue, title = article.text.split(' - ', 1)
        except ValueError:
            continue  # Skip malformed entries
        
        issue = issue.replace('no', 'No')
        data.append([href, issue, title])
    
    return pd.DataFrame(data, columns=['pdf_url', 'issue', 'title'])


def add_section(df: pd.DataFrame) -> pd.DataFrame:
    """Assign sections to articles based on their titles.

    Args:
        df: DataFrame containing article information.

    Returns:
        DataFrame with added section column.
    """
    df['section'] = 'artikel'
    
    df.loc[df['title'].str.contains('Boekbespreking', na=False), 
           'section'] = 'boekbespreking'
    df.loc[df['title'].str.contains(
        'Rectificatie|In memoriam|Ontvangen publicaties|'
        'Van Doorns Indische lessen', 
        na=False), 'section'] = 'algemeen'
    
    return df


def add_id_and_year(df: pd.DataFrame) -> pd.DataFrame:
    """Generate unique article IDs and extract publication years.

    Args:
        df: DataFrame containing article information.

    Returns:
        DataFrame with added ID and year columns.
    """
    df['year'] = df['issue'].str.extract('(\d+)').astype(float) + 1925
    
    # Forward fill missing years
    df['year'] = df['year'].ffill().astype(int)
    df['issue'] = df['issue'] + ' (' + df['year'].astype(str) + ')'
    
    df['id'] = ('article-' + df.groupby('issue').cumcount()
                .add(1).astype(str).str.zfill(2))
    
    return df


def main() -> None:
    """Execute the main scraping workflow."""
    article_urls = get_article_urls(BASE_URL)
    df = (urls_to_df(article_urls)
          .pipe(add_section)
          .pipe(add_id_and_year)
          .assign(database=2))
    
    df.to_csv('scraped_data2.csv', index=False)
    print("Scraping complete. Data saved to 'scraped_data2.csv'")


if __name__ == "__main__":
    main()