"""Web scraper for M&M journal articles from AUP Online archive.

This module provides functionality to scrape article metadata, abstracts,
and PDF links from the Amsterdam University Press Online journal archive
for Mens en Maatschappij.
"""

from typing import List, Optional, Tuple

import httpx
import pandas as pd
import requests
from bs4 import BeautifulSoup
from tqdm.notebook import tqdm

# Global constants
BASE_URL: str = ('https://www.aup-online.com/content/journals/00259454/'
                 'browse?page=previous-issues')
ARCHIVE_URL: str = 'https://www.aup-online.com'


def get_list_of_volumes(base_url: str) -> List[Tuple[str, str]]:
    """Fetch list of available volumes from the archive page.
    
    Args:
        base_url: URL of the journal archive page.
        
    Returns:
        List of tuples containing (volume_name, volume_url).
    """
    response = httpx.get(base_url)
    soup = BeautifulSoup(response.content, 'html.parser')
    
    volume_items = soup.find_all('li', class_='volume-item')
    
    return [
        (a_tag.get_text(strip=True).replace('\r', ''),
         ARCHIVE_URL + a_tag['href'].split('&showDates')[0])
        for item in volume_items if (a_tag := item.find('a'))
    ]


def get_list_of_issues(
    volume_items: List[Tuple[str, str]]
) -> List[Tuple[str, str, str, str, str]]:
    """Extract individual issue details from volume pages.
    
    Args:
        volume_items: List of tuples containing volume information.
        
    Returns:
        List of tuples containing (volume, volume_url, issue_url, 
        issue_name, issue_month).
    """
    results = []
    for volume, volume_url in volume_items:
        response = httpx.get(volume_url)
        soup = BeautifulSoup(response.content, 'html.parser')
    
        for issue in soup.find_all('li'):
            a_tag = issue.find('a')
            if not a_tag:
                continue
            
            issue_url = ARCHIVE_URL + a_tag.get('href')
            issue_name = issue.find('span', class_='issuenumber')
            issue_month = issue.find('span', class_='issueyear')
            
            if issue_name and issue_month:
                results.append((
                    volume,
                    volume_url,
                    issue_url,
                    issue_name.get_text(strip=True),
                    issue_month.get_text(strip=True).split('\n')[1]
                ))
    
    return results


def extract_article_info(article: BeautifulSoup) -> Tuple[
    Optional[str], Optional[str], Optional[str], Optional[str], str, Optional[str]
]:
    """Extract metadata from an article element.
    
    Args:
        article: BeautifulSoup object containing article HTML.
        
    Returns:
        Tuple containing (title, eng_title, doi_url, authors, section, abstract).
    """
    doi_tag = article.find(
        'a', href=True, 
        string=lambda s: 'doi.org' in s if s else False
    )
    doi_url = doi_tag['href'] if doi_tag else None
    
    title_tag = article.find('h3')
    title = title_tag.get_text(strip=True) if title_tag else None
    section = ('boekbespreking' if title and 'Boekbespreking' in title 
              else 'algemeen')
    
    info_tag = article.find('div', class_='js-desc')
    eng_title, abstract = None, None
    if info_tag:
        info_parts = [p.get_text(strip=True).replace(' .', '.') 
                     for p in info_tag.find_all('p')]
        if info_parts:
            abstract = info_parts[-1]
            eng_title = (' -- '.join(info_parts[:-1]) + '.' 
                        if len(info_parts) > 1 else None)
    
    authors = ', '.join(
        a.get_text(strip=True) 
        for a in article.find_all('a', class_='nonDisambigAuthorLink')
    ) if article.find_all('a', class_='nonDisambigAuthorLink') else None
    
    return title, eng_title, doi_url, authors, section, abstract


def get_article_data(
    issue_items: List[Tuple[str, str, str, str, str]]
) -> pd.DataFrame:
    """Fetch article metadata from issue pages.
    
    Args:
        issue_items: List of tuples containing issue information.
        
    Returns:
        DataFrame containing article metadata.
    """
    results = []
    
    for volume, _, issue_url, issue_name, issue_month in issue_items:
        response = httpx.get(issue_url)
        soup = BeautifulSoup(response.content, 'html.parser')
        
        article_items = soup.select_one(
            "#main-content-container div.issue-listing "
            "div.panel-body div.publistwrapper"
        )
        if not article_items:
            continue
        
        for article in article_items.find_all(
            'ul', class_='list-unstyled', role='listitem'
        ):
            article_info = extract_article_info(article)
            results.append({
                'volume': volume,
                'issue': issue_name, 
                'month': issue_month,
                'title': article_info[0],
                'eng_title': article_info[1],
                'doi_url': article_info[2],
                'authors': article_info[3],
                'section': article_info[4],
                'abstract': article_info[5]
            })
    
    return pd.DataFrame(results)


def add_pdf_url(df: pd.DataFrame) -> pd.DataFrame:
    """Add PDF download URLs to the DataFrame.
    
    Args:
        df: DataFrame containing article information.
        
    Returns:
        DataFrame with added PDF URLs.
    """
    results = []
    
    for _, row in tqdm(df.iterrows(), total=df.shape[0]):
        response = requests.get(row['doi_url'])
        soup = BeautifulSoup(response.content, 'html.parser')
        form = soup.find(
            'form', 
            class_='ft-download-content__form ft-download-content__form--pdf '
                   'js-ft-download-form'
        )
        
        pdf_url = (requests.compat.urljoin(ARCHIVE_URL, form.get('action')) 
                  if form else None)
        results.append({'id': row['id'], 'pdf_url': pdf_url})
    
    return df.merge(pd.DataFrame(results), how='inner')


def format_issue(df: pd.DataFrame) -> pd.DataFrame:
    """Format the issue column to include volume numbers.
    
    Args:
        df: DataFrame containing article information.
        
    Returns:
        DataFrame with formatted issue column.
    """
    issue_fix = (
        df['volume']
        .str.replace('(2018 - 2019)', '(2018)')
        .str.split(' ', expand=True)
        .set_axis(['temp', 'volume', 'year'], axis=1)
        .assign(
            year=lambda x: x['year'].str[1:-1],
            issue=df['issue'].str.split(' ').str[-1],
        )
        .assign(
            issue=lambda x: ('Vol ' + x['volume'] + ' No ' + x['issue'] + 
                           ' (' + x['year'] + ')')
        )
        .drop('temp', axis=1)
    )

    for col in issue_fix.columns:
        df[col] = issue_fix[col]
    
    return df


def main() -> None:
    """Execute the main scraping workflow."""
    volume_items = get_list_of_volumes(BASE_URL)
    issue_items = get_list_of_issues(volume_items)
        
    df = get_article_data(issue_items)
    df['id'] = 'article-' + (90000 + df.index).astype(str)
    df['database'] = 3
    
    df = add_pdf_url(df).pipe(format_issue)
    df.to_csv('scraped_data3.csv', index=False)
    
    print("Scraping complete. Data saved to 'scraped_data3.csv'")


if __name__ == "__main__":
    main()