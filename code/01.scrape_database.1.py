"""Web scraper for M&M journal articles from University of Groningen Press.

This module provides functionality to scrape article metadata, abstracts,
and PDF links from the M&M journal archive.
"""

from typing import Optional, List

import pandas as pd
import requests
from bs4 import BeautifulSoup
from httpx import get as httpx_get
from tqdm.notebook import tqdm


class TrialContextManager:
    """Custom context manager to handle exceptions without interrupting execution."""

    def __enter__(self) -> None:
        pass

    def __exit__(self, *args) -> bool:
        return True


trial = TrialContextManager()


def parse_items(items: List, section: str = 'algemeen') -> pd.DataFrame:
    """Parse article items from HTML elements.

    Args:
        items: List of HTML elements containing article information.
        section: Section name for the articles. Defaults to 'algemeen'.

    Returns:
        DataFrame containing parsed article information.
    """
    articles = []

    for i, item in enumerate(items):
        if i % 2 == 0:  # Process every second item
            title_element = item.find("div", class_="title")
            if not title_element:
                continue

            link = title_element.find('a')
            article_id = link.get('id')
            url = link.get('href')
            with trial:
                title = link.string.strip()

            authors = item.find("div", class_="authors").string.strip().split('\t')
            pages = (None if section == 'algemeen' 
                    else item.find("div", class_="pages").string.strip())

            articles.append(pd.DataFrame({
                'id': article_id,
                'title': title,
                'authors': authors,
                'pages': pages,
                'url': url
            }, index=[0]))

    return (pd.concat(articles, ignore_index=True).assign(section=section) 
            if articles else pd.DataFrame())


def scrape_article(url: str) -> pd.DataFrame:
    """Scrape article details from the given URL.

    Args:
        url: URL of the article page.

    Returns:
        DataFrame containing article details.
    """
    response = requests.get(url)
    soup = BeautifulSoup(response.content, 'lxml')

    articles = []
    for section in soup.find_all("div", class_="section"):
        section_title = section.h2.get_text(strip=True).lower()
        elements = section.find_all('ul', class_="cmp_article_list articles")

        for element in elements:
            with trial:
                articles.append(
                    parse_items(element.find_all('li'), section=section_title))

    return pd.concat(articles, ignore_index=True) if articles else pd.DataFrame()


def scrape_archive(url: str) -> pd.DataFrame:
    """Scrape archive page for issue titles and URLs.

    Args:
        url: URL of the archive page.

    Returns:
        DataFrame containing issue information.
    """
    response = httpx_get(url)
    soup = BeautifulSoup(response.content, 'lxml')

    listings = soup.find_all("a", class_="title")
    return pd.DataFrame({
        'issue': [listing.get_text(strip=True) for listing in listings],
        'url': [listing.get('href') for listing in listings]
    })


def parse_text(
    soup: BeautifulSoup,
    tag: str,
    class_name: str,
    replace_text: Optional[str] = None
) -> Optional[str]:
    """Extract and clean text from a specific tag in BeautifulSoup object.

    Args:
        soup: BeautifulSoup object containing the HTML.
        tag: HTML tag to search for.
        class_name: CSS class name to match.
        replace_text: Text to remove from the result, if any.

    Returns:
        Cleaned text string or None if element not found.
    """
    section = soup.find(tag, class_=class_name)
    if section:
        text = section.get_text(strip=True).replace('\n', '').replace('\t', '')
        if replace_text:
            text = text.replace(replace_text, '')
        return text
    return None


def scrape_article_details(df: pd.DataFrame) -> pd.DataFrame:
    """Scrape additional details for each article in DataFrame.

    Args:
        df: DataFrame containing article URLs.

    Returns:
        DataFrame with added abstract, tags, and date information.
    """
    for index, row in tqdm(df.iterrows(), total=df.shape[0]):
        response = httpx_get(row['url'])
        soup = BeautifulSoup(response.content, 'lxml')
        df.at[index, 'abstract'] = parse_text(
            soup, 'section', 'item abstract', 'Samenvatting')
        df.at[index, 'tags'] = parse_text(
            soup, 'section', 'item keywords', 'Trefwoorden:')
        df.at[index, 'date'] = parse_text(
            soup, 'div', 'item published', 'Gepubliceerd')

    return df


def get_pdf_url(row: pd.Series) -> Optional[str]:
    """Extract PDF URL from an article page.

    Args:
        row: DataFrame row containing article URL.

    Returns:
        PDF download URL or None if not found.
    """
    response = httpx_get(row['url'])
    if response.status_code != 200:
        response = requests.get(row['url'])

    soup = BeautifulSoup(response.content, 'lxml')
    pdf_link = soup.find('a', class_='obj_galley_link pdf')

    if pdf_link:
        response = httpx_get(pdf_link.get('href'))
        if response.status_code != 200:
            response = requests.get(pdf_link.get('href'))

        soup = BeautifulSoup(response.content, 'lxml')
        download_link = soup.find('a', class_='download')
        return download_link.get('href') if download_link else None

    return None


def main() -> None:
    """Execute the main scraping workflow."""
    base_url = "https://ugp.rug.nl/MenM/issue/archive"
    hold = []

    # Scrape archive pages
    for i in range(1, 17):
        url = base_url if i == 1 else f"{base_url}/{i}"
        hold.append(scrape_archive(url))

    archive_df = pd.concat(hold, ignore_index=True)

    # Scrape articles per issue
    hold = []
    for _, row in tqdm(archive_df.iterrows(), total=archive_df.shape[0]):
        hold.append(scrape_article(row['url']).assign(issue=row['issue']))

    articles_df = pd.concat(hold, ignore_index=True)

    # Scrape additional details
    articles_df = scrape_article_details(articles_df)

    # Retrieve PDF URLs
    for i, row in tqdm(articles_df.iterrows(), total=articles_df.shape[0]):
        articles_df.at[i, 'pdf_url'] = get_pdf_url(row)

    articles_df['database'] = 1
    articles_df.to_csv('scraped_data1.csv', index=False)
    print("Scraping complete. Data saved to 'scraped_data.csv'")


if __name__ == "__main__":
    main()