"""Module for patching and cleaning scraped database 2 articles.

This module provides functionality to parse and clean article metadata from
markdown files, including titles, abstracts, and author information.
"""

from typing import Dict, Union

import os
import re
import pandas as pd
from pathlib import Path


def read_in_first_page(row: pd.Series, clean: bool = True) -> str:
    """Read and extract first page content from cleaned markdown file.

    Args:
        row: Series containing article metadata including year and id.

    Returns:
        String containing first page content.
    """
    file = 'clean_text.md' if clean else "text.md"
    
    file_path = os.path.join(
        'data', 'text-images', row['year'], row['id'], file
    )
    
    with open(file_path) as f:
        fp = f.read()
        fp = fp.split('-----', 1)[0]

    return fp


def read_in_last_page(row: pd.Series) -> str:
    """Read and extract last page content from markdown file.

    Args:
        row: Series containing article metadata including year and id.

    Returns:
        String containing last page content.
    """
    file_path = os.path.join(
        'data', 'text-images', row['year'], row['id'], 'text.md'
    )
    
    with open(file_path) as f:
        lp = f.read()
        lp = lp.split('-----')[-2]

    return lp


def demarkdownify(text: str) -> str:
    """Remove markdown formatting from text.

    Args:
        text: String containing markdown formatted text.

    Returns:
        Clean text with markdown formatting removed.
    """
    return (text.replace('**', '')
            .replace('*', '')
            .replace('_', '')
            .replace('-\n', ''))


def parse_title(row: pd.Series) -> Union[str, None]:
    """Extract article title from first page content.

    Args:
        row: Series containing article metadata.

    Returns:
        title as a string
    """
    title=None
    
    fp = demarkdownify(read_in_first_page(row, clean=False))
    top_of_front_page = re.split(r"(?:#+\ +)?Summary", fp, maxsplit=1)[0]
    
    match = re.search(r'^# (.*)', top_of_front_page, re.MULTILINE)
    if match:
        title = match.group(1)

    if row['section'] == 'artikel':
        pattern = r'^# (.*)\n{1,2}(?:(?:## (.*))|(?:_(.*)_))'
        match = re.search(pattern, top_of_front_page, re.MULTILINE)
        if match:
            title = f"{match.group(1)}: {match.group(2)}"

    if not title:
        title = row['title']

    return title

def parse_abstract(row: pd.Series) -> Union[str, None]:
    """Parse abstract from first page content.

    Args:
        row: Series containing article metadata.

    Returns:
        abstract as a string.
    """
    abstract = None

    fp = demarkdownify(read_in_first_page(row, clean=False))
    
    if 'Summary' in fp:
        bottom_of_front_page = re.split(r"(?:#+\ +)?Summary", fp, maxsplit=1)[-1]
        summary_text = re.split(r"#{2,}", bottom_of_front_page, maxsplit=1)[0].strip()
        abstract = summary_text.split('\n\n', 1)[-1]
        
    return abstract


def parse_authors(row: pd.Series) -> Union[str, None]:
    """Extract author information from page content.

    Args:
        row: Series containing article metadata.

    Returns:
        authors list as a string
    """
    authors = None

    fp = demarkdownify(read_in_first_page(row, clean=False))
    lp = demarkdownify(read_in_last_page(row))

    if "Summary" in fp:
        top_of_front_page = re.split(r"(?:#+\ +)?Summary", fp, maxsplit=1)[0]
        if match := re.search(r"^(.*)\[\d]$", top_of_front_page, re.MULTILINE):
            authors = match.group(1).strip()
            authors = authors.replace(' en ', ', ')
            
    if authors is None:
        lp_ = lp.split('## L')[0]
        pattern = r'(?:[\.\?]\s{2,}([^0-9#&\_\r\n]{3,})\s*\d*\s*$)'
        match = re.search(pattern, lp_)
        if match:
            authors = match.group(1)
    
    return authors

def parse_articles(df2):
    hold = []
    for i, row in df2.iterrows():
        row['title'] = parse_title(row)
        row['abstract'] = parse_abstract(row)
        row['authors'] = parse_authors(row)
        hold.append(row)
    
    df2 = pd.concat(hold, axis=1).T
    return df2


def main() -> None:
    """Execute main data processing workflow."""
    file_path = Path('data') / 'scrape_data_combined.csv'

    df2 = (
        pd.read_csv(file_path)
        .assign(year=lambda df_: df_['year'].astype(str))
        .loc[lambda df_: df_['database'] == 2, :]
        .pipe(parse_articles)
        .assign(
            section=lambda df_: df_['section']
                .mask(df_['id'] == 'article-20097', 'boekbespreking'),
            authors=lambda df_: df_['authors']
                .mask(df_['id'] == 'article-20098', 'H. Schijf')
                .mask(df_['id'] == 'article-20098', 'Frans L. Leeuw')
        )
    ) 
    
    df13 = (
        pd.read_csv(file_path)
        .assign(year=lambda df_: df_['year'].astype(str))
        .loc[lambda df_: df_['database'] != 2, :]
    )
    
    df = pd.concat([df13, df2], axis=0).sort_index()
    df.to_csv(file_path, index=False)


if __name__ == "__main__":
    main()









