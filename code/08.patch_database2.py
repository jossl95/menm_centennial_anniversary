"""Module for patching and cleaning scraped database 2 articles.

This module provides functionality to parse and clean article metadata from
markdown files, including titles, abstracts, and author information.
"""

from typing import Dict

import os
import re
import pandas as pd


def read_in_first_page(row: Dict) -> str:
    """Read and extract first page content from cleaned markdown file.

    Args:
        row: Dictionary containing article metadata including year and id.

    Returns:
        String containing first page content.
    """
    file_path = os.path.join(
        'data', 'text-images', row['year'], row['id'], 'clean_text.md'
    )
    
    with open(file_path) as f:
        fp = f.read()
        fp = fp.split('-----', 1)[0]

    return fp


def read_in_last_page(row: Dict) -> str:
    """Read and extract last page content from markdown file.

    Args:
        row: Dictionary containing article metadata including year and id.

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
            .replace('_', '')
            .replace('-\n', '')
            .replace('\n', ' '))


def parse_abstract(fp: str, row: Dict) -> Dict:
    """Parse abstract and English title from first page content.

    Args:
        fp: First page content string.
        row: Dictionary containing article metadata.

    Returns:
        Updated row dictionary with abstract and English title.

    Raises:
        ValueError: If required sections are not found in expected format.
    """
    if 'Summary\n' in fp:
        try:
            summary_text = fp.split('Summary\n\n', 1)[1]
            summary_end = '\n\n##' if '\n\n##' in summary_text else '_\n\n'
            abstract_section = summary_text.split(summary_end, 1)[0]
            eng_title, abstract_text = abstract_section.split('\n\n', 1)
        except IndexError:
            raise ValueError('Markdown format unexpected: required sections not found.')
        
        row['eng_title'] = demarkdownify(eng_title)
        row['abstract'] = demarkdownify(abstract_text)
    return row


def downcast_headings(fp: str, row: Dict) -> str:
    """Adjust heading levels in markdown content for specific articles.

    Args:
        fp: First page content string.
        row: Dictionary containing article metadata.

    Returns:
        String with adjusted heading levels.
    """
    if row['id'] in ['article-20063', 'article-20079']:
        fp = (fp.replace('## ', '# ')
              .replace('### ', '## ')
              .replace('#### ', '### '))

    if row['id'] in ['article-20179', 'article-20144', 'article-20140']:
        fp = (fp.replace('## ', '__ ')
              .replace('# ', '## ')
              .replace('__ ', '# ')
              .replace('### ', '### '))
    return fp


def parse_title(fp: str, row: Dict) -> Dict:
    """Extract article title from first page content.

    Args:
        fp: First page content string.
        row: Dictionary containing article metadata.

    Returns:
        Updated row dictionary with title information.
    """
    top_of_front_page = fp.split('Summary\n\n')[0]
    
    match = re.search(r'^# (.*)', top_of_front_page, re.MULTILINE)
    if match:
        row['title'] = match.group(1)

    if row['section'] == 'artikel':
        pattern = r'^# (.*)\n{1,2}(?:(?:## (.*))|(?:_(.*)_))'
        match = re.search(pattern, top_of_front_page, re.MULTILINE)
        if match:
            row['title'] = f"{match.group(1)}: {match.group(2)}"

    return row


def parse_authors(fp: str, lp: str, row: Dict) -> Dict:
    """Extract author information from page content.

    Args:
        fp: First page content string.
        lp: Last page content string.
        row: Dictionary containing article metadata.

    Returns:
        Updated row dictionary with author information.
    """
    row['authors'] = None
    
    patterns = [
        r'(?:(?:^# .*\n\s*## .*\s*(_[^#].*)\n$)|'
        r'(?:^# .*\n\s*_[^#].*_\s*(_.*$))|(?:^# .*\n\s*(_.*$)))',
        r'^(_.*)\s*#{2,4} Summary$',
        r'(?:[\.\?]\s{2,}([^0-9#&\_\r\n]{3,})\s*\d*\s*$)',
        r'(?:(?:\[.\?]\s{2,}([^0-9#&\_\r\n]{3,})\s*\d*\s*$)|'
        r'(?:\.\s{2,}([^0-9#&\_\r\n]{3,})\s*## \w*))'
    ]

    for pattern in patterns:
        if row['authors'] is None:
            match = re.search(pattern, fp if patterns.index(pattern) < 2 else lp, 
                            re.MULTILINE)
            if match:
                row['authors'] = match.group(1)

    if row['authors'] is None:
        lp_ = lp.split('## L')[0]
        pattern = r'(?:[\.\?]\s{2,}([^0-9#&\_\r\n]{3,})\s*\d*\s*$)'
        match = re.search(pattern, lp_)
        if match:
            row['authors'] = match.group(1)
    
    return row


def parse_scrape_data() -> pd.DataFrame:
    """Parse and clean scraped article data from database 2.

    Returns:
        DataFrame containing cleaned article metadata.
    """
    file_path = os.path.join('data', 'scrape_data_combined.csv')
    df = pd.read_csv(file_path).assign(year=lambda df_: df_['year'].astype(str))
    
    articles = (df.loc[lambda df_: df_['database'] == 2, :]
                .assign(authors=None))
    
    hold = []
    for _, row in articles.iterrows():
        row = row.to_dict()
        lp = read_in_last_page(row)
        fp = read_in_first_page(row)
        fp = downcast_headings(fp, row)
        row = parse_title(fp, row)
        if row['section'] == 'artikel':
            row = parse_abstract(fp, row)
            row = parse_authors(fp, lp, row)
        hold.append(row)
    
    return pd.DataFrame(hold)


def main() -> None:
    """Execute main data processing workflow."""
    df = pd.read_csv(os.path.join('data', 'scrape_data_combined.csv'))
    sdf = df.loc[lambda df_: df_['database'] != 2, :]
    
    ndf = (parse_scrape_data()
           .assign(section=lambda df_: df_['section']
                  .mask(df_['id'] == 'article-20097', 'boekbespreking'),
                  authors=lambda df_: df_['authors']
                  .mask(df_['id'] == 'article-20098', 'H. Schijf')
                  .mask(df_['id'] == 'article-20098', 'Frans L. Leeuw')))

    df = pd.concat([sdf, ndf], axis=0).sort_index()
    file_path = os.path.join('data', 'scrape_data_combined.csv')
    df.to_csv(file_path, index=False)


if __name__ == "__main__":
    main()