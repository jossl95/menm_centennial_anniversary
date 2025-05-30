"""Module for harmonizing scraped article data.

This module provides functionality to clean and standardize article metadata,
including author names, issue labels, and topic classification.
"""

import os
import pandas as pd

from pathlib import Path
from bertopic import BERTopic
from IPython.display import display, Markdown
from nameparser import HumanName
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from thefuzz import fuzz, process


def harmonize_issue_label(df: pd.DataFrame) -> pd.DataFrame:
    """Standardize issue labels and extract volume/issue information.

    Args:
        df: DataFrame containing article metadata.

    Returns:
        DataFrame with standardized issue labels and extracted components.
    """
    df['issue_label'] = (
        df['issue']
        .str.replace('No1', 'No 1')
        .str.replace('3/4', '3')
        .str.replace('1/2', '1')
        .mask(lambda x: x == 'thema: Anthropogenese', 'Vol 18 Nr 3 (1942)')
        .mask(lambda x: x == 'Steinmetznummer', 'Vol 18 Nr 3 (1942)')
    )
    
    issue_parts = (
        df['issue_label']
        .mask(lambda x: x == 'Jubileumnummer')
        .str.split(' ', expand=True)
        .set_axis(
            ['volume_prefix', 'volume', 'issue_prefix', 'issue', 'year'],
            axis=1
        )
        .assign(year=lambda df_: df_['year'].str[1:5])
    )
    
    df['volume'] = issue_parts['volume'].astype(str)
    df['issue'] = issue_parts['issue'].astype(str)
    df['year'] = issue_parts['year'].astype(str)

    return df


def clean_authors(df: pd.DataFrame) -> pd.DataFrame:
    """Clean and standardize author names.

    Args:
        df: DataFrame containing article metadata.

    Returns:
        DataFrame with cleaned author names.
    """
    name_replacements = {
        '_': '',
        ' en ': ', ',
        'M., M.': 'M. en M.',
        '& Maat': 'en Maat',
        'Mensch, Maatschappij': 'Mensch en Maatschappij',
        # ... additional replacements ...
    }
    
    df['authors_list'] = (
        df['authors']
        .str.replace('|'.join(name_replacements.keys()), 
                    lambda x: name_replacements[x.group()], 
                    regex=True)
        .str.split('[').str[0]
        .str.split(r'(?:(?:\,\s)|(?:\&))', regex=True)
        .mask(df['id'] == 'article-11893')
    )
    return df


def parse_names(df: pd.DataFrame) -> pd.DataFrame:
    """Parse author names into components.

    Args:
        df: DataFrame containing article metadata.

    Returns:
        DataFrame with parsed author names.
    """
    authors = (
        df[['id', 'authors_list']]
        .dropna()
        .explode('authors_list')
        .rename({'authors_list': 'name'}, axis=1)
        .assign(
            name_has_comma=lambda df_: df_['name'].str.contains(','),
            name_chunks=lambda df_: df_['name'].str.split(',')
                .mask(~df_['name_has_comma'])
        )
    )
    
    name_fixes = (
        authors['name_chunks'].str[1].str.strip() + " " +
        authors['name_chunks'].str[0].str.strip()
    )
    
    authors = (
        authors
        .assign(
            name=lambda df_: df_['name']
                .mask(df_['name_has_comma'], name_fixes)
                .str.replace('.', '. ')
                .str.replace('  ', ' ')
        )
        .drop(['name_chunks'], axis=1)
    )
    return authors


def add_clean_name(authors: pd.DataFrame) -> pd.DataFrame:
    """Add standardized clean names using nameparser.

    Args:
        authors: DataFrame containing author information.

    Returns:
        DataFrame with added clean name column.
    """
    parsed_names = (
        authors
        .assign(
            first=authors['name'].apply(lambda name: HumanName(name).first).astype(str),
            middle=authors['name'].apply(lambda name: HumanName(name).middle),
            last=authors['name'].apply(lambda name: HumanName(name).last),
            initial = lambda x: x['first'].str[0] + '.'
        )
    )
    
    authors['clean_name'] = (
        parsed_names[['initial', 'middle', 'last']]
        .astype(str)
        .agg(' '.join, axis=1)
        .str.replace('  ', ' ')
        .str.replace(' . ', ' ')
        .mask(
            authors['name'].str.contains('edactie'),
            'Redactie M&M'
        )
        .mask(
            authors['name'].str.contains('ereniging') |
            authors['name'].str.contains('ureau voor') |
            (authors['name'] == 'AT'),
            authors['name']
        )
    )

    return authors


def homogenize_names(authors: pd.DataFrame) -> pd.DataFrame:
    """Harmonize similar author names using fuzzy matching.

    Args:
        authors: DataFrame containing author information.

    Returns:
        DataFrame with harmonized author names.
    """
    matches = []
    names_set = set(authors['clean_name'].unique())
    
    for _, row in authors.iterrows():
        match = process.extract(row['clean_name'], names_set)
        similar_names = [name for name, ratio in match if ratio > 92][:2]
        if len(similar_names) > 1:
            matches.append(similar_names)
    
    # Filter valid matches
    name_mappings = {
        sorted(m, key=len)[0]: sorted(m, key=len)[1] 
        for m in matches
        if (_is_valid_name_match(m[0], m[1]))
    }
    
    authors['clean_name'] = (
        authors['clean_name']
        .mask(
            authors['clean_name'].isin(name_mappings.keys()),
            authors['clean_name'].map(name_mappings)
        )
    )
    return authors


def _is_valid_name_match(name1: str, name2: str) -> bool:
    """Check if two names should be considered matches.

    Args:
        name1: First author name.
        name2: Second author name.

    Returns:
        Boolean indicating if names should be matched.
    """
    exclusions = {'Hagedoorn', 'A. Bierens', 'Hzn', 'Blokland', 'Roos'}
    return (
        name1[0] == name2[0] and
        not any(excl in name1 for excl in exclusions)
    )


def main() -> None:
    """Execute main data harmonization workflow."""
    file_path = Path('data') / 'scrape_data_combined.csv'
    df = pd.read_csv(file_path)

    # Clean and process data
    df = (
        df
        .sort_values('id')
        .reset_index(drop=True)
        .assign(
            section=lambda df_: df_['section']
                .mask(lambda x: x == 'artikelen', 'artikel')
                .mask(lambda x: x == 'boekbesprekingen', 'boekbespreking'),
            url=lambda df_: df_['url']
                .mask(lambda x: x.isna(), df_['doi_url'])
        )
        .pipe(harmonize_issue_label)
        .pipe(clean_authors)
    )

    # Process authors
    authors = (
        df
        .pipe(parse_names)
        .pipe(add_clean_name)
        .pipe(homogenize_names)
        .drop(['name', 'name_has_comma'], axis=1)
        .rename({'clean_name': 'name'}, axis=1)
    )

    authors_file = Path('data') / 'article_author_link.xlsx'
    authors.to_excel(authors_file, index=False)

    # Update article data with processed authors
    df = (
        df
        .drop(['authors_list', 'authors'], axis=1)
        .merge(
            authors
                .groupby('id')
                .agg(list)
                .reset_index()
                .rename({'name': 'authors_list'}, axis=1)
                .assign(
                    authors=lambda df_: 
                        df_['authors_list'].transform(lambda x: ', '.join(x)),
                ),
            on='id',
            how='left'
        )
        .assign(
            section=lambda df_: df_['section']
                .mask(df_['authors'] == 'Redactie M&M', 'algemeen')
                .mask(
                    ~df_['abstract'].isna() & 
                    (df_['section'] == 'algemeen'), 
                    'artikel'
                ),
            year=lambda df_: df_['year'].mask(lambda x: x == 'None', '1975'),
            page_count=lambda df_: df_['page_count'].astype('Int64')
        )
    )

    # Save processed data
    cols = [
        'id', 'year', 'issue_label', 'section', 'title', 'authors',
        'abstract', 'tags', 'page_count', 'url', 'pdf_url', 
    ]

    articles_file = Path('data') / 'articles_metadata.xlsx'
    df[cols].to_excel(articles_file, index=False)

    # Save decade-specific files
    for decade_start in range(1925, 2025, 10):
        decade_end = decade_start + 10
        decade_df = df.loc[
            (df['year'].astype(int) >= decade_start) & 
            (df['year'].astype(int) < decade_end),
            cols
        ]

        decade_file = Path('decades') / f'articles {decade_start}-{decade_end}.xlsx'
        decade_df.to_excel(decade_file, index=False)


if __name__ == "__main__":
    main()