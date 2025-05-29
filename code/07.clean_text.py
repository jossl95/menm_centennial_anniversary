"""Module topic modelling on clean scraped data.

This module provides functionality to further clean the scraped
data and perform topic modelling on these data.
"""

import os
import re
from pathlib import Path
from typing import List, Optional, Union

import pandas as pd
from multiprocess import Pool, cpu_count
from tqdm.notebook import tqdm

DATA_DIR = Path('data')
OUTDIR = DATA_DIR / 'text-images'
OVERWRITE = True

def add_pdf_downloaded_variable(df: pd.DataFrame) -> pd.DataFrame:
    """Adds a boolean column indicating if PDFs were downloaded.

    Args:
        df: pandas DataFrame containing article information.

    Returns:
        DataFrame with new pdf_downloaded column.
    """
    pdf_base = Path('data') / 'pdf'
    years = [
        year.name for year in sorted(pdf_base.iterdir())
        if year.is_dir() and len(year.name) == 4
    ]

    articles = [
        pdf.stem
        for year in years
        for pdf in (pdf_base / year).glob('*.pdf')
        if pdf.is_file()
    ]

    df['pdf_downloaded'] = df['id'].isin(articles)
    return df

def add_pdf_parsed_variable(df: pd.DataFrame) -> pd.DataFrame:
    """Adds a boolean column indicating if PDFs were parsed.

    Args:
        df: pandas DataFrame containing article information.

    Returns:
        DataFrame with new pdf_parsed column.
    """
    base_dir = Path(OUTDIR)
    years = [
        year.name for year in sorted(base_dir.iterdir())
        if year.is_dir() and len(year.name) == 4
    ]

    articles = [
        article.name
        for year in years
        for article in (base_dir / year).iterdir()
        if (article / 'text.md').is_file()
    ]

    df['pdf_parsed'] = df['id'].isin(articles)
    return df

def read_text(row: pd.Series) -> Optional[str]:
    text_dir = Path(os.path.join(OUTDIR, str(row['year']), row['id']))
    file = text_dir / 'text.md'

    if file.exists():
        with open(file) as f:
            return f.read()
    return None

def read_texts_with_workers(df: pd.DataFrame, num_workers: Optional[int] = None) -> pd.DataFrame:
    """Run text extraction concurrently using multiple CPU cores.

    Args:
        df: DataFrame containing article information.
        num_workers: Number of worker processes to use. Defaults to CPU count.

    Returns:
        DataFrame with added 'docs' column.
    """
    if num_workers is None:
        num_workers = cpu_count()

    with Pool(processes=num_workers) as pool:
        docs = list(tqdm(
            pool.imap(read_text, [row for _, row in df.iterrows()]),
            total=df.shape[0]
        ))

    df['docs'] = docs
    return df

def clean_markdown(text: str) -> str:
    text = re.sub(r'(\**\_*)|(\_\*\**)', '', text, flags=re.MULTILINE)
    text = re.sub(r'(?<!\.)\n\n(?<![A-Z])', '\n', text)
    text = re.sub(r'[\t\r\f\v]+', ' ', text)
    text = re.sub(r'[\u00AD]+', '-', text)
    return text

# def slice_title_section(text: str) -> str:
#     split_pattern = r"\n(?:#{2,}\ +)(?:\d{1}(?:\.|\:| |\W)*)?(?:In)+.*\n"
#     chunks = re.split(split_pattern, text, maxsplit=1)
#     text = chunks[-1]

#     return text

def clean_footer_and_header(page: str) -> str:
    line_patterns = [
        r"^(?:#+\ +)?(?:MENS|Mens|mens) +(?:&|[a-z]{2}) +(?:MAATSCHAPPIJ|Maatschappij|maatschappij)$",
        r"MENSens & MAATSCHAPPIJaatschappij",
        r"^g$",
        r"\ *(?:\d{4},?\ +)?jaargang\ +\d+,\ *nr\.?\ *\d\ *",
        r"^.*Jaargang\ +\d+.*$",
        r"^(?:MENS|Mens|mens).*\d+$",
        r"^(?:#+\ +|\d+)?VOL\.[^a-z]*$",
        r"^.*VOL\..*$",
        r"^(?:#+\ +)?Guest *(.*$)",
        r"^(?:#+\ +|\d+)?IP[^a-zA-Z]*$",
        r"^(?:#+\ +|\d+)?DOI[^a-z]*$",
        r"^(?:#+\ +)?\d+$",
        r"^https:\/\/doi.org\/10\.\d{4,5}\/[a-zA-Z0-9.]{,22}$",
        r"^\u00a9.*$"
    ]
    for pattern in line_patterns:
        page = re.sub(pattern, '', page, flags=re.MULTILINE).strip()

    patterns = [
        r"^[0-9]+[^a-z\n]+",
        r"^[^a-z\n]+[0-9]+",
        r"[^a-z\n]+[0-9]+$"
    ]
    for pattern in patterns:
        page = re.sub(pattern, '', page).strip()

    return page

def clean_title_page(page: str) -> str:
    """Cleans raw markdown strings.

    Args:
        page: String containing clean text of one page.

    Returns:
        Cleaned string.
    """
    patterns = [
        r"^(?:[A-Z0-9]+|#+)(?:\s+[^a-z\n\(]+)?(?:\(.*\)(?:[^\n]+)?)?$",
        r"^(?!-+$)(?:[^a-z\n])*$",
        r"^(?:(?:door)|(?:d o o r )|(?:DOOR)).*(?:(?:[\.\)])|(?:[A-Z])){1}$"
    ]
    for pattern in patterns:
        page = re.sub(pattern, '', page, flags=re.MULTILINE).strip()

    return page

def clean_full_text(text: str, row: pd.Series) -> str:
    """Cleans raw markdown strings.

    Args:
        text: String containing clean text.
        row: Series containing article metadata.

    Returns:
        Cleaned string.
    """
    text = clean_markdown(text)
    # text = slice_title_section(text)
    pages = text.split("-----")

    clean_pages = []
    for i, page in enumerate(pages):
        page = page.strip()
        page = clean_footer_and_header(page)
        page = clean_title_page(page)
        if page != '':
            clean_pages.append(page)

    return '\n\n-----\n\n'.join(clean_pages)

def format_clean_text(text: str, row: pd.Series) -> str:
    """Formats clean texts.

    Args:
        text: String containing clean text.
        row: Series containing article metadata.

    Returns:
        Formatted string.
    """
    title, abstract = '', ''

    if isinstance(row['title'], str):
        title = f"# {row['title'].upper()}"
        
    if isinstance(row['abstract'], str):
        abstract = f"## ABSTRACT: {row['abstract']}"
        
    if isinstance(text, str):
        text = f"## 1. Inleiding\n{text}"

    elements = [item for item in [title, abstract, text] if item != '']
    return '\n\n'.join(elements)

def dump_text(text: str, row: pd.Series) -> None:
    """Writes cleaned text to file.

    Args:
        text: String containing text to write.
        row: Series containing article metadata.
    """
    out_path = OUTDIR / row['year'] / row['id']
    out_path.mkdir(parents=True, exist_ok=True)
    out_file = out_path / 'clean_text.md'
    out_file.write_text(text, encoding='utf-8')

def task(row: pd.Series) -> None:
    """Process the raw markdown texts.

    Args:
        row: Series containing article metadata.
    """
    text = read_text(row)
    text = clean_full_text(text, row)
    text = format_clean_text(text, row)
    dump_text(text, row)

def process_with_workers(df: pd.DataFrame, num_workers: Optional[int] = None) -> None:
    """Runs text cleaning tasks concurrently using multiple CPU cores.

    Args:
        df: pandas DataFrame containing articles to process.
        num_workers: Optional number of worker processes to use.
    """
    if num_workers is None:
        num_workers = cpu_count()

    with Pool(processes=num_workers) as pool:
        list(tqdm(
            pool.imap(task, [row for _, row in df.iterrows()]),
            total=df.shape[0]
        ))

def main() -> None:
    """Execute the main text extraction workflow."""
    file_path = DATA_DIR / 'scrape_data_combined.csv'
    df = (
        pd.read_csv(file_path)
        .assign(year=lambda df_: df_['year'].astype(str))
        .pipe(add_pdf_downloaded_variable)
        .pipe(add_pdf_parsed_variable)
        .loc[lambda df_: (df_['pdf_downloaded'] & df_['pdf_parsed'])]
    )

    process_with_workers(df)

if __name__ == '__main__':
    main()
