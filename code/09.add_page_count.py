"""Module for adding page counts to article metadata.

This module provides functionality to extract page count information from PDF files
and add it to the article metadata database.
"""
from typing import List, Optional, Tuple

import os
import warnings
from pathlib import Path

import pandas as pd
from multiprocess import Pool, cpu_count
from pypdf import PdfReader
from tqdm.notebook import tqdm

# Suppress warnings
warnings.filterwarnings('ignore')


def get_page_count(row: pd.Series) -> Optional[int]:
    """Extract page count from PDF file.

    Args:
        row: Series containing article metadata with year and id.

    Returns:
        Tuple of (article_id, page_count), where page_count may be None
        if the PDF cannot be read.
    """
    pdf_file = Path('data') / 'pdf' / str(row['year']) / f"{row['id']}.pdf"
    page_count = pd.NA
    
    if pdf_file.exists():
        try:
            with PdfReader(pdf_file) as doc:
                page_count = len(doc.pages)
        except:  # Catch any PDF reading errors
            pass

    return (row['id'], page_count)


def process_with_workers(
    df: pd.DataFrame, 
    num_workers: Optional[int] = None
) -> List[Tuple[str, Optional[int]]]:
    """Run page count extraction concurrently using multiple CPU cores.

    Args:
        df: DataFrame containing article metadata.
        num_workers: Number of worker processes to use. Defaults to CPU count.

    Returns:
        List of tuples containing (article_id, page_count) for each article.
    """
    if num_workers is None:
        num_workers = cpu_count()
    
    with Pool(processes=num_workers) as pool:
        results = list(
            tqdm(
                pool.imap(
                    get_page_count, 
                    [row for _, row in df.iterrows()]
                ), total = df.shape[0]
            )
        )

    return(results)


def main() -> None:
    """Execute the main page count extraction workflow."""
    file_path = Path('data') / 'scrape_data_combined.csv'
    
    # Read and sort input data
    df = pd.read_csv(file_path).sort_values('id')
    
    # Extract page counts
    results = pd.DataFrame(
        process_with_workers(df), 
        columns=['id', 'page_count']
    )

    df['page_count'] = results['page_count']
    
    df.to_csv(file_path, index=False)


if __name__ == "__main__":
    main()