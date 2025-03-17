"""Module for extracting text from PDF articles.

This module provides functionality to extract text content from PDF files
and save it as markdown files in an organized directory structure.
"""

from typing import List, Optional

import os
import pandas as pd
from pathlib import Path

import pymupdf4llm
from multiprocess import Pool, cpu_count
from tqdm.notebook import tqdm


def is_downloaded(df: pd.DataFrame) -> List[bool]:
    """Check if PDFs exist for each row in the DataFrame.

    Args:
        df: DataFrame containing article information with year and id columns.

    Returns:
        List of booleans indicating if each PDF file exists.
    """
    base_dir = os.path.join('data', 'pdf')
    downloaded = []
    
    for _, row in df.iterrows():
        file_path = os.path.join(base_dir, row['year'], f"{row['id']}.pdf")
        downloaded.append(os.path.isfile(file_path))

    return downloaded


def extract_text_from_pdf(pdf_file: str) -> str:
    """Extract text content from PDF file and convert to markdown.

    Args:
        pdf_file: Path to the PDF file.

    Returns:
        Extracted text in markdown format.
    """
    return pymupdf4llm.to_markdown(
        pdf_file, 
        margins=(0., 10., 0., 70.),
        show_progress=False
    )


def dump_text(text: str, out_dir: str) -> None:
    """Save text content to a markdown file.

    Args:
        text: Text content to save.
        out_dir: Directory path where to save the file.
    """
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    
    outfile = out_path / "text.md"
    if not outfile.exists():
        outfile.write_text(text, encoding='utf-8')


def task(row: pd.Series) -> None:
    """Process a single article's PDF file.

    Args:
        row: Series containing article metadata.
    """
    pdf_dir = os.path.join('data', 'pdf', row['year'])
    pdf_file = os.path.join(pdf_dir, f"{row['id']}.pdf")
    out_dir = os.path.join('data', 'text-images', row['year'], row['id'])
    
    os.makedirs(out_dir, exist_ok=True)
    
    if not os.path.isfile(os.path.join(out_dir, 'text.md')):
        md_text = extract_text_from_pdf(pdf_file)
        dump_text(md_text, out_dir)


def process_with_workers(df: pd.DataFrame, num_workers: Optional[int] = None) -> None:
    """Run text extraction concurrently using multiple CPU cores.

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
    """Execute the main text extraction workflow."""
    file_path = os.path.join('data', 'scrape_data_combined.csv')
    df = (
        pd.read_csv(file_path)
        .assign(year=lambda df_: df_['year'].astype(str))
        .loc[lambda df_: ~df_['pdf_url'].isna() & is_downloaded(df_), :]
    )

    process_with_workers(df)


if __name__ == "__main__":
    main()