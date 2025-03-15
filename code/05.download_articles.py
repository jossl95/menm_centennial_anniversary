import os
import requests
import httpx
import re
import pandas as pd
from tqdm.notebook import tqdm
from multiprocess import Pool, cpu_count

def create_article_path(year, article_id):
    """Creates and returns the directory path for storing an article PDF."""
    issue_path = os.path.join('data', article_id )
    article_path = os.path.join('data', 'pdf', year)
    
    # Create necessary directories if they don't exist
    os.makedirs(article_path, exist_ok=True)
    
    return os.path.join(article_path, f"{article_id}.pdf")

def download_article(row):
    """Downloads a PDF article and saves it in the appropriate directory."""
    legacy = False if int(row['database']) == 2 else True
    
    file_path = create_article_path(row['year'], row['id'])
    if not os.path.isfile(file_path):
        # Select appropriate HTTP client
        response = requests.get(row['pdf_url']) if legacy else httpx.get(row['pdf_url'])
        
        # Proceed only if the request is successful
        if (response.status_code == 200):
            # Save the PDF file
            with open(file_path, 'wb') as file:
                file.write(response.content)

def validate_issue_format(series):
    """Creates a boolean array asserting if each entry in the series matches the expected issue format."""
    pattern = re.compile(r'Vol\s*(\d+)\s*(?:No|Nr)\s*([\d-]+)\s*\((\d{4})\)')
    return series.apply(lambda x: bool(pattern.fullmatch(x)))

def fix_year(df):
    df['valid_issue'] = df['issue'].pipe(validate_issue_format)
    
    df['year'] = (
        df
        .loc[df['valid_issue'], 'issue']
        .str.split('(').str[-1]
        .str.replace(')', '')
    )

    return(df.drop('valid_issue', axis=1))

def task(row):
    download_article(row)

def process_with_workers(df, num_workers=None):
    """Runs the task function concurrently using multiple CPU cores via multiprocess."""
    if num_workers is None:
        num_workers = cpu_count()  # Use all available CPU cores
    
    with Pool(processes=num_workers) as pool:
        list(tqdm(pool.imap(task, [row for _, row in df.iterrows()]), total=df.shape[0]))

df = (
    pd.read_csv(
        os.path.join('data', 'scrape_data_combined.csv')
    )
    # exclude rows without a pdf_url
    .loc[lambda df_: ~df_['pdf_url'].isna(), :]
    # cast year to string for path_names
    .assign(year = lambda df_: df_['year'].astype(str))
)

process_with_workers(df, num_workers=8)

