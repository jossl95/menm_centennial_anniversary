import os
import pandas as pd
from pathlib import Path
from tqdm.notebook import tqdm
from pypdf import PdfReader
from multiprocess import Pool, cpu_count

import warnings
warnings.filterwarnings('ignore')

def get_page_count(row):
    pdf_file = f"data/pdf/{row['year']}/{row['id']}.pdf"
    page_count = None
    
    if os.path.isfile(pdf_file):
        try:
            with PdfReader(pdf_file) as doc:
                page_count = len(doc.pages)
        except:
            pass

    return((row['id'], page_count))

def process_with_workers(df, num_workers=None):
    """Runs the task function concurrently using multiple CPU cores via multiprocess."""

    if num_workers is None:
        num_workers = cpu_count()  # Use all available CPU cores
    
    with Pool(processes=num_workers) as pool:
        res = list(
            tqdm(pool.imap(
                    get_page_count, 
                    [row for _, row in df.iterrows()]
                ), total=df.shape[0]
            )
        )

    return(res)

file_path = os.path.join('data', 'scrape_data_combined.csv')

df = (
    pd.read_csv(file_path)
    .sort_values('id')
)

res = pd.DataFrame(process_with_workers(df), columns =['id', 'page_count'])

df.merge(res).to_csv(file_path, index=False)