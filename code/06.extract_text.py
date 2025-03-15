import os
import pandas as pd
import pymupdf
import pymupdf4llm
from pathlib import Path
from tqdm.notebook import tqdm
from IPython.display import display, Markdown
from multiprocess import Pool, cpu_count

def is_downloaded(df):
    BASEDIR = os.path.join('data', 'pdf')
    hold = []
    for i, row in df.iterrows():
        year = row['year']
        id_ = row['id']
        file_path = os.path.join(BASEDIR, year, id_ + '.pdf' )
        file_exists = os.path.isfile(file_path)
        hold.append(file_exists)

    return(hold)

def extract_text_from_pdf(pdf_file):
    md_text = pymupdf4llm.to_markdown(
        pdf_file, 
        margins=(0., 10., 0., 70.),
        show_progress=False
    )

    return(md_text)
    
def dump_text(text, out_dir):
    out_path = Path(out_dir)
    # Create directory (and any missing parent directories) if needed
    out_path.mkdir(parents=True, exist_ok=True)
    
    outfile = out_path / "text.md"
    # Write only if the file doesn't already exist
    if not outfile.exists():
        outfile.write_text(text, encoding='utf-8')

def task(row):
    pdf_dir = os.path.join('data', 'pdf', row['year'])
    pdf_file = os.path.join(pdf_dir, row['id'] + '.pdf')
    out_dir = os.path.join('data', 'text-images', row['year'], row['id'])
    os.makedirs(out_dir, exist_ok=True)
    
    if not os.path.isfile(os.path.join(out_dir, 'text.md')):
        md_text = extract_text_from_pdf(pdf_file)
        dump_text(md_text, out_dir)

def process_with_workers(df, num_workers=None):
    """Runs the task function concurrently using multiple CPU cores via multiprocess."""
    if num_workers is None:
        num_workers = cpu_count()  # Use all available CPU cores
    
    with Pool(processes=num_workers) as pool:
        list(tqdm(pool.imap(task, [row for _, row in df.iterrows()]), total=df.shape[0]))

file = os.path.join('data', 'scrape_data_combined.csv')
df = (
    pd.read_csv(file)
    .assign(
        year=lambda df_: df_['year'].astype(str)
    )
    .loc[
        lambda df_: ~df_['pdf_url'].isna() & is_downloaded(df_)
        ,:
    ]
)

process_with_workers(df)


    



# #38
# row = df.iloc[100, :]
# pdf_dir = os.path.join('data', 'pdf', row['year'].astype(str))
# pdf_file = os.path.join(pdf_dir, row['id'] + '.pdf')
# out_dir = os.path.join('data', 'text-images', row['year'].astype(str), row['id'])
# os.makedirs(out_dir, exist_ok=True)

# md_text = pymupdf4llm.to_markdown(
#     pdf_file, 
#     margins=(0., 45., 0., 40.),
#     show_progress=False
# )

# dump_text(md_text, out_dir)

# Markdown(md_text)





