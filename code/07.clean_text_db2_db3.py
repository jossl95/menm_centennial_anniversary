import itertools
import re
import os
import pandas as pd
from pathlib import Path
from tqdm.notebook import tqdm
from IPython.display import display, Markdown
from multiprocess import Pool, cpu_count

BASEDIR = os.path.join('data', 'text-images')

def add_pdf_downloaded_variable(df):
    # scan all downloaded pdf
    years = [
        year 
            for year in sorted(os.listdir(os.path.join('data', 'pdf')))
                if (len(year) == 4)
    ]
    
    articles  = {year: os.listdir(os.path.join('data', 'pdf', year)) for year in years}
    articles = [
        id_.split('.')[0]
            for year, ids in articles.items() 
            for id_ in ids 
                if os.path.isfile(os.path.join('data', 'pdf' , year, id_) )
    ]

    # add pdf downloaded variable
    df['pdf_downloaded'] = df['id'].isin(articles)
    return(df)

def add_pdf_parsed_variable(df):
    # scan all parsed texts
    years = [
        year 
            for year in sorted(os.listdir(BASEDIR)) 
                if (len(year) == 4)
    ]
    
    articles  = {year: os.listdir(os.path.join(BASEDIR, year)) for year in years}
    articles = [
        id_
            for year, ids in articles.items() 
            for id_ in ids 
                if os.path.isfile(os.path.join(BASEDIR, year, id_, 'text.md') )
    ]

    df['pdf_parsed'] = df['id'].isin(articles)
    return(df)

def write_updated_df(df, file_path):
    df.to_csv(file_path, index=False)
    return(df)

def clean_text(md_text):
    # Define patterns to remove
    patterns = [
        r"^\n?_\d{4}, jaargang \d+, nr\.? \d+_\n",  # Flexible header removal
        r"^\n_Mens & Maatschappij_\n",  # "Mens & Maatschappij" standalone header
        r"^\nMens & Maatschappij\n",  # "Mens & Maatschappij" standalone header
        r"^MENS & MAATSCHAPPIJ$",  # "Mens & Maatschappij" standalone header
        r"^MENS & MAATSCHAPPIJ \d{1,3}\.\d \(\d{4}\) \d{1,3}-\d{1,3}", # 1st page header
        r"^\n_https:\/\/doi.org\/10\.\d{4,5}\/[A-Z0-9.]+_\n", # 1st page doi link
        r"-_\n_",  # Stray underscore cleanup
        r"^- ",  # Dashes followed by space
        r"^\n\d+\n*-{3,10}",  # Footer: Page number followed by '-----'
        r"^-{3,10}\n\n[A-Z &\-\.:;!]*", # capitalized header db3
        r".*?\n\s*-{5,}\s*\n", # Footer: credits followed by '-----'
    ]

    # Apply all patterns
    for pattern in patterns[:-3]:
        md_text = re.sub(pattern, "", md_text, flags=re.MULTILINE)

    md_text = re.sub(patterns[-1], "-----\n", md_text, flags=re.MULTILINE)
    md_text = re.sub(patterns[-2], "-----\n", md_text, flags=re.MULTILINE)
    md_text = re.sub(patterns[-2], "-----\n", md_text, flags=re.MULTILINE)

    # general pattern of roque headers and footers
    md_text = re.sub(r"^\n.*\n*-{2,10}", "\n-----\n", md_text, flags=re.MULTILINE)
    md_text = re.sub(r"^\n-{2,10}\n{1,2}.*\n{2,4}", "\n-----\n\n", md_text, flags=re.MULTILINE)
    md_text = re.sub(r"^-{5,10}", "\n\n-----\n\n", md_text, flags=re.MULTILINE)

    # come unicode white space sliped in, which are taken out
    md_text = re.sub(r"^[\u0008\u0020]{2,5}", "", md_text, flags=re.MULTILINE)

    return md_text

def dump_text(text, out_dir):
    out_path = Path(out_dir)    
    outfile = out_path / "clean_text.md"
    # Write only if the file doesn't already exist
    # if not outfile.exists():
    #     outfile.write_text(text, encoding='utf-8')
    outfile.write_text(text, encoding='utf-8')

def task(row):
    file_path = os.path.join(BASEDIR, row['year'], row['id'], 'text.md')
    
    with open(file_path) as f:
        md_text = f.read()
        md_text = clean_text(md_text)
    
    out_dir = os.path.join(BASEDIR, row['year'], row['id'])
    dump_text(md_text, out_dir)

def process_with_workers(df, num_workers=None):
    """Runs the task function concurrently using multiple CPU cores via multiprocess."""
    if num_workers is None:
        num_workers = cpu_count()  # Use all available CPU cores
    
    with Pool(processes=num_workers) as pool:
        list(tqdm(pool.imap(task, [row for _, row in df.iterrows()]), total=df.shape[0]))



file_path = os.path.join('data', 'scrape_data_combined.csv')
selected_df = (
    pd.read_csv(file_path)
    .assign(
        year = lambda df_: df_['year'].astype(str),
    )
    .pipe(add_pdf_downloaded_variable)
    .pipe(add_pdf_parsed_variable)
    .pipe(write_updated_df, file_path)
    .loc[
        lambda df_: df_['pdf_downloaded'] 
                    & df_['pdf_parsed'] 
                    & df_['database'].isin([2, 3])
    ]
)

process_with_workers(selected_df)

row = selected_df.loc[lambda df_: df_['id'] == 'article-30370'].iloc[0,:]

# de text is nog niet perfect schoon, maar al zeker bruikbaar
# de in sommige bestanden zitten nog headers en footers.
# veel al is dit de titel van het stuk. 
# het is een mogelijkheid om een functie te maken die de title
# uit de tekst haalt, als deze niet met een "# " begint, 
# het gevaar hiervan is dat sommige bestanden de title niet juist is 
# opgemaakt. 





