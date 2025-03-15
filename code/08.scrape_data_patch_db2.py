import os
import re
import pandas as pd
from IPython.display import display, Markdown


def read_in_first_page(row):
    # set file_path
    file_path = os.path.join(
        'data', 'text-images', row['year'], row['id'], 'clean_text.md'
    )
    
    # read file and only select the first page
    with open(file_path) as f:
        fp = f.read()
        fp = fp.split('-----', 1)[0]

    return(fp)

def read_in_last_page(row):
    # set file_path
    file_path = os.path.join(
        'data', 'text-images', row['year'], row['id'], 'text.md'
    )
    
    # read file and only select the last page
    with open(file_path) as f:
        lp = f.read()
        lp = lp.split('-----')[-2]\

    return(lp)

def demarkdownify(text):
    return text.replace('**', '').replace('_', '').replace('-\n', '').replace('\n', ' ')

def parse_abstract(fp, row):
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
    return(row)

def downcast_headings(fp, row):
    if row['id'] in ['article-20063', 'article-20079']:
        fp = (
            fp
            .replace('## ', '# ')
            .replace('### ', '## ')
            .replace('#### ', '### ')
        )

    if row['id'] in ['article-20179', 'article-20144', 'article-20140']:
        fp = (
            fp
            .replace('## ' , '__ ' )
            .replace('# '  , '## ' )
            .replace('__ ' , '# '  )
            .replace('### ', '### ')
        )
    return(fp)

def parse_title(fp, row):
    top_of_front_page = fp.split('Summary\n\n')[0]
    
    # if no subtitle exists, then try only maintitle
    match = re.search(r'^# (.*)', top_of_front_page, re.MULTILINE)
    if match:
        row['title'] = match.group(1)

    if row['section'] == 'artikel':
        #first check first if there is a subtitle
        pattern = r'^# (.*)\n{1,2}(?:(?:## (.*))|(?:_(.*)_))'
        match = re.search(pattern, top_of_front_page, re.MULTILINE)
        if match:
            row['title'] = match.group(1) + ': ' + str(match.group(2))

    return(row)

def parse_authors(fp, lp, row):
    # set authors to none:
    row['authors'] = None
    
    pattern = r'(?:(?:^# .*\n\s*## .*\s*(_[^#].*)\n$)|(?:^# .*\n\s*_[^#].*_\s*(_.*$))|(?:^# .*\n\s*(_.*$)))'
    match = re.search(pattern, fp, re.MULTILINE)
    if match:
        row['authors'] = match.group(1)
        
    if row['authors'] is None:
        pattern = r'^(_.*)\s*#{2,4} Summary$'   
        match = re.search(pattern, fp, re.MULTILINE)
        if match:
            row['authors'] = match.group(1)

    if row['authors'] is None:
        pattern = r'(?:[\.\?]\s{2,}([^0-9#&\_\r\n]{3,})\s*\d*\s*$)'
        match = re.search(pattern, lp)
        if match:
            row['authors'] = match.group(1)
    
    if row['authors'] is None:
        pattern = r'(?:(?:\[.\?]\s{2,}([^0-9#&\_\r\n]{3,})\s*\d*\s*$)|(?:\.\s{2,}([^0-9#&\_\r\n]{3,})\s*## \w*))'
        match = re.search(pattern, lp, re.MULTILINE)
        if match:
            row['authors'] = match.group(1)

    if row['authors'] is None:
        lp_  = lp.split('## L')[0]
        pattern = r'(?:[\.\?]\s{2,}([^0-9#&\_\r\n]{3,})\s*\d*\s*$)'
        match = re.search(pattern, lp_)
        if match:
            row['authors'] = match.group(1)
    
    return(row)

def parse_scrape_data():
    file_path = os.path.join('data', 'scrape_data_combined.csv')
    df = (
        pd.read_csv(file_path)
        .assign(
            year = lambda df_: df_['year'].astype(str),
        )
    )
    
    a = df.loc[lambda df_: df_['database'] == 2, :]\
        .assign(authors = None)
    
    hold = []
    for i, row in a.iterrows():
        row = row.to_dict()
        lp = read_in_last_page(row)
        fp = read_in_first_page(row)
        fp = downcast_headings(fp, row)
        row = parse_title(fp, row)
        if row['section'] == 'artikel':
            row = parse_abstract(fp, row)
            row = parse_authors(fp, lp, row)
        hold.append(row)
    
    
    ndf = pd.DataFrame(hold)
    return(ndf)


df = pd.read_csv(os.path.join('data', 'scrape_data_combined.csv'))

sdf = df.loc[lambda df_: df_['database'] != 2, :]
ndf = (
    parse_scrape_data()
    .assign(
        section = lambda df_: df_['section']\
            .mask(df_['id'] == 'article-20097', 'boekbespreking'),
        authors = lambda df_: df_['authors']\
            .mask(df_['id'] == 'article-20098', 'H. Schijf')\
            .mask(df_['id'] == 'article-20098', 'Frans L. Leeuw')
    )
)
ndf.to_excel('temp.xlsx', index=False)

df = pd.concat([sdf, ndf], axis=0).sort_index()
file_path = os.path.join('data', 'scrape_data_combined.csv')
df.to_csv(file_path, index=False)



