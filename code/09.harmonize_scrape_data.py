import os
import re
import pandas as pd
from IPython.display import display, Markdown

# packages for name harmonization
from nameparser import HumanName
from thefuzz import fuzz
from thefuzz import process

# packages for topic harmonization
from bertopic import BERTopic
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

def harmonize_issue_label(df):
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
        .str.split(' ', expand=True)\
        .set_axis(
            ['volume_prefix', 'volume', 'issue_prefix', 'issue', 'year'],
            axis=1
        )
        .assign(year = lambda df_: df_['year'].str[1:5])
    )
    
    df['volume'] = issue_parts['volume'].astype(str)
    df['issue'] = issue_parts['issue'].astype(str)
    df['year'] = issue_parts['year'].astype(str)

    return(df)

def clean_authors(df):
    df['authors_list'] = (
        df['authors']
        .str.replace('_','')
        .str.split('[').str[0]
        .str.replace(' en ', ', ')
        .str.replace('M., M.', 'M. en M.')
        .str.replace('& Maat', 'en Maat')
        .str.replace('Mensch, Maatschappij', 'Mensch en Maatschappij')
        .str.replace('Naters, M.', 'Naters,M.')
        .str.replace('Isselt, E.', 'Isselt,E.')
        .str.replace('Blankenstein, Th.', 'Blankenstein,Th.')
        .str.replace('Anderson, C.', 'Anderson,C.')
        .str.replace('Maesen.C.', 'Maesen,C.')
        .str.replace('Wallenburg H.', 'Wallenburg,H.')
        .str.replace('Pijper, W.', 'Pijper,W.')
        .str.replace('Neijens, P.', 'Neijens,P.')
        .str.replace('Doorne-Huiskes, Anneke van', 'Doorne-Huiskes,Anneke van')
        .str.replace('Hraba.Joseph', 'Hraba,Joseph')
        .str.replace('Kraaykamp, G.', 'Kraaykamp,G.')
        .str.replace('Akkerman, W.', 'Akkerman,W.')
        .str.replace('  Maassen. G. H.', ' Maassen,G. H.')
        .str.replace('Schippers, J. J.', 'Schippers,J. J.')
        .str.replace('Arts, Wil', 'Arts,Wil') 
        .str.replace('Werfhorst, Herman', 'Werfhorst,Herman')
        .str.replace('Manuela, du', 'Manuela du')
        .str.replace('Met medewerking van ', '')
        .str.replace('Fr.', 'F.')
        .str.replace(' ir.', '')
        .str.replace('Yildiz1', 'Yildiz')
        .str.replace('[1]', '')
        .str.replace('MSc', '')
        .str.split(r'(?:(?:\,\s)|(?:\&))', regex=True)
        .mask(df['id'] == 'article-11893')
    )
    return(df)

def parse_names(df):
    authors = (
        df[['id', 'authors_list']]
        .dropna()
        .explode('authors_list')
        .rename({'authors_list': 'name'}, axis=1)
        .assign(
            name_has_comma = lambda df_: df_['name'].str.contains(','),
            name_chunks = lambda df_: df_['name'].str.split(',')
                .mask(~df_['name_has_comma'])
        )
    )
    
    name_fixes = (
        (authors['name_chunks']
            .str[1].str.strip())
        + " " 
        + (authors['name_chunks']
            .str[0].str.strip())
    )
    
    authors = (
        authors
        .assign(
            name = lambda df_: df_['name']
                .mask(df_['name_has_comma'], name_fixes)
                .str.replace('.', '. ')
                .str.replace('  ', ' ')
        )
        .drop(['name_chunks'], axis=1)
    )
    return(authors)

def add_clean_name(authors):
    a = (
        authors
        .assign(
            first = authors['name'].apply(lambda name: HumanName(name).first),
            middle = authors['name'].apply(lambda name: HumanName(name).middle),
            last = authors['name'].apply(lambda name: HumanName(name).last),
            initial = lambda x: x['first'].str[0] + '.'
        )
    )
    
    authors['clean_name'] = (
        a[['initial', 'middle', 'last']]
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

    return(authors)

def homogenize_names(authors):
    matches = []
    for i, row in authors.iterrows():
        names_set = {name for name in authors['clean_name'].unique()}
        
        match = process.extract(row['clean_name'], names_set)
        match = [name for name, ratio in match if ratio > 92][:2]
        if len(match) > 1:
            matches.append(match)
    
    
    # restrict the matches to 
    # - have the same initial
    # - not overwrite separate authors with the same name
    matches = {
        sorted(m, key=len)[0]: sorted(m, key=len)[1] 
        for m in matches
            if (m[0][0] == m[1][0])
            & ('Hagedoorn' not in m[0])
            & ('A. Bierens' not in m[0])
            & ('Hzn' not in m[0])
            & ('Blokland' not in m[0])
            & (m[0] != 'Roos')
    }
    
    
    authors['clean_name'] = (
        authors['clean_name']
        .mask(
            authors['clean_name'].isin(matches.keys()),
            authors['clean_name'].map(matches)
        )
    )
    return(authors)

def harmonize_issue_label(df):
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
        .str.split(' ', expand=True)\
        .set_axis(
            ['volume_prefix', 'volume', 'issue_prefix', 'issue', 'year'],
            axis=1
        )
        .assign(year = lambda df_: df_['year'].str[1:5])
    )
    
    df['volume'] = issue_parts['volume'].astype(str)
    df['issue'] = issue_parts['issue'].astype(str)
    df['year'] = issue_parts['year'].astype(str)

    return(df)

def add_topics(articles):
    a = articles.loc[~articles['abstract'].isna(), ['id', 'year', 'abstract']]
    
    stop_words = stopwords.words('english')
    stop_woorden = stopwords.words('dutch')
    
    new_docs = []
    for doc in a['abstract']:
        word_tokens = word_tokenize(doc)
        filtered_words = [
            w for w in word_tokens 
                if not (w.lower() in stop_words) 
                and not (w.lower() in stop_woorden)
        ]
        new_doc = ' '.join(filtered_words)
        new_docs.append(new_doc)
    
    topic_model = BERTopic()
    topics, probs = topic_model.fit_transform(new_docs)
    a['topic_id'] = topics
    a['topic_prob'] = probs
    
    
    
    def topic_labels(df):
        desc = df['Representation']
        if(('migrant' in desc) or ('immigrants' in desc)):
            return 'Migratie en Integratie'
        elif(('unemployment' in desc) or ('workers' in desc) or ('economics' in desc)):
            return 'Economie'
        elif(('stratification' in desc) or ('school' in desc)):
            return 'Stratificatie'
        elif('politic' in desc):
            return 'Politiek'
        elif(('method' in desc) or ('models' in desc)):
            return 'Methoden'
        elif(('church' in desc) or ('religion' in desc)):
            return 'Religie'
        elif(('crime' in desc) or ('deviant' in desc)):
            return 'Criminaliteit'
        elif(('marriage' in desc) or ('household' in desc) or ('family' in desc)):
            return 'Familie'
        elif(('Culture' in desc) or ('cultural' in desc)):
            return 'Cultuur'
        elif(('relation' in desc) or ('network' in desc)):
            return 'Sociale Netwerken'
        else:
            return 'Overig'
    
    topic_order = [
        'Familie', 'Stratificatie', 'Methoden', 'Economie',
        'Migratie en Integratie', 'Religie', 'Criminaliteit',
        'Sociale Netwerken', 'Cultuur', 'Overig'
    ]
    
    topic_data = (
        a.merge(
            topic_model.get_topic_info()[['Topic', 'Representation']], 
            left_on='topic_id', right_on='Topic'
        )
        .assign(
            y = lambda df_: df_['year'].astype(int),
            year = lambda df_: df_['year'].astype(str),
            topic_label = lambda df_: 
                df_.apply(topic_labels, axis=1)
                .pipe(pd.Categorical, categories=topic_order, ordered=True),
            topic_order = lambda df_:
                df_['topic_label'].pipe(pd.factorize)[0]
        )
    )
    
    topic_data.loc[lambda x: x['topic_label'] != 'Overig', 'topic_label'].value_counts()
    
    return (
        articles
        .merge(
            topic_data[['id', 'topic_label', 'topic_order']],
            how='left'
        )
    )

file_path = os.path.join('data', 'scrape_data_combined.csv')

df = (
    pd.read_csv(file_path)
    .sort_values('id')
    .reset_index(drop=True)
    .assign(
        section = lambda df_: df_['section']\
            .mask(lambda x: x == 'artikelen', 'artikel')\
            .mask(lambda x: x == 'boekbesprekingen', 'boekbespreking'),
        url = lambda df_: df_['url']\
            .mask(lambda x: x.isna(), df_['doi_url'])
    )
    .pipe(harmonize_issue_label)
    .pipe(clean_authors)
)

# create an authors dataframe with clean homogenized names
authors = (
    df
    .pipe(parse_names)
    .pipe(add_clean_name)
    .pipe(homogenize_names) # this is somewhat computationally intensive
    .drop(['name', 'name_has_comma'], axis=1)
    .rename({'clean_name': 'name'}, axis=1)
)


out_file = os.path.join('data', 'article_author_link.xlsx')
authors.to_excel(out_file, index=False)


# merge the homogenized names back in to the articles dataframe
df = (
    df
    .drop(['authors_list', 'authors'], axis=1)
    .merge(
        authors
            .groupby('id').agg(list)
            .reset_index()
            .rename({'name': 'authors_list'} ,axis=1)
            .assign(
                 authors = lambda df_: 
                    df_['authors_list'].transform(lambda x: ', '.join(x)),
            ),
        on='id',
        how='left'
    )
    .assign(
        section = lambda df_: df_['section']
            .mask(df_['authors'] == 'Redactie M&M', 'algemeen')
            .mask(
                ~df_['abstract'].isna() & 
                (df_['section'] == 'algemeen'), 
                'artikel'
            ),
        year = lambda df_: df_['year'].mask(lambda x: x == 'None', '1975')
    )
    .pipe(add_topics)
)

cols = [
    'year', 'id', 'issue_label', 'title', 'authors', 'section',
    'abstract', 'eng_title', 'tags', 'topic_label', 'topic_order', 
    'url', 'pdf_url', 
]
out_file = os.path.join('data', 'articles.xlsx')
df.loc[:, cols].to_excel(out_file, index=False)


for ll in range(1925, 2025, 10):
    df['year'] = df['year'].pipe(pd.to_numeric)
    ul = ll + 10

    (
        df
        .loc[
            lambda df_: (df['year'] >= ll) & (df_['year'] < ul), 
            cols
        ]
        .to_excel(os.path.join('decades', f"articles {ll}-{ul}.xlsx"), index=False)
    )






            