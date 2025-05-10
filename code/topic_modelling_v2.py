"""Module topic modelling on cleaned scraped data

This module provides functionality to further clean the scraped
data and performance topic modelling on these data.
"""

from typing import List, Optional

import os
import pandas as pd
from pathlib import Path

from bertopic import BERTopic
from IPython.display import display, Markdown
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from multiprocess import Pool, cpu_count
from tqdm.notebook import tqdm

BASEDIR = os.path.join('data', 'text-images')  # Use hyphen instead of space

def read_text(row):
    text_dir = Path(os.path.join(BASEDIR, str(row['year']), row['id']))
    file = text_dir / "clean_text.md"
    
    # Preprocess abstracts
    manual_stop_words = [
        'den', 'wel', 'wij', 'echter', 'ten', 'zeer', 'slechts', 'welke', 'alle', 'mensch',
        'tussen', 'sociale', 'onderzoek', 'aantal', 'joden', 'komt', 'menschen', 'model',
        'mensen', 'vrouwen', 'social', 'kinderen', 'effect', 'minder', 'nederland', 'tabel'
    ]
    stop_words = set(stopwords.words('english') + stopwords.words('dutch') + manual_stop_words)

    if file.exists():
        # read in text
        with open(file) as f:
            doc = f.read()
        
            # exclude stopwords
            processed_doc = ' '.join(
                word.lower() for word in word_tokenize(doc)
                if word.lower() not in stop_words
            )
    else:
        processed_doc = None

    return(processed_doc)

def read_texts_with_workers(df: pd.DataFrame, num_workers: Optional[int] = None) -> None:
    """Run text extraction concurrently using multiple CPU cores.

    Args:
        df: DataFrame containing article information.
        num_workers: Number of worker processes to use. Defaults to CPU count.
    """
    if num_workers is None:
        num_workers = cpu_count()
    
    with Pool(processes=num_workers) as pool:
        texts = list(tqdm(
            pool.imap(read_text, [row for _, row in df.iterrows()]),
            total=df.shape[0]
        ))

    df['text'] = texts
    return(df)

articles_file = os.path.join('data', 'articles.xlsx')
data = (
    pd.read_excel(articles_file)
    .loc[lambda df_: df_['section'] == 'artikel']
    .assign(text = None)
    .pipe(read_texts_with_workers)
    .dropna(subset='text')
)

seed_topic_list = [
    'Familie', 'Stratificatie', 'Methoden', 'Economie',
    'Migratie en Integratie', 'Religie', 'Criminaliteit',
    'Sociale Netwerken', 'Cultuur', 'Overig'
]


# Generate topics
topic_model = BERTopic(seed_topic_list=seed_topic_list)
topics, probs = topic_model.fit_transform(data['text'])
# abstracts['topic_id'] = topics
# abstracts['topic_prob'] = probs



