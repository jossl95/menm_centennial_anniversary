"""Module topic modelling on cleaned scraped data

This module provides functionality to further clean the scraped
data and performance topic modelling on these data.
"""

from typing import List, Optional

import os
import sys
import pandas as pd
import numpy as np

from pathlib import Path
from multiprocess import Pool, cpu_count
from mlx_lm import load, generate
from contextlib import contextmanager
from tqdm import tqdm

BASEDIR = Path('data') / 'text-images' 
# os.environ["TOKENIZERS_PARALLELISM"] = "true"
# model, tokenizer = load("mlx-community/gemma-3-4b-it-8bit")

def read_text(row):
    text_dir = BASEDIR / str(row['year']) / row['id']
    file = text_dir / "text.md"
    
    doc = None
    if file.exists():
        # read in text
        with open(file) as f:
            doc = f.read()

    return(doc)

@contextmanager
def suppress_output():
    """Suppress all console output within this block."""
    with open(os.devnull, "w") as devnull:
        old_out, old_err = sys.stdout, sys.stderr
        sys.stdout, sys.stderr = devnull, devnull
        try:
            yield
        finally:
            sys.stdout, sys.stderr = old_out, old_err

class ArticleSummarizer:
    """ A class to summarize articles using a language model. """

    def __init__(
        self,
        model_id: str = "mlx-community/gemma-3-4b-it-8bit",
        max_tokens_abstract: int = 400,
        temperature: float = 0.0,
        top_p: float = 0.95
    ) -> None:
        """Initialize with a quantized Mistral-7B-Instruct model.

        Args:
            model_id: Pretrained MLX-LM model identifier.
            max_tokens_abstract: Max tokens for abstract generation.
            temperature: Sampling temperature.
            top_p: Nucleus sampling probability.
        """
        with suppress_output():
            self.model, self.tokenizer = load(model_id)
        self._gen_kwargs = {
            "temperature": temperature,
            "top_p": top_p,
            "verbose": False,
        }
        self._max_tokens_abstract = max_tokens_abstract

        # System prompts for chat API
        self._system_abstract = (
            "Je bent een sociaal wetenschappelijk assistent die gespecialiseerd is "
            "in het genereren van academische abstracts. Wanneer je de tekst van een "
            "artikel ontvangt, produceer je een beknopt, accuraat abstract in "
            "formeel academisch Nederlands. Volg de gebruikelijke sociologische "
            "structuur: Achtergrond; Onderzoeksdoel of -vraag; Data en "
            "Methodologie; Resultaten; Conclusie. Als de tekst een abstract of summary "
            "bevat, vertaal deze dan naar het Nederlands. "   
        )

    def generate_abstract(self, text:str) -> str:
        """Generate an abstract for the given text.
        Args:
            text: The text to summarize.
        Returns:
            The generated abstract.
        """

        # Define the user prompt
        abstract_message = (
            "Maak een abstract van het onderstaande artikel volgens deze richtlijnen:\n"
            "1. Structuur: Achtergrond; Onderzoeksdoel of -vraag; Data en "
            "Methodologie; Resultaten; Conclusie\n"
            "2. Maximaal 300 woorden\n"
            "3. Alleen spaties, geen tabs geen nieuwe regels\n"
            "4. Taal: Nederlands\n"
            "5. Geef als antwoord enkel en alleen de abstract, zonder enige toelichting of toevoeging\n\n"
            "6. Geef een platte tekst zonder Markdown opmaak\n"
            "Artikel (Markdown):\n```markdown\n"
            f"{text}\n""```"
        )

        messages = [
            {
                "role": "system",
                "content": [{"type": "text", "text": self._system_abstract}]
            },
            {
                "role": "user",
                "content": [{"type": "text", "text": abstract_message}]
            }
        ]

        prompt = self.tokenizer.apply_chat_template(
            messages, add_generation_prompt=True
        )

        result = generate(
            self.model,
            self.tokenizer,
            prompt=prompt,
            max_tokens=self._max_tokens_abstract
        )

        return result.strip()

def task(row: pd.Series) -> pd.Series:
    
    model = ArticleSummarizer()
    row['abstract_llm'] = model.generate_abstract(text)

    return row

articles_file = Path('data') / 'articles_metadata.xlsx'
data = (
    pd.read_excel(articles_file)
    .loc[lambda df_: df_['section'] == 'artikel']
)


hold = []
for _, row in tqdm(data.iterrows(), total=len(data)):
    if row.page_count < 5:
        # skip too‐short pubs entirely
        continue   

    text = read_text(row)
    if len(text) < 200:
        continue
    
    row = task(row)
    hold.append(row)


data = pd.concat(hold, axis=1).T






"""
ARCHIVE: 

# def generate_abstract(text: str) -> str:

#     messages = [
#         {"role": "system", 
#          "content": (
#                 "You are a scholarly abstract generator specializing in sociological "
#                 "research. When given an article in Markdown, you produce a concise, "
#                 "accurate abstract in formal academic Dutch. "
#                 "Follow the conventional sociological structure: "
#                 "Background; Objective; Methods; Results; Conclusion."
#             )
#         },
#         {"role": "user", 
#          "content": (
#                 "Maak een abstract van het onderstaande artikel volgens deze richtlijnen:\n"
#                 "1. Structuur: Background; Objective; Methods; Results; Conclusion\n"
#                 "2. Maximaal 400 woorden\n"
#                 "3. Enkel spaties, geen regeleinden\n"
#                 "4. Taal: Nederlands\n"
#                 "5. Geef alleen de abstract, zonder verdere toelichting\n\n"
#                 "Artikel (Markdown):\n"
#                 "```markdown\n"
#                 f"{text}\n"
#                 "```"
#             )
#         }
#     ]
    
#     # initialize prompt
#     prompt = tokenizer.apply_chat_template(
#         messages, 
#         add_generation_prompt=True, 
#         enable_thinking=True
#     )

#     # generate text
#     text = generate(
#         model,
#         tokenizer,
#         prompt=prompt,
#         max_tokens=1200,
#         verbose=False
#     )

#     # remove thoughts
#     text = re.sub(r"<think>(?:.|\n)*<\/think>", '', text).strip()
#     return text

def generate_discipline(text: str) -> str:

    disciplines = [
        "Filosofie",
        "Geschiedenis",
        "Culturele Antropologie",
        "Psychologie en/of Sociale Psychologie",
        "Fysische Antropologie en/of Eugenetica",
        "Rechtsgeleerdheid",
        "Criminologie",
        "Sociale Geografie en/of Sociografie",
        "Demografie",
        "Economie",
        "Politicologie",
        "Sociologie",
        "Methodologie en Onderzoekstechnieken"
    ]
    
    messages = [
        {
            "role": "system",
            "content": (
                "Je bent een classificatie-assistent die academische artikelen toewijst "
                "aan precies één van deze disciplines: "
                + "; ".join(disciplines) +
                "."
            )
        },
        {
            "role": "user",
            "content": (
                "Classificeer het onderstaande artikel. "
                "Geef **alleen** de naam van de discipline, zonder toelichting.\n\n"
                "Artikel (Markdown):\n"
                "```markdown\n"
                f"{text}\n"
                "```"
            )
        }
    ]
    
    # initialize prompt
    prompt = tokenizer.apply_chat_template(
        messages, 
        add_generation_prompt=True, 
        enable_thinking=False
    )

    # generate text
    text = generate(
        model,
        tokenizer,
        prompt=prompt,
        verbose=False
    )

    return text

def task(row):
    text = read_text(row)
    row['discipline'] = generate_discipline(text)
    # row['abstract'] = generate_abstract(text)
    return row[['id', 'title', 'abstract', 'tags', 'discipline']]

def generate_abstracts_with_workers(df: pd.DataFrame, num_workers: Optional[int] = None) -> None:
    if num_workers is None:
        num_workers = cpu_count()
    
    with Pool(processes=num_workers) as pool:
        rows = list(tqdm(
            pool.imap(read_text, [row for _, row in df.iterrows()]),
            total=df.shape[0]
        ))

    return(df)
"""