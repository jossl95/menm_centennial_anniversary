""" Module for generating abstracts for all articles.

    This module reads article metadata from an Excel file, summarizes each
    article's text using Google's Gemini Pro model, and saves the summaries
    to Markdown files. It requires the Google Generative AI library and a
    valid API key.
"""

from typing import Text, Optional

import os
import getpass
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import google.generativeai as genai

# Set global variables
BASEDIR = Path('data') / 'text-images'
ENVVAR = "GOOGLE_API_KEY"
OVERWRITE = False

def set_api_key(envvar: Optional[Text] = None) -> None:
    """Sets the API key as an environment variable.
    Attempts to get the key from an existing environment variable.
    If not found, prompts the user securely using getpass.
    """
    envvar = envvar or ENVVAR
    api_key = getpass.getpass(f"Please enter your {envvar}: ")
    os.environ[envvar] = api_key

def get_api_key(envvar: Optional[Text] = None) -> Optional[Text]:
    """Retrieves the API key from environment variables.

    Returns:
    The API key string if found, otherwise None.
    """
    envvar = envvar or ENVVAR
    return os.environ.get(envvar)


class ArticleSummarizer:
    """A class to summarize articles using Google's Gemini Pro model."""

    def __init__(
        self,
        model: Text = "gemini-1.5-flash",
        envvar: Optional[Text] = None,
    ):
        """Initializes the ArticleSummarizer with a specified Gemini model.

        Args:
            model_name: The name of the Gemini model to use.
            temperature: The sampling temperature for the model.
        """
        self.model = model,
        self.envvar = envvar or ENVVAR
        self.configure_api_key()
        self._llm = genai.GenerativeModel(model_name=model)
 
    def configure_api_key(self, overwrite: bool = False) -> Optional[Text]:
        """Configures the API key for authentication.

        Args:
            overwrite: If True, forces re-entry of the API key even if it
                already exists in the environment.

        Returns:
            The API key string if successfully configured, otherwise None.
        """
        _key_not_configured = self.envvar not in os.environ
        if overwrite or _key_not_configured:
            set_api_key(self.envvar)

        api_key = get_api_key(self.envvar)
        genai.configure(api_key=api_key)
        return api_key
    
    def set_instructions(self, instructions: Optional[Text] = None) -> Text:
        """Sets the system instructions for the summarization model."""
        if not instructions: 
            instructions = """
            Jij bent een deskundige academische schrijver van abstracts. Jouw 
            taak is om een beknopte, informatieve en goed gestructureerde 
            abstract te genereren voor wetenschappelijke artikelen.
    
            Vereisten voor de abstract:
            1. **Doel:** De abstract moet duidelijk het hoofddoel, de 
               doelstelling of de onderzoeksvraag van het document vermelden.
            2. **Methoden (indien van toepassing):** Beschrijf kort de gebruikte
               methodologie, benadering of belangrijkste technieken.
            3. **Belangrijkste Bevindingen/Resultaten:** Vat de belangrijkste
               bevindingen, ontdekkingen of uitkomsten samen.
            4. **Conclusie/Implicaties:** Vermeld de belangrijkste conclusie(s)
            5. **Beknoptheid:** De abstract moet tussen de 200 en 300 woorden
               zijn. Geef prioriteit aanduidelijkheid.
            6. **Op Zichzelf Staand:** De abstract moet volledig begrijpelijk
               zijn zonder het hele document te hoeven lezen.
            7. **Toon:** Gebruik een objectieve, formele en academische toon.
               Schrijf de abstract in het Nederlands
            8. **Trefwoorden (Optioneel maar Aanbevolen):** Stel 3-5 relevante
               trefwoorden voor die deessentie van het document weergeven, 
               gescheiden door komma's, aan het einde van de abstract.
            
            Interne stappen (uit te voeren vóór generatie):
            * Identificeer het kernargument of het centrale thema van het
              document.
            * Extraheer de primaire doelstelling en het belangrijkste probleem
              dat wordt behandeld.
            * Identificeer de gebruikte onderzoeksmethoden en technieken
              (indien relevant).
            * Bepaal de meest opvallende resultaten en bewijsmateriaal.
            * Formuleer de overkoepelende conclusie en de betekenis ervan.
            * Synthetiseer deze informatie tot een samenhangend verhaal dat
              logisch stroomt.
            
            Genereer de abstract nu, strikt vasthoudend aan het aantal woorden
            en de inhoudsvereisten.
            """.replace("\n            ", "\n")
        
        return instructions

    def summarize_articles(self, text_file, out_file, verbose=False) -> None:
        """Summarizes the article text and saves it to a file.
        Args:
            row: A pandas Series containing article metadata.
        """
        instructions = self.set_instructions() 
        
        if verbose:
            with open(text_file, "r") as f:
                text = f.read()
            prompt = [instructions, text]
        else:
            prompt = [instructions, genai.upload_file(text_file)]
        
        
        response = self._llm.generate_content(prompt)

        if response:
            with open(out_file, "w") as f:
                f.write(response.text.strip())

def main():
    """Reads article metadata, summarizes each Markdown text file, and saves the summaries.

    Raises:
        FileNotFoundError: If the metadata file is not found.
    """
    articles_file = Path('data') / 'articles_metadata.xlsx'
    data = (
        pd.read_excel(articles_file)
        .loc[lambda df_: df_['section'] == 'artikel']
        .loc[lambda df_: df_['page_count'] > 5]
    )

    for _, row in tqdm(data.iterrows(), total=len(data)):
        text_dir = BASEDIR / str(row["year"]) / str(row["id"])
        text_file = text_dir / "clean_text.md"
        out_file = text_dir / "summary.md"
        
        if not text_file.exists():
            continue

        if out_file.exists():
            continue

        try:
            summarizer = ArticleSummarizer()
            summarizer.summarize_articles(text_file, out_file)
        except Exception as e:
            print(f"Error summarizing text for {str(row["id"])}")
            summarizer = ArticleSummarizer()
            summarizer.summarize_articles(text_file, out_file, verbose=True)
            continue
    
if __name__ == "__main__":
    main()
