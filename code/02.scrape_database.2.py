from bs4 import BeautifulSoup
import requests
import pandas as pd

class TrialContextManager:
    """Custom context manager to handle exceptions without interrupting execution."""
    def __enter__(self):
        pass
    
    def __exit__(self, *args):
        return True

trial = TrialContextManager()

BASE_URL = 'https://www.aup.nl/en/journal/mens-en-maatschappij/back-issues'
ARCHIVE_URL = 'https://journal-archive.aup.nl'

def get_article_urls(base_url):
    """Fetches article URLs from the base page."""
    response = requests.get(base_url)
    soup = BeautifulSoup(response.content, 'lxml')
    
    # Find all links that contain 'journal-downloads'
    return [a for a in soup.find_all("a") if 'journal-downloads' in a.get('href', '')]

def urls_to_df(article_urls):
    """Converts extracted article URLs into a DataFrame."""
    data = []
    
    for article in article_urls:
        href = ARCHIVE_URL + article.get('href').replace('/journal-downloads', '')
        try:
            issue, title = article.text.split(' - ', 1)
        except ValueError:
            continue  # Skip malformed entries
        
        issue = issue.replace('no', 'No')
        data.append([href, issue, title])
    
    return pd.DataFrame(data, columns=['pdf_url', 'issue', 'title'])

def add_section(df):
    """Assigns sections to articles based on their titles."""
    df['section'] = 'artikel'
    
    df.loc[df['title'].str.contains('Boekbespreking', na=False), 'section'] = 'boekbespreking'
    df.loc[df['title'].str.contains(
        'Rectificatie|In memoriam|Ontvangen publicaties|Van Doorns Indische lessen', na=False), 'section'] = 'algemeen'
    
    return df

def add_id_and_year(df):
    """Generates unique article IDs and extracts publication years."""
    df['year'] = df['issue'].str.extract('(\d+)').astype(float) + 1925
    df['year'] = df['year'].fillna(method='ffill').astype(int)  # Forward fill missing years
    df['issue'] = df['issue'] + ' (' + df['year'].astype(str) + ')'
    
    df['id'] = 'article-' + df.groupby('issue').cumcount().add(1).astype(str).str.zfill(2)
    
    return df

def main():
    """Main execution function."""
    article_urls = get_article_urls(BASE_URL)
    df = urls_to_df(article_urls)
    df = df.pipe(add_section).pipe(add_id_and_year).assign(database=2)
    df.to_csv('scraped_data2.csv', index=False)
    print("Scraping complete. Data saved to 'scraped_data2.csv'")

if __name__ == "__main__":
    main()
