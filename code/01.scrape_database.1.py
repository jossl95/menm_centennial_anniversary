from bs4 import BeautifulSoup
import requests
import httpx
import pandas as pd
from tqdm.notebook import tqdm

class TrialContextManager:
    """Custom context manager to handle exceptions without interrupting execution."""
    def __enter__(self):
        pass
    
    def __exit__(self, *args):
        return True

trial = TrialContextManager()

def parse_items(items, section='algemeen'):
    """Parses article items from HTML elements."""
    articles = []
    
    for i, item in enumerate(items):
        if i % 2 == 0:  # Process every second item
            title_element = item.find("div", class_="title")
            if not title_element:
                continue
            
            link = title_element.find('a')
            article_id = link.get('id')
            url = link.get('href')
            with trial:
                title = link.string.strip()
            
            authors = item.find("div", class_="authors").string.strip().split('\t')
            pages = None if section == 'algemeen' else item.find("div", class_="pages").string.strip()
            
            articles.append(pd.DataFrame({
                'id': article_id, 'title': title, 'authors': authors, 'pages': pages, 'url': url
            }, index=[0]))
    
    return pd.concat(articles, ignore_index=True).assign(section=section) if articles else pd.DataFrame()

def scrape_article(url):
    """Scrapes article details from the given URL."""
    response = requests.get(url)
    soup = BeautifulSoup(response.content, 'lxml')
    
    articles = []
    for section in soup.find_all("div", class_="section"):
        section_title = section.h2.get_text(strip=True).lower()
        elements = section.find_all('ul', class_="cmp_article_list articles")
        
        for element in elements:
            with trial: articles.append(parse_items(element.find_all('li'), section=section_title))
    
    return pd.concat(articles, ignore_index=True) if articles else pd.DataFrame()

def scrape_archive(url):
    """Scrapes archive page for issue titles and URLs."""
    response = httpx.get(url)
    soup = BeautifulSoup(response.content, 'lxml')
    
    listings = soup.find_all("a", class_="title")
    return pd.DataFrame({
        'issue': [listing.get_text(strip=True) for listing in listings],
        'url': [listing.get('href') for listing in listings]
    })

def parse_text(soup, tag, class_name, replace_text=None):
    """Extracts and cleans text from a specific tag in BeautifulSoup object."""
    section = soup.find(tag, class_=class_name)
    if section:
        text = section.get_text(strip=True).replace('\n', '').replace('\t', '')
        if replace_text:
            text = text.replace(replace_text, '')
        return text
    return None

def scrape_article_details(df):
    """Scrapes additional details (abstract, tags, date) for each article in DataFrame."""
    for index, row in tqdm(df.iterrows(), total=df.shape[0]):
        response = httpx.get(row['url'])
        soup = BeautifulSoup(response.content, 'lxml')
        df.at[index, 'abstract'] = parse_text(soup, 'section', 'item abstract', 'Samenvatting')
        df.at[index, 'tags'] = parse_text(soup, 'section', 'item keywords', 'Trefwoorden:')
        df.at[index, 'date'] = parse_text(soup, 'div', 'item published', 'Gepubliceerd')
    
    return df

def get_pdf_url(row):
    """Extracts PDF URL from an article page."""
    response = httpx.get(row['url'])
    if response.status_code != 200:
        response = requests.get(row['url'])
    
    soup = BeautifulSoup(response.content, 'lxml')
    pdf_link = soup.find('a', class_='obj_galley_link pdf')
    
    if pdf_link:
        response = httpx.get(pdf_link.get('href'))
        if response.status_code != 200:
            response = requests.get(pdf_link.get('href'))
        
        soup = BeautifulSoup(response.content, 'lxml')
        download_link = soup.find('a', class_='download')
        return download_link.get('href') if download_link else None
    
    return None

def main():
    base_url = "https://ugp.rug.nl/MenM/issue/archive"
    hold = []
    
    # Scrape archive pages
    for i in range(1, 17):
        url = base_url if i == 1 else f"{base_url}/{i}"
        hold.append(scrape_archive(url))
    
    archive_df = pd.concat(hold, ignore_index=True)
    
    # Scrape articles per issue
    hold = []
    for _, row in tqdm(archive_df.iterrows(), total=archive_df.shape[0]):
        hold.append(scrape_article(row['url']).assign(issue=row['issue']))
    
    articles_df = pd.concat(hold, ignore_index=True)
    
    # Scrape additional details
    articles_df = scrape_article_details(articles_df)
    
    # Retrieve PDF URLs
    for i, row in tqdm(articles_df.iterrows(), total=articles_df.shape[0]):
        articles_df.at[i, 'pdf_url'] = get_pdf_url(row)
    
    articles_df['database'] = 1
    articles_df.to_csv('scraped_data1.csv', index=False)
    print("Scraping complete. Data saved to 'scraped_data.csv'")

if __name__ == "__main__":
    main()


