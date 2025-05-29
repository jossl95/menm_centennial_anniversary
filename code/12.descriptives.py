import os
import re
import pandas as pd
import numpy as np
from IPython.display import display, Markdown
from tqdm.notebook import tqdm
import altair as alt
import arviz as az

import statsmodels.api as sm
lowess = sm.nonparametric.lowess

@alt.theme.register("menm_theme", enable=True)
def menm_theme() -> alt.theme.ThemeConfig:
    theme = {
        "config": {
            
            "view": {
                "continuousWidth": 600, 
                "continuousHeight": 300},
            
            "axis": {
                "labelFontSize": 11,
                "titleFontSize": 11,
            },
            
            "legend": {
                "labelFontSize": 11,
                "titleFontSize": 11,
            },
            
            "title": {
                "fontSize": 14,
            },
            
            "range": {
               "category": [
                    "#DDDDDD",  # very light gray
                    "#111111",  # very dark gray
                    "#C8C8C8",  # added lighter-medium gray
                    "#222222",  # added very dark-medium gray
                    "#BBBBBB",  # lighter gray
                    "#333333",  # darker-medium gray
                    "#A0A0A0",  # added medium-light gray
                    "#4D4D4D",  # dark-medium gray
                    "#888888",  # medium-light gray
                    "#707070",  # medium gray
                    
                ],
                "ordinal": {"scheme": "greys"},
                "heatmap": {"scheme": "greys"},
                "ramp": {"scheme": "greys"}
            }
        }
    }
    
    return (theme)

def compute_hdi_for_group(group, hdi_prob=0.95):
    results = {}
    # Loop over each column in the group, skipping the grouping column
    for col in group.columns:
        if col != "x":
            lower, upper = az.hdi(group[col].values, hdi_prob=hdi_prob)
            results[f"ll.{str(hdi_prob)[-2:]}"] = lower
            results[f"ul.{str(hdi_prob)[-2:]}"] = upper
    return pd.Series(results)


def save_plot(plot, title):
    plot.save(f'figures/{title}.pdf')
    plot.save(f'figures/{title}.svg')

articles_file = os.path.join('data', 'articles.xlsx')
articles = pd.read_excel(articles_file)
articles_selection = (articles['page_count'] > 3) & (articles['page_count'] < 75)

authors_file = os.path.join('data', 'article_author_link.xlsx')
authors = pd.read_excel(authors_file)

# article counts ------------------------------------------------------------------

article_counts = (
    articles
    .assign(year = lambda df_: df_['year'].astype(str))
    .groupby(['year', 'section']).size()
    .reset_index()
    .rename({0: 'n'}, axis=1)
)

plot = (
    alt.Chart(article_counts)
    .mark_bar(size=7)
    .encode(
        alt.X('year:T').title('Jaar'),
        alt.Y('n:Q').title('Aantal Publicaties').stack(True),
        alt.Color('section:O').title('Type Publicaties').legend(orient='bottom')
    )
    .configure_scale(barBandPaddingInner=0)
)
save_plot(plot, 'article_counts')

(
    articles
    .assign(
        decade = lambda df_: df_['year']
            .astype(int)
            .pipe(
                pd.cut, 
                bins=[y for y in range(1935, 2035, 10)]
            ),
        year = lambda df_: df_['year'].astype(str)
    )
    .groupby(['decade','section'], observed=False).size()
    .unstack(level=1)
    .to_excel(os.path.join('tables', "section_counts.xlsx"))
)

# page counts ------------------------------------------------------------------


data = (
    articles
    .loc[lambda df_: (df_['section']=='artikel') & articles_selection]
)

x = data['year'].values
y = data['page_count'].values
x_grid = np.linspace(x.min(), x.max(), len(np.unique(x)))
n_bootstraps = 10000
predictions = np.zeros((n_bootstraps, len(x_grid)))
frac = 0.1

for i in tqdm(range(n_bootstraps), total = n_bootstraps):
    data_sample = data.sample(n=len(data), replace=True)
    loess_fit = lowess(data_sample['page_count'], data_sample['year'], frac=frac)
    predictions[i] = np.interp(x_grid, loess_fit[:, 0], loess_fit[:, 1])

pred_data = pd.DataFrame(predictions, columns=np.int64(x_grid))

# Compute summary statistics for each grid point.
pdata = (
    pd.DataFrame({
        'year': x_grid,
        'mean': pred_data.mean(axis=0).values,
        'll': az.hdi(pred_data.values, hdi_prob=0.95).T[0],
        'ul': az.hdi(pred_data.values, hdi_prob=0.95).T[1]
    })
    .assign(year = lambda df_: df_['year'].astype(int).astype(str))
)

base = (
    alt.Chart(pdata)
    .encode(
        alt.X('year:T').title('Jaar'),
        alt.Y('mean:Q').title("Aantal Pagina's"),
    )
)

line = base.mark_line()

ci = (
    base
    .mark_area(opacity=0.6)
    .encode(
        alt.Y('ll:Q'),
        alt.Y2('ul:Q')
    )
)

plot = ci + line
save_plot(plot, 'page_counts')

(
    articles
    .loc[lambda df_: (df_['section']=='artikel') & articles_selection]
    .assign(
        decade = lambda df_: df_['year']
            .astype(int)
            .pipe(
                pd.cut, 
                bins=[y for y in range(1935, 2035, 10)]
            ),
        year = lambda df_: df_['year'].astype(str)
    )
    .groupby('decade', observed=False)['page_count']
    .agg(['size', 'mean', 'std'])
    .to_excel(os.path.join('tables', "page_counts.xlsx"))
)

# author counts ------------------------------------------------------------------

author_counts = (
    articles
    .loc[lambda df_: df_['id'].isin(authors['id'].unique())]
    .assign(
        year = lambda df_: df_['year'].astype(str),
        n_authors = lambda df_: df_['authors'].str.split(', ').apply(len)
    )
    .groupby(['year', 'n_authors']).size()
    .reset_index()
    .rename({0: 'n'}, axis=1)
)

plot = (
    alt.Chart(author_counts)
    .mark_bar(size=7)
    .encode(
        alt.X('year:T').title('Jaar'),
        alt.Y('n:Q').title('Aantal Publicaties').stack('normalize'),
        alt.Color('n_authors:O').title('Aantal Auteurs').legend(orient='bottom')
    )
)

save_plot(plot, 'author_counts')

(
    articles
    .loc[lambda df_: df_['id'].isin(authors['id'].unique())]
    .assign(
        decade = lambda df_: df_['year']
            .astype(int)
            .pipe(
                pd.cut, 
                bins=[y for y in range(1935, 2035, 10)]
            ),
        year = lambda df_: df_['year'].astype(str),
        n_authors = lambda df_: df_['authors'].str.split(', ').apply(len)
    )
    .groupby(['decade','n_authors'], observed=False).size()
    .unstack(level=1)
    .to_excel(os.path.join('tables', "author_counts.xlsx"))
)

# most published counts -----------------------------------------------------------

most_published_authors = authors['name'].value_counts().head(7)
hold = []
for author in most_published_authors.index:
    cumcount_author = (
        articles
        .assign(year = lambda df_: df_['year'].astype(str))
        .loc[lambda df_: df_['id'].isin(authors['id'].unique())]
        .loc[lambda df_: df_['authors'].str.contains(author)]
        .groupby('year').size().cumsum()\
        .reset_index()
        .rename({0: 'n'}, axis=1)
        .assign(name=author)
    )

    hold.append(cumcount_author)

plot = (
    alt.Chart(pd.concat(hold))
    .mark_line(interpolate='step-after')
    .encode(
        alt.X('year:T').title('Jaar'),
        alt.Y('n:Q').title('Aantal Publicaties'),
        alt.Color('name:O').title('Auteurs').legend(orient='bottom', columns=5)
    )
)

save_plot(plot, 'most_published_authors')

most_published_authors.to_excel(os.path.join('tables', "most_published_authors.xlsx"))

# topic_counts -----------------------------------------------------------

pdata=articles[~articles['topic_label'].isna()]

plot = (
    alt.Chart(pdata)
    .transform_density(
        'year',
        as_=['year', 'density'],
        groupby=['topic_label'],
        extent= [1925, 2025],
        # counts = True,
        resolve='independent',
        steps=200
    ).mark_area(opacity=0.7).encode(
        alt.X('year:Q')
            .axis(format='i')
            .scale(domain=[1925, 2025])
            .title('Jaar'),
        alt.Y('density:Q')
            .stack('normalize')
            .title('Prevalentie'),
        alt.Color('topic_label:N')
            .legend(orient='bottom', columns=5)
            .title('Onderwerpen')
    )
)

save_plot(plot, 'topic_prevalence')

