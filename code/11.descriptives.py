import os
import re
import pandas as pd
from IPython.display import display, Markdown
import altair as alt

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

from formulaic import Formula
import pymc as pm
import numpy as np
import nutpie
import arviz as az

data = (
    articles
    .loc[lambda df_: (df_['section']=='artikel') & articles_selection]
)

formula = Formula("bs(year, df=7, degree=6)")
B = formula.get_model_matrix(data).drop('Intercept', axis=1)

# PyMC spline regression model
# with pm.Model() as spline_model:
#     # Priors for spline coefficients
#     coef = pm.Normal("coef", mu=0, sigma=10, shape=spline_basis.shape[1])
#     intercept = pm.Normal("intercept", mu=a["page_count"].mean(), sigma=10)
#     sigma = pm.HalfNormal("sigma", sigma=10)

#     # Expected value
#     mu = intercept + pm.math.dot(spline_basis.values, coef)
#     obs = pm.Normal("obs", mu=mu, sigma=sigma, observed=a["page_count"])

COORDS = {"splines": np.arange(B.shape[1])}

with pm.Model(coords=COORDS) as model:
    x = pm.Data('x', value=data.year)
    a = pm.StudentT("a", nu=10, mu=10, sigma=1)
    w = pm.StudentT("w", nu=10, mu=0, sigma=1, dims="splines")
    mu = pm.Deterministic("mu", a + pm.math.dot(B, w.T))
    sigma = pm.HalfStudentT("sigma", nu=5, sigma=1)
    
    alpha = pm.StudentT("alpha", nu=5, mu=0, sigma=1.5)
    mean = pm.Deterministic(
        "mean", mu + sigma * pm.math.sqrt(2/np.pi) 
        * (alpha/pm.math.sqrt(1 + pm.math.sqr(alpha)))
    )

    y = pm.SkewNormal(
        "y", mu=mu, sigma=sigma, alpha=alpha, 
        observed=data.page_count, shape=len(data.year)
    )

compiled_model = nutpie.compile_pymc_model(model)
trace = nutpie.sample(compiled_model)

with model:
    ppc = pm.sample_posterior_predictive(trace, var_names=["y"], random_seed=42)


pred = (
    ppc['posterior_predictive']
    .to_dataframe()
    .unstack(level=[0, 1])
    .assign(x = data.year.values)
    .stack(level=[1, 2])
    .reset_index()
    .assign(x = lambda df_: df_.groupby('y_dim_2')['x'].transform(lambda x: x.max()))
    .dropna()
    .set_index(['y_dim_2', 'chain', 'draw'])\
    .groupby('x')
)


m = pred.mean()
ci89 = pred.apply(compute_hdi_for_group, 0.89)
ci60 = pred.apply(compute_hdi_for_group, 0.60)

pdata = (
    pd.concat([m, ci89, ci60], axis=1)
    .reset_index()
    .assign(x = lambda df_: df_['x'].astype(int).astype('str'))
    .melt(id_vars=['x', 'y'])
    .assign(
        level = lambda df_: 
            df_['variable'].str.split('.').str[-1].map({'6': '60', '89': '89'}),
        variable = lambda df_: df_['variable'].str.split('.').str[0]
    )
    .pivot(index=['x', 'y', 'level'], columns='variable', values='value')
    .reset_index()
)

base = (
    alt.Chart(pdata)
    .encode(
        alt.X('x:T').title('Jaar'),
        alt.Y('y:Q').title("Aantal Pagina's"),
    )
)

line = (
    base
    .mark_line()
    .transform_filter(
        alt.FieldEqualPredicate(field='level', equal='89')
    )
)

ci89 = (
    base
    .mark_area(opacity=0.6)
    .encode(
        alt.Y('ll:Q'),
        alt.Y2('ul:Q')
    )
    .transform_filter(
        alt.FieldEqualPredicate(field='level', equal='89')
    )
)

ci60 = (
    base
    .mark_area(opacity=0.6)
    .encode(
        alt.Y('ll:Q'),
        alt.Y2('ul:Q')
    )
    .transform_filter(
        alt.FieldEqualPredicate(field='level', equal='60')
    )
)

plot = ci60 + ci89 + line

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
    .agg(['mean', 'std'])
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

