import os
import re
import altair as alt
import pandas as pd
import numpy as np
import arviz as az

from pathlib import Path
from typing import Optional
from multiprocess import Pool, cpu_count

import statsmodels.api as sm
from tqdm.notebook import tqdm

lowess = sm.nonparametric.lowess
alt.data_transformers.enable("vegafusion")

# Constants
BASEDIR = Path("data/text-images")
FIGDIR = Path("figures")

# Register Altair theme
@alt.theme.register("menm_theme", enable=True)
def menm_theme() -> alt.theme.ThemeConfig:
    theme = {
        "config": {
            "view": {
                "continuousWidth": 650,
                "continuousHeight": 350
            },
            "axis": {
                "labelFontSize": 11,
                "titleFontSize": 11
            },
            "legend": {
                "labelFontSize": 11,
                "titleFontSize": 11
            },
            "title": {"fontSize": 14},
        }
    }
    return theme


def save_plot(plot, title):
    plot.save(f"figures/{title}.pdf")
    plot.save(f"figures/{title}.svg")



def bootstrap_lowess(data, var='page_count', frac=0.1, resolution=1):
    x = data["year"].values
    y = data[var].values
    k = len(np.unique(x)) * resolution - resolution if resolution != 1 else len(np.unique(x))
    
    x_grid = np.linspace(x.min(), x.max(), k)
    n_bootstraps = 10000
    predictions = np.zeros((n_bootstraps, len(x_grid)))
    
    for i in tqdm(range(n_bootstraps), total=n_bootstraps):
        data_sample = data.sample(n=len(data), replace=True)
        loess_fit = lowess(
            data_sample[var], data_sample["year"], frac=frac
        )
        predictions[i] = np.interp(x_grid, loess_fit[:, 0], loess_fit[:, 1])
    
    pred_data = pd.DataFrame(predictions, columns=x_grid)
    
    pdata = pd.DataFrame({
        "year": x_grid,
        "mean": pred_data.mean(axis=0).values,
        "ll": az.hdi(pred_data.values, hdi_prob=0.95).T[0],
        "ul": az.hdi(pred_data.values, hdi_prob=0.95).T[1],
    }).assign(
        dt = lambda df_: fractional_year_to_datetime(df_['year'])
    )

    return pdata

def fractional_year_to_datetime(fy):
    year = fy // 1
    remainder = fy - year
    start_of_year = pd.to_datetime(year, format='%Y')
    next_year = pd.to_datetime(year+1, format='%Y')
    delta = next_year - start_of_year
    return start_of_year + remainder * delta

def read_method(row) -> Optional[str]:
    """Read the summary markdown file for a given article row."""
    method_path = BASEDIR / str(row["year"]) / row["id"] / "method.md"
    if method_path.exists():
        return method_path.read_text(encoding="utf-8")
    return None


def read_method_parallel(
    data: pd.DataFrame, num_workers: Optional[int] = None
) -> pd.DataFrame:
    """Read all summaries in parallel using multiprocessing."""
    num_workers = num_workers or cpu_count()
    with Pool(processes=num_workers) as pool:
        methods = list(
            tqdm(
                pool.imap(read_method, [row for _, row in data.iterrows()]),
                total=len(data)
            )
        )
    data["method"] = methods
    return data
    

data = pd.read_excel(os.path.join('data', 'articles.xlsx'))
data2 = pd.read_excel(os.path.join('data', 'articles_metadata.xlsx'))
authors = pd.read_excel(os.path.join("data", "article_author_link.xlsx"))

selection = lambda d: (
    (d["page_count"] > 3) & (d["page_count"] < 75)
)


# Article counts ---------------------------------------------------------

pdata = (
    data
    .assign(
        year = lambda df_: df_['year'].astype(str),
        dt = lambda df_: pd.to_datetime(df_['year'], format="%Y")
    )
    .groupby(["dt", "section"]).size()
    .reset_index()
    .rename({0: "n"}, axis=1)
)



plot = (
    alt.Chart(pdata)
    .mark_area(interpolate="step-after")
    .encode(
        alt.X("dt:T").title("Jaar"),
        alt.Y("n:Q").title("Aantal Publicaties").stack(True),
        alt.Color("section:O")
            .title("Type Publicaties")
            .legend(orient="bottom")
    )
)

save_plot(plot, "article_counts")

(
    data.loc[
        lambda df_: (df_["section"] == "artikel") & selection(df_)
    ]
    .assign(
        decade=lambda df_: df_["year"].astype(int).pipe(
            pd.cut, bins=[y for y in range(1935, 2035, 10)]
        ),
        year=lambda df_: df_["year"].astype(str)
    )
    .groupby(["decade", "section"], observed=False).size()
    .unstack(level=1)
    .to_excel(os.path.join("tables", "section_counts.xlsx"))
)


pdata = bootstrap_lowess(data2, resolution=1, frac=0.4)
base = alt.Chart(pdata).encode(
    alt.X("dt:T").title("Jaar"),
    alt.Y("mean:Q").title("Aantal Pagina's")
)

line = base.mark_line(interpolate="basis")
ci = base.mark_area(opacity=0.6, interpolate="basis").encode(
    alt.Y("ll:Q"),
    alt.Y2("ul:Q")
)

plot = ci + line
save_plot(plot, "page_counts")


#--
topic_counts = (
    data2
    .groupby(["year", "topic_label"]).size()
    .reset_index()
    .rename({0: "n"}, axis=1)
    .pivot(index='year', columns='topic_label', values='n')
    .fillna(0)
    .reset_index()
)

hold = []
for topic in data2['topic_label'].unique():
    slice = topic_counts[['year', topic]]
    slice = bootstrap_lowess(slice, topic, frac=0.075, resolution=2)
    slice['topic_label'] = topic
    hold.append(slice)

pdata = pd.concat(hold, axis=0)

plot = (
    alt.Chart(pdata)
    .mark_area(interpolate="basis")
    .encode(
        alt.X("dt:T", title="Jaar"),
            # .axis(format="i")
            # .scale(domain=[1925, 2023]),
        alt.Y("ul:Q", title="Prevalentie")
            .stack("normalize"),
        alt.Color("topic_label:N", title="Topic")
            .scale(scheme="category20")
            .legend(orient="bottom", columns=6)
    )
)

save_plot(plot, "topic_prevalence")

tdata = (
    topic_counts
    .set_index('year')
    .stack().to_frame()
    .set_axis(['count'],axis=1)
    .reset_index()
    .assign(dt = lambda df_: pd.to_datetime(df_['year'].astype(str), format="%Y"))
)

plot = (
    alt.Chart(tdata)
    .mark_area(interpolate='step-after')
    .encode(
        alt.X('dt:T').title(None),
        alt.Y('count:Q').title(None),
        alt.Color("topic_label:N", title="Topic")
            .scale(scheme="category20")
            .legend(orient="bottom", columns=6)
    )
    .properties(height=100, width = 215)
    .facet(alt.Row('topic_label').title("Count"), columns=3)
    .configure_facet(
        spacing=7.5
    )
    .configure_header(
        # labelFontSize=0
        labelBaseline="top",
        labelFontSize=11,
        labelAnchor="start",
        labelPadding=5,
        labelFontWeight='bold',
        titleOrient='left'
    )
)

save_plot(plot, "topic_prevalence_facet")


