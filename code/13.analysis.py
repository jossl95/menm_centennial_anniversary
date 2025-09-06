"""Analysis and visualization of article metadata.

This module performs statistical analysis and visualization of article metadata,
including:
- Research method trends over time
- Article counts and page lengths
- Topic modeling results visualization

The module uses bootstrapped LOWESS smoothing for trend analysis and 
Altair for visualization.
"""

import os
import re
from pathlib import Path
from typing import Optional

import altair as alt
import arviz as az
import numpy as np
import pandas as pd
import statsmodels.api as sm
from multiprocess import Pool, cpu_count
from tqdm.notebook import tqdm

# Constants
BASEDIR: Path = Path("data/text-images")
FIGDIR: Path = Path("figures")
DATADIR: Path = Path("data")

# Enable VegaFusion for Altair
alt.data_transformers.enable("vegafusion")


@alt.theme.register("menm_theme", enable=True)
def menm_theme() -> alt.theme.ThemeConfig:
    """Creates a custom Altair theme configuration.
    
    Returns:
        Dictionary containing theme configuration settings.
    """
    return {
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
            "header": {"labelFontWeight": "bold"},
            "title": {"fontSize": 14},
        }
    }


def save_plot(plot: alt.Chart, title: str) -> None:
    """Saves an Altair chart in multiple formats.
    
    Args:
        plot: The Altair chart to save
        title: Base filename for the saved files (without extension)
    """
    plot.save(FIGDIR / f"{title}.pdf")
    plot.save(FIGDIR / f"{title}.svg")
    plot.save(FIGDIR / f"{title}.png", ppi=500)


def bootstrap_lowess(
    data: pd.DataFrame,
    var: str = 'page_count',
    frac: float = 0.1,
    resolution: int = 1
) -> pd.DataFrame:
    """Performs bootstrapped LOWESS smoothing on time series data.
    
    Args:
        data: DataFrame containing time series data
        var: Column name of variable to smooth
        frac: Fraction of data used for smoothing at each point
        resolution: Number of points between each observed year
    
    Returns:
        DataFrame with smoothed values and confidence intervals
    """
    x = data["year"].values
    y = data[var].values
    k = (len(np.unique(x)) * resolution - resolution 
         if resolution != 1 else len(np.unique(x)))
    
    x_grid = np.linspace(x.min(), x.max(), k)
    n_bootstraps = 10000
    predictions = np.zeros((n_bootstraps, len(x_grid)))
    
    for i in tqdm(range(n_bootstraps), total=n_bootstraps):
        data_sample = data.sample(n=len(data), replace=True)
        loess_fit = sm.nonparametric.lowess(
            data_sample[var],
            data_sample["year"],
            frac=frac
        )
        predictions[i] = np.interp(x_grid, loess_fit[:, 0], loess_fit[:, 1])
    
    pred_data = pd.DataFrame(predictions, columns=x_grid)
    
    return pd.DataFrame({
        "year": x_grid,
        "mean": pred_data.mean(axis=0).values,
        "ll": az.hdi(pred_data.values, hdi_prob=0.95).T[0],
        "ul": az.hdi(pred_data.values, hdi_prob=0.95).T[1],
    }).assign(dt=lambda df_: fractional_year_to_datetime(df_['year']))


def fractional_year_to_datetime(fy: float) -> pd.Timestamp:
    """Converts fractional year to datetime.
    
    Args:
        fy: Fractional year value (e.g., 2020.5)
    
    Returns:
        Corresponding datetime
    """
    year = fy // 1
    remainder = fy - year
    start_of_year = pd.to_datetime(year, format='%Y')
    next_year = pd.to_datetime(year + 1, format='%Y')
    delta = next_year - start_of_year
    return start_of_year + remainder * delta


def read_method(row: pd.Series) -> Optional[str]:
    """Reads the method markdown file for a given article.
    
    Args:
        row: Series containing article metadata
    
    Returns:
        Contents of method file if it exists, None otherwise
    """
    method_path = BASEDIR / str(row["year"]) / row["id"] / "method.md"
    if method_path.exists():
        return method_path.read_text(encoding="utf-8")
    return None


def read_method_parallel(
    data: pd.DataFrame,
    num_workers: Optional[int] = None
) -> pd.DataFrame:
    """Reads all method files in parallel using multiprocessing.
    
    Args:
        data: DataFrame containing article metadata
        num_workers: Number of parallel processes to use
    
    Returns:
        DataFrame with added 'method' column
    """
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

def process_research_methods(data: pd.DataFrame) -> pd.DataFrame:
    """Processes and analyzes research methods data.
    
    Args:
        data: DataFrame containing article metadata
    
    Returns:
        Processed DataFrame with method proportions
    """
    methods_df = (
        data
        .pipe(read_method_parallel)
        .assign(
            method=lambda d_: d_['method']
                .str.title()
                .str.split('\n').str[0]
                .str.replace('Literatuuroverzicht', 'Literatuuronderzoek')
                .str.replace('Kwalitatief, Mixed-Methods', 'Mixed-Methods')
        )
        .groupby('year')['method']
        .value_counts(normalize=True)
        .unstack().fillna(0).stack()
        .to_frame(name='proportion')
        .reset_index()
    )

    results = []
    for method in methods_df['method'].unique():
        if method == 'Overig':
            continue
        processed = (
            methods_df.loc[methods_df['method'] == method]
            .pipe(bootstrap_lowess, var='proportion', frac=0.14)
            .assign(
                year=lambda d_: d_['year'].astype(int),
                method=method
            )
        )
        results.append(processed)
    
    return pd.concat(results)

def plot_research_methods(source: pd.DataFrame) -> None:
    """Creates visualization of research methods trends.
    
    Args:
        source: DataFrame containing processed methods data
    """
    base = (
        alt.Chart(source)
        .encode(
            alt.X('year:T').title('Jaar'),
            alt.Y('mean:Q')
                .title(None)
                .axis(format='%')
                .scale(domain=(0, 1)),
            alt.Color('method:N')
                .title('Methodologie')
                .legend(None)
        )
    )

    line = base.mark_line()
    area = (
        base
        .mark_area(opacity=.3, clip=True)
        .encode(
            alt.Y('ll'),
            alt.Y2('ul')
        )
    )

    plot = (
        alt.layer(line, area)
        .properties(
            width=150,
            height=350
        )
        .facet(
            alt.Row(
                'method:N',
                header=alt.Header(labelAnchor='start', labelPadding=5)
            ).title(None),
            columns=4
        )
        .configure_facet(spacing=7.5)
    )

    save_plot(plot, "method_counts")

def plot_article_counts(data: pd.DataFrame) -> None:
    """Creates visualization of article counts over time.
    
    Args:
        data: DataFrame containing article metadata
    """
    pdata = (
        data
        .assign(
            year=lambda df_: df_['year'].astype(str),
            dt=lambda df_: pd.to_datetime(df_['year'], format="%Y")
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


def export_decade_counts(
    data: pd.DataFrame, 
    selection_func: callable
) -> None:
    """Exports article counts by decade to Excel.
    
    Args:
        data: DataFrame containing article metadata
        selection_func: Function to filter articles
    """
    (
        data.loc[
            lambda df_: (df_["section"] == "artikel") & selection_func(df_)
        ]
        .assign(
            decade=lambda df_: df_["year"].astype(int).pipe(
                pd.cut, bins=range(1935, 2035, 10)
            ),
            year=lambda df_: df_["year"].astype(str)
        )
        .groupby(["decade", "section"], observed=False).size()
        .unstack(level=1)
        .to_excel(DATADIR / "section_counts.xlsx")
    )


def plot_page_counts(data: pd.DataFrame) -> None:
    """Creates visualization of page counts over time.
    
    Args:
        data: DataFrame containing article metadata
    """
    pdata = bootstrap_lowess(data, resolution=1, frac=0.4)
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


def process_topic_counts(data: pd.DataFrame) -> pd.DataFrame:
    """Processes topic count data for visualization.
    
    Args:
        data: DataFrame containing article metadata with topics
    
    Returns:
        Processed DataFrame with topic prevalence over time
    """
    topic_counts = (
        data
        .groupby(["year", "topic_label"]).size()
        .reset_index()
        .rename({0: "n"}, axis=1)
        .pivot(index='year', columns='topic_label', values='n')
        .fillna(0)
        .reset_index()
    )

    results = []
    for topic in data['topic_label'].unique():
        slice_data = topic_counts[['year', topic]]
        smoothed = bootstrap_lowess(
            slice_data, 
            topic, 
            frac=0.075, 
            resolution=2
        )
        smoothed['topic_label'] = topic
        results.append(smoothed)

    return pd.concat(results, axis=0)


def plot_topic_prevalence(data: pd.DataFrame) -> None:
    """Creates visualization of topic prevalence over time.
    
    Args:
        data: DataFrame containing processed topic data
    """
    plot = (
        alt.Chart(data)
        .mark_area(interpolate="basis")
        .encode(
            alt.X("dt:T", title="Jaar"),
            alt.Y("ul:Q", title="Prevalentie")
                .stack("normalize"),
            alt.Color("topic_label:N", title="Topic")
                .scale(scheme="category20")
                .legend(orient="bottom", columns=6)
        )
    )
    save_plot(plot, "topic_prevalence")


def plot_topic_facets(topic_counts: pd.DataFrame) -> None:
    """Creates faceted visualization of topic counts.
    
    Args:
        topic_counts: DataFrame containing topic count data
    """
    tdata = (
        topic_counts
        .set_index('year')
        .stack().to_frame()
        .set_axis(['count'], axis=1)
        .reset_index()
        .assign(
            dt=lambda df_: pd.to_datetime(df_['year'].astype(str), format="%Y")
        )
    )

    area = (
        alt.Chart(tdata)
        .mark_area(interpolate='step-after')
        .encode(
            alt.X('dt:T').title(None),
            alt.Y('count:Q').title(None),
            alt.Color('topic_label:N', title='Topic')
                .scale(scheme='category20')
                .legend(orient='bottom', columns=6)
        )
        .properties(height=100, width=215)
    )

    label = (
        alt.Chart(tdata)
        .transform_aggregate(n='count()', groupby=['topic_label'])
        .mark_text(
            align='left', 
            baseline='top', 
            fontSize=11, 
            fontWeight='bold'
        )
        .encode(
            text='topic_label:N',
            x=alt.value(5),
            y=alt.value(5)
        )
    )

    plot = (
        alt.layer(area, label)
        .facet(
            alt.Row(
                'topic_label:N',
                header=alt.Header(labelExpr="''", labels=False, title=None)
            ),
            columns=3
        )
        .configure_facet(spacing=7.5)
    )

    save_plot(plot, "topic_prevalence_facet")


def main() -> None:
    """Runs the complete analysis pipeline for article metadata."""
    # Load Data
    data = pd.read_excel(DATADIR / 'articles.xlsx')
    data2 = pd.read_excel(DATADIR / 'articles_metadata.xlsx')
    authors = pd.read_excel(DATADIR / 'article_author_link.xlsx')

    def selection(d: pd.DataFrame) -> pd.Series:
        """Filters articles based on page count criteria.
        
        Args:
            d: DataFrame containing article metadata
            
        Returns:
            Boolean series for filtering
        """
        return (d["page_count"] > 3) & (d["page_count"] < 75)
    
    # Research Methods Analysis
    methods_data = process_research_methods(data2)
    plot_research_methods(methods_data)
    
    # Article Counts Analysis
    plot_article_counts(data)
    export_decade_counts(data, selection)
    
    # Page Counts Analysis
    plot_page_counts(data2)
    
    # Topic Analysis
    plot_topic_prevalence(data2)
    plot_topic_facets(data2)


if __name__ == "__main__":
    main()