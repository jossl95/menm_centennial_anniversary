"""Topic modeling on cleaned scraped data.

This module extracts article summaries and applies BERTopic modeling.
"""

from pathlib import Path
from typing import Optional

import pandas as pd
import numpy as np
from scipy.stats import zscore
from tqdm import tqdm
from multiprocess import Pool, cpu_count

from nltk.corpus import stopwords
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.cluster import KMeans

from bertopic import BERTopic
from bertopic.vectorizers import ClassTfidfTransformer
from bertopic.representation import (
    MaximalMarginalRelevance,
    PartOfSpeech,
)
from umap import UMAP

import altair as alt


# Constants
BASEDIR = Path("data/text-images")
FIGDIR = Path("figures")


@alt.theme.register("menm_theme", enable=True)
def menm_theme() -> alt.theme.ThemeConfig:
    """Custom Altair theme configuration."""
    return {
        "config": {
            "view": {
                "continuousWidth": 650,
                "continuousHeight": 350
            },
            "axis": {"labelFontSize": 11, "titleFontSize": 11},
            "legend": {"labelFontSize": 11, "titleFontSize": 11},
            "title": {"fontSize": 14}
        }
    }


def save_plot(plot: alt.Chart, title: str) -> None:
    """Saves Altair chart as PDF and SVG."""
    plot.save(FIGDIR / f"{title}.pdf")
    plot.save(FIGDIR / f"{title}.svg")


def read_summary(row) -> Optional[str]:
    """Read the summary markdown file for a given article row."""
    summary_path = BASEDIR / str(row["year"]) / row["id"] / "summary.md"
    if summary_path.exists():
        return summary_path.read_text(encoding="utf-8")
    return None


def read_summaries_parallel(
    data: pd.DataFrame, num_workers: Optional[int] = None
) -> pd.DataFrame:
    """Read all summaries in parallel using multiprocessing."""
    num_workers = num_workers or cpu_count()
    with Pool(processes=num_workers) as pool:
        docs = list(
            tqdm(
                pool.imap(read_summary, [row for _, row in data.iterrows()]),
                total=len(data)
            )
        )
    data["summaries"] = docs
    return data


def load_data() -> pd.DataFrame:
    """Load and filter article metadata, then read summaries."""
    df = pd.read_excel(Path("data/articles_metadata.xlsx"))
    df = df.query("section == 'artikel' and page_count > 5")
    return read_summaries_parallel(df)


def configure_bertopic(data: pd.DataFrame) -> tuple[BERTopic, list[float]]:
    """Configure the BERTopic model and return it with embeddings."""
    embedding_model = SentenceTransformer(
        "intfloat/multilingual-e5-large-instruct"
    )
    embeddings = embedding_model.encode(
        data["summaries"].tolist(), show_progress_bar=True
    )

    stop_words = (
        stopwords.words("english") + stopwords.words("dutch") + [
            "abstract", "abstracts", "Abstract", "Abstracts",
            "samenvatting", "samenvattingen", "artikel", "paper",
            "studie", "werden", "aantal"
        ]
    )

    vectorizer_model = CountVectorizer(stop_words=stop_words)
    ctfidf_model = ClassTfidfTransformer(reduce_frequent_words=True)
    representation_model = [
        MaximalMarginalRelevance(diversity=0.25),
        PartOfSpeech("en_core_web_sm")
    ]
    umap_model = UMAP(
        n_neighbors=60, n_components=6, min_dist=0.0,
        metric="cosine", random_state=89
    )
    cluster_model = KMeans(
        n_clusters=40, random_state=42, n_init="auto"
    )

    topic_model = BERTopic(
        embedding_model=embedding_model,
        vectorizer_model=vectorizer_model,
        representation_model=representation_model,
        umap_model=umap_model,
        ctfidf_model=ctfidf_model,
        hdbscan_model=cluster_model,
        top_n_words=10,
        verbose=True,
        min_topic_size=5,
        n_gram_range=(1, 3),
        nr_topics="auto"
    )

    return topic_model, embeddings


def main():
    """Run the full topic modeling pipeline."""
    data = load_data()
    data = data[data["summaries"].notna()]

    topic_model, embeddings = configure_bertopic(data)
    topics, _ = topic_model.fit_transform(
        data["summaries"].tolist(), embeddings
    )

    data["topic"] = topics
    return data, topic_model, embeddings


if __name__ == "__main__":
    data, topic_model, embeddings = main()


# Save topic lookup and metadata
lookup = topic_model.get_topic_info()
lookup["Label"] = [
    "Sociologie", "Arbeidssociologie", "Onderwijs", "Anthropologie",
    "Methoden en Statistiek", "Politicologie", "Demografie",
    "Criminologie", "Anthropologie", "Anthropologie",
    "Religiewetenschappen", "Psychologie", "Rechtssociologie",
    "Krijgsmacht", "Eugenetica", "Cultuursociologie",
    "Religiewetenschappen"
]
lookup.to_excel("data/topic_lookup.xlsx", index=False)

map_labels = lookup.set_index("Topic")["Label"].to_dict()
data["topic_label"] = data["topic"].map(map_labels)
data.to_excel("data/articles_metadata.xlsx", index=False)

# Create 2D projection for visualization
umap_model = UMAP(
    n_neighbors=40, n_components=2, min_dist=0.0, metric="cosine"
).fit(embeddings)
embeddings_2d = umap_model.embedding_

viz_df = data.loc[:, ["id", "year", "topic", "topic_label"]].assign(
    x=zscore(embeddings_2d[:, 0]),
    y=zscore(embeddings_2d[:, 1])
)

viz_df.to_excel("data/topic_data.xlsx", index=False)

# Plot topic cloud
plot = (
    alt.Chart(viz_df)
    .mark_point(filled=True, size=25, opacity=1)
    .encode(
        alt.X("x:Q", title=None)
            .scale(domain=[-2.3, 2.4])
            .axis(None),
        alt.Y("y:Q", title=None)
            .scale(domain=[-2.2, 2.2])
            .axis(None),
        alt.Color("topic_label:N", title="Topic")
            .scale(scheme="category20")
            .legend(orient="bottom", columns=6)
    )
    .configure_view(stroke=None)
    .properties(width=675, height=375)
)
save_plot(plot, "topic_cloud")

# Plot topic prevalence
plot = (
    alt.Chart(viz_df)
    .transform_density(
        "year", as_=["year", "density"],
        groupby=["topic_label"],
        extent=[1925, 2025], resolve="shared"
    )
    .mark_area()
    .encode(
        alt.X("year:Q", title="Jaar")
            .axis(format="i")
            .scale(domain=[1925, 2025]),
        alt.Y("density:Q", title="Prevalentie")
            .stack("normalize"),
        alt.Color("topic_label:N", title="Topic")
            .scale(scheme="category20")
            .legend(orient="bottom", columns=6)
    )
)
save_plot(plot, "topic_prevalence")
