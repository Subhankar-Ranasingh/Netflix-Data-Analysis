# netflix_eda.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# ---------- Settings ----------
sns.set_theme(context="notebook")
plt.rcParams["figure.figsize"] = (10, 6)
OUT = Path("figures")
OUT.mkdir(parents=True, exist_ok=True)

# ---------- Load Dataset ----------
df = pd.read_csv("data/netflix_titles.csv", encoding="utf-8")
df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]

# ---------- Data Cleaning ----------
for col in ["country", "rating", "duration", "listed_in"]:
    if col in df.columns:
        df[col] = df[col].fillna("Unknown").astype(str).str.strip()

if "date_added" in df.columns:
    df["date_added"] = pd.to_datetime(df["date_added"], errors="coerce")
    df["year_added"] = df["date_added"].dt.year
else:
    df["year_added"] = np.nan

if "release_year" not in df.columns:
    df["release_year"] = np.nan

# ---------- Helper to Save Plots ----------
def saveplot(path):
    plt.tight_layout()
    plt.savefig(OUT / path, dpi=200, bbox_inches="tight")
    plt.close()

# ---------- 1) Movies vs TV Shows ----------
type_counts = df["type"].value_counts().sort_values(ascending=False)
ax = sns.barplot(x=type_counts.index, y=type_counts.values)
ax.set_xlabel("")
ax.set_ylabel("Count")
ax.set_title("Movies vs TV Shows on Netflix")
for i, v in enumerate(type_counts.values):
    ax.text(i, v, f"{v}", ha="center", va="bottom", fontsize=9)
saveplot("01_movies_vs_tv.png")

# ---------- 2) Top 10 Countries ----------
countries = (
    df.assign(country=df["country"].str.split(","))
      .explode("country")
)
countries["country"] = countries["country"].str.strip()
top_countries = (
    countries[countries["country"].ne("Unknown")]["country"]
    .value_counts()
    .head(10)
    .sort_values(ascending=True)
)
ax = top_countries.plot(kind="barh")
ax.set_xlabel("Titles")
ax.set_ylabel("")
ax.set_title("Top 10 Content-Producing Countries")
saveplot("02_top_countries.png")

# ---------- 3) Netflix Growth by Year ----------
year_counts = (
    df.dropna(subset=["year_added"])
      .groupby("year_added")["title"].count()
      .sort_index()
)
ax = year_counts.plot(kind="line", marker="o")
ax.set_xlabel("Year Added to Netflix")
ax.set_ylabel("Number of Titles")
ax.set_title("Netflix Catalog Growth Over Years")
saveplot("03_growth_by_year_added.png")

# ---------- 4) Top Genres ----------
genres = (
    df.assign(listed_in=df["listed_in"].str.split(","))
      .explode("listed_in")
)
genres["listed_in"] = genres["listed_in"].str.strip()
top_genres = (
    genres[genres["listed_in"].ne("Unknown")]
    ["listed_in"].value_counts().head(15).sort_values()
)
ax = top_genres.plot(kind="barh")
ax.set_xlabel("Titles")
ax.set_ylabel("")
ax.set_title("Top 15 Genres / Categories on Netflix")
saveplot("04_top_genres.png")

# ---------- 5) Movie Duration Distribution ----------
movies = df[df["type"].eq("Movie")].copy()
movies["duration_minutes"] = (
    movies["duration"].str.extract(r"(\d+)").astype(float)
)
ax = movies["duration_minutes"].dropna().plot(kind="hist", bins=30)
ax.set_xlabel("Duration (minutes)")
ax.set_title("Movie Duration Distribution")
saveplot("05_movie_duration_hist.png")

# ---------- 6) Movie Duration by Decade ----------
movies["decade"] = (movies["release_year"] // 10 * 10).astype("Int64")
recent_decades = movies.dropna(subset=["decade"])
recent_decades = recent_decades[recent_decades["decade"] >= 1980]
ax = sns.boxplot(
    data=recent_decades,
    x="decade",
    y="duration_minutes"
)
ax.set_xlabel("Decade")
ax.set_ylabel("Minutes")
ax.set_title("Movie Duration by Decade")
saveplot("06_movie_duration_by_decade.png")

# ---------- 7) TV Show Seasons ----------
tv = df[df["type"].eq("TV Show")].copy()
tv["seasons"] = tv["duration"].str.extract(r"(\d+)").astype(float)
if tv["seasons"].notna().sum() > 0:
    ax = tv["seasons"].dropna().plot(kind="hist", bins=15)
    ax.set_xlabel("Seasons")
    ax.set_title("TV Show Seasons Distribution")
    saveplot("07_tv_seasons_hist.png")

# ---------- Generate Insights ----------
ins = []
if not type_counts.empty:
    dominant = type_counts.idxmax()
    pct = type_counts.max() / type_counts.sum() * 100
    ins.append(f"{dominant} lead with ~{pct:.1f}% of total titles.")

if not top_countries.empty:
    tcountry = top_countries.sort_values(ascending=False).index[:3].tolist()
    ins.append("Top content countries: " + ", ".join(tcountry) + ".")

if len(year_counts) > 1:
    recent_growth = year_counts.iloc[-1] / year_counts.max() * 100
    ins.append(f"Catalog peak additions around {year_counts.idxmax()}; latest year is ~{recent_growth:.0f}% of peak.")

if movies["duration_minutes"].notna().any():
    mean_dur = movies["duration_minutes"].mean()
    ins.append(f"Avg movie duration ≈ {mean_dur:.0f} minutes.")

(Path("insights.txt")).write_text(
    "Key Insights:\n- " + "\n- ".join(ins),
    encoding="utf-8"
)

print("Saved figures:")
for p in sorted(OUT.glob("*.png")):
    print(" -", p.as_posix())
print("\nInsights saved to insights.txt")
