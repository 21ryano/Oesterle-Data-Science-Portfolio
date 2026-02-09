# INSTRUCTIONS ON HOW TO RUN THIS APP
# First, in "Command Prompt" on the Terminal, type "cd PortfolioUpdate1"
# Second, still in "Command prompt", type "streamlit run final.py"


# Import Everything
import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


# Abbreviations Key
label_map = {
    'pts': 'Points Per Game',
    'ast': 'Assists Per Game',
    'reb': 'Rebounds Per Game',
    'gp': 'Games Played',
    'player_name': 'Player',
    'player_height': 'Height (cm)',
    'player_weight': 'Weight (kgs)',
    'age': 'Age',
    'oreb_pct': 'Offensive Rebound %',
    'dreb_pct': 'Defensive Rebound %',
    'usg_pct': 'Usage %',
    'net_rating': 'Net Rating'
}


# App Title
st.title("🏀 Basketball Statistics")


# Load Dataset
df = pd.read_csv("all_seasons.csv")


# Sidebar Navigation
st.sidebar.title("Navigation")
section = st.sidebar.radio(
    "Go to Section:",
    [
        "🏀 Basketball Statistics Home",
        "🌎 Player Demographics",
        "🏆 Top 10 Player Leaders",
        "📈 Distribution of Stats/Game",
        "📊 Statistics by Season",
        "💪 Performance Comparisons",
    ]
)


# Season + Team Filters
st.sidebar.title("Filters")
seasons = df['season'].unique()
selected_seasons = st.sidebar.multiselect(
    "Deselect Season(s):", options=seasons, default=seasons
)
teams = df['team_abbreviation'].unique()
selected_teams = st.sidebar.multiselect(
    "Deselect Team(s):", options=teams, default=teams
)
df_filtered = df[
    df['season'].isin(selected_seasons) &
    df['team_abbreviation'].isin(selected_teams)
]



# SECTION 1: Home
if section == "🏀 Basketball Statistics Home":
    st.write("Welcome to the Basketball Statistics Home!")
    st.write("Here you will find a dataset of individual player statistics for seasons between 1996-2023. Feel free to scroll through the dataset or type in an individual player name. Each row represents the statistics of an individual player for a particular season with points, rebounds, and assists being averaged.")

    player_name_input = st.text_input("**Search by Player:**")

    if player_name_input:
        # Filter the dataframe by player name (case insensitive)
        df_display = df_filtered[df_filtered['player_name'].str.contains(player_name_input, case=False, na=False)]
    else:
        df_display = df_filtered

    st.dataframe(df_display)


# SECTION 2: Player Demographics
elif section == "🌎 Player Demographics":
    st.header("🌎 Player Demographics")

    st.write("### Total Players by Team")
    st.bar_chart(df_filtered['team_abbreviation'].value_counts())

    st.write("### Players by Age")
    age_counts = df_filtered['age'].value_counts().sort_index()
    chart = age_counts.plot.bar(figsize=(10, 6))
    plt.xlabel(label_map['age'])
    plt.xticks(rotation=45)
    plt.ylabel("Number of Players")
    st.pyplot(chart.get_figure())
    chart.get_figure().clf()

    st.write("### Players by Region")
    europe_countries = {
        'France', 'Spain', 'Italy', 'Germany', 'Russia', 'Greece', 'Croatia',
        'Serbia', 'Turkey', 'Lithuania', 'Slovenia', 'Czech Republic', 'Belgium',
        'Sweden', 'Ukraine', 'Poland', 'Netherlands', 'Finland', 'Austria'
    }
    africa_countries = {
        'Nigeria', 'Senegal', 'Cameroon', 'South Africa', 'Angola', 'Morocco',
        'Tunisia', 'Egypt', 'Ghana', 'Ivory Coast', 'Mali', 'Algeria'
    }
    df_filtered['region'] = df_filtered['country'].apply(
        lambda x: 'USA' if x == 'USA'
        else ('Europe' if x in europe_countries
        else ('Africa' if x in africa_countries else 'Other'))
    )
    chart = df_filtered['region'].value_counts().plot.pie(
        autopct='%1.1f%%', figsize=(6, 6), ylabel=''
    )
    st.pyplot(chart.get_figure())
    chart.get_figure().clf()

    st.write("### Players per Draft Year (excluding undrafted)")
    df_drafted_only = df_filtered[
        df_filtered['draft_year'].notna()
        & (df_filtered['draft_year'] != "")
        & (df_filtered['draft_year'] != "Undrafted")
    ]

    st.write("### Drafted vs Undrafted Players")
    chart = pd.Series(
        [df_drafted_only.shape[0], df_filtered.shape[0] - df_drafted_only.shape[0]],
        index=["Drafted", "Undrafted"]
    ).plot.pie(autopct='%1.1f%%', figsize=(6, 6), ylabel='')
    st.pyplot(chart.get_figure())
    chart.get_figure().clf()


# SECTION 3: Top 10 Player Leaders
elif section == "🏆 Top 10 Player Leaders":
    st.header("🏆 Top 10 Player Leaders")

    # Top 10 Points in any single season
    st.write("### Top 10 Single-Season Point Leaders")
    pts_per_season = (df_filtered['pts'] * df_filtered['gp']).groupby([df_filtered['season'], df_filtered['player_name']]).sum()
    top_pts = pts_per_season.sort_values(ascending=False).head(10)
    chart = top_pts.plot.bar(figsize=(10,6))
    plt.xlabel("Season / Player")
    plt.ylabel("Total Points")
    plt.xticks(rotation=45)
    st.pyplot(chart.get_figure())
    chart.get_figure().clf()

    # Top 10 Assists in any single season
    st.write("### Top 10 Single-Season Assist Leaders")
    ast_per_season = (df_filtered['ast'] * df_filtered['gp']).groupby([df_filtered['season'], df_filtered['player_name']]).sum()
    top_ast = ast_per_season.sort_values(ascending=False).head(10)
    chart = top_ast.plot.bar(figsize=(10,6))
    plt.xlabel("Season / Player")
    plt.ylabel("Total Assists")
    plt.xticks(rotation=45)
    st.pyplot(chart.get_figure())
    chart.get_figure().clf()

    # Top 10 Rebounds in any single season
    st.write("### Top 10 Single-Season Rebound Leaders")
    reb_per_season = (df_filtered['reb'] * df_filtered['gp']).groupby([df_filtered['season'], df_filtered['player_name']]).sum()
    top_reb = reb_per_season.sort_values(ascending=False).head(10)
    chart = top_reb.plot.bar(figsize=(10,6))
    plt.xlabel("Season / Player")
    plt.ylabel("Total Rebounds")
    plt.xticks(rotation=45)
    st.pyplot(chart.get_figure())
    chart.get_figure().clf()


# SECTION 4: Distribution of Stats/Game
elif section == "📈 Distribution of Stats/Game":
    st.header("📈 Distribution of Stats/Game")

    # Distribution of Points per Game
    st.write("### Distribution of Points Per Game")
    sns_plot = sns.histplot(df_filtered['pts'], bins=30, kde=True)
    sns_plot.set_xlabel(label_map['pts'])
    sns_plot.set_ylabel("Frequency")
    st.pyplot(sns_plot.get_figure())
    sns_plot.get_figure().clf()

    # Distribution of Assists per Game
    st.write("### Distribution of Assists Per Game")
    sns_plot = sns.histplot(df_filtered['ast'], bins=30, kde=True)
    sns_plot.set_xlabel(label_map['ast'])
    sns_plot.set_ylabel("Frequency")
    st.pyplot(sns_plot.get_figure())
    sns_plot.get_figure().clf()

    # Distribution of Rebounds per Game
    st.write("### Distribution of Rebounds Per Game")
    sns_plot = sns.histplot(df_filtered['reb'], bins=30, kde=True)
    sns_plot.set_xlabel(label_map['reb'])
    sns_plot.set_ylabel("Frequency")
    st.pyplot(sns_plot.get_figure())
    sns_plot.get_figure().clf()


# SECTION 5: Statistics by Season
elif section == "📊 Statistics by Season":
    st.header("📊 Statistics by Season")

    st.write("### Average Points per Game per Season")
    avg_pts = df_filtered.groupby('season')['pts'].mean()
    sns_plot = sns.lineplot(x=avg_pts.index, y=avg_pts.values, marker='o')
    sns_plot.set_xlabel("Season")
    sns_plot.set_ylabel("Points Per Game")
    plt.xticks(rotation=45)
    st.pyplot(sns_plot.get_figure())
    sns_plot.get_figure().clf()

    st.write("### Average Assists per Game per Season")
    avg_ast = df_filtered.groupby('season')['ast'].mean()
    sns_plot = sns.lineplot(x=avg_ast.index, y=avg_ast.values, marker='o')
    sns_plot.set_xlabel("Season")
    sns_plot.set_ylabel("Assists Per Game")
    plt.xticks(rotation=45)
    st.pyplot(sns_plot.get_figure())
    sns_plot.get_figure().clf()

    st.write("### Average Rebounds per Game per Season")
    avg_reb = df_filtered.groupby('season')['reb'].mean()
    sns_plot = sns.lineplot(x=avg_reb.index, y=avg_reb.values, marker='o')
    sns_plot.set_xlabel("Season")
    sns_plot.set_ylabel("Rebounds Per Game")
    plt.xticks(rotation=45)
    st.pyplot(sns_plot.get_figure())
    sns_plot.get_figure().clf()

    st.write("### Average Games Played Per Season")
    avg_gp = df_filtered.groupby('season')['gp'].mean()
    sns_plot = sns.lineplot(x=avg_gp.index, y=avg_gp.values, marker='o')
    sns_plot.set_xlabel("Season")
    sns_plot.set_ylabel(label_map['gp'])
    plt.xticks(rotation=45)
    st.pyplot(sns_plot.get_figure())
    sns_plot.get_figure().clf()


# SECTION 6: Performance Comparisons
elif section == "💪 Performance Comparisons":
    st.header("💪 Performance Comparisons")

    st.write("### Age vs Points")
    sns_plot = sns.scatterplot(data=df_filtered, x='age', y='pts', alpha=0.7)
    sns_plot.set_xlabel(label_map['age'])
    sns_plot.set_ylabel(label_map['pts'])
    st.pyplot(sns_plot.get_figure())
    sns_plot.get_figure().clf()

    st.write("### Height vs Rebounds")
    sns_plot = sns.scatterplot(data=df_filtered, x='player_height', y='reb', alpha=0.7)
    sns_plot.set_xlabel(label_map['player_height'])
    sns_plot.set_ylabel(label_map['reb'])
    st.pyplot(sns_plot.get_figure())
    sns_plot.get_figure().clf()

    st.write("### Usage vs Rating")
    sns_plot = sns.scatterplot(data=df_filtered, x='usg_pct', y='net_rating', alpha=0.7)
    sns_plot.set_xlabel(label_map['usg_pct'])
    sns_plot.set_ylabel(label_map['net_rating'])
    st.pyplot(sns_plot.get_figure())
    sns_plot.get_figure().clf()