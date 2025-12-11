import streamlit as st
import pandas as pd
import random
import os

# Try importing kagglehub, but don't crash if it's missing (fallback to demo mode)
try:
    import kagglehub
    KAGGLE_AVAILABLE = True
except ImportError:
    KAGGLE_AVAILABLE = False

# --- App Configuration ---
st.set_page_config(
    page_title="CineMatch",
    page_icon="🎬",
    layout="centered"
)

# --- Mock Data (Fallback) ---
MOCK_MOVIES = [
    {"title": "The Dark Knight", "year": 2008, "genres": "Action|Crime|Drama", "avg_rating": 9.0, "overview": "When the menace known as the Joker wreaks havoc and chaos on the people of Gotham, Batman must accept one of the greatest psychological and physical tests of his ability to fight injustice."},
    {"title": "Inception", "year": 2010, "genres": "Action|Adventure|Sci-Fi", "avg_rating": 8.8, "overview": "A thief who steals corporate secrets through the use of dream-sharing technology is given the inverse task of planting an idea into the mind of a C.E.O."},
    {"title": "Interstellar", "year": 2014, "genres": "Adventure|Drama|Sci-Fi", "avg_rating": 8.6, "overview": "A team of explorers travel through a wormhole in space in an attempt to ensure humanity's survival."},
    {"title": "Parasite", "year": 2019, "genres": "Drama|Thriller", "avg_rating": 8.6, "overview": "Greed and class discrimination threaten the newly formed symbiotic relationship between the wealthy Park family and the destitute Kim clan."},
    {"title": "Spirited Away", "year": 2001, "genres": "Animation|Adventure|Family", "avg_rating": 8.6, "overview": "During her family's move to the suburbs, a sullen 10-year-old girl wanders into a world ruled by gods, witches, and spirits."},
    {"title": "Pulp Fiction", "year": 1994, "genres": "Crime|Drama", "avg_rating": 8.9, "overview": "The lives of two mob hitmen, a boxer, a gangster and his wife, and a pair of diner bandits intertwine in four tales of violence and redemption."},
    {"title": "The Matrix", "year": 1999, "genres": "Action|Sci-Fi", "avg_rating": 8.7, "overview": "When a beautiful stranger leads computer hacker Neo to a forbidding underworld, he discovers the shocking truth--the life he knows is the elaborate deception of an evil cyber-intelligence."},
    {"title": "Forrest Gump", "year": 1994, "genres": "Drama|Romance", "avg_rating": 8.8, "overview": "The presidencies of Kennedy and Johnson, the events of Vietnam, Watergate and other historical events unfold through the perspective of an Alabama man with an IQ of 75."},
    {"title": "Everything Everywhere All At Once", "year": 2022, "genres": "Action|Adventure|Comedy", "avg_rating": 8.0, "overview": "A middle-aged Chinese immigrant is swept up into an insane adventure in which she alone can save the existence by exploring other universes."},
    {"title": "Dune", "year": 2021, "genres": "Action|Adventure|Sci-Fi", "avg_rating": 8.0, "overview": "A noble family becomes embroiled in a war for control over the galaxy's most valuable asset while its heir becomes troubled by visions of a dark future."},
    {"title": "The Godfather", "year": 1972, "genres": "Crime|Drama", "avg_rating": 9.2, "overview": "The aging patriarch of an organized crime dynasty transfers control of his clandestine empire to his reluctant son."},
    {"title": "Schindler's List", "year": 1993, "genres": "Biography|Drama|History", "avg_rating": 8.9, "overview": "In German-occupied Poland during World War II, industrialist Oskar Schindler gradually becomes concerned for his Jewish workforce after witnessing their persecution by the Nazis."},
    {"title": "Whiplash", "year": 2014, "genres": "Drama|Music", "avg_rating": 8.5, "overview": "A promising young drummer enrolls at a cut-throat music conservatory where his dreams of greatness are mentored by an instructor who will stop at nothing to realize a student's potential."}
]

# --- Data Loading Functions ---

def load_mock_data():
    """Loads the fallback mock dataset."""
    df = pd.DataFrame(MOCK_MOVIES)
    df['genres_list'] = df['genres'].str.split('|')
    return df

@st.cache_data
def load_kaggle_data():
    """Downloads and processes the dataset from Kaggle."""
    if not KAGGLE_AVAILABLE:
        raise ImportError("kagglehub not installed")
    
    # Download dataset
    path = kagglehub.dataset_download("parasharmanas/movie-recommendation-system")
    
    # Load CSVs
    movies = pd.read_csv(f"{path}/movies.csv")
    ratings = pd.read_csv(f"{path}/ratings.csv")
    
    # Extract Year from Title string "Toy Story (1995)"
    movies['year'] = movies['title'].str.extract(r'\((\d{4})\)')
    movies['year'] = pd.to_numeric(movies['year'], errors='coerce')
    
    # Calculate Average Ratings
    avg_ratings = ratings.groupby('movieId')['rating'].mean().round(1)
    
    # Merge DataFrames
    df = pd.merge(movies, avg_ratings, on='movieId')
    df.rename(columns={'rating': 'avg_rating'}, inplace=True)
    
    # Process Genres (String "Action|Comedy" -> List ["Action", "Comedy"])
    df['genres_list'] = df['genres'].str.split('|')
    
    # Clean Data
    df = df.dropna(subset=['year', 'avg_rating'])
    
    # Add generic overview if missing
    if 'overview' not in df.columns:
         df['overview'] = "No plot summary available for this title."
         
    return df

# --- Main Application ---

def main():
    st.title("🍿 CineMatch")
    st.write("Filter your preferences and let us pick a movie for you.")

    # --- Data Source Selection ---
    with st.sidebar:
        st.header("Settings")
        
        # Check for Kaggle keys
        has_kaggle_auth = os.environ.get('KAGGLE_USERNAME') and os.environ.get('KAGGLE_KEY')
        
        data_source = st.radio(
            "Data Source",
            ["Demo Data", "Kaggle Dataset"],
            index=0 if not has_kaggle_auth else 1
        )

    # --- Load Data Logic ---
    df = pd.DataFrame()
    
    if data_source == "Demo Data":
        df = load_mock_data()
    else:
        try:
            with st.spinner("Downloading data..."):
                df = load_kaggle_data()
            st.success(f"Loaded {len(df):,} movies.")
        except Exception as e:
            st.error("Could not load Kaggle data. Switched to Demo Mode.")
            df = load_mock_data()

    if df.empty:
        st.stop()

    # --- Sidebar Filters ---
    with st.sidebar:
        st.divider()
        st.header("Filters")
        
        # Genres
        all_genres = sorted(list(set([g for sublist in df['genres_list'] for g in sublist])))
        if "no genres listed" in all_genres: all_genres.remove("no genres listed")
        selected_genre = st.selectbox("Genre", ["Any"] + all_genres)
        
        # Rating
        min_rating = st.slider("Min Rating", 1.0, 10.0, 7.0, 0.5)
        
        # Year
        min_year = int(df['year'].min())
        max_year = int(df['year'].max())
        default_start = max(1980, min_year)
        selected_years = st.slider("Year Range", min_year, max_year, (default_start, max_year))

    # --- Filtering Logic ---
    filtered_df = df[
        (df['avg_rating'] >= min_rating) & 
        (df['year'] >= selected_years[0]) & 
        (df['year'] <= selected_years[1])
    ]

    if selected_genre != "Any":
        filtered_df = filtered_df[filtered_df['genres_list'].apply(
            lambda x: selected_genre in x if isinstance(x, list) else False
        )]

    # --- Result Section ---
    st.divider()
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.metric("Movies Found", len(filtered_df))
    
    with col2:
        if st.button("🎲 Randomize Movie", type="primary", use_container_width=True):
            if not filtered_df.empty:
                movie = filtered_df.sample(1).iloc[0]
                st.session_state['selected_movie'] = movie
            else:
                st.session_state['selected_movie'] = None
                st.warning("No movies found!")

    # Display Selection
    if st.session_state.get('selected_movie') is not None:
        movie = st.session_state['selected_movie']
        
        with st.container(border=True):
            st.header(movie['title'])
            st.caption(f"{int(movie['year'])} • {movie['genres'].replace('|', ', ')}")
            
            c1, c2 = st.columns(2)
            c1.metric("Rating", f"{movie['avg_rating']}/10")
            
            st.write("### Plot")
            st.write(movie.get('overview', 'No plot available.'))
            
            # Links
            search_query = movie['title'].replace(' ', '+')
            st.markdown(f"""
            [Search Google](https://www.google.com/search?q={search_query}+movie) | 
            [IMDb](https://www.imdb.com/find?q={search_query}) | 
            [Trailer](https://www.youtube.com/results?search_query={search_query}+trailer)
            """)

if __name__ == "__main__":
    main()