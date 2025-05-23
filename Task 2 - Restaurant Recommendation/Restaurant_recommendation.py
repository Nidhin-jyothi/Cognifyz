import streamlit as st
import pandas as pd
import re
from sklearn.preprocessing import MinMaxScaler
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# Load data
df = pd.read_csv('data.csv')

# Drop irrelevant columns
df = df.drop(columns=[
    'Restaurant ID', 'Longitude', 'Latitude', 'Rating color',
    'Rating text', 'Votes', 'Switch to order menu', 'Is delivering now'
])

df = df.dropna(subset=['Cuisines'])

# Map binary features
df['Has Table booking'] = df['Has Table booking'].map({'Yes': 1, 'No': 0})
df['Has Online delivery'] = df['Has Online delivery'].map({'Yes': 1, 'No': 0})

# Preserve original values for display
df['Original Cost'] = df['Average Cost for two']
df['Original Rating'] = df['Aggregate rating']

# Normalize for embeddings
scaler = MinMaxScaler()
df[['Average Cost for two', 'Aggregate rating']] = scaler.fit_transform(df[['Average Cost for two', 'Aggregate rating']])

# Create embedding text
def create_metadata(row):
    cuisines = row['Cuisines'].replace(',', ' ')
    return f"{row['City']} {cuisines} cost:{row['Average Cost for two']} rating:{row['Aggregate rating']} price:{row['Price range']} table:{row['Has Table booking']} delivery:{row['Has Online delivery']}"

df['metadata'] = df.apply(create_metadata, axis=1)

# Generate embeddings
model = SentenceTransformer('all-MiniLM-L6-v2')
embeddings = model.encode(df['metadata'].tolist(), show_progress_bar=True)

# Function to extract fields from natural language 
def parse_input(text):
    city_match = re.search(r"in (\w+)", text, re.IGNORECASE)
    cuisines = re.findall(r'\b(Italian|Mexican|Chinese|Indian|Thai|American|Asian|South Indian|North Indian|Continental|Biryani|Pizza)\b', text, re.IGNORECASE)
    rating_match = re.search(r"rating (?:above|greater than|minimum)? ?(\d+(\.\d+)?)", text, re.IGNORECASE)
    price_match = re.search(r"price range (\d)", text)
    table_booking = int(bool(re.search(r"table booking", text, re.IGNORECASE)))
    delivery = int(bool(re.search(r"online delivery", text, re.IGNORECASE)))

    return {
        'city': city_match.group(1) if city_match else 'Bangalore',
        'cuisines': list(set(cuisines)) or ['Italian'],
        'min_rating': float(rating_match.group(1)) if rating_match else 3.5,
        'price_range': int(price_match.group(1)) if price_match else 2,
        'table_booking': table_booking,
        'online_delivery': delivery
    }

# Build metadata from structured input
def build_user_metadata(user_pref):
    cuisine_str = " ".join(user_pref['cuisines'])
    return f"{user_pref['city']} {cuisine_str} cost:0.5 rating:{user_pref['min_rating']/5} price:{user_pref['price_range']} table:{user_pref['table_booking']} delivery:{user_pref['online_delivery']}"

# Recommendation function
def get_recommendations(user_embedding, top_n=5):
    sim_scores = cosine_similarity([user_embedding], embeddings)[0]
    top_indices = sim_scores.argsort()[-top_n:][::-1]
    results = df.iloc[top_indices][[
        'Restaurant Name', 'City', 'Cuisines',
        'Original Cost', 'Original Rating'
    ]].rename(columns={
        'Original Cost': 'Cost for Two',
        'Original Rating': 'Rating'
    })
    results['Cost for Two'] = results['Cost for Two'].apply(lambda x: f"₹{int(x)}")
    results['Rating'] = results['Rating'].round(1)
    return results

# Streamlit UI 
st.title("Restaurant Recommender")
user_input = st.text_area("What are you looking for?", 
                          "I need a restaurant in Bangalore with Italian and Mexican cuisine, rating above 3.5, price range 2, and that offers table booking and online delivery.")

if st.button("Find Recommendations"):
    user_pref = parse_input(user_input)
    user_text = build_user_metadata(user_pref)
    user_embedding = model.encode([user_text])[0]
    recommendations = get_recommendations(user_embedding, top_n=5)

    st.subheader("Top Recommendations:")
    st.write(recommendations)
