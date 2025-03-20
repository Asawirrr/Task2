import streamlit as st
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline

# Load pre-trained FinBERT model and tokenizer
model_name = "ProsusAI/finbert"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)

# Create Sentiment Analysis Pipeline
nlp_pipeline = pipeline("text-classification", model=model, tokenizer=tokenizer)

# Streamlit UI
st.title("📈 Financial News Sentiment Analyzer 📰")
st.write("Enter a financial news headline to analyze its sentiment.")

# Input text box
user_input = st.text_area("Enter Financial News Headline:", "")

if st.button("Analyze Sentiment"):
    if user_input.strip():
        # Predict sentiment
        prediction = nlp_pipeline(user_input)[0]

         

        # Correct sentiment mapping
        sentiment_mapping = {"positive": "Positive", "negative": "Negative", "neutral": "Neutral"}

        # Extract label and score
        sentiment_label = prediction["label"].lower()  # Ensure lowercase matching
        confidence_score = prediction["score"]

        # Get correct sentiment text
        sentiment_text = sentiment_mapping.get(sentiment_label, "Unknown")

        # Display results
        st.subheader(f"🧐 Sentiment: {sentiment_text}")
        st.write(f"🔍 Confidence Score: {confidence_score:.4f}")
    else:
        st.warning("Please enter a headline to analyze.")
