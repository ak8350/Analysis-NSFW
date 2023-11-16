import streamlit as st
from transformers import pipeline

# Load the sentiment analysis pipeline
classifier = pipeline("sentiment-analysis", model="michellejieli/NSFW_text_classification", revision="main")


# Streamlit app
def main():
    st.title("Sentiment Analysis App")

    # User input for text
    user_input = st.text_area("Enter the text for sentiment analysis:")

    # Analyze sentiment button
    if st.button("Analyze Sentiment"):
        if user_input:
            # Perform sentiment analysis
            result = analyze_sentiment(user_input)
            st.subheader("Sentiment Analysis Result:")
            st.write(f"Text: {user_input}")
            st.write(f"Sentiment: {result['label']} (Confidence: {result['score']:.4f})")
        else:
            st.warning("Please enter text for sentiment analysis.")

# Function to analyze sentiment
def analyze_sentiment(text):
    result = classifier(text)[0]
    return result

if __name__ == "__main__":
    main()
