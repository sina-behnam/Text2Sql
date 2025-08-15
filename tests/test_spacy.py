import spacy
import pandas as pd
# Load transformer-based English model
nlp = spacy.load("en_core_web_trf")

text = (
    "I don't have three cats but I have 2 dogs. "
    "We will not adopt 10 more animals because we won't have space for five larger ones. "
    "He doesn't need 3 cars, and she did not buy seven books. "
    "They won't raise 12 chickens since they do not own 2 coops."
)

doc = nlp(text)

# Initialize list to store token information
tokens_data = []

# Detect negation and numbers
for i, token in enumerate(doc):
    if token.dep_ == "neg":
        tokens_data.append({
            "text": token.text,
            "type": "negation",
            "position": i,
            "pos": token.pos_,
            "dep": token.dep_
        })
        print(f"Negation word: {token.text}")
    
    if token.like_num:
        tokens_data.append({
            "text": token.text,
            "type": "number",
            "position": i,
            "pos": token.pos_,
            "dep": token.dep_
        })
        print(f"Number: {token.text}")

# Create DataFrame
tokens_df = pd.DataFrame(tokens_data)

# Count by type
if not tokens_df.empty:
    type_counts = tokens_df['type'].value_counts()
    print("\nToken counts:")
    print(type_counts)
    
    print("\nTokens DataFrame:")
    print(tokens_df)
else:
    print("\nNo negation words or numbers found.")