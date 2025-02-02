# Import necessary libraries
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from transformers import AutoTokenizer, AutoModelForCausalLM
from sklearn.metrics.pairwise import cosine_similarity
from IPython.display import display

# Function to evaluate conversational models
def evaluate_model(model_name, conversations):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)

    # Set padding token if missing
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    embeddings = []
    for text in conversations:
        inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
        with torch.no_grad():
            outputs = model(**inputs)
        embeddings.append(outputs.logits.mean(dim=1).numpy())  # Extract embeddings

    # Compute cosine similarities
    similarities = cosine_similarity(np.vstack(embeddings))
    return similarities

# Simple conversational dataset
conversations = [
    "Hi there!",
    "How have you been?",
    "Tell me something funny."
]

# **Alternative lightweight models with similar properties**
models = [
    "gpt2",  # Alternative to distilgpt2
    "facebook/blenderbot_small-90M",  # Alternative to BlenderBot 90M
    "microsoft/DialoGPT-small"  # Alternative to DialoGPT-small
]

# Store similarity scores and dummy values for inference speed & model size
model_scores = []
for model_name in models:
    print(f"Evaluating model: {model_name}")
    similarity = evaluate_model(model_name, conversations)
    aggregate_score = similarity.mean()  # Mean similarity score

    model_scores.append({
        "model": model_name,
        "mean_similarity": aggregate_score,
        "inference_speed": np.random.uniform(0.9, 1.0),  # Simulated speed
        "model_size": np.random.uniform(0.7, 1.0)  # Simulated model size
    })

# Convert results to DataFrame
df = pd.DataFrame(model_scores)

# TOPSIS function for ranking models
def topsis(scores, weights):
    scores = np.array(scores)
    norm_scores = scores / np.sqrt((scores**2).sum(axis=0))  # Normalize scores
    ideal_best = norm_scores.max(axis=0)
    ideal_worst = norm_scores.min(axis=0)

    # Compute distances from ideal best and worst
    dist_best = np.sqrt(((norm_scores - ideal_best)**2).sum(axis=1))
    dist_worst = np.sqrt(((norm_scores - ideal_worst)**2).sum(axis=1))

    # Calculate TOPSIS scores
    ranks = dist_worst / (dist_best + dist_worst)
    return ranks

# Apply TOPSIS ranking
scores = df[["mean_similarity", "inference_speed", "model_size"]].values
weights = [0.5, 0.3, 0.2]  # Custom weights
df["TOPSIS Score"] = topsis(scores, weights)

# Rank models based on TOPSIS score
df["Rank"] = df["TOPSIS Score"].rank(ascending=False)
df = df.sort_values(by="TOPSIS Score", ascending=False)

# Display the results as a table
display(df)

# Save results to CSV
df.to_csv("topsis_results.csv", index=False)

# Plot the results and save it as PNG
plt.figure(figsize=(8, 4))
sns.barplot(x=df["model"], y=df["TOPSIS Score"], palette="coolwarm")
plt.xlabel("Pre-trained Models")
plt.ylabel("TOPSIS Score")
plt.title("Ranking of Conversational Models using TOPSIS")
plt.xticks(rotation=20)

# Save the graph as a PNG file
plt.savefig("topsis_results_graph.png", format="png")

# Show the graph
plt.show()
