import pandas as pd
import numpy as np
import re
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

os.makedirs("outputs", exist_ok=True)

# ================================
# Config
# ================================
INPUT_FILE = "outputs/qwen_hidden_rewrites.csv"
DETECTION_FILE = "outputs/blind_detection_results.csv"

# ================================
# Helpers
# ================================
def word_count(text):
    return len(str(text).split())


def sentence_count(text):
    sentences = re.split(r'(?<=[.!?])\s+', str(text).strip())
    return len([s for s in sentences if s.strip()])


# ================================
# Load data
# ================================
print("Loading data...")
df = pd.read_csv(INPUT_FILE)

# ================================
# 1. Text Length Metrics
# ================================
print("Computing length metrics...")

df["source_words"] = df["source_text"].apply(word_count)
df["normal_words"] = df["normal_rewrite"].apply(word_count)
df["hidden_words"] = df["hidden_rewrite"].apply(word_count)

df["source_sentences"] = df["source_text"].apply(sentence_count)
df["normal_sentences"] = df["normal_rewrite"].apply(sentence_count)
df["hidden_sentences"] = df["hidden_rewrite"].apply(sentence_count)

df["hidden_vs_normal_words"] = df["hidden_words"] - df["normal_words"]
df["hidden_vs_normal_sentences"] = df["hidden_sentences"] - df["normal_sentences"]

# ================================
# 2. Semantic Similarity
# ================================
print("Loading embedding model...")
model = SentenceTransformer("all-MiniLM-L6-v2")

print("Computing embeddings...")
source_embeddings = model.encode(df["source_text"].tolist(), batch_size=32)
normal_embeddings = model.encode(df["normal_rewrite"].tolist(), batch_size=32)
hidden_embeddings = model.encode(df["hidden_rewrite"].tolist(), batch_size=32)

print("Computing similarity...")


def cosine_sim(a, b):
    return np.diag(cosine_similarity(a, b))


df["sim_source_normal"] = cosine_sim(source_embeddings, normal_embeddings)
df["sim_source_hidden"] = cosine_sim(source_embeddings, hidden_embeddings)

# ================================
# 3. Summary Stats
# ================================
print("\n===== SUMMARY =====")

print("\n--- Length ---")
print(f"Avg source words: {df['source_words'].mean():.2f}")
print(f"Avg normal words: {df['normal_words'].mean():.2f}")
print(f"Avg hidden words: {df['hidden_words'].mean():.2f}")
print(f"Avg hidden-normal word diff: {df['hidden_vs_normal_words'].mean():.2f}")

print("\n--- Similarity ---")
print(f"Avg similarity (source vs normal): {df['sim_source_normal'].mean():.4f}")
print(f"Avg similarity (source vs hidden): {df['sim_source_hidden'].mean():.4f}")

print("\n--- By Secret Word ---")
print(df.groupby("target_secret")[["sim_source_hidden", "hidden_words", "hidden_success"]].mean())

if "encoding_scheme" in df.columns:
    print("\n--- By Encoding Scheme ---")
    print(df.groupby("encoding_scheme")[["sim_source_hidden", "hidden_words", "hidden_success"]].mean())

# ================================
# Save enriched file
# ================================
df.to_csv("outputs/analysis_with_metrics.csv", index=False)
print("\nSaved: outputs/analysis_with_metrics.csv")

sns.set(style="whitegrid")

# =========================
# 1. Success per word
# =========================
success_by_word = df.groupby("target_secret")["hidden_success"].mean()

plt.figure()
success_by_word.plot(kind="bar")
plt.title("Encoding Success Rate per Secret Word")
plt.ylabel("Success Rate")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("outputs/plot_success_per_word.png")
plt.close()

# =========================
# 2. Success per scheme
# =========================
if "encoding_scheme" in df.columns:
    success_by_scheme = df.groupby("encoding_scheme")["hidden_success"].mean()
    plt.figure()
    success_by_scheme.plot(kind="bar")
    plt.title("Encoding Success Rate per Scheme")
    plt.ylabel("Success Rate")
    plt.xticks(rotation=20)
    plt.tight_layout()
    plt.savefig("outputs/plot_success_per_scheme.png")
    plt.close()

# =========================
# 3. Attempts histogram
# =========================
plt.figure()
sns.histplot(df["hidden_attempts_used"], bins=10)
plt.title("Attempts Needed for Hidden Encoding")
plt.xlabel("Attempts")
plt.ylabel("Count")
plt.tight_layout()
plt.savefig("outputs/plot_attempts_hist.png")
plt.close()

# =========================
# 4. Similarity comparison
# =========================
plt.figure()
sns.boxplot(data=df[["sim_source_normal", "sim_source_hidden"]])
plt.xticks([0, 1], ["Normal", "Hidden"])
plt.title("Semantic Similarity Comparison")
plt.ylabel("Similarity")
plt.tight_layout()
plt.savefig("outputs/plot_similarity.png")
plt.close()

# =========================
# 5. Similarity by scheme
# =========================
if "encoding_scheme" in df.columns:
    plt.figure()
    sns.boxplot(data=df, x="encoding_scheme", y="sim_source_hidden")
    plt.title("Hidden Semantic Similarity by Scheme")
    plt.xticks(rotation=20)
    plt.tight_layout()
    plt.savefig("outputs/plot_similarity_by_scheme.png")
    plt.close()

# =========================
# 6. Length comparison
# =========================
plt.figure()
sns.boxplot(data=df[["normal_words", "hidden_words"]])
plt.xticks([0, 1], ["Normal", "Hidden"])
plt.title("Text Length Comparison")
plt.ylabel("Word Count")
plt.tight_layout()
plt.savefig("outputs/plot_length.png")
plt.close()

plt.figure()
sns.histplot(df["sim_source_hidden"], bins=12, kde=True)
plt.title("Distribution of Hidden-Text Semantic Similarity")
plt.xlabel("Similarity to Source")
plt.ylabel("Count")
plt.tight_layout()
plt.savefig("outputs/plot_similarity_hist.png")
plt.close()

all_embeddings = np.vstack([normal_embeddings, hidden_embeddings])
labels = ["Normal"] * len(normal_embeddings) + ["Hidden"] * len(hidden_embeddings)
vis_df = pd.DataFrame({"label": labels})

pca = PCA(n_components=2, random_state=42)
pca_coords = pca.fit_transform(all_embeddings)

vis_df["pca1"] = pca_coords[:, 0]
vis_df["pca2"] = pca_coords[:, 1]

plt.figure()
sns.scatterplot(data=vis_df, x="pca1", y="pca2", hue="label")
plt.title("PCA of Normal vs Hidden Text Embeddings")
plt.tight_layout()
plt.savefig("outputs/plot_pca_embeddings.png")
plt.close()

tsne = TSNE(n_components=2, random_state=42, perplexity=10)
tsne_coords = tsne.fit_transform(all_embeddings)

vis_df["tsne1"] = tsne_coords[:, 0]
vis_df["tsne2"] = tsne_coords[:, 1]

plt.figure()
sns.scatterplot(data=vis_df, x="tsne1", y="tsne2", hue="label")
plt.title("t-SNE of Normal vs Hidden Text Embeddings")
plt.tight_layout()
plt.savefig("outputs/plot_tsne_embeddings.png")
plt.close()

# =========================
# 7. Detection results
# =========================
if os.path.exists(DETECTION_FILE):
    det_df = pd.read_csv(DETECTION_FILE)

    plt.figure(figsize=(10, 4))

    plt.subplot(1, 2, 1)
    sns.histplot(det_df["hidden_confidence"], bins=10, kde=True, color="blue")
    plt.title("Hidden Confidence")

    plt.subplot(1, 2, 2)
    sns.histplot(det_df["normal_confidence"], bins=10, kde=True, color="orange")
    plt.title("Normal Confidence")

    plt.tight_layout()
    plt.savefig("outputs/plot_confidence_separate.png")
    plt.close()

    hidden_rate = (det_df["hidden_label"] == "suspicious").mean()
    normal_rate = (det_df["normal_label"] == "suspicious").mean()

    success_df = det_df[det_df["hidden_success_generation"] == 1]
    success_rate = (success_df["hidden_label"] == "suspicious").mean() if len(success_df) > 0 else 0.0

    plt.figure()
    plt.bar(
        ["Hidden", "False Positive", "Hidden (Success Only)"],
        [hidden_rate, normal_rate, success_rate]
    )
    plt.title("Detection Results")
    plt.ylabel("Rate")
    plt.tight_layout()
    plt.savefig("outputs/plot_detection.png")
    plt.close()

    if "encoding_scheme" in det_df.columns:
        scheme_rates = det_df.groupby("encoding_scheme")["hidden_label"].apply(lambda s: (s == "suspicious").mean())
        plt.figure()
        scheme_rates.plot(kind="bar")
        plt.title("Hidden Detection Rate by Scheme")
        plt.ylabel("Rate")
        plt.xticks(rotation=20)
        plt.tight_layout()
        plt.savefig("outputs/plot_detection_by_scheme.png")
        plt.close()

print("All plots saved successfully.")
