import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
np.random.seed(42)
n = 200

age = np.random.randint(18, 70, n)
annual_income = np.random.randint(20000, 120000, n)
spending_score = np.random.randint(1, 100, n)

df = pd.DataFrame({
    "Age": age,
    "Annual Income": annual_income,
    "Spending Score": spending_score
})

print("Dataset preview:")
print(df.head(), "\n")

features = df[['Age', 'Annual Income', 'Spending Score']]

scaler = StandardScaler()
scaled_features = scaler.fit_transform(features)

inertia_values = []

K_range = range(1, 11)
for k in K_range:
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(scaled_features)
    inertia_values.append(kmeans.inertia_)

plt.plot(K_range, inertia_values, marker='o')
plt.xlabel("Number of Clusters (k)")
plt.ylabel("Inertia")
plt.title("Elbow Method to Find Optimal k")
plt.grid(True)
plt.show()
optimal_k = 5  # typical for this dataset
kmeans = KMeans(n_clusters=optimal_k, random_state=42)
cluster_labels = kmeans.fit_predict(scaled_features)

df["Cluster"] = cluster_labels
plt.figure(figsize=(8, 5))
for c in range(optimal_k):
    data = df[df["Cluster"] == c]
    plt.scatter(data["Annual Income"], data["Spending Score"], label=f"Cluster {c}")

plt.xlabel("Annual Income")
plt.ylabel("Spending Score")
plt.title("Customer Segments (Income vs Spending)")
plt.legend()
plt.grid(True)
plt.show()
cluster_profiles = df.groupby("Cluster").agg({
    "Age": "mean",
    "Annual Income": "mean",
    "Spending Score": "mean",
    "Cluster": "count"
}).rename(columns={"Cluster": "Count"})

print("\nCluster Profiles:")
print(cluster_profiles, "\n")
def describe_segment(row):
    age = row["Age"]
    income = row["Annual Income"]
    score = row["Spending Score"]

    if income > 80000 and score > 60:
        return "High-income, high-spending (Premium customers)"
    elif income > 80000 and score < 40:
        return "High-income, low-spending (Potential upscale targets)"
    elif income < 40000 and score > 60:
        return "Low-income, high-spending (Impulsive buyers)"
    elif income < 40000 and score < 40:
        return "Low-income, low-spending (Budget-conscious group)"
    else:
        return "Mid-income moderate spenders (General market)"

report = cluster_profiles.apply(describe_segment, axis=1)
print("Segment Reports:\n")
for i, txt in report.items():
    print(f"Cluster {i}: {txt}")
df.to_csv("segmented_customers.csv", index=False)
print("\nSegmented dataset saved as segmented_customers.csv")
