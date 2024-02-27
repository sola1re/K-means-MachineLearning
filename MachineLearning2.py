import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import pandas as pd

df = pd.read_csv('StudentsPerformance.csv')

X = df[['math score', 'reading score', 'writing score']]

scaler = StandardScaler()
X_norm = scaler.fit_transform(X)

inertias = []
for k in range(2, 10):
    kmeans = KMeans(n_clusters=k, random_state=11)
    kmeans.fit(X_norm)
    inertias.append(kmeans.inertia_)

plt.plot(range(2, 10), inertias, marker='o')
plt.xlabel('Number of clusters')
plt.ylabel('Inertia')
plt.title('Inertia vs. Number of Clusters')
plt.show()

# The optimal number of clusters can be determined by looking for a "knee" in the plot,
# where the inertia decreases significantly from one value of K to the next.
