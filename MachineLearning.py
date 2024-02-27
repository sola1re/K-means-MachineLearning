from sklearn.cluster import KMeans
from sklearn import cluster, datasets, mixture
from sklearn import preprocessing
from sklearn.datasets import make_blobs
import plotly.express as px
import pandas as pd
from sklearn.datasets import make_moons


#X, y = make_blobs(n_samples=100, centers=3, cluster_std=1, n_features=2,center_box=(-5.0, 5.0),shuffle=True, random_state = 11)

X, y =make_moons(n_samples=200, noise=0.05,random_state=0)


# Convert to dataframe:
dfB = pd.DataFrame(X, columns = ['X','Y'])
print(dfB)

# Plot data:
plot = px.scatter(dfB, x="X", y="Y")
plot.update_layout(   
    title={'text':"Data",
           'xanchor':'center',
           'yanchor':'top',
           'x':0.5})
plot.show()

kmeans = KMeans(
    init="random",
    n_clusters=3,
    n_init=10,
    max_iter=300,
    random_state=11
)

kmeans.fit(dfB)
print(kmeans.cluster_centers_)

from sklearn.datasets import make_moons
X, y =make_moons(n_samples=200, noise=0.05,random_state=0)
