from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

from price_features import features

def run_kmeans(features_df, k=3):
    scaler = StandardScaler()
    scaled = scaler.fit_transform(features_df)

    model = KMeans(n_clusters=k, random_state=42, n_init=10)
    clusters = model.fit_predict(scaled)

    result = features_df.copy()
    print(result)
    result["cluster"] = clusters
    print(result)
    return result, model, scaler
    

run_kmeans = run_kmeans(features)


