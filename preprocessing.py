from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

def scale_data(X):
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    return X_scaled, scaler

def apply_pca(X_scaled, n_components=0.95):
    pca = PCA(n_components=n_components, random_state=42)
    X_pca = pca.fit_transform(X_scaled)
    print(f"📉 PCA reduced dimensions: {X_pca.shape[1]}")
    return X_pca, pca
