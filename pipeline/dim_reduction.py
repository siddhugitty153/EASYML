"""
Dimensionality Reduction Module
---------------------------------
Reduces number of features while preserving structure.
Supports PCA, LDA, t-SNE, UMAP, and Autoencoders.
"""

import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.manifold import TSNE


class DimensionalityReducer:
    """Reduce feature space dimensionality."""

    def __init__(self):
        self.models = {}
        self.report = {
            'method': None,
            'input_dims': 0,
            'output_dims': 0,
            'explained_variance': None,
            'actions': []
        }

    def get_report(self):
        return self.report

    def apply_pca(self, df: pd.DataFrame, n_components: int = None,
                  variance_ratio: float = 0.95) -> pd.DataFrame:
        """
        Apply PCA. If n_components is None, keep enough for variance_ratio.
        """
        numeric = df.select_dtypes(include=[np.number])
        non_numeric = df.select_dtypes(exclude=[np.number])

        data = numeric.dropna()
        if len(data) == 0 or len(data.columns) < 2:
            return df

        if n_components is None:
            pca = PCA(n_components=variance_ratio, svd_solver='full')
        else:
            n_components = min(n_components, len(data.columns), len(data))
            pca = PCA(n_components=n_components)

        transformed = pca.fit_transform(data)
        self.models['pca'] = pca

        pca_cols = [f'PC{i+1}' for i in range(transformed.shape[1])]
        result = pd.DataFrame(transformed, columns=pca_cols, index=data.index)

        # Re-attach non-numeric columns
        for col in non_numeric.columns:
            result[col] = df[col]

        self.report['method'] = 'pca'
        self.report['input_dims'] = len(numeric.columns)
        self.report['output_dims'] = transformed.shape[1]
        self.report['explained_variance'] = [round(float(v), 4) for v in pca.explained_variance_ratio_]
        self.report['total_variance_explained'] = round(float(sum(pca.explained_variance_ratio_)), 4)
        self.report['actions'].append(
            f'PCA: {len(numeric.columns)} → {transformed.shape[1]} dims '
            f'({self.report["total_variance_explained"]*100:.1f}% variance)'
        )
        return result

    def apply_lda(self, df: pd.DataFrame, target_col: str,
                  n_components: int = None) -> pd.DataFrame:
        """Apply Linear Discriminant Analysis (requires target labels)."""
        numeric = df.select_dtypes(include=[np.number])
        if target_col not in numeric.columns:
            return df

        X = numeric.drop(columns=[target_col]).dropna()
        y = df.loc[X.index, target_col]

        n_classes = len(y.unique())
        max_components = min(n_classes - 1, len(X.columns))
        if max_components < 1:
            return df

        n_components = min(n_components or max_components, max_components)
        lda = LDA(n_components=n_components)
        transformed = lda.fit_transform(X, y)
        self.models['lda'] = lda

        lda_cols = [f'LD{i+1}' for i in range(transformed.shape[1])]
        result = pd.DataFrame(transformed, columns=lda_cols, index=X.index)
        result[target_col] = y

        non_numeric = df.select_dtypes(exclude=[np.number])
        for col in non_numeric.columns:
            result[col] = df.loc[X.index, col]

        self.report['method'] = 'lda'
        self.report['input_dims'] = len(X.columns)
        self.report['output_dims'] = transformed.shape[1]
        if hasattr(lda, 'explained_variance_ratio_'):
            self.report['explained_variance'] = [round(float(v), 4) for v in lda.explained_variance_ratio_]
        self.report['actions'].append(f'LDA: {len(X.columns)} → {transformed.shape[1]} dims')
        return result

    def apply_tsne(self, df: pd.DataFrame, n_components: int = 2,
                   perplexity: float = 30.0) -> pd.DataFrame:
        """Apply t-SNE for visualization (2D or 3D)."""
        numeric = df.select_dtypes(include=[np.number])
        non_numeric = df.select_dtypes(exclude=[np.number])

        data = numeric.dropna()
        if len(data) == 0:
            return df

        n_components = min(n_components, 3)
        perplexity = min(perplexity, max(5.0, len(data) / 4))

        tsne = TSNE(n_components=n_components, perplexity=perplexity, random_state=42)
        transformed = tsne.fit_transform(data)

        tsne_cols = [f'tSNE{i+1}' for i in range(n_components)]
        result = pd.DataFrame(transformed, columns=tsne_cols, index=data.index)

        for col in non_numeric.columns:
            result[col] = df.loc[data.index, col]

        self.report['method'] = 'tsne'
        self.report['input_dims'] = len(numeric.columns)
        self.report['output_dims'] = n_components
        self.report['actions'].append(f't-SNE: {len(numeric.columns)} → {n_components} dims')
        return result

    def apply_umap(self, df: pd.DataFrame, n_components: int = 2,
                   n_neighbors: int = 15) -> pd.DataFrame:
        """Apply UMAP for dimensionality reduction."""
        try:
            import umap
        except ImportError:
            self.report['actions'].append('UMAP skipped: umap-learn not installed')
            return df

        numeric = df.select_dtypes(include=[np.number])
        non_numeric = df.select_dtypes(exclude=[np.number])

        data = numeric.dropna()
        if len(data) == 0:
            return df

        reducer = umap.UMAP(n_components=n_components, n_neighbors=n_neighbors, random_state=42)
        transformed = reducer.fit_transform(data)
        self.models['umap'] = reducer

        umap_cols = [f'UMAP{i+1}' for i in range(n_components)]
        result = pd.DataFrame(transformed, columns=umap_cols, index=data.index)

        for col in non_numeric.columns:
            result[col] = df.loc[data.index, col]

        self.report['method'] = 'umap'
        self.report['input_dims'] = len(numeric.columns)
        self.report['output_dims'] = n_components
        self.report['actions'].append(f'UMAP: {len(numeric.columns)} → {n_components} dims')
        return result

    def apply_autoencoder(self, df: pd.DataFrame, encoding_dim: int = 10,
                          epochs: int = 50) -> pd.DataFrame:
        """Apply autoencoder-based dimensionality reduction."""
        try:
            import tensorflow as tf
            from tensorflow import keras
        except ImportError:
            self.report['actions'].append('Autoencoder skipped: tensorflow not installed')
            return df

        numeric = df.select_dtypes(include=[np.number])
        non_numeric = df.select_dtypes(exclude=[np.number])

        data = numeric.dropna()
        if len(data) == 0 or len(data.columns) < 2:
            return df

        input_dim = len(data.columns)
        encoding_dim = min(encoding_dim, input_dim - 1)
        if encoding_dim < 1:
            encoding_dim = 1

        # Normalize input
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(data)

        # Build autoencoder
        input_layer = keras.layers.Input(shape=(input_dim,))
        # Encoder
        encoded = keras.layers.Dense(max(input_dim // 2, encoding_dim + 1), activation='relu')(input_layer)
        encoded = keras.layers.Dense(encoding_dim, activation='relu', name='encoding')(encoded)
        # Decoder
        decoded = keras.layers.Dense(max(input_dim // 2, encoding_dim + 1), activation='relu')(encoded)
        decoded = keras.layers.Dense(input_dim, activation='linear')(decoded)

        autoencoder = keras.Model(input_layer, decoded)
        encoder = keras.Model(input_layer, autoencoder.get_layer('encoding').output)

        autoencoder.compile(optimizer='adam', loss='mse')
        autoencoder.fit(X_scaled, X_scaled, epochs=epochs, batch_size=32,
                        verbose=0, validation_split=0.1)

        transformed = encoder.predict(X_scaled, verbose=0)
        self.models['autoencoder'] = {'encoder': encoder, 'scaler': scaler}

        ae_cols = [f'AE{i+1}' for i in range(transformed.shape[1])]
        result = pd.DataFrame(transformed, columns=ae_cols, index=data.index)

        for col in non_numeric.columns:
            result[col] = df.loc[data.index, col]

        self.report['method'] = 'autoencoder'
        self.report['input_dims'] = input_dim
        self.report['output_dims'] = encoding_dim
        self.report['actions'].append(f'Autoencoder: {input_dim} → {encoding_dim} dims ({epochs} epochs)')
        return result

    def reduce(self, df: pd.DataFrame, method: str = 'pca',
               config: dict = None) -> pd.DataFrame:
        """
        Apply dimensionality reduction.

        Config keys vary by method:
            pca:  n_components, variance_ratio
            lda:  target_col, n_components
            tsne: n_components, perplexity
            umap: n_components, n_neighbors
            autoencoder: encoding_dim, epochs
        """
        config = config or {}

        if method == 'pca':
            return self.apply_pca(df, **{k: v for k, v in config.items()
                                         if k in ('n_components', 'variance_ratio')})
        elif method == 'lda':
            target = config.get('target_col')
            if not target:
                self.report['actions'].append('LDA skipped: target_col required')
                return df
            return self.apply_lda(df, target, n_components=config.get('n_components'))
        elif method == 'tsne':
            return self.apply_tsne(df, n_components=config.get('n_components', 2),
                                   perplexity=config.get('perplexity', 30.0))
        elif method == 'umap':
            return self.apply_umap(df, n_components=config.get('n_components', 2),
                                   n_neighbors=config.get('n_neighbors', 15))
        elif method == 'autoencoder':
            return self.apply_autoencoder(df, encoding_dim=config.get('encoding_dim', 10),
                                          epochs=config.get('epochs', 50))
        return df
