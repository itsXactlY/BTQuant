
import numpy as np
import polars as pl
from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import permutation_importance
from sklearn.preprocessing import RobustScaler
from rich.console import Console

console = Console()

def extract_and_select_features(df: pl.DataFrame, n_features: int = 15) -> tuple:

    try:

        feature_cols = [col for col in df.columns if col.startswith((
            'RSI', 'MACD', 'ATR', 'ADX', 'CCI', 'Stoch', 'WillR',
            'MOM', 'ROC', 'OBV', 'MFI', 'CMO', 'SAR'
        ))]

        if len(feature_cols) == 0:
            console.print("❌ No indicator columns found. Available columns:", df.columns)
            return None, None

        console.print(f"🔍 Feature selection on {len(feature_cols)} candidates...")

        df_with_target = df.with_columns(
            (pl.col('close').shift(-1) / pl.col('close') - 1.0).alias('target')
        )

        if 'target' not in df_with_target.columns:
            console.print("❌ Target creation failed")
            return None, None

        df_clean = df_with_target.select(['target'] + feature_cols).drop_nulls()

        console.print(f"📊 Clean data: {len(df_clean):,} rows (from {len(df_with_target)})")

        if len(df_clean) < 100:
            console.print(f"❌ Insufficient clean data: {len(df_clean)} < 100")
            return None, None

        try:
            X = df_clean.select(feature_cols).to_numpy()
            y = df_clean['target'].to_numpy()
        except Exception as e:
            console.print(f"❌ Data extraction failed: {e}")
            return None, None

        if X.shape[0] != y.shape[0]:
            console.print(f"❌ Shape mismatch: X={X.shape[0]}, y={y.shape[0]}")
            return None, None

        if X.shape[1] != len(feature_cols):
            console.print(f"❌ Feature count mismatch: X.shape[1]={X.shape[1]}, expected={len(feature_cols)}")
            return None, None

        split_idx = int(0.8 * len(X))
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]

        if len(X_train) < 50 or len(X_test) < 20:
            console.print(f"❌ Insufficient split sizes: train={len(X_train)}, test={len(X_test)}")
            return None, None

        console.print(f"Train/Test split: {len(X_train):,}/{len(X_test):,} rows")

        try:
            rf = RandomForestRegressor(n_estimators=50, random_state=42, n_jobs=-1)
            rf.fit(X_train, y_train)
        except Exception as e:
            console.print(f"❌ RF training failed: {e}")
            return None, None

        try:
            perm = permutation_importance(
                rf, X_test, y_test, n_repeats=5, random_state=42,
                scoring='r2', n_jobs=-1
            )
        except Exception as e:
            console.print(f"❌ Permutation importance failed: {e}")
            return None, None

        importances = perm.importances_mean
        top_idx = np.argsort(importances)[-n_features:]
        selected = [feature_cols[i] for i in top_idx if importances[i] > 0.01]

        if len(selected) < n_features:
            candidates = [feature_cols[i] for i in np.argsort(importances)[::-1]
                          if feature_cols[i] not in selected]
            to_add = candidates[:(n_features - len(selected))]
            selected += to_add

        selected = list(dict.fromkeys(selected))[:n_features]

        if len(selected) == 0:
            console.print("❌ Feature selection produced 0 features")
            return None, None

        try:
            if len(selected) > 1:
                sel_idx = [feature_cols.index(f) for f in selected]
                corr_matrix = np.corrcoef(X[:, sel_idx].T)
                to_keep = set(range(len(selected)))

                for i in range(len(selected)):
                    for j in range(i + 1, len(selected)):
                        if abs(corr_matrix[i, j]) > 0.85:
                            if importances[feature_cols.index(selected[i])] > importances[feature_cols.index(selected[j])]:
                                to_keep.discard(j)
                            else:
                                to_keep.discard(i)

                selected = [selected[i] for i in sorted(to_keep)]
        except Exception as e:
            console.print(f"⚠️ Correlation filtering failed: {e}. Using unfiltered features.")

        if len(selected) == 0:
            console.print("❌ All features removed by correlation filtering")
            return None, None

        try:
            sel_idx = [feature_cols.index(s) for s in selected]
            scaler = RobustScaler().fit(X[:, sel_idx])
        except Exception as e:
            console.print(f"❌ Scaler fitting failed: {e}")
            return None, None

        console.print(f"✅ Selected {len(selected)} features: {selected[:5]}...")

        return selected, scaler

    except Exception as e:
        console.print(f"❌ CRITICAL ERROR in extract_and_select_features: {e}")
        import traceback
        traceback.print_exc()
        return None, None

