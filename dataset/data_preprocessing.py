import pandas as pd
import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.preprocessing import LabelEncoder
from ucimlrepo import fetch_ucirepo


def load_bank_marketing():
    # source: https://archive.ics.uci.edu/dataset/222/bank+marketing
    # The dataset contains 45211 samples and 16 features, with a binary target variable 'y' indicating whether the client subscribed to a term deposit
    # feature engineering source: https://www.kaggle.com/code/tajgerka/bank-marketing-dataset-subs-prediction#Data-Cleaning-&-Feature-Engineering
    # V6 - balance: client's account balance (numeric)
    # V13 - campaign: number of contacts performed during this campaign and for this client(numeric, includes last contact)
    # V14 - pdays
    data = fetch_openml(name="bank-marketing", version=1, as_frame=True, parser="auto")
    df = data.frame.copy()
    df = df.dropna()

    df["Class"] = (df["Class"] == "2").astype(int)  # classes are strings '1' and '2', convert to binary 0 and 1

    # Preprocessing to improve the prediction
    df['pdays_clean'] = df['V14'].replace(-1, 999)
    df['was_contacted_before'] = (df['V14'] != -1).astype(int)

    df['balance_log'] = np.sign(df['V6']) * np.log1p(np.abs(df['V6']))

    upper_limit = df['V13'].quantile(0.95)
    df['camp_capped'] = np.where(df['V13'] > upper_limit, upper_limit, df['V13'])

    df = df.drop(columns=['V6', 'V13', 'V14'])  # drop original columns after feature engineering

    # Separate features and target
    X = df.drop(columns=["Class"])
    y = df["Class"]

    # Encode categorical columns
    cat_cols = X.select_dtypes(include=["category", "object"]).columns
    # X = pd.get_dummies(X, columns=cat_cols, drop_first=False, dtype=float)
    for col in cat_cols:
        X[col] = LabelEncoder().fit_transform(X[col].astype(str))

    X = X.values.astype(float)
    y = y.values

    print(f"N samples is {y.shape}, positive class is {y.sum()}, percentage is {y.mean():.4f}")
    return X, y


def load_adult_income():
    print("Downloading Adult Income dataset...")
    data = fetch_openml(name="adult", version=2, as_frame=True, parser="auto")
    df = data.frame.copy()

    # Target: '>50K' -> 1, '<=50K' -> 0
    target_col = data.target_names[0]
    y = (data.target.str.strip().str.replace(".", "", regex=False) == ">50K").astype(int)

    X = df.drop(columns=[target_col])

    # Drop rows with missing values (marked as '?' in this dataset)
    mask = ~(X == "?").any(axis=1)
    X = X[mask].reset_index(drop=True)
    y = y[mask].reset_index(drop=True)

    # Encode categoricals
    cat_cols = X.select_dtypes(include=["category", "object"]).columns
    # X = pd.get_dummies(X, columns=cat_cols, drop_first=False, dtype=float)
    for col in cat_cols:
        X[col] = LabelEncoder().fit_transform(X[col].astype(str))

    X = X.values.astype(float)
    y = y.values
    return X, y


def load_gamma_telescope():
    print("Downloading MAGIC Gamma Telescope dataset...")
    magic_gamma_telescope = fetch_ucirepo(id=159)

    # data (as pandas dataframes)
    X = magic_gamma_telescope.data.features.values.astype(float)
    y = magic_gamma_telescope.data.targets

    y = (y == "g").values.astype(int)  # gamma=1, hadron=0

    return X, y


def make_windows(sequence, targets, seq_len, step=1):
    X, y = [], []
    for start in range(0, len(sequence) - seq_len + 1, step):
        X.append(sequence[start:start + seq_len])
        y.append(targets[start + seq_len - 1])
    return np.stack(X), np.array(y)


def load_occupancy(seq_len=50, step=1):
    # the dataset has three recording sessions as time series
    # the time series are clumped together with their header rows
    repo = fetch_ucirepo(id=357)
    X_raw = repo.data.features.copy()
    y_raw = repo.data.targets.copy()
    X_raw["Occupancy"] = y_raw["Occupancy"].values

    # detect repeated header rows to identify session boundaries
    header_rows = X_raw.index[X_raw["date"] == "Temperature"].tolist()
    boundaries = [0] + header_rows + [len(X_raw)]

    sessions = []
    for start, end in zip(boundaries, boundaries[1:]):
        session = X_raw.iloc[start:end]
        session = session[session["date"] != "Temperature"].reset_index(drop=True)
        session = session.dropna()
        sessions.append(session)

    # sessions[0]=train, sessions[1]=test1, sessions[2]=test2
    def preprocess(session):
        session = session.copy()
        session["date"] = pd.to_datetime(session["date"])
        session["seconds_since_midnight"] = (
            session["date"].dt.hour * 3600 +
            session["date"].dt.minute * 60 +
            session["date"].dt.second
        ).astype(float)
        # session["day_of_week"] = session["date"].dt.dayofweek.astype(float)
        # session["is_weekend"] = (session["date"].dt.dayofweek >= 5).astype(float)
        session = session.drop(columns=["date"])
        return session

    sessions = [preprocess(s) for s in sessions]

    feature_cols = [c for c in sessions[0].columns if c != "Occupancy"]

    def window_session(session):
        return make_windows(
            session[feature_cols].values.astype(np.float32),
            session["Occupancy"].values.astype(np.int64),
            seq_len, step
        )

    X_train1, y_train1 = window_session(sessions[0])
    X_test,   y_test   = window_session(sessions[1])
    X_train2, y_train2 = window_session(sessions[2])

    X = np.concatenate([X_train1, X_train2, X_test]).transpose(0, 2, 1)
    y = np.concatenate([y_train1, y_train2, y_test])
    train_idx = len(X_train1) + len(X_train2)

    print(f"features      : {feature_cols}")
    print(f"X shape       : {X.shape}  (N, features, seq_len)")
    print(f"train_idx     : {train_idx}")
    print(f"class balance : {y.mean():.1%} occupied")

    return X, y, train_idx


def load_air_quality(seq_len=50, step=1, target_col="CO(GT)"):
    """
    UCI Air Quality dataset — regression.
    target: ground truth CO concentration in mg/m^3 (default).
    missing values encoded as -200, rows with missing target are dropped.
    returns X, y with shape (N, features, seq_len) and train_idx=None (random split)
    """
    print(f"Loading air quality dataset with target {target_col}...")
    repo = fetch_ucirepo(id=360)
    X_raw = repo.data.features.copy()

    # replace -200 (missing value) with NaN, apply forward fill (use the value from the nearest point)
    X_raw = X_raw.replace(-200, np.nan).ffill()

    # drop NMHC(GT) — >90% missing
    X_raw = X_raw.drop(columns=[c for c in X_raw.columns if "NMHC" in c])

    # parse datetime
    X_raw["DateTime"] = pd.to_datetime(
        X_raw["Date"].astype(str) + " " + X_raw["Time"].astype(str),
        # dayfirst=True
        format="mixed"  # handle both D/M/Y and M/D/Y formats (dataset has inconsistent date formats)
    )
    X_raw = X_raw.drop(columns=["Date", "Time"])

    X_raw["seconds_since_midnight"] = (
        X_raw["DateTime"].dt.hour * 3600 +
        X_raw["DateTime"].dt.minute * 60 +
        X_raw["DateTime"].dt.second
    ).astype(float)

    X_raw["day_of_week"] = X_raw["DateTime"].dt.dayofweek.astype(float)
    X_raw["month"] = X_raw["DateTime"].dt.month.astype(float)
    X_raw["is_weekend"] = (X_raw["DateTime"].dt.dayofweek >= 5).astype(float)
    X_raw = X_raw.drop(columns=["DateTime"])

    # forward fill remaining missing sensor values (small gaps in a continuous signal)
    feature_cols = [c for c in X_raw.columns if c != target_col]
    X_raw[feature_cols] = X_raw[feature_cols].ffill()

    X_win, y_win = make_windows(
        X_raw[feature_cols].values.astype(np.float32),
        X_raw[target_col].values.astype(np.float32),
        seq_len, step
    )

    # (N, seq_len, features) -> (N, features, seq_len)
    X_win = X_win.transpose(0, 2, 1)

    print(f"features      : {feature_cols}")
    print(f"X shape       : {X_win.shape}  (N, features, seq_len)")
    print(f"target        : {target_col}")

    return X_win, y_win

