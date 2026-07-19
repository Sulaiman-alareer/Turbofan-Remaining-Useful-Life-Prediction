import pandas as pd

RUL_CLIP_VALUE = 125

COLUMNS = (
    ["engine_id", "cycle"]
    + [f"setting_{i}" for i in range(1, 4)]
    + [f"sensor_{i}" for i in range(1, 22)]
)

FEATURE_COLUMNS = (
    ["cycle"]
    + [f"setting_{i}" for i in range(1, 4)]
    + [f"sensor_{i}" for i in range(1, 22)]
)


def load_train_data(path, clip_rul=True):
    df = pd.read_csv(path, sep=r"\s+", header=None, names=COLUMNS)

    max_cycle = df.groupby("engine_id")["cycle"].max().reset_index()
    max_cycle.columns = ["engine_id", "max_cycle"]

    df = df.merge(max_cycle, on="engine_id")
    df["RUL"] = df["max_cycle"] - df["cycle"]

    if clip_rul:
        df["RUL"] = df["RUL"].clip(upper=RUL_CLIP_VALUE)

    df = df.drop(columns=["max_cycle"])
    return df


def load_test_data(test_path, rul_path, clip_rul=True):
    test_df = pd.read_csv(test_path, sep=r"\s+", header=None, names=COLUMNS)
    rul_df = pd.read_csv(rul_path, sep=r"\s+", header=None, names=["true_RUL"])

    last_cycle = test_df.groupby("engine_id")["cycle"].max().reset_index()
    last_cycle.columns = ["engine_id", "last_cycle"]

    test_last = test_df.merge(last_cycle, on="engine_id")
    test_last = test_last[test_last["cycle"] == test_last["last_cycle"]]
    test_last = test_last.drop(columns=["last_cycle"])

    test_last = test_last.sort_values("engine_id").reset_index(drop=True)
    test_last["RUL"] = rul_df["true_RUL"]

    if clip_rul:
        test_last["RUL"] = test_last["RUL"].clip(upper=RUL_CLIP_VALUE)

    return test_last


def split_features_target(df):
    x = df[FEATURE_COLUMNS]
    y = df["RUL"]
    return x, y


if __name__ == "__main__":
    train_fd001 = load_train_data("data/train_FD001.txt")
    test_fd001 = load_test_data("data/test_FD001.txt", "data/RUL_FD001.txt")

    train_fd003 = load_train_data("data/train_FD003.txt")
    test_fd003 = load_test_data("data/test_FD003.txt", "data/RUL_FD003.txt")

    print("RUL clipping value:", RUL_CLIP_VALUE)
    print("FD001 Train:", train_fd001.shape)
    print("FD001 Test:", test_fd001.shape)
    print("FD003 Train:", train_fd003.shape)
    print("FD003 Test:", test_fd003.shape)
    print("\nFD001 RUL stats after clipping:")
    print(train_fd001["RUL"].describe())
    print("\nFD001 sample:")
    print(train_fd001.head())
