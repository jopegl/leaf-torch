import pandas as pd

import pandas as pd

def complete_csv():
    df = pd.read_csv("FINAL_all_leaf_metrics.csv")

    rer_cols = [
        "area_rer",
        "area_annot_rer",
        "pred_perimeter_rer",
        "perim_annot_rer",
        "length_rer",
        "length_annot_rer"
    ]

    def detect_outliers_iqr(series):
        Q1 = series.quantile(0.25)
        Q3 = series.quantile(0.75)
        IQR = Q3 - Q1

        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR

        return (series < lower) | (series > upper)

    # cria uma coluna de outlier para cada métrica
    for col in rer_cols:
        df[f"is_outlier_{col}"] = detect_outliers_iqr(df[col])

    # opcional: quantidade total de flags
    outlier_cols = [f"is_outlier_{col}" for col in rer_cols]

    df["n_outlier_metrics"] = df[outlier_cols].sum(axis=1)

    # opcional: outlier global
    df["is_outlier_global"] = df["n_outlier_metrics"] >= 2

    df.to_csv("dados_com_outliers_flag.csv", index=False)

    print(df[outlier_cols + ["n_outlier_metrics", "is_outlier_global"]].sum())

if __name__ == '__main__':
    complete_csv()