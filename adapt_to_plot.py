import os
import shutil
import pandas as pd

# Carrega seu CSV original

def adapt_to_plot():

    df = pd.read_csv("dados_com_outliers_flag.csv")

    rows = []

    for _, r in df.iterrows():

        base = {
            "image_name": r["image_name"],
            "fold": r["fold"],
            "species": r["species"],
            "dist": r["dist"],
            "pattern_size": r["pattern-size"],
            "is_outlier": r["is_outlier_global"],
        }

        # AREA
        rows.append({
            **base,
            "metric": "area",
            "real": r["real_area_cm2"],
            "annot": r["area_annot"],
            "pred": r["pred_area_cm2"],
            "annot_RER": r["area_annot_rer"],
            "pred_RER": r["area_rer"],
            "method_RER": r["area_method_rer"],
        })

        # PERIMETER
        rows.append({
            **base,
            "metric": "perimeter",
            "real": r["real_perimeter_cm"],
            "annot": r["perim_annot"],
            "pred": r["pred_perimeter_cm"],
            "annot_RER": r["perim_annot_rer"],
            "pred_RER": r["pred_perimeter_rer"],
            "method_RER": r["perim_method_rer"],
        })

        # LENGTH
        rows.append({
            **base,
            "metric": "length",
            "real": r["real_length_cm"],
            "annot": r["length_annot"],
            "pred": r["pred_length_cm"],
            "annot_RER": r["length_annot_rer"],
            "pred_RER": r["length_rer"],
            "method_RER": r["length_method_rer"],
        })

        # WIDTH
        rows.append({
        **base,
        "metric": "width",
        "real": r["real_width_cm"],
        "annot": r["width_annot"],
        "pred": r["pred_width_cm"],
        "annot_RER": r["width_annot_rer"],
        "pred_RER": r["width_rer"],
        "method_RER": r["width_method_rer"],
        })

    # Novo DataFrame
    df_new = pd.DataFrame(rows)

    # Salva CSV adaptado
    df_new.to_csv("dados_pra_grafico.csv", index=False)

    print("✅ CSV gerado: dados_pra_grafico.csv")

    # =========================================================
    # CRIA CSV SOMENTE COM OUTLIERS
    # =========================================================

    df_outliers = df[df["is_outlier_global"] == True]

    df_outliers.to_csv("imagens_outliers.csv", index=False)

    print(f"✅ CSV de outliers gerado: imagens_outliers.csv")
    print(f"Total de outliers: {len(df_outliers)}")

    # =========================================================
    # OPCIONAL: COPIAR IMAGENS DOS OUTLIERS PARA UMA PASTA
    # =========================================================

    INPUT_IMAGE_DIR = "images"  # ajuste aqui
    OUTPUT_DIR = "outlier_images"

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    for _, r in df_outliers.iterrows():

        img_name = r["image_name"]

        src = os.path.join(INPUT_IMAGE_DIR, img_name)
        dst = os.path.join(OUTPUT_DIR, img_name)

        if os.path.exists(src):
            shutil.copy2(src, dst)

    print(f"✅ Imagens dos outliers copiadas para: {OUTPUT_DIR}")

