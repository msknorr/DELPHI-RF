import os
import pandas as pd
from sklearn.model_selection import StratifiedKFold
import numpy as np
import inspect

from config import TrainGlobalConfig
from score2 import calculate_score2

config = TrainGlobalConfig()


def create_hchs_dataframe():
    IMAGE_DIR = "D:/mediapipe/"
    print("LEN imagedir", len(os.listdir(IMAGE_DIR)))
    arr = []
    for pat in os.listdir(IMAGE_DIR):
        files = os.listdir(IMAGE_DIR + pat)
        faces = [x for x in files if "face" in x]
        fronts = [x for x in files if "front" in x]
        backs = [x for x in files if "back" in x]
        faces = faces[0] if len(faces) > 0 else np.nan
        fronts = fronts[0] if len(fronts) > 0 else np.nan
        backs = backs[0] if len(backs) > 0 else np.nan
        paths = [IMAGE_DIR + pat + "/", faces, fronts, backs, pat]
        arr.append(paths)

    df = pd.DataFrame(arr, columns=["base_dir", "face", "front", "back", "Pseu_Haut"])

    df = df[~df.index.isin(df[(df.face.isna()) & (df.front.isna()) & (df.back.isna())].index)]  # drop 2 without images

    df["face_locs"] = df["face"].str.split(".").str[0] + ".npy"

    haut_keys = pd.read_excel("C:/Users/uhz/Documents/pseudonyme.xlsx")
    df = df.merge(haut_keys, on="Pseu_Haut")

    target_columns = ["HCH_SVSEX0001", "HCH_SVAGE0001", "HCH_SVWEIGHT",
                      "HCH_SVHEIGHT", "HCH_SVSYS0001", "HCH_SVPY0002", "HCH_SVAH0001", "HCH_SVDM0001",
                      "HCH_SVRS0001", "HDL", "Chol", "HCH_SVHL0001"]
    variables = pd.read_csv("D:/tabelle/variables.csv", low_memory=False)  # [["P_DSC_Ney_160_233_D-PSN(", target]]
    variables = variables[["P_DSC_Ney_160_233_D-PSN(", *target_columns]]
    variables["Pseu_Daten"] = variables["P_DSC_Ney_160_233_D-PSN("]
    variables = variables.iloc[:, 1:]
    df2 = df.merge(variables, on="Pseu_Daten")

    for col in target_columns:
        df2.loc[df2[col] == "NP", col] = np.nan  # "NaN"
        df2.loc[df2[col] == "Messung nicht möglich.", col] = np.nan  # "NaN"

        df2.loc[df2[col] == "1 - ja", col] = 1
        df2.loc[df2[col] == "0 - nein", col] = 0

        df2[col] = df2[col].astype(float)

    for v in df2[target_columns].values.flatten():
        try:
            float(v)
        except:
            print("Erroneous value:", v)

    # encode BMI
    df2["HCH_SVBMI0001"] = df2["HCH_SVWEIGHT"] / ((df2["HCH_SVHEIGHT"] / 100) ** 2)
    df2.loc[df2["HCH_SVBMI0001"] > 30, "Obesity"] = 1
    df2.loc[df2["HCH_SVBMI0001"] <= 30, "Obesity"] = 0

    # encode Pack Years as regression task
    df2.loc[df2["HCH_SVPY0002"] == 0, "HCH_SVPY0002"] = np.nan  # only keep packyears of smokers, otherwise distribution skewed
    df2["neversmoker"] = np.nan
    df2.loc[(df2["HCH_SVPY0002"] == 0), "neversmoker"] = 1
    df2.loc[(df2["HCH_SVPY0002"] > 0), "neversmoker"] = 0
    df2.loc[(df2["HCH_SVRS0001"] == 1), "neversmoker"] = 0

    # score2
    score2_list = []
    risk_list = []
    for i in range(len(df2)):
        sub = df2.iloc[i]
        score, risk = calculate_score2(sub, verbose=False)
        score2_list.append(score)
        risk_list.append(risk)
    df2["score2"] = score2_list
    df2["score2risk"] = risk_list

    df2 = df2.fillna("NaN")
    return df2.reset_index(drop=True)


def define_targets():
    class Target:
        def __init__(self, name, isClasfc, out_dim, loss_weight=1.0):
            self.name = name
            self.isClassification = isClasfc
            self.out_dim = out_dim
            self.loss_weight = loss_weight

        def __iter__(self):
            for name in dir(self):
                value = getattr(self, name)
                if not name.startswith('__') and not inspect.ismethod(value):
                    yield name, value

    TARGETS = [Target("HCH_SVSEX0001", isClasfc=True, out_dim=2, loss_weight=1),
               Target("HCH_SVAGE0001", isClasfc=False, out_dim=1, loss_weight=0.2),
               Target("HCH_SVAH0001", isClasfc=True, out_dim=2, loss_weight=1),
               Target("HCH_SVDM0001", isClasfc=True, out_dim=2, loss_weight=1),
               Target("HCH_SVBMI0001", isClasfc=False, out_dim=1, loss_weight=0.25),
               Target("HCH_SVRS0001", isClasfc=True, out_dim=2, loss_weight=1),
               Target("HCH_SVPY0002", isClasfc=False, out_dim=1, loss_weight=0.075),
               Target("HCH_SVSYS0001", isClasfc=False, out_dim=1, loss_weight=0.05),
               Target("HDL", isClasfc=False, out_dim=1, loss_weight=0.05),
               Target("Chol", isClasfc=False, out_dim=1, loss_weight=0.025),
               Target("HCH_SVHL0001", isClasfc=True, out_dim=2, loss_weight=1),  # new 30.12.22
               Target("Obesity", isClasfc=True, out_dim=2, loss_weight=1),  # new 30.12.22

               ]
    # TARGETS.append(Target("neversmoker", isClasfc=True, out_dim=2, loss_weight=1))

    pickable_dict = {target.name: dict(target) for target in TARGETS}
    return pickable_dict


def add_folds(df, num_folds, random_state):
    df2 = df.copy()
    df2.loc[df2["HCH_SVAGE0001"] == "NaN", "HCH_SVAGE0001"] = 62  # mean

    n_bins = 7
    bins = np.linspace(0, np.percentile(df2["HCH_SVAGE0001"].values, 95), n_bins)  # np.percentile in case of outliers
    df2["binned_age"] = np.digitize(df2["HCH_SVAGE0001"].values, bins)

    df["fold"] = np.nan
    skf = StratifiedKFold(n_splits=num_folds, shuffle=True, random_state=random_state)
    for f, (train_id, test_id) in enumerate(skf.split(df, df2["binned_age"].values)):
        df.loc[test_id, "fold"] = f
    df["fold"] = df["fold"].astype(int)
    return df


def load_and_unshuffle_dataset():
    df = create_hchs_dataframe()
    df = add_folds(df, config.num_folds, random_state=0)
    arr = []
    for f in range(config.num_folds):
        arr.append(df[df["fold"] == f])
    df = pd.concat(arr).reset_index(drop=True)
    # df = df.replace("NaN", np.nan)
    return df.reset_index(drop=True)
