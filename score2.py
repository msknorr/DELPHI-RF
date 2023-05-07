import numpy as np

score2_scores = {"woman": {"non_smoker":  # woman, non smoker
                               {"age6569": np.array([8, 8, 9, 9, 7, 7, 7, 7, 5, 6, 6, 6, 5, 5, 5, 5]).reshape(4, -1),
                                "age6064": np.array([6, 6, 7, 7, 5, 5, 5, 6, 4, 4, 4, 5, 3, 3, 4, 4]).reshape(4, -1),
                                "age5559": np.array([4, 5, 5, 5, 3, 4, 4, 4, 3, 3, 3, 3, 2, 2, 3, 3]).reshape(4, -1),
                                "age5054": np.array([3, 4, 4, 4, 3, 3, 3, 3, 2, 2, 2, 3, 2, 2, 2, 2]).reshape(4, -1),
                                "age4549": np.array([2, 3, 3, 3, 2, 2, 2, 3, 1, 2, 2, 2, 1, 1, 1, 1]).reshape(4, -1),
                                "age4044": np.array([2, 2, 2, 3, 1, 2, 2, 2, 1, 1, 1, 1, 1, 1, 1, 1]).reshape(4, -1)},
                           "smoker":
                               {"age6569": np.array([12, 12, 13, 13, 10, 10, 11, 11, 8, 9, 9, 9, 7, 7, 7, 8]).reshape(4,
                                                                                                                      -1),
                                "age6064": np.array([10, 10, 11, 11, 8, 8, 9, 9, 6, 7, 7, 8, 5, 6, 6, 6]).reshape(4,
                                                                                                                  -1),
                                "age5559": np.array([8, 8, 9, 10, 6, 7, 7, 8, 5, 5, 6, 6, 4, 4, 5, 5]).reshape(4, -1),
                                "age5054": np.array([6, 7, 7, 8, 5, 5, 6, 6, 4, 4, 5, 5, 3, 3, 4, 4]).reshape(4, -1),
                                "age4549": np.array([5, 5, 6, 7, 4, 4, 5, 5, 3, 3, 4, 4, 2, 2, 3, 3]).reshape(4, -1),
                                "age4044": np.array([4, 4, 5, 6, 3, 3, 4, 4, 2, 3, 3, 3, 2, 2, 2, 2]).reshape(4, -1)}
                           },
                 "man": {"non_smoker":  # woman, non smoker
                             {"age6569": np.array([11, 12, 12, 13, 10, 10, 11, 11, 8, 9, 9, 9, 7, 7, 7, 8]).reshape(4,
                                                                                                                    -1),
                              "age6064": np.array([8, 9, 10, 11, 7, 8, 8, 9, 6, 6, 7, 8, 5, 5, 6, 6]).reshape(4, -1),
                              "age5559": np.array([7, 7, 8, 9, 5, 6, 7, 8, 4, 5, 5, 6, 4, 4, 4, 5]).reshape(4, -1),
                              "age5054": np.array([5, 6, 7, 8, 4, 5, 5, 6, 3, 4, 4, 5, 3, 3, 3, 4]).reshape(4, -1),
                              "age4549": np.array([4, 5, 6, 6, 3, 4, 4, 5, 2, 3, 3, 4, 2, 2, 3, 3]).reshape(4, -1),
                              "age4044": np.array([3, 4, 5, 5, 2, 3, 3, 4, 2, 2, 3, 3, 1, 2, 2, 2]).reshape(4, -1)},
                         "smoker":
                             {"age6569": np.array(
                                 [15, 16, 17, 19, 13, 14, 15, 16, 11, 12, 13, 13, 9, 10, 11, 11]).reshape(4, -1),
                              "age6064": np.array([13, 14, 15, 17, 10, 11, 13, 14, 9, 10, 10, 11, 7, 8, 9, 10]).reshape(
                                  4, -1),
                              "age5559": np.array([10, 12, 13, 15, 9, 10, 11, 12, 7, 8, 9, 10, 6, 6, 7, 8]).reshape(4,
                                                                                                                    -1),
                              "age5054": np.array([9, 10, 11, 13, 7, 8, 9, 10, 6, 6, 7, 8, 4, 5, 6, 7]).reshape(4, -1),
                              "age4549": np.array([7, 8, 10, 11, 6, 7, 8, 9, 4, 5, 6, 7, 3, 4, 5, 5]).reshape(4, -1),
                              "age4044": np.array([6, 7, 8, 10, 5, 5, 6, 8, 3, 4, 5, 6, 3, 3, 4, 5]).reshape(4, -1)}
                         },
                 }


def calculate_score2(sub, verbose=True):
    sub = sub.replace("NaN", np.nan)

    mmol_hdl = sub["HDL"] / 38.6473
    mmol_chol = sub["Chol"] / 38.6473
    mmol_non_hdl = mmol_chol - mmol_hdl

    if not np.isnan(sub["HCH_SVSEX0001"]):
        sex = "woman" if int(sub["HCH_SVSEX0001"]) == 1 else "man"
    else:
        if verbose:
            print("NaN encountered in sex:", sub["HCH_SVSEX0001"])
        return np.nan, np.nan

    if not np.isnan(sub["HCH_SVRS0001"]):
        smoker = "smoker" if int(sub["HCH_SVRS0001"]) == 1 else "non_smoker"
    else:
        if verbose:
            print("NaN encountered in smoker:", sub["HCH_SVRS0001"])
        return np.nan, np.nan

    if not np.isnan(sub["HCH_SVSYS0001"]):
        rr = int(sub["HCH_SVSYS0001"])
    else:
        if verbose:
            print("NaN encountered in SyS:", sub["HCH_SVSYS0001"])
        return np.nan, np.nan

    if not np.isnan(sub["HCH_SVAGE0001"]):
        age = int(sub["HCH_SVAGE0001"])
    else:
        if verbose:
            print("NaN encountered in Age:", sub["HCH_SVAGE0001"])
        return np.nan, np.nan

    if (age >= 40) and (age < 45):
        age = "age4044"
    elif (age >= 45) and (age < 50):
        age = "age4549"
    elif (age >= 50) and (age < 55):
        age = "age5054"
    elif (age >= 55) and (age < 60):
        age = "age5559"
    elif (age >= 60) and (age < 65):
        age = "age6064"
    elif (age >= 65) and (age < 70):
        age = "age6569"
    else:
        if verbose:
            print("wrong age:", age)
        return np.nan, np.nan

    square = score2_scores[sex][smoker][age]

    if (rr >= 100) and (rr < 120):
        rr_idx = 3
    elif (rr >= 120) and (rr < 140):
        rr_idx = 2
    elif (rr >= 140) and (rr < 160):
        rr_idx = 1
    elif (rr >= 160) and (rr < 180):
        rr_idx = 0
    else:
        if verbose:
            print("wrong RR:", rr)
        return np.nan, np.nan

    row = square[rr_idx]

    if (mmol_non_hdl >= 3) and (mmol_non_hdl < 4):
        chol_idx = 0
    elif (mmol_non_hdl >= 4) and (mmol_non_hdl < 5):
        chol_idx = 1
    elif (mmol_non_hdl >= 5) and (mmol_non_hdl < 6):
        chol_idx = 2
    elif (mmol_non_hdl >= 6) and (mmol_non_hdl < 7):
        chol_idx = 3
    else:
        if verbose:
            print("wrong chol:", mmol_non_hdl)
        return np.nan, np.nan

    score = row[chol_idx]

    age = int(sub["HCH_SVAGE0001"])
    if (age) < 50:
        if (score >= 0) and (score < 3):
            risk = "green"
        elif (score >= 3) and (score < 8):
            risk = "yellow"
        elif (score >= 8):
            risk = "red"
    else:
        if (score >= 0) and (score < 5):
            risk = "green"
        elif (score >= 5) and (score < 10):
            risk = "yellow"
        elif (score >= 10):
            risk = "red"

    return (score, risk)
