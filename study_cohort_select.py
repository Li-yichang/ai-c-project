import pandas as pd

# 讀 ICU stays
icu = pd.read_csv("_icustays.csv", usecols=["subject_id", "hadm_id", "stay_id", "intime", "outtime"])
icu["intime"] = pd.to_datetime(icu["intime"])
icu["outtime"] = pd.to_datetime(icu["outtime"])

# 每個病人只保留第一次 ICU stay
icu = icu.sort_values("intime").drop_duplicates("subject_id", keep="first")

# 計算 ICU 停留時間（小時）
icu["los_hr"] = (icu["outtime"] - icu["intime"]).dt.total_seconds() / 3600
icu = icu[icu["los_hr"] >= 6]

# 合併出院時間
adm = pd.read_csv("_admissions.csv", usecols=["subject_id", "hadm_id", "dischtime", "hospital_expire_flag"])
adm["dischtime"] = pd.to_datetime(adm["dischtime"])

icu = pd.merge(icu, adm, on=["subject_id", "hadm_id"], how="inner")

icu.to_csv("filtered_icu_stays.csv", index=False)

print(f"總共篩選出 {len(icu)} 筆 ICU stays")
