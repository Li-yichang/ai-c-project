import pandas as pd
import numpy as np

patients = pd.read_csv('_patients.csv')
icu = pd.read_csv('filtered_icu_stays.csv')  
chart = pd.read_csv('_chartevents.csv')
labevents = pd.read_csv('_labevents.csv', low_memory=False)
diagnoses = pd.read_csv('_diagnoses_icd.csv')
d_icd = pd.read_csv('_d_icd_diagnoses.csv')

# === Age & Gender ===
features = icu.merge(patients[['subject_id', 'anchor_age', 'gender']], on='subject_id', how='left')
features = features.rename(columns={'anchor_age': 'age'})

# === BMI Calculation ===
bmi_chart = chart[chart['itemid'].isin([226707, 226730, 226512, 224639])]
bmi_records = []

for _, row in features.iterrows():
    sid, stay_id = row['subject_id'], row['stay_id']
    data = bmi_chart[(bmi_chart['subject_id'] == sid) & (bmi_chart['stay_id'] == stay_id)]

    # 先找 cm 單位的身高 (226730)，若無再用 inch (226707)
    height_cm = data[data['itemid'] == 226730]['valuenum'].dropna().mean()
    if pd.isna(height_cm):
        height_in = data[data['itemid'] == 226707]['valuenum'].dropna().mean()
        height_cm = height_in * 2.54 if pd.notnull(height_in) else np.nan

    # 體重
    weight = data[data['itemid'].isin([226512, 224639])]['valuenum'].dropna().mean()

    # BMI 計算
    bmi = np.nan
    if pd.notnull(height_cm) and pd.notnull(weight) and height_cm > 0:
        height_m = height_cm / 100
        bmi = weight / (height_m ** 2)

    bmi_records.append({'stay_id': stay_id, 'bmi': bmi})

bmi_df = pd.DataFrame(bmi_records)
features = features.merge(bmi_df, on='stay_id', how='left')
 
# lab items
lab_items = {
    'bun': 51006,
    'alk_phos': 50863,
    'bilirubin': 50885,
    'creatinine': 50912,
    'glucose': 50931,
    'platelets': 51265,
    'hemoglobin': 51222
}

lab_data = labevents[labevents['itemid'].isin(lab_items.values())].copy()
lab_data['charttime'] = pd.to_datetime(lab_data['charttime'])

itemid_to_name = {v: k for k, v in lab_items.items()}
lab_data['lab_name'] = lab_data['itemid'].map(itemid_to_name)

lab_merged = lab_data.merge(features[['subject_id', 'stay_id', 'intime']], on='subject_id', how='inner')
lab_merged['intime'] = pd.to_datetime(lab_merged['intime'])
lab_merged['charttime'] = pd.to_datetime(lab_merged['charttime'])

lab_merged = lab_merged[
    (lab_merged['charttime'] >= lab_merged['intime']) &
    (lab_merged['charttime'] <= lab_merged['intime'] + pd.Timedelta(hours=6))
]

lab_stats = lab_merged.groupby(['stay_id', 'lab_name'])['valuenum'].agg(['mean', 'min', 'max', 'std']).unstack()
lab_stats.columns = [f'{lab}_{stat}' for lab, stat in lab_stats.columns]
lab_stats = lab_stats.reset_index()

features = features.merge(lab_stats, on='stay_id', how='left')

# vital signs
vital_items = {
    'heart_rate': 220045,
    'resp_rate': 220210,
    'map': 220052,
    'temperature': 223762,
    'sbp': 220179
}

vital_data = chart[chart['itemid'].isin(vital_items.values())].copy()
vital_data['charttime'] = pd.to_datetime(vital_data['charttime'])

itemid_to_name_v = {v: k for k, v in vital_items.items()}
vital_data['vital_name'] = vital_data['itemid'].map(itemid_to_name_v)

vital_merged = vital_data.merge(features[['subject_id', 'stay_id', 'intime']], on=['subject_id', 'stay_id'], how='inner')

vital_merged['charttime'] = pd.to_datetime(vital_merged['charttime'])
vital_merged['intime'] = pd.to_datetime(vital_merged['intime'])

vital_merged = vital_merged[
    (vital_merged['charttime'] >= vital_merged['intime']) &
    (vital_merged['charttime'] <= vital_merged['intime'] + pd.Timedelta(hours=6))
]

vital_stats = vital_merged.groupby(['stay_id', 'vital_name'])['valuenum'].agg(['mean', 'min', 'max', 'std']).unstack()
vital_stats.columns = [f'{vital}_{stat}' for vital, stat in vital_stats.columns]
vital_stats = vital_stats.reset_index()

features = features.merge(vital_stats, on='stay_id', how='left')

# === Previous Diagnoses (multi-hot encoding top 10) ===
prev_diag = diagnoses.merge(icu[['subject_id', 'hadm_id']], on='subject_id')
prev_diag = prev_diag[prev_diag['hadm_id_x'] < prev_diag['hadm_id_y']]
prev_diag = prev_diag.rename(columns={'hadm_id_x': 'prev_hadm_id', 'hadm_id_y': 'current_hadm_id'})
 
top_codes = prev_diag['icd_code'].value_counts().nlargest(10).index.tolist()
for code in top_codes:
    col_name = f'diag_{code}'
    prev_diag[col_name] = (prev_diag['icd_code'] == code).astype(int)

diag_features = prev_diag.groupby('subject_id')[[f'diag_{code}' for code in top_codes]].max().reset_index()
features = features.merge(diag_features, on='subject_id', how='left')
 
# === Label: In-Hospital Mortality ===
features['label'] = features['hospital_expire_flag']

features.to_csv('features_todo2.csv', index=False)
