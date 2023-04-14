{{ config(materialized='view') }}

with icustay as (
    select subject_id,
    hadm_id,
    stay_id,
    gender,
    dod,
    cast(admittime as TIMESTAMP) as admittime,
    cast(dischtime as TIMESTAMP) as dischtime,
    los_hospital,
    admission_age,
    ethnicity,
    hospital_expire_flag, 	
    hospstay_seq,
    first_hosp_stay,
    cast(icu_intime as TIMESTAMP) as icu_intime,
    cast(icu_outtime as TIMESTAMP) as icu_outtime,
    los_icu,
    icustay_seq,
    first_icu_stay 
    from mimic_derived.icustay_detail
),
diabetes as (
    select subject_id,hadm_id,diabetes_without_cc,diabetes_with_cc 
    from mimic_derived.charlson
),
weight as (
    select stay_id,weight 
    from mimic_derived.first_day_weight
),
icustay_expanded as (
    select t1.*,t2.diabetes_without_cc,t2.diabetes_with_cc,t3.weight
    from icustay t1
    left join 
    diabetes t2
    on t1.subject_id = t2.subject_id and t1.hadm_id = t2.hadm_id
    left join 
    weight t3
    on t3.stay_id = t1.stay_id
)

-- distinct because some glucose are in both lab source and chartevent source
-- and so appear twice because not filtered out due to stay_id being both missing and 
-- non-missing
select DISTINCT glc.SUBJECT_ID,glc.HADM_ID,coalesce(glc.stay_id,icu.stay_id) as stay_id,
    glc.TIMER,glc.GLCTIMER,glc.GLC,glc.GLC_LAB,glc.GLCSOURCE,
    glc.STARTTIME,glc.ENDTIME,glc.INPUT,glc.INPUT_HRS,glc.InsulinType,glc.EVENT,glc.INFXSTOP,
    icu.diabetes_without_cc,icu.diabetes_with_cc,icu.weight,
    icu.gender,icu.dod,icu.admittime,icu.dischtime,icu.los_hospital,icu.admission_age,
    icu.ethnicity,icu.hospital_expire_flag,icu.hospstay_seq,
    icu.first_hosp_stay,icu.icu_intime,icu.icu_outtime,icu.los_icu,icu.icustay_seq,icu.first_icu_stay 
from 
{{ ref('stg_glucose_insulin_concat') }} glc
join 
icustay_expanded icu
on glc.SUBJECT_ID = icu.SUBJECT_ID AND
    glc.HADM_ID = icu.HADM_ID and 
    glc.TIMER > icu.icu_intime and 
    glc.TIMER < icu.icu_outtime
order by SUBJECT_ID,HADM_ID,stay_id,TIMER