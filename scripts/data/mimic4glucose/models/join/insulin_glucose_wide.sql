{{ config(materialized='table') }}

with tab as (
    select SUBJECT_ID,
        HADM_ID,
        stay_id,
        TIMER,
        max(timer_next) as timer_next,
        max(GLC) as glc,
        max(glc_lab) as glc_lab,
        max(GLCTIMER) as GLCTIMER,
        max(GLCSOURCE) as GLCSOURCE,
        gender,
        diabetes_without_cc,
        diabetes_with_cc,
        weight,
        dod,
        admittime,
        dischtime,
        los_hospital,
        admission_age,
        ethnicity,
        hospital_expire_flag,
        hospstay_seq,
        first_hosp_stay,
        icu_intime,
        icu_outtime,
        los_icu,
        icustay_seq,
        first_icu_stay
    from {{ ref('insulin_glucose_long') }}
    group by SUBJECT_ID,
        HADM_ID,
        stay_id,
        TIMER,
        gender,
        diabetes_without_cc,
        diabetes_with_cc,
        weight,
        dod,
        admittime,
        dischtime,
        los_hospital,
        admission_age,
        ethnicity,
        hospital_expire_flag,
        hospstay_seq,
        first_hosp_stay,
        icu_intime,
        icu_outtime,
        los_icu,
        icustay_seq,
        first_icu_stay
),
infusion as (
    select SUBJECT_ID,
        HADM_ID,
        stay_id,
        TIMER,
        starttime,
        endtime,
        INPUT_HRS
    from {{ ref('insulin_glucose_long') }}
    where lower(event) = 'infusion'
),
bolus_push as (
    select SUBJECT_ID,
        HADM_ID,
        stay_id,
        TIMER,
        sum(INPUT) as input_bolus_push
    from {{ ref('insulin_glucose_long') }}
    where lower(event) = 'bolus_push'
    group by SUBJECT_ID,
            HADM_ID,
            stay_id,
            TIMER
),
bolus_injection as (
    select SUBJECT_ID,
        HADM_ID,
        stay_id,
        TIMER,
        sum(INPUT) as input_bolus_injection
    from {{ ref('insulin_glucose_long') }}
    where lower(event) = 'bolus_injection'
    group by SUBJECT_ID,
            HADM_ID,
            stay_id,
            TIMER
)

select tab.SUBJECT_ID,
        tab.HADM_ID,
        tab.stay_id,
        tab.TIMER,
        timer_next,
        (CASE WHEN LAG (tab.TIMER,1) over (partition by tab.stay_id order by tab.TIMER) is null then icu_intime
            ELSE LAG (tab.TIMER,1) over (partition by tab.stay_id order by tab.TIMER) END) as timer_prev,
        GLC,
        LEAD (GLC,1) over (partition by tab.stay_id order by tab.TIMER) as glc_next,
        glc_lab,
        GLCTIMER,
        GLCSOURCE,
        bp.input_bolus_push,
        bi.input_bolus_injection,
        i.input_hrs,
        i.starttime,
        i.endtime,
        gender,
        diabetes_without_cc,
        diabetes_with_cc,
        weight,
        dod,
        admittime,
        dischtime,
        los_hospital,
        admission_age,
        ethnicity,
        hospital_expire_flag,
        hospstay_seq,
        first_hosp_stay,
        icu_intime,
        icu_outtime,
        los_icu,
        icustay_seq,
        first_icu_stay
from tab
left join
infusion i
on i.timer = tab.timer and i.stay_id=tab.stay_id or (tab.timer >= i.starttime and tab.timer < i.endtime and i.stay_id=tab.stay_id)
left join 
bolus_push bp 
on bp.timer = tab.timer and bp.stay_id=tab.stay_id
left join 
bolus_injection bi
on bi.timer = tab.timer and bi.stay_id=tab.stay_id
order by stay_id, timer