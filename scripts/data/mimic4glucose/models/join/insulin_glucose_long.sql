{{ config(materialized='table') }}

-- rejoin infusions to the data
with infusions as (
    select *
    from {{ ref('match_glucose_insulin')}}
    where lower(event) = 'infusion'
),
tab as (
    select SUBJECT_ID,HADM_ID,stay_id,
            TIMER,
            LEAD (timer,1) over (partition by stay_id order by timer) as timer_next,
            coalesce(GLC,glc_al) as GLC,
            glc_lab,
            coalesce(GLCTIMER,GLCTIMER_AL) as GLCTIMER,
            coalesce(GLCSOURCE,GLCSOURCE_al) as GLCSOURCE,
            STARTTIME,
            ENDTIME,
            INPUT,
            INPUT_HRS,
            InsulinType,
            EVENT,
            INFXSTOP,
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
    from {{ ref('match_glucose_insulin')}} t1
                              -- drop lab measures
    where (Repeated is null and lower(coalesce(GLCSOURCE,GLCSOURCE_al)) = 'fingerstick') or InsulinType is not null
    order by stay_id, timer
)

select SUBJECT_ID,HADM_ID,stay_id,
            TIMER,
            timer_next,
            GLC,
            glc_lab,
            GLCTIMER,
            GLCSOURCE,
            STARTTIME,
            ENDTIME,
            INPUT,
            INPUT_HRS,
            InsulinType,
            EVENT,
            INFXSTOP,
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
where timer_next is not null
order by stay_id, timer