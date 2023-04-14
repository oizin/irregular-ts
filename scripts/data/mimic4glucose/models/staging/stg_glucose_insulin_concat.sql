{{ config(materialized='view') }}

with glucose as (
    select SUBJECT_ID,
    HADM_ID,
    stay_id,
        charttime as GLCTIMER,
        glucose as glc,
        glucose_lab as glc_lab,
        GLCSOURCE,
            CAST(NULL as TIMESTAMP) as STARTTIME,
            CAST(NULL as TIMESTAMP) as ENDTIME,
            NULL as INPUT,
            NULL as INPUT_HRS,
            CAST(NULL as STRING) as INSULINTYPE,
            CAST(NULL as STRING) as EVENT,
            NULL as INFXSTOP
    from {{ ref('stg_glucose_clean') }}
),
insulin as (
        select SUBJECT_ID,
            HADM_ID,
            stay_id,
            CAST(NULL as TIMESTAMP) as GLCTIMER,
            NULL as GLC,
            NULL as glc_lab,
            CAST(NULL as STRING) as GLCSOURCE,
            STARTTIME,
            ENDTIME,
            AMOUNT as INPUT,
            RATE as INPUT_HRS,
            InsulinType as InsulinType,
            InsulinAdmin as EVENT,
            INFXSTOP 
    from {{ ref('stg_insulin_clean') }}
)

select SUBJECT_ID,HADM_ID,stay_id,GLCTIMER as TIMER,GLCTIMER,GLC,glc_lab,GLCSOURCE,STARTTIME,ENDTIME,INPUT,INPUT_HRS,InsulinType,EVENT,INFXSTOP
from glucose
union all
select SUBJECT_ID,HADM_ID,stay_id,STARTTIME as TIMER,GLCTIMER,GLC,glc_lab,GLCSOURCE,STARTTIME,ENDTIME,INPUT,INPUT_HRS,InsulinType,EVENT,INFXSTOP
from insulin
