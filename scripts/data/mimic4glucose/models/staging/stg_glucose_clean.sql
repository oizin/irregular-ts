{{ config(materialized='view') }}

-- some lab measurements are also present in the chartevents table
with fingerstick as (
    select DISTINCT SUBJECT_ID,HADM_ID,stay_id,charttime,glucose as glucose_finger
    from {{ ref('stg_glucose_raw') }}
    where ITEM_GLC in (807,811,1529,225664) AND
        	glucose < 500
),
lab as (
    select DISTINCT SUBJECT_ID,HADM_ID,stay_id,charttime,glucose as glucose_lab
    from {{ ref('stg_glucose_raw') }}
    where ITEM_GLC in (3745,220621,50931,50809,226537) AND
        	glucose < 1000
),
lab_finger as (
    select SUBJECT_ID,HADM_ID,
            MAX(stay_id) as stay_id, -- stay_id might be null so grouping by not proper
            charttime,
            max(glucose_finger) as glucose_finger,
            max(glucose_lab) as glucose_lab
    from (
        select SUBJECT_ID,HADM_ID,stay_id,CHARTTIME,glucose_finger,NULL as glucose_lab
        from fingerstick
        union distinct
        select SUBJECT_ID,HADM_ID,stay_id,CHARTTIME,NULL as glucose_finger,glucose_lab
        from lab
    ) tab
    group by SUBJECT_ID,HADM_ID,charttime
)
select SUBJECT_ID,HADM_ID,stay_id,charttime,
        coalesce(glucose_finger,glucose_lab) as glucose,
        glucose_lab,
        (case when glucose_finger is not null then 'fingerstick'
            else 'blood' end) as glcsource
from lab_finger



-- another group by.... OR maybe this is the only place a group by should be...