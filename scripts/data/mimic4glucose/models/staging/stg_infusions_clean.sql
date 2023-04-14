{{ config(materialized='view') }}

with infusions as (
    select SUBJECT_ID,HADM_ID,stay_id,STARTTIME,ENDTIME,AMOUNT,coalesce(RATE,ORIGINALRATE) as rate,ORIGINALRATE,ITEMID,ORDERCATEGORYNAME,InsulinType,InsulinAdmin,INFXSTOP 
    from {{ ref('stg_insulin_raw') }}
    where upper(InsulinAdmin)='INFUSION' and upper(InsulinType)='SHORT'
),
infusions99 as (
    select APPROX_COUNT_DISTINCT(RATE) as rate_99
    from infusions
)

select *
from infusions
where rate < (select rate_99 from infusions99) and rate > 0
