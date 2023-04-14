{{ config(materialized='view') }}

with bolus as (
    select SUBJECT_ID,HADM_ID,stay_id,STARTTIME,ENDTIME,AMOUNT,RATE,ORIGINALRATE,ITEMID,ORDERCATEGORYNAME,InsulinType,InsulinAdmin,INFXSTOP 
    from {{ ref('stg_insulin_raw') }}
    where upper(InsulinAdmin) in ('BOLUS_PUSH','BOLUS_INJECTION') and upper(InsulinType)='SHORT'
),
bolus99 as (
    select APPROX_COUNT_DISTINCT(amount) as amount_99
    from bolus
    group by InsulinType
)

select SUBJECT_ID,HADM_ID,stay_id,STARTTIME,ENDTIME,
    case 
        WHEN upper(InsulinType) = 'SHORT' and amount < (select amount_99 from bolus99 where upper(InsulinType) = 'SHORT') then amount
        WHEN upper(InsulinType) = 'INTERMEDIATE' and amount < (select amount_99 from bolus99 where upper(InsulinType) = 'INTERMEDIATE') then amount
        WHEN upper(InsulinType) = 'LONG' and amount < (select amount_99 from bolus99 where upper(InsulinType) = 'LONG') then amount
    end as amount,
    RATE,ORIGINALRATE,ITEMID,ORDERCATEGORYNAME,InsulinType,InsulinAdmin,INFXSTOP,
from bolus
where amount > 0