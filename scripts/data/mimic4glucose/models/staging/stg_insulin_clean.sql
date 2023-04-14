{{ config(materialized='view') }}

select DISTINCT SUBJECT_ID,HADM_ID,stay_id,STARTTIME,ENDTIME,AMOUNT,RATE,ORIGINALRATE,ITEMID,ORDERCATEGORYNAME,InsulinType,InsulinAdmin,INFXSTOP 
from {{ ref('stg_bolus_clean') }}
union all
select DISTINCT SUBJECT_ID,HADM_ID,stay_id,STARTTIME,ENDTIME,AMOUNT,RATE,ORIGINALRATE,ITEMID,ORDERCATEGORYNAME,InsulinType,InsulinAdmin,INFXSTOP 
from {{ ref('stg_infusions_clean') }}
