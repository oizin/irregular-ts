-- are there any duplicated measures? Note there may be several measures at the same time
select STAY_ID, GLCTIMER,count(*) as N
from {{ ref('stg_glucose_insulin_icustay') }}
where GLCTIMER is not null
group by STAY_ID,GLCTIMER,GLC,glcsource
having N > 1