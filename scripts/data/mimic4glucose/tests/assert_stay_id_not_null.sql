select *
from {{ ref('stg_glucose_insulin_icustay') }}
where stay_id is null