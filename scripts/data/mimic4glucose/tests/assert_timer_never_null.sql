select *
from {{ ref('stg_glucose_insulin_concat') }}
where timer is null