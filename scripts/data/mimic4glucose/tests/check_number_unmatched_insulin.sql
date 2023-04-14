select *
from {{ ref('match_glucose_insulin') }}
where InsulinType is not null and rule is null
