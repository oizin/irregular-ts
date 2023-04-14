
with t1 as (
    select *, 
        TIMESTAMP_DIFF(endtime, starttime, MINUTE)/60.0 as enteral_length
    from {{ ref('stg_nutrition_ent_raw') }}
)

select stay_id,hadm_id,subject_id,
    starttime as timer,
    enteral_length,
    starttime,
    endtime,
    (amount_enteral / enteral_length) as enteral_rate,
    (cho_enteral / enteral_length) as cho_enteral,
    (dextrose_enteral / enteral_length) as dextrose_enteral,
    (fat_enteral / enteral_length) as fat_enteral,
    (protein_enteral / enteral_length) as protein_enteral,
    (fibre_enteral / enteral_length) as fibre_enteral,
    (calorie_enteral / enteral_length) as calorie_enteral
from t1