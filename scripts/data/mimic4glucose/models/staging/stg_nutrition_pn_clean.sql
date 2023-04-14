
with tab_tpn as (
    select *, 
        TIMESTAMP_DIFF(endtime, starttime, MINUTE)/60.0 as tpn_length
    from {{ ref('stg_nutrition_pn_raw') }}
)

select stay_id,
    starttime as timer,
    (case when tpn is not null then tpn_length else null end) as tpn_length,
    starttime,
    endtime,
    dextrose_fluid,
    (tpn / tpn_length) as tpn_rate
from tab_tpn