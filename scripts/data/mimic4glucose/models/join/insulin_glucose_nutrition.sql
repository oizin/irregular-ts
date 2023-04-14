{{ config(materialized='table') }}

with tpn as (
    select stay_id, starttime,endtime,tpn_rate
    from {{ ref('stg_nutrition_pn_clean') }}
    where tpn_rate is not null
),
dextrose_fluid as (
    select stay_id, starttime,dextrose_fluid
    from {{ ref('stg_nutrition_pn_clean') }}
    where dextrose_fluid is not null
),
dextrose_fluid1 as (
    select t1.stay_id,timer,sum(dextrose_fluid) as dextrose_fluid
    from {{ ref('insulin_glucose_wide') }} t1
    left join dextrose_fluid t2
    on t1.stay_id = t2.stay_id and
        t2.starttime < t1.timer and
        t2.starttime >= t1.timer_prev
    group by stay_id,timer
),
enteral as (
    select t1.stay_id,
            t1.timer,        
            avg(t2.enteral_rate) as enteral_rate,
            avg(t2.cho_enteral) as cho_enteral,
            avg(t2.dextrose_enteral) as dextrose_enteral,
            avg(t2.fat_enteral) as fat_enteral,
            avg(t2.protein_enteral) as protein_enteral,
            avg(t2.fibre_enteral) as fibre_enteral,
            avg(t2.calorie_enteral) as calorie_enteral,
            min(t2.starttime) as starttime,
            max(t2.endtime) as endtime,
    from {{ ref('insulin_glucose_wide') }} t1
    left join {{ ref('stg_nutrition_ent_clean') }} t2
    on t1.stay_id = t2.stay_id and
        t2.starttime <= t1.timer and
        t2.endtime > t1.timer
    group by stay_id,timer
),
curr_tab as (
    select t1.*,
            t2.enteral_rate,
            t2.cho_enteral,
            t2.dextrose_enteral,
            t2.fat_enteral,
            t2.protein_enteral,
            t2.fibre_enteral,
            t2.calorie_enteral,
            t2.starttime as ent_starttime,
            t2.endtime as ent_endtime,
            t3.tpn_rate,
            t4.dextrose_fluid
    from {{ ref('insulin_glucose_wide') }} t1
    left join enteral t2
    on t1.stay_id = t2.stay_id and
        t2.timer = t1.timer 
    left join tpn t3
    on t1.stay_id = t3.stay_id and
        t3.starttime <= t1.timer and
        t3.endtime > t1.timer
    left join dextrose_fluid1 t4
    on t1.stay_id = t4.stay_id and
        t4.timer = t1.timer 
    order by stay_id,timer
),
removals as (
    -- remove patients with multiple of the same treatment occuring simulataneuosly
    select distinct stay_id from 
    (
    SELECT stay_id,timer,count(*) as n 
    FROM curr_tab
    group by stay_id,timer
    having n > 1
    ) t
)

select *
from curr_tab
where stay_id not in (select stay_id from removals)

