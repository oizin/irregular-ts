/* Extract feeding data from MIMIC
 * 
 *
 *
 *
220952	Dextrose 50%	Dextrose 50%	metavision	inputevents_mv
225947	Dextrose PN	    Dextrose PN	    metavision	inputevents_mv
228140	Dextrose 20%	Dextrose 20%	metavision	inputevents_mv
228141	Dextrose 30%	Dextrose 30%	metavision	inputevents_mv
228142	Dextrose 40%	Dextrose 40%	metavision	inputevents_mv
220949	Dextrose 5%	    Dextrose 5%	    metavision	inputevents_mv
220950	Dextrose 10%	Dextrose 10%	metavision	inputevents_mv

225823	D5 1/2NS
225825	D5NS
225827	D5LR
225941	D5 1/4NS

227090	Lipids 10%	Lipids 10%	metavision	inputevents_mv	Nutrition - Parenteral	mL	Solution
222190	Ranitidine	Ranitidine	metavision	inputevents_mv	Nutrition - Parenteral	mg	Solution
225916	TPN w/ Lipids	TPN w/ Lipids	metavision	inputevents_mv	Nutrition - Parenteral	mL	Solution
225801	Lipids 20%	Lipids 20%	metavision	inputevents_mv	Nutrition - Parenteral	mL	Solution
225969	Lipids (additive)	Lipids (additive)	metavision	inputevents_mv	Nutrition - Parenteral	grams	Solution
225917	TPN without Lipids	TPN without Lipids	metavision	inputevents_mv	Nutrition - Parenteral	mL	Solution
225920	Peripheral Parenteral Nutrition	Peripheral Parenteral Nutrition	metavision	inputevents_mv	Nutrition - Parenteral	mL	Solution
225947	Dextrose PN	Dextrose PN	metavision	inputevents_mv	Nutrition - Parenteral	grams	Solution
225948	Amino Acids	Amino Acids	metavision	inputevents_mv	Nutrition - Parenteral	grams	Solution
*/

with pn_0 as (
	select im.stay_id,
	hadm_id,
	cast(starttime as timestamp) as starttime,
	cast(endtime as timestamp) as endtime,
	(case 
		when itemid in (220949,225823,225825,225827,225941) and amount >= 0 then amount*0.05
		when itemid in (220950) and amount >= 0 then amount*0.1
		when itemid in (228140) and amount >= 0 then amount*0.2 
		when itemid in (228141) and amount >= 0 then amount*0.3 
		when itemid in (228142) and amount >= 0 then amount*0.4 
		when itemid in (220952) and amount >= 0 then amount*0.5 
	else null end) as dextrose_fluid,
	(case when itemid in (225916,225917) and amount >= 0 then amount 
		else null end) as tpn,
	(case 
		when itemid = 225916 and amount >= 0 then 1
		when itemid = 225917 and amount >= 0 then 0
		else null end) as tpn_lipids,
	(case 
		when itemid in (225947) and amount >= 0 then amount 
		else null end) as dextrose_tpn,
	(case 
		when itemid in (225969) and amount >= 0 then amount 
		else null end) as lipids_tpn,
	(case
		when itemid in (225948) and amount >= 0 then amount 
		else null end) as amino_acids_tpn,
	(case 
		when itemid in (227090) and amount >= 0 then amount*0.1
		when itemid in (225801) and amount >= 0 then amount*0.2
		else null end) as lipids_10_20,
	amountuom as original_amountuom,
	(case 
		when itemid in (220949,220950,228140,228141,228142,220952) 
		and lower(amountuom) = 'ml' 
		then 'grams'
	else amountuom end) as amountuom,
	ordercategoryname ,
	ordercategorydescription
	from `mimic_icu.inputevents` im 
	where itemid in (220949,220950,228140,228141,228142,220952,  --solution dextrose
					 225947,  --parenteral dextrose
					 227090,225801,  --parenteral lipid 10%, 20%
					 225969,  --lipids TPN
					 225948,  --amino acids TPN
					 225916,225917  --total parenteral nutrition
					 ) and 
	statusdescription is distinct from 'Rewritten'
	order by stay_id, starttime
)
select pn_0.stay_id,
	pn_0.starttime,
	pn_0.endtime,
	sum(dextrose_fluid) as dextrose_fluid,
	sum(tpn) as tpn,
	max(tpn_lipids) as tpn_lipids,
	sum(dextrose_tpn) as dextrose_tpn,
	sum(lipids_tpn) as lipids_tpn,
	sum(amino_acids_tpn) as amino_acids_tpn,
	sum(lipids_10_20) as lipids_10_20
from pn_0
group by pn_0.stay_id, pn_0.starttime, pn_0.endtime
order by pn_0.stay_id, pn_0.starttime, pn_0.endtime