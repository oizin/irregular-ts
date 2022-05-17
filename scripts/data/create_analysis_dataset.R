###############################################################################
# Create analysis dataset for glucose prediction in ICU project
#
#
###############################################################################

# libraries
library(data.table)

# import
# blood glucose
# bg = fread("data/raw/glucose_insulin_pair.csv")
# names(bg) = tolower(names(bg))
bg = fread("data/raw/glucose_insulin_ICU.csv")
names(bg) = tolower(names(bg))
bg = bg[!glcsource == "BLOOD"]

# static info ------------------------------------------------------------------

# icustays - join #1
static_info = fread("data/raw/static_variables.csv")
static_info = static_info[icustay_id %in% bg$icustay_id]
bg = bg[static_info[,.(icustay_id,intime=intime.x,outtime,first_careunit,
                       admission_type,ethnicity,diagnosis,gender,dm,dmcx)],
        on="icustay_id"]
rm(static_info); gc()


bg[,intime:=lubridate::as_datetime(intime)]
bg[,timer:=lubridate::as_datetime(timer)]
bg[,starttime:=lubridate::as_datetime(starttime)]
bg[,endtime:=lubridate::as_datetime(endtime)]
bg[,glctimer_al:=lubridate::as_datetime(glctimer_al)]
bg[,glctimer:=lubridate::as_datetime(glctimer)]


bg[,timer := as.numeric(difftime(timer,intime,units = "mins"))]
bg[,starttime := as.numeric(difftime(starttime,intime,units = "mins"))/60]
bg[,endtime := as.numeric(difftime(endtime,intime,units = "mins"))/60]
bg[,glctimer_al := as.numeric(difftime(glctimer_al,intime,units = "mins"))/60]
bg[,glctimer := as.numeric(difftime(glctimer,intime,units = "mins"))/60]


bg[,timer_hr := (timer/60)]

# outcome
msk = (bg$glctimer_al[-1] == bg$glctimer[-nrow(bg)]) & 
  (bg$icustay_id[-1] == bg$icustay_id[-nrow(bg)])
msk[is.na(msk)] = FALSE
bg[c(FALSE,msk),glc_al := NA]


bg[event == "BOLUS_INYECTION",starttime := NA]
bg[event == "BOLUS_INYECTION",endtime := NA]

bg[,starttime := nafill(x = starttime,type = "locf"),by=icustay_id]
bg[,endtime := nafill(x = endtime,type = "locf"),by=icustay_id]
bg[,input_hrs := nafill(x = input_hrs,type = "locf"),by=icustay_id]
bg[,input_hrs := nafill(x = input_hrs,type = "locf"),by=icustay_id]

bg[endtime < timer_hr,starttime := NA]
bg[endtime < timer_hr,input_hrs := NA]
bg[endtime < timer_hr,endtime := NA]

bg[,glc_1 := shift(glc,-1),by=icustay_id]
bg[,dt := c(diff(timer_hr),NA),by=icustay_id]
bg[,msk := fifelse(is.na(glc_1),1,0,0)]
bg[dt == 0,msk := 1]
bg[,.(icustay_id,timer_hr,event,starttime,endtime,input_hrs,input,glc,dt,glc_1,msk)]

bg[event == "INFUSION",input := NA]

bg[,injection := fifelse(event == "BOLUS_INYECTION",1,0,0)]

setnafill(bg,cols=c("input_hrs","input"), fill=0)

bg[,.(icustay_id,timer_hr,injection,input_hrs,input,glc,dt,glc_1,msk)]

bg[,cutoff := max(timer_hr[msk == 0]),by=icustay_id]

bg = bg[(timer_hr < cutoff) & cutoff != -Inf]

fwrite(x = bg[,.(icustay_id,timer_hr,injection,input_hrs,input,glc,dt,glc_1,msk)],
       "data/analysis.csv")
