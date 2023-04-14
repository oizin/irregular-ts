{{ config(materialized='view') }}

WITH pg AS(
    SELECT p1.*

    -- Column GLC_AL that would gather paired glucose values according to the proposed rules
    ,(CASE
        -- 1ST CLAUSE
        -- When previous and following rows are glucose readings, select the glucose value that 
        -- has the shortest time distance to insulin bolus/infusion.
        
        -- Identify an insulin event either bolus or infusion
        WHEN upper(p1.EVENT) IN('BOLUS_INJECTION', 'BOLUS_PUSH', 'INFUSION')
            -- Identify preceding and posterior glucose reading
            AND (LAG(upper(p1.GLCSOURCE),1) OVER(w)) IN('BLOOD', 'FINGERSTICK')
            AND (LEAD(upper(p1.GLCSOURCE),1) OVER(w)) IN('BLOOD', 'FINGERSTICK')
            AND ( -- Posterior glucose has a longer time-gap to insulin than the posterior
                    ABS(TIMESTAMP_DIFF((LAG(p1.TIMER,1) OVER(w)), p1.TIMER, MINUTE)) <= 
                    ABS(TIMESTAMP_DIFF((LEAD(p1.TIMER,1) OVER(w)), p1.TIMER, MINUTE))
                    )
            -- Time-gap between glucose and insulin, should be equal or less than 90 minutes
            AND ( -- Preceding glucose
                ABS(TIMESTAMP_DIFF((LAG(p1.TIMER,1) OVER(w)), p1.TIMER, MINUTE)) <= 90)
            -- Preceding glucose should be equal or greater than 90 mg/dL
            AND (LAG(p1.GLC,1) OVER(w)) >= 90
            -- Posterior glucose value is lower than the preceding glucose
            AND LAG(p1.GLC,1) OVER(w) >= LEAD(p1.GLC,1) OVER(w)
        -- Return the PRECEDING glucose measurement that gathers the previous conditions
        THEN (LAG(p1.GLC,1) OVER(w))
        
        -- %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        -- 2ND CLAUSE
        -- In case the posterior glucose reading is higher than the preceding
        
        -- Identify an insulin event either bolus or infusion
        WHEN upper(p1.EVENT) IN('BOLUS_INJECTION', 'BOLUS_PUSH', 'INFUSION')
            -- Identify preceding and posterior glucose measurements
            AND (LAG(upper(p1.GLCSOURCE),1) OVER(w)) IN('BLOOD', 'FINGERSTICK')
            AND (LEAD(upper(p1.GLCSOURCE),1) OVER(w)) IN('BLOOD', 'FINGERSTICK')
            AND ( -- Preceding glucose has a longer OR equal time-gap to insulin than the posterior
                    ABS(TIMESTAMP_DIFF((LAG(p1.TIMER,1) OVER(w)), p1.TIMER, MINUTE)) > 
                    ABS(TIMESTAMP_DIFF((LEAD(p1.TIMER,1) OVER(w)), p1.TIMER, MINUTE))
                    )
            -- Time-gap between glucose and insulin, should be equal or less than 90 minutes
                -- Preceding glucose
            AND ABS(TIMESTAMP_DIFF((LAG(p1.TIMER,1) OVER(w)), p1.TIMER, MINUTE)) <= 90
                -- Posterior glucose
            AND ABS(TIMESTAMP_DIFF((LEAD(p1.TIMER,1) OVER(w)), p1.TIMER, MINUTE)) <= 90
            -- Posterior glucose should be equal or greater than 90 mg/dL
            AND (LEAD(p1.GLC,1) OVER(w)) >= 90
            -- Posterior glucose values is higher than the preceding glucose
            AND LAG(p1.GLC,1) OVER(w) < LEAD(p1.GLC,1) OVER(w) 
        -- Return the POSTERIOR glucose measurement that gathers the previous conditions
        THEN (LEAD(p1.GLC,1) OVER(w))
        
        -- %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        -- 3RD CLAUSE
        -- When previous timestamp is an insulin bolus/infusion event
        
        -- Identify an insulin event either bolus or infusion
        WHEN upper(p1.EVENT) IN('BOLUS_INJECTION', 'BOLUS_PUSH', 'INFUSION')
            -- Regular insulin or short-acting
            AND upper(p1.INSULINTYPE) IN('SHORT')
            -- Identify preceding glucose reading 2 rows above and regular insulin
            AND (LAG(upper(p1.GLCSOURCE),2) OVER(w)) IN('BLOOD', 'FINGERSTICK')
            AND (LAG(upper(p1.INSULINTYPE),2) OVER(w)) IN('SHORT')
            -- One row above there is another insulin event
            AND (LAG(upper(p1.EVENT),1) OVER(w)) IN('BOLUS_INJECTION','BOLUS_PUSH','INFUSION')
            -- Identify posterior glucose reading 1 row below
            AND (LEAD(upper(p1.GLCSOURCE),1) OVER(w)) IN('BLOOD', 'FINGERSTICK')
            AND ( -- Preceding glucose has a shortime or equal time-gap to insulin than the posterior
                ABS(TIMESTAMP_DIFF((LAG(p1.TIMER,2) OVER(w)), p1.TIMER, MINUTE)) <= 
                ABS(TIMESTAMP_DIFF((LEAD(p1.TIMER,1) OVER(w)), p1.TIMER, MINUTE))
                )
            -- Preceding glucose 2 rows above occured up to 90 minutes before
            AND ABS(TIMESTAMP_DIFF((LAG(p1.TIMER,2) OVER(w)), p1.TIMER, MINUTE)) <= 90
            -- Preceding glucose 2 rows above is equal or greater than 90 min
            AND (LAG(p1.GLC,2) OVER(w)) >= 90
            -- Posterior glucose value is lower than the preceding glucose 2 rows above
            AND LAG(p1.GLC,2) OVER(w) >= LEAD(p1.GLC,1) OVER(w)
        -- Return the preceding glucose value 2 rows above
        THEN (LAG(p1.GLC,2) OVER(w))
        
        -- %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        -- 4TH CLAUSE
        -- When previous timestamp is for Insulin bolus/infusion but posterior glucose
        -- is higher than the preceding glucose 2 rows above.
        
        -- Identify an insulin event either bolus or infusion
        WHEN upper(p1.EVENT) IN('BOLUS_INJECTION', 'BOLUS_PUSH', 'INFUSION')
            -- Regular insulin or short-acting
            AND upper(p1.INSULINTYPE) IN('SHORT')
            -- Identify preceding glucose reading 2 rows above
            AND (LAG(upper(p1.GLCSOURCE),2) OVER(w)) IN('BLOOD', 'FINGERSTICK')
            -- One row above there is another regular insulin
            AND (LAG(upper(p1.EVENT),1) OVER(w)) IN('BOLUS_INJECTION','BOLUS_PUSH','INFUSION')
            AND (LAG(upper(p1.INSULINTYPE),1) OVER(w)) IN('SHORT')
            -- Identify posterior glucose reading 1 row below
            AND (LEAD(upper(p1.GLCSOURCE),1) OVER(w)) IN('BLOOD', 'FINGERSTICK')
            AND ( -- Preceding glucose has a longer time-gap to insulin than the posterior
                ABS(TIMESTAMP_DIFF((LAG(p1.TIMER,2) OVER(w)), p1.TIMER, MINUTE)) > 
                ABS(TIMESTAMP_DIFF((LEAD(p1.TIMER,1) OVER(w)), p1.TIMER, MINUTE))
                )
            -- Posterior glucose occurs within 90 minutes
            AND ABS(TIMESTAMP_DIFF(LEAD(p1.timer,1) OVER(w), p1.timer, MINUTE)) <= 90
            -- Preceding glucose 2 rows above occured up to 90 minutes before
            AND ABS(TIMESTAMP_DIFF((LAG(p1.TIMER,2) OVER(w)), p1.TIMER, MINUTE)) <= 90
            -- Posterior glucose reading is greater or equal to 90 mg/dL
            AND (LEAD(p1.GLC,1) OVER(w)) >= 90
            -- Posterior glucose value is higher than the preceding glucose 2 rows above
            AND LAG(p1.GLC,2) OVER(w) < LEAD(p1.GLC,1) OVER(w)
        -- Return the POSTERIOR glucose measurement that gathers the previous conditions
        THEN (LEAD(p1.GLC,1) OVER(w))
        
        -- %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        -- 5TH CLAUSE
        -- When posterior timestamp is for Insulin bolus/infusion but preceding is glucose
        -- and there is a glucose 2 rows below.

        -- Identify an insulin event either bolus or infusion
        WHEN upper(p1.EVENT) IN('BOLUS_INJECTION', 'BOLUS_PUSH', 'INFUSION')
            -- Regular insulin or short-acting
            AND upper(p1.INSULINTYPE) IN('SHORT')
            -- Identify preceding glucose reading 1 row above
            AND (LAG(upper(p1.GLCSOURCE),1) OVER(w)) IN('BLOOD', 'FINGERSTICK')
            -- Identify posterior glucose reading 2 rows below
            AND (LEAD(upper(p1.GLCSOURCE),2) OVER(w)) IN('BLOOD', 'FINGERSTICK')
            -- One row BELOW there is another regular insulin
            AND (LEAD(upper(p1.EVENT),1) OVER(w)) IN('BOLUS_INJECTION','BOLUS_PUSH','INFUSION')
            AND (LEAD(upper(p1.INSULINTYPE),1) OVER(w)) IN('SHORT')
            AND ( -- Preceding glucose has a shorter OR equal time-gap to insulin than the posterior
                ABS(TIMESTAMP_DIFF((LAG(p1.TIMER,1) OVER(w)), p1.TIMER, MINUTE)) <= 
                ABS(TIMESTAMP_DIFF((LEAD(p1.TIMER,2) OVER(w)), p1.TIMER, MINUTE))
                )
            -- Preceding glucose reading is greater or equal to 90 mg/dL
            AND (LAG(p1.GLC,1) OVER(w)) >= 90
            -- Preceding glucose 2 rows above occured up to 90 minutes before
            AND ABS(TIMESTAMP_DIFF((LAG(p1.TIMER,1) OVER(w)), p1.TIMER, MINUTE)) <= 90
            -- Posterior glucose value (2 rows below) is lower than the preceding glucose 1 row above
            AND LAG(p1.GLC,1) OVER(w) >= LEAD(p1.GLC,2) OVER(w)
        -- Return the PRECEDING glucose (1 row above) measurement that gathers the previous conditions
        THEN (LAG(p1.GLC,1) OVER(w)) 
        
        -- %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        -- 6TH CLAUSE
        -- When posterior glucose reading (2 rows below) is higher than preceding glucose.
        
        -- Identify an insulin event either bolus or infusion
        WHEN upper(p1.EVENT) IN('BOLUS_INJECTION', 'BOLUS_PUSH', 'INFUSION')
            -- Identify preceding glucose reading 1 row above
            AND (LAG(upper(p1.GLCSOURCE),1) OVER(w)) IN('BLOOD', 'FINGERSTICK')
            -- Identify posterior glucose reading 2 rows below
            AND (LEAD(upper(p1.GLCSOURCE),2) OVER(w)) IN('BLOOD', 'FINGERSTICK')
            -- One row BELOW there is another insulin event
            AND (LEAD(upper(p1.EVENT),1) OVER(w)) IN('BOLUS_INJECTION','BOLUS_PUSH','INFUSION')
            AND ( -- Preceding glucose has a longer time-gap to insulin than the posterior
                ABS(TIMESTAMP_DIFF((LAG(p1.TIMER,1) OVER(w)), p1.TIMER, MINUTE)) > 
                ABS(TIMESTAMP_DIFF((LEAD(p1.TIMER,2) OVER(w)), p1.TIMER, MINUTE))
                )
            -- Posterior glucose reading is greater or equal to 90 mg/dL
            AND (LEAD(p1.GLC,2) OVER(w)) >= 90
            -- Posterior glucose (2 rows below) occurs within 90 minutes
            AND ABS(TIMESTAMP_DIFF(LEAD(p1.timer,2) OVER(w), p1.timer, MINUTE)) <= 90
            -- Preceding glucose 1 row above occures up to 90 minutes before
            AND ABS(TIMESTAMP_DIFF((LAG(p1.TIMER,1) OVER(w)), p1.TIMER, MINUTE)) <= 90
            -- Posterior glucose value (2 rows below) is higher than the preceding glucose 1 row above
            AND LAG(p1.GLC,1) OVER(w) < LEAD(p1.GLC,2) OVER(w)
        -- Return the POSTERIOR glucose (2 rows below) measurement that gathers the previous conditions
        THEN (LEAD(p1.GLC,2) OVER(w))

        -- %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        -- 7TH CLAUSE
        -- When it is the last insulin dose and record in an ICU stay

        -- Identify an insulin event either bolus or infusion
        WHEN upper(p1.EVENT) IN('BOLUS_INJECTION', 'BOLUS_PUSH', 'INFUSION')
            -- Identify preceding glucose reading
            AND (LAG(upper(p1.GLCSOURCE),1) OVER(w)) IN('BLOOD', 'FINGERSTICK')
            -- Time-gap between preceding glucose and insulin, should be equal or less than 90 minutes
            AND (ABS(TIMESTAMP_DIFF((LAG(p1.TIMER,1) OVER(w)), p1.TIMER, MINUTE)) <= 90)
            -- Preceding glucose should be equal or greater than 90 mg/dL
            AND (LAG(p1.GLC,1) OVER(w)) >= 90
        -- Return the PRECEDING glucose measurement that gathers the previous conditions
        THEN (LAG(p1.GLC,1) OVER(w))

        -- %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        -- 8TH CLAUSE
        -- When there is no preceding glucose reading within 90 min, but there is a posterior 
        -- glucose within 90 min

        -- Identify an insulin event either bolus or infusion
        WHEN upper(p1.EVENT) IN('BOLUS_INJECTION', 'BOLUS_PUSH', 'INFUSION')
            -- Regular insulin or short-acting
            AND upper(p1.INSULINTYPE) IN('SHORT')
            -- Identify preceding glucose reading 1 row above
            AND (LAG(upper(p1.GLCSOURCE),1) OVER(w)) IN('BLOOD', 'FINGERSTICK')
            -- Identify posterior glucose reading 1 row below
            AND (LEAD(upper(p1.GLCSOURCE),1) OVER(w)) IN('BLOOD', 'FINGERSTICK')
            -- Time-gap between preceding glucose and insulin is greater than 90 minutes
            AND (ABS(TIMESTAMP_DIFF((LAG(p1.TIMER,1) OVER(w)), p1.TIMER, MINUTE)) > 90)
            -- Time-gap between posterior glucose and insulin is equal or less than 90 minutes
            AND (ABS(TIMESTAMP_DIFF((LEAD(p1.TIMER,1) OVER(w)), p1.TIMER, MINUTE)) <= 90)
            -- Posterior glucose should be equal or greater than 90 mg/dL
            AND (LEAD(p1.GLC,1) OVER(w)) >= 90
        -- Return the POSTERIOR glucose (1 rows below) measurement that gathers the previous conditions
        THEN (LEAD(p1.GLC,1) OVER(w))

        
        -- Otherwise, return null value and finish CASE clause
        ELSE null END
    ) AS GLC_AL

    -- ---------------------------------------------------------------------------------------------
    -- Column GLCTIMER_AL that would gather the timestamp of the paired glucose reading
    , (CASE 
        -- 1ST CLAUSE
        -- When previous and following rows are glucose readings,vselect the glucose value that 
        -- has the shortest time distance to insulin bolus/infusion.

        -- Identify an insulin event either bolus or infusion
        WHEN upper(p1.EVENT) IN('BOLUS_INJECTION', 'BOLUS_PUSH', 'INFUSION')
            -- Identify preceding and posterior glucose reading
            AND (LAG(upper(p1.GLCSOURCE),1) OVER(w)) IN('BLOOD', 'FINGERSTICK')
            AND (LEAD(upper(p1.GLCSOURCE),1) OVER(w)) IN('BLOOD', 'FINGERSTICK')
            AND ( -- Posterior glucose has a longer time-gap to insulin than the posterior
                ABS(TIMESTAMP_DIFF((LAG(p1.TIMER,1) OVER(w)), p1.TIMER, MINUTE)) <= 
                ABS(TIMESTAMP_DIFF((LEAD(p1.TIMER,1) OVER(w)), p1.TIMER, MINUTE))
                )
            -- Time-gap between glucose and insulin, should be equal or less than 90 minutes
            AND ( -- Preceding glucose
                ABS(TIMESTAMP_DIFF((LAG(p1.TIMER,1) OVER(w)), p1.TIMER, MINUTE)) <= 90)
            -- Preceding glucose should be equal or greater than 90 mg/dL
            AND (LAG(p1.GLC,1) OVER(w)) >= 90
            -- Posterior glucose value is lower than the preceding glucose
            AND LAG(p1.GLC,1) OVER(w) >= LEAD(p1.GLC,1) OVER(w)
        -- Return the PRECEDING glucose measurement that gathers the previous conditions
        THEN (LAG(p1.TIMER,1) OVER(w))
        
        -- %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        -- 2ND CLAUSE
        -- In case the posterior glucose reading is higher than the preceding
        
        -- Identify an insulin event either bolus or infusion
        WHEN upper(p1.EVENT) IN('BOLUS_INJECTION', 'BOLUS_PUSH', 'INFUSION')
            -- Identify preceding and posterior glucose measurements
            AND (LAG(upper(p1.GLCSOURCE),1) OVER(w)) IN('BLOOD', 'FINGERSTICK')
            AND (LEAD(upper(p1.GLCSOURCE),1) OVER(w)) IN('BLOOD', 'FINGERSTICK')
            AND ( -- Preceding glucose has a longer OR equal time-gap to insulin than the posterior
                ABS(TIMESTAMP_DIFF((LAG(p1.TIMER,1) OVER(w)), p1.TIMER, MINUTE)) > 
                ABS(TIMESTAMP_DIFF((LEAD(p1.TIMER,1) OVER(w)), p1.TIMER, MINUTE))
                )
            -- Time-gap between glucose and insulin, should be equal or less than 90 minutes
                -- Preceding glucose
            AND ABS(TIMESTAMP_DIFF((LAG(p1.TIMER,1) OVER(w)), p1.TIMER, MINUTE)) <= 90
                -- Posterior glucose
            AND ABS(TIMESTAMP_DIFF((LEAD(p1.TIMER,1) OVER(w)), p1.TIMER, MINUTE)) <= 90
            -- Posterior glucose should be equal or greater than 90 mg/dL
            AND (LEAD(p1.GLC,1) OVER(w)) >= 90
            -- Posterior glucose values is higher than the preceding glucose
            AND LAG(p1.GLC,1) OVER(w) < LEAD(p1.GLC,1) OVER(w) 
        -- Return the POSTERIOR glucose measurement that gathers the previous conditions
        THEN (LEAD(p1.TIMER,1) OVER(w))
        
        -- %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        -- 3RD CLAUSE
        -- When previous timestamp is an insulin bolus/infusion event
        
        -- Identify an insulin event either bolus or infusion
        WHEN upper(p1.EVENT) IN('BOLUS_INJECTION', 'BOLUS_PUSH', 'INFUSION')
            -- Regular insulin or short-acting
            AND upper(p1.INSULINTYPE) IN('SHORT')
            -- Identify preceding glucose reading 2 rows above and regular insulin
            AND (LAG(upper(p1.GLCSOURCE),2) OVER(w)) IN('BLOOD', 'FINGERSTICK')
            AND (LAG(upper(p1.INSULINTYPE),2) OVER(w)) IN('SHORT')
            -- One row above there is another insulin event
            AND (LAG(upper(p1.EVENT),1) OVER(w)) IN('BOLUS_INJECTION','BOLUS_PUSH','INFUSION')
            -- Identify posterior glucose reading 1 row below
            AND (LEAD(upper(p1.GLCSOURCE),1) OVER(w)) IN('BLOOD', 'FINGERSTICK')
            AND ( -- Preceding glucose has a shortime or equal time-gap to insulin than the posterior
                ABS(TIMESTAMP_DIFF((LAG(p1.TIMER,2) OVER(w)), p1.TIMER, MINUTE)) <= 
                ABS(TIMESTAMP_DIFF((LEAD(p1.TIMER,1) OVER(w)), p1.TIMER, MINUTE))
                )
            -- Preceding glucose 2 rows above occured up to 90 minutes before
            AND ABS(TIMESTAMP_DIFF((LAG(p1.TIMER,2) OVER(w)), p1.TIMER, MINUTE)) <= 90
            -- Preceding glucose 2 rows above is equal or greater than 90 min
            AND (LAG(p1.GLC,2) OVER(w)) >= 90
            -- Posterior glucose value is lower than the preceding glucose 2 rows above
            AND LAG(p1.GLC,2) OVER(w) >= LEAD(p1.GLC,1) OVER(w)
        -- Return the preceding glucose value 2 rows above
        THEN (LAG(p1.TIMER,2) OVER(w))
        
        -- %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        -- 4TH CLAUSE
        -- When previous timestamp is for Insulin bolus/infusion but posterior glucose
        -- is higher than the preceding glucose 2 rows above.
        
        -- Identify an insulin event either bolus or infusion
        WHEN upper(p1.EVENT) IN('BOLUS_INJECTION', 'BOLUS_PUSH', 'INFUSION')
            -- Regular insulin or short-acting
            AND upper(p1.INSULINTYPE) IN('SHORT')
            -- Identify preceding glucose reading 2 rows above
            AND (LAG(upper(p1.GLCSOURCE),2) OVER(w)) IN('BLOOD', 'FINGERSTICK')
            -- One row above there is another regular insulin
            AND (LAG(upper(p1.EVENT),1) OVER(w)) IN('BOLUS_INJECTION','BOLUS_PUSH','INFUSION')
            AND (LAG(upper(p1.INSULINTYPE),1) OVER(w)) IN('SHORT')
            -- Identify posterior glucose reading 1 row below
            AND (LEAD(upper(p1.GLCSOURCE),1) OVER(w)) IN('BLOOD', 'FINGERSTICK')
            AND ( -- Preceding glucose has a longer time-gap to insulin than the posterior
                ABS(TIMESTAMP_DIFF((LAG(p1.TIMER,2) OVER(w)), p1.TIMER, MINUTE)) > 
                ABS(TIMESTAMP_DIFF((LEAD(p1.TIMER,1) OVER(w)), p1.TIMER, MINUTE))
                )
            -- Posterior glucose occurs within 90 minutes
            AND ABS(TIMESTAMP_DIFF(LEAD(p1.timer,1) OVER(w), p1.timer, MINUTE)) <= 90
            -- Preceding glucose 2 rows above occured up to 90 minutes before
            AND ABS(TIMESTAMP_DIFF((LAG(p1.TIMER,2) OVER(w)), p1.TIMER, MINUTE)) <= 90
            -- Posterior glucose reading is greater or equal to 90 mg/dL
            AND (LEAD(p1.GLC,1) OVER(w)) >= 90
            -- Posterior glucose value is higher than the preceding glucose 2 rows above
            AND LAG(p1.GLC,2) OVER(w) < LEAD(p1.GLC,1) OVER(w)
        -- Return the POSTERIOR glucose measurement that gathers the previous conditions
        THEN (LEAD(p1.TIMER,1) OVER(w))
        
        -- %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        -- 5TH CLAUSE
        -- When posterior timestamp is for Insulin bolus/infusion but preceding is glucose
        -- and there is a glucose 2 rows below.

        -- Identify an insulin event either bolus or infusion
        WHEN upper(p1.EVENT) IN('BOLUS_INJECTION', 'BOLUS_PUSH', 'INFUSION')
            -- Regular insulin or short-acting
            AND upper(p1.INSULINTYPE) IN('SHORT')
            -- Identify preceding glucose reading 1 row above
            AND (LAG(upper(p1.GLCSOURCE),1) OVER(w)) IN('BLOOD', 'FINGERSTICK')
            -- Identify posterior glucose reading 2 rows below
            AND (LEAD(upper(p1.GLCSOURCE),2) OVER(w)) IN('BLOOD', 'FINGERSTICK')
            -- One row BELOW there is another regular insulin
            AND (LEAD(upper(p1.EVENT),1) OVER(w)) IN('BOLUS_INJECTION','BOLUS_PUSH','INFUSION')
            AND (LEAD(upper(p1.INSULINTYPE),1) OVER(w)) IN('SHORT')
            AND ( -- Preceding glucose has a shorter OR equal time-gap to insulin than the posterior
                ABS(TIMESTAMP_DIFF((LAG(p1.TIMER,1) OVER(w)), p1.TIMER, MINUTE)) <= 
                ABS(TIMESTAMP_DIFF((LEAD(p1.TIMER,2) OVER(w)), p1.TIMER, MINUTE))
                )
            -- Preceding glucose reading is greater or equal to 90 mg/dL
            AND (LAG(p1.GLC,1) OVER(w)) >= 90
            -- Preceding glucose 2 rows above occured up to 90 minutes before
            AND ABS(TIMESTAMP_DIFF((LAG(p1.TIMER,1) OVER(w)), p1.TIMER, MINUTE)) <= 90
            -- Posterior glucose value (2 rows below) is lower than the preceding glucose 1 row above
            AND LAG(p1.GLC,1) OVER(w) >= LEAD(p1.GLC,2) OVER(w)
        -- Return the PRECEDING glucose (1 row above) measurement that gathers the previous conditions
        THEN (LAG(p1.TIMER,1) OVER(w)) 
        
        -- %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        -- 6TH CLAUSE
        -- When posterior glucose reading (2 rows below) is higher than preceding glucose.
        
        -- Identify an insulin event either bolus or infusion
        WHEN upper(p1.EVENT) IN('BOLUS_INJECTION', 'BOLUS_PUSH', 'INFUSION')
            -- Regular insulin or short-acting
            AND upper(p1.INSULINTYPE) IN('SHORT')
            -- Identify preceding glucose reading 1 row above
            AND (LAG(upper(p1.GLCSOURCE),1) OVER(w)) IN('BLOOD', 'FINGERSTICK')
            -- Identify posterior glucose reading 2 rows below
            AND (LEAD(upper(p1.GLCSOURCE),2) OVER(w)) IN('BLOOD', 'FINGERSTICK')
            -- One row BELOW there is another regular insulin
            AND (LEAD(upper(p1.EVENT),1) OVER(w)) IN('BOLUS_INJECTION','BOLUS_PUSH','INFUSION')
            AND (LEAD(upper(p1.INSULINTYPE),1) OVER(w)) IN('SHORT')
            AND ( -- Preceding glucose has a longer time-gap to insulin than the posterior
                ABS(TIMESTAMP_DIFF((LAG(p1.TIMER,1) OVER(w)), p1.TIMER, MINUTE)) > 
                ABS(TIMESTAMP_DIFF((LEAD(p1.TIMER,2) OVER(w)), p1.TIMER, MINUTE))
                )
            -- Posterior glucose reading is greater or equal to 90 mg/dL
            AND (LEAD(p1.GLC,2) OVER(w)) >= 90
            -- Posterior glucose (2 rows below) occurs within 90 minutes
            AND ABS(TIMESTAMP_DIFF(LEAD(p1.timer,2) OVER(w), p1.timer, MINUTE)) <= 90
            -- Preceding glucose 1 row above occures up to 90 minutes before
            AND ABS(TIMESTAMP_DIFF((LAG(p1.TIMER,1) OVER(w)), p1.TIMER, MINUTE)) <= 90
            -- Posterior glucose value (2 rows below) is higher than the preceding glucose 1 row above
            AND LAG(p1.GLC,1) OVER(w) < LEAD(p1.GLC,2) OVER(w)
        -- Return the POSTERIOR glucose (2 rows below) measurement that gathers the previous conditions
        THEN (LEAD(p1.TIMER,2) OVER(w))

        -- %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        -- 7TH CLAUSE
        -- When it is the last insulin dose and record in an ICU stay

        -- Identify an insulin event either bolus or infusion
        WHEN upper(p1.EVENT) IN('BOLUS_INJECTION', 'BOLUS_PUSH', 'INFUSION')
            -- Identify preceding glucose reading
            AND (LAG(upper(p1.GLCSOURCE),1) OVER(w)) IN('BLOOD', 'FINGERSTICK')
            -- Time-gap between preceding glucose and insulin, should be equal or less than 90 minutes
            AND (ABS(TIMESTAMP_DIFF((LAG(p1.TIMER,1) OVER(w)), p1.TIMER, MINUTE)) <= 90)
            -- Preceding glucose should be equal or greater than 90 mg/dL
            AND (LAG(p1.GLC,1) OVER(w)) >= 90
        -- Return the PRECEDING glucose measurement that gathers the previous conditions
        THEN (LAG(p1.TIMER,1) OVER(w))

        -- %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        -- 8TH CLAUSE
        -- When there is no preceding glucose reading within 90 min, but there is a posterior 
        -- glucose within 90 min

        -- Identify an insulin event either bolus or infusion
        WHEN upper(p1.EVENT) IN('BOLUS_INJECTION', 'BOLUS_PUSH', 'INFUSION')
            -- Regular insulin or short-acting
            AND upper(p1.INSULINTYPE) IN('SHORT')
            -- Identify preceding glucose reading 1 row above
            AND (LAG(upper(p1.GLCSOURCE),1) OVER(w)) IN('BLOOD', 'FINGERSTICK')
            -- Identify posterior glucose reading 1 row below
            AND (LEAD(upper(p1.GLCSOURCE),1) OVER(w)) IN('BLOOD', 'FINGERSTICK')
            -- Time-gap between preceding glucose and insulin is greater than 90 minutes
            AND (ABS(TIMESTAMP_DIFF((LAG(p1.TIMER,1) OVER(w)), p1.TIMER, MINUTE)) > 90)
            -- Time-gap between posterior glucose and insulin is equal or less than 90 minutes
            AND (ABS(TIMESTAMP_DIFF((LEAD(p1.TIMER,1) OVER(w)), p1.TIMER, MINUTE)) <= 90)
            -- Posterior glucose should be equal or greater than 90 mg/dL
            AND (LEAD(p1.GLC,1) OVER(w)) >= 90
        -- Return the timestamp of the POSTERIOR glucose (1 rows below) measurement that gathers the 
        -- previous conditions
        THEN (LEAD(p1.TIMER,1) OVER(w))

        -- Otherwise, return null value and finish CASE clause
        ELSE null END
    ) AS GLCTIMER_AL

    -- -----------------------------------------------------------------------------------------------
    -- Column GLCSOURCE_AL that would indicate whether is fingerstick or lab analyzer sample of 
    -- the paired glucose reading
    , (CASE
        -- 1ST CLAUSE
        -- When previous and following rows are glucose readings,vselect the glucose value that 
        -- has the shortest time distance to insulin bolus/infusion.

        -- Identify an insulin event either bolus or infusion
        WHEN upper(p1.EVENT) IN('BOLUS_INJECTION', 'BOLUS_PUSH', 'INFUSION')
            -- Identify preceding and posterior glucose reading
            AND (LAG(upper(p1.GLCSOURCE),1) OVER(w)) IN('BLOOD', 'FINGERSTICK')
            AND (LEAD(upper(p1.GLCSOURCE),1) OVER(w)) IN('BLOOD', 'FINGERSTICK')
            AND ( -- Posterior glucose has a longer time-gap to insulin than the posterior
                ABS(TIMESTAMP_DIFF((LAG(p1.TIMER,1) OVER(w)), p1.TIMER, MINUTE)) <= 
                ABS(TIMESTAMP_DIFF((LEAD(p1.TIMER,1) OVER(w)), p1.TIMER, MINUTE))
                )
            -- Time-gap between glucose and insulin, should be equal or less than 90 minutes
            AND ( -- Preceding glucose
                ABS(TIMESTAMP_DIFF((LAG(p1.TIMER,1) OVER(w)), p1.TIMER, MINUTE)) <= 90)
            -- Preceding glucose should be equal or greater than 90 mg/dL
            AND (LAG(p1.GLC,1) OVER(w)) >= 90
            -- Posterior glucose value is lower than the preceding glucose
            AND LAG(p1.GLC,1) OVER(w) >= LEAD(p1.GLC,1) OVER(w)
        -- Return the PRECEDING glucose measurement that gathers the previous conditions
        THEN (LAG(upper(p1.GLCSOURCE),1) OVER(w))
        
        -- %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        -- 2ND CLAUSE
        -- In case the posterior glucose reading is higher than the preceding
        
        -- Identify an insulin event either bolus or infusion
        WHEN upper(p1.EVENT) IN('BOLUS_INJECTION', 'BOLUS_PUSH', 'INFUSION')
            -- Identify preceding and posterior glucose measurements
            AND (LAG(upper(p1.GLCSOURCE),1) OVER(w)) IN('BLOOD', 'FINGERSTICK')
            AND (LEAD(upper(p1.GLCSOURCE),1) OVER(w)) IN('BLOOD', 'FINGERSTICK')
            AND ( -- Preceding glucose has a longer OR equal time-gap to insulin than the posterior
                ABS(TIMESTAMP_DIFF((LAG(p1.TIMER,1) OVER(w)), p1.TIMER, MINUTE)) > 
                ABS(TIMESTAMP_DIFF((LEAD(p1.TIMER,1) OVER(w)), p1.TIMER, MINUTE))
                )
            -- Time-gap between glucose and insulin, should be equal or less than 90 minutes
                -- Preceding glucose
            AND ABS(TIMESTAMP_DIFF((LAG(p1.TIMER,1) OVER(w)), p1.TIMER, MINUTE)) <= 90
                -- Posterior glucose
            AND ABS(TIMESTAMP_DIFF((LEAD(p1.TIMER,1) OVER(w)), p1.TIMER, MINUTE)) <= 90
            -- Posterior glucose should be equal or greater than 90 mg/dL
            AND (LEAD(p1.GLC,1) OVER(w)) >= 90
            -- Posterior glucose values is higher than the preceding glucose
            AND LAG(p1.GLC,1) OVER(w) < LEAD(p1.GLC,1) OVER(w) 
        -- Return the POSTERIOR glucose measurement that gathers the previous conditions
        THEN (LEAD(upper(p1.GLCSOURCE),1) OVER(w))
        
        -- %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        -- 3RD CLAUSE
        -- When previous timestamp is an insulin bolus/infusion event
        
        -- Identify an insulin event either bolus or infusion
        WHEN upper(p1.EVENT) IN('BOLUS_INJECTION', 'BOLUS_PUSH', 'INFUSION')
            -- Regular insulin or short-acting
            AND upper(p1.INSULINTYPE) IN('SHORT')
            -- Identify preceding glucose reading 2 rows above and regular insulin
            AND (LAG(upper(p1.GLCSOURCE),2) OVER(w)) IN('BLOOD', 'FINGERSTICK')
            AND (LAG(upper(p1.INSULINTYPE),2) OVER(w)) IN('SHORT')
            -- One row above there is another insulin event
            AND (LAG(upper(p1.EVENT),1) OVER(w)) IN('BOLUS_INJECTION','BOLUS_PUSH','INFUSION')
            -- Identify posterior glucose reading 1 row below
            AND (LEAD(upper(p1.GLCSOURCE),1) OVER(w)) IN('BLOOD', 'FINGERSTICK')
            AND ( -- Preceding glucose has a shortime or equal time-gap to insulin than the posterior
                ABS(TIMESTAMP_DIFF((LAG(p1.TIMER,2) OVER(w)), p1.TIMER, MINUTE)) <= 
                ABS(TIMESTAMP_DIFF((LEAD(p1.TIMER,1) OVER(w)), p1.TIMER, MINUTE))
                )
            -- Preceding glucose 2 rows above occured up to 90 minutes before
            AND ABS(TIMESTAMP_DIFF((LAG(p1.TIMER,2) OVER(w)), p1.TIMER, MINUTE)) <= 90
            -- Preceding glucose 2 rows above is equal or greater than 90 min
            AND (LAG(p1.GLC,2) OVER(w)) >= 90
            -- Posterior glucose value is lower than the preceding glucose 2 rows above
            AND LAG(p1.GLC,2) OVER(w) >= LEAD(p1.GLC,1) OVER(w)
        -- Return the preceding glucose value 2 rows above
        THEN (LAG(upper(p1.GLCSOURCE),2) OVER(w))
        
        -- %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        -- 4TH CLAUSE
        -- When previous timestamp is for Insulin bolus/infusion but posterior glucose
        -- is higher than the preceding glucose 2 rows above.
        
        -- Identify an insulin event either bolus or infusion
        WHEN upper(p1.EVENT) IN('BOLUS_INJECTION', 'BOLUS_PUSH', 'INFUSION')
            -- Regular insulin or short-acting
            AND upper(p1.INSULINTYPE) IN('SHORT')
            -- Identify preceding glucose reading 2 rows above
            AND (LAG(upper(p1.GLCSOURCE),2) OVER(w)) IN('BLOOD', 'FINGERSTICK')
            -- One row above there is another regular insulin
            AND (LAG(upper(p1.EVENT),1) OVER(w)) IN('BOLUS_INJECTION','BOLUS_PUSH','INFUSION')
            AND (LAG(upper(p1.INSULINTYPE),1) OVER(w)) IN('SHORT')
            -- Identify posterior glucose reading 1 row below
            AND (LEAD(upper(p1.GLCSOURCE),1) OVER(w)) IN('BLOOD', 'FINGERSTICK')
            AND ( -- Preceding glucose has a longer time-gap to insulin than the posterior
                ABS(TIMESTAMP_DIFF((LAG(p1.TIMER,2) OVER(w)), p1.TIMER, MINUTE)) > 
                ABS(TIMESTAMP_DIFF((LEAD(p1.TIMER,1) OVER(w)), p1.TIMER, MINUTE))
                )
            -- Posterior glucose occurs within 90 minutes
            AND ABS(TIMESTAMP_DIFF(LEAD(p1.timer,1) OVER(w), p1.timer, MINUTE)) <= 90
            -- Preceding glucose 2 rows above occured up to 90 minutes before
            AND ABS(TIMESTAMP_DIFF((LAG(p1.TIMER,2) OVER(w)), p1.TIMER, MINUTE)) <= 90
            -- Posterior glucose reading is greater or equal to 90 mg/dL
            AND (LEAD(p1.GLC,1) OVER(w)) >= 90
            -- Posterior glucose value is higher than the preceding glucose 2 rows above
            AND LAG(p1.GLC,2) OVER(w) < LEAD(p1.GLC,1) OVER(w)
        -- Return the POSTERIOR glucose measurement that gathers the previous conditions
        THEN (LEAD(upper(p1.GLCSOURCE),1) OVER(w))
        
        -- %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        -- 5TH CLAUSE
        -- When posterior timestamp is for Insulin bolus/infusion but preceding is glucose
        -- and there is a glucose 2 rows below.

        -- Identify an insulin event either bolus or infusion
        WHEN upper(p1.EVENT) IN('BOLUS_INJECTION', 'BOLUS_PUSH', 'INFUSION')
            -- Regular insulin or short-acting
            AND upper(p1.INSULINTYPE) IN('SHORT')
            -- Identify preceding glucose reading 1 row above
            AND (LAG(upper(p1.GLCSOURCE),1) OVER(w)) IN('BLOOD', 'FINGERSTICK')
            -- Identify posterior glucose reading 2 rows below
            AND (LEAD(upper(p1.GLCSOURCE),2) OVER(w)) IN('BLOOD', 'FINGERSTICK')
            -- One row BELOW there is another regular insulin
            AND (LEAD(upper(p1.EVENT),1) OVER(w)) IN('BOLUS_INJECTION','BOLUS_PUSH','INFUSION')
            AND (LEAD(upper(p1.INSULINTYPE),1) OVER(w)) IN('SHORT')
            AND ( -- Preceding glucose has a shorter OR equal time-gap to insulin than the posterior
                ABS(TIMESTAMP_DIFF((LAG(p1.TIMER,1) OVER(w)), p1.TIMER, MINUTE)) <= 
                ABS(TIMESTAMP_DIFF((LEAD(p1.TIMER,2) OVER(w)), p1.TIMER, MINUTE))
                )
            -- Preceding glucose reading is greater or equal to 90 mg/dL
            AND (LAG(p1.GLC,1) OVER(w)) >= 90
            -- Preceding glucose 2 rows above occured up to 90 minutes before
            AND ABS(TIMESTAMP_DIFF((LAG(p1.TIMER,1) OVER(w)), p1.TIMER, MINUTE)) <= 90
            -- Posterior glucose value (2 rows below) is lower than the preceding glucose 1 row above
            AND LAG(p1.GLC,1) OVER(w) >= LEAD(p1.GLC,2) OVER(w)
        -- Return the PRECEDING glucose (1 row above) measurement that gathers the previous conditions
        THEN (LAG(upper(p1.GLCSOURCE),1) OVER(w)) 
        
        -- %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        -- 6TH CLAUSE
        -- When posterior glucose reading (2 rows below) is higher than preceding glucose.
        
        -- Identify an insulin event either bolus or infusion
        WHEN upper(p1.EVENT) IN('BOLUS_INJECTION', 'BOLUS_PUSH', 'INFUSION')
            -- Regular insulin or short-acting
            AND upper(p1.INSULINTYPE) IN('SHORT')
            -- Identify preceding glucose reading 1 row above
            AND (LAG(upper(p1.GLCSOURCE),1) OVER(w)) IN('BLOOD', 'FINGERSTICK')
            -- Identify posterior glucose reading 2 rows below
            AND (LEAD(upper(p1.GLCSOURCE),2) OVER(w)) IN('BLOOD', 'FINGERSTICK')
            -- One row BELOW there is another regular insulin
            AND (LEAD(upper(p1.EVENT),1) OVER(w)) IN('BOLUS_INJECTION','BOLUS_PUSH','INFUSION')
            AND (LEAD(upper(p1.INSULINTYPE),1) OVER(w)) IN('Short')
            AND ( -- Preceding glucose has a longer time-gap to insulin than the posterior
                ABS(TIMESTAMP_DIFF((LAG(p1.TIMER,1) OVER(w)), p1.TIMER, MINUTE)) > 
                ABS(TIMESTAMP_DIFF((LEAD(p1.TIMER,2) OVER(w)), p1.TIMER, MINUTE))
                )
            -- Posterior glucose reading is greater or equal to 90 mg/dL
            AND (LEAD(p1.GLC,2) OVER(w)) >= 90
            -- Posterior glucose (2 rows below) occurs within 90 minutes
            AND ABS(TIMESTAMP_DIFF(LEAD(p1.timer,2) OVER(w), p1.timer, MINUTE)) <= 90
            -- Preceding glucose 1 row above occures up to 90 minutes before
            AND ABS(TIMESTAMP_DIFF((LAG(p1.TIMER,1) OVER(w)), p1.TIMER, MINUTE)) <= 90
            -- Posterior glucose value (2 rows below) is higher than the preceding glucose 1 row above
            AND LAG(p1.GLC,1) OVER(w) < LEAD(p1.GLC,2) OVER(w)
        -- Return the POSTERIOR glucose (2 rows below) measurement that gathers the previous conditions
        THEN (LEAD(upper(p1.GLCSOURCE),2) OVER(w))

        -- %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        -- 7TH CLAUSE
        -- When it is the last insulin dose and record in an ICU stay

        -- Identify an insulin event either bolus or infusion
        WHEN upper(p1.EVENT) IN('BOLUS_INJECTION', 'BOLUS_PUSH', 'INFUSION')
            -- Identify preceding glucose reading
            AND (LAG(upper(p1.GLCSOURCE),1) OVER(w)) IN('BLOOD', 'FINGERSTICK')
            -- Time-gap between preceding glucose and insulin, should be equal or less than 90 minutes
            AND (ABS(TIMESTAMP_DIFF((LAG(p1.TIMER,1) OVER(w)), p1.TIMER, MINUTE)) <= 90)
            -- Preceding glucose should be equal or greater than 90 mg/dL
            AND (LAG(p1.GLC,1) OVER(w)) >= 90
        -- Return the PRECEDING glucose measurement that gathers the previous conditions
        THEN (LAG(upper(p1.GLCSOURCE),1) OVER(w))

        -- %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        -- 8TH CLAUSE
        -- When there is no preceding glucose reading within 90 min, but there is a posterior 
        -- glucose within 90 min

        -- Identify an insulin event either bolus or infusion
        WHEN upper(p1.EVENT) IN('BOLUS_INJECTION', 'BOLUS_PUSH', 'INFUSION')
            -- Regular insulin or short-acting
            AND upper(p1.INSULINTYPE) IN('SHORT')
            -- Identify preceding glucose reading 1 row above
            AND (LAG(upper(p1.GLCSOURCE),1) OVER(w)) IN('BLOOD', 'FINGERSTICK')
            -- Identify posterior glucose reading 1 row below
            AND (LEAD(upper(p1.GLCSOURCE),1) OVER(w)) IN('BLOOD', 'FINGERSTICK')
            -- Time-gap between preceding glucose and insulin is greater than 90 minutes
            AND (ABS(TIMESTAMP_DIFF((LAG(p1.TIMER,1) OVER(w)), p1.TIMER, MINUTE)) > 90)
            -- Time-gap between posterior glucose and insulin is equal or less than 90 minutes
            AND (ABS(TIMESTAMP_DIFF((LEAD(p1.TIMER,1) OVER(w)), p1.TIMER, MINUTE)) <= 90)
            -- Posterior glucose should be equal or greater than 90 mg/dL
            AND (LEAD(p1.GLC,1) OVER(w)) >= 90
        -- Return the whether is figerstick or lab analyzer the POSTERIOR glucose (1 rows below) measurement 
        -- that gathers the previous conditions
        THEN (LEAD(upper(p1.GLCSOURCE),1) OVER(w))

        -- Otherwise, return null value and finish CASE clause
        ELSE null END
    ) AS GLCSOURCE_AL

    -- ---------------------------------------------------------------------------------------------
    -- Column RULE that indicateS which pairing rule is applied for the i^th case
    , (CASE
        -- 1ST CLAUSE
        -- When previous and following rows are glucose readings,vselect the glucose value that 
        -- has the shortest time distance to insulin bolus/infusion.

            -- Identify an insulin event either bolus or infusion
        WHEN upper(p1.EVENT) IN('BOLUS_INJECTION', 'BOLUS_PUSH', 'INFUSION')
            -- Identify preceding and posterior glucose reading
            AND (LAG(upper(p1.GLCSOURCE),1) OVER(w)) IN('BLOOD', 'FINGERSTICK')
            AND (LEAD(upper(p1.GLCSOURCE),1) OVER(w)) IN('BLOOD', 'FINGERSTICK')
            AND ( -- Posterior glucose has a longer time-gap to insulin than the posterior
                ABS(TIMESTAMP_DIFF((LAG(p1.TIMER,1) OVER(w)), p1.TIMER, MINUTE)) <= 
                ABS(TIMESTAMP_DIFF((LEAD(p1.TIMER,1) OVER(w)), p1.TIMER, MINUTE))
                )
            -- Time-gap between glucose and insulin, should be equal or less than 90 minutes
            AND ( -- Preceding glucose
                ABS(TIMESTAMP_DIFF((LAG(p1.TIMER,1) OVER(w)), p1.TIMER, MINUTE)) <= 90)
            -- Preceding glucose should be equal or greater than 90 mg/dL
            AND (LAG(p1.GLC,1) OVER(w)) >= 90
            -- Posterior glucose value is lower than the preceding glucose
            AND LAG(p1.GLC,1) OVER(w) >= LEAD(p1.GLC,1) OVER(w)
        -- Return the PRECEDING glucose measurement that gathers the previous conditions
        THEN 1
        
        -- %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        -- 2ND CLAUSE
        -- In case the posterior glucose reading is higher than the preceding
        
        -- Identify an insulin event either bolus or infusion
        WHEN upper(p1.EVENT) IN('BOLUS_INJECTION', 'BOLUS_PUSH', 'INFUSION')
            -- Regular insulin or short-acting
            AND upper(p1.INSULINTYPE) IN('SHORT')
            -- Identify preceding and posterior glucose measurements
            AND (LAG(upper(p1.GLCSOURCE),1) OVER(w)) IN('BLOOD', 'FINGERSTICK')
            AND (LEAD(upper(p1.GLCSOURCE),1) OVER(w)) IN('BLOOD', 'FINGERSTICK')
            AND ( -- Preceding glucose has a longer OR equal time-gap to insulin than the posterior
                ABS(TIMESTAMP_DIFF((LAG(p1.TIMER,1) OVER(w)), p1.TIMER, MINUTE)) > 
                ABS(TIMESTAMP_DIFF((LEAD(p1.TIMER,1) OVER(w)), p1.TIMER, MINUTE))
                )
            -- Time-gap between glucose and insulin, should be equal or less than 90 minutes
                -- Preceding glucose
            AND ABS(TIMESTAMP_DIFF((LAG(p1.TIMER,1) OVER(w)), p1.TIMER, MINUTE)) <= 90
                -- Posterior glucose
            AND ABS(TIMESTAMP_DIFF((LEAD(p1.TIMER,1) OVER(w)), p1.TIMER, MINUTE)) <= 90
            -- Posterior glucose should be equal or greater than 90 mg/dL
            AND (LEAD(p1.GLC,1) OVER(w)) >= 90
            -- Posterior glucose values is higher than the preceding glucose
            AND LAG(p1.GLC,1) OVER(w) < LEAD(p1.GLC,1) OVER(w) 
        -- Return the POSTERIOR glucose measurement that gathers the previous conditions
        THEN 3
        
        -- %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        -- 3RD CLAUSE
        -- When previous timestamp is an insulin bolus/infusion event
        
        -- Identify an insulin event either bolus or infusion
        WHEN upper(p1.EVENT) IN('BOLUS_INJECTION', 'BOLUS_PUSH', 'INFUSION')
            -- Regular insulin or short-acting
            AND upper(p1.INSULINTYPE) IN('SHORT')
            -- Identify preceding glucose reading 2 rows above and regular insulin
            AND (LAG(upper(p1.GLCSOURCE),2) OVER(w)) IN('BLOOD', 'FINGERSTICK')
            AND (LAG(upper(p1.INSULINTYPE),2) OVER(w)) IN('SHORT')
            -- One row above there is another insulin event
            AND (LAG(upper(p1.EVENT),1) OVER(w)) IN('BOLUS_INJECTION','BOLUS_PUSH','INFUSION')
            -- Identify posterior glucose reading 1 row below
            AND (LEAD(upper(p1.GLCSOURCE),1) OVER(w)) IN('BLOOD', 'FINGERSTICK')
            AND ( -- Preceding glucose has a shortime or equal time-gap to insulin than the posterior
                ABS(TIMESTAMP_DIFF((LAG(p1.TIMER,2) OVER(w)), p1.TIMER, MINUTE)) <= 
                ABS(TIMESTAMP_DIFF((LEAD(p1.TIMER,1) OVER(w)), p1.TIMER, MINUTE))
                )
            -- Preceding glucose 2 rows above occured up to 90 minutes before
            AND ABS(TIMESTAMP_DIFF((LAG(p1.TIMER,2) OVER(w)), p1.TIMER, MINUTE)) <= 90
            -- Preceding glucose 2 rows above is equal or greater than 90 min
            AND (LAG(p1.GLC,2) OVER(w)) >= 90
            -- Posterior glucose value is lower than the preceding glucose 2 rows above
            AND LAG(p1.GLC,2) OVER(w) >= LEAD(p1.GLC,1) OVER(w)
        -- Return the preceding glucose value 2 rows above
        THEN 4
        
        -- %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        -- 4TH CLAUSE
        -- When previous timestamp is for Insulin bolus/infusion but posterior glucose
        -- is higher than the preceding glucose 2 rows above.
        
        -- Identify an insulin event either bolus or infusion
        WHEN upper(p1.EVENT) IN('BOLUS_INJECTION', 'BOLUS_PUSH', 'INFUSION')
            -- Regular insulin or short-acting
            AND upper(p1.INSULINTYPE) IN('SHORT')
            -- Identify preceding glucose reading 2 rows above
            AND (LAG(upper(p1.GLCSOURCE),2) OVER(w)) IN('BLOOD', 'FINGERSTICK')
            -- One row above there is another regular insulin
            AND (LAG(upper(p1.EVENT),1) OVER(w)) IN('BOLUS_INJECTION','BOLUS_PUSH','INFUSION')
            AND (LAG(upper(p1.INSULINTYPE),1) OVER(w)) IN('SHORT')
            -- Identify posterior glucose reading 1 row below
            AND (LEAD(upper(p1.GLCSOURCE),1) OVER(w)) IN('BLOOD', 'FINGERSTICK')
            AND ( -- Preceding glucose has a longer time-gap to insulin than the posterior
                ABS(TIMESTAMP_DIFF((LAG(p1.TIMER,2) OVER(w)), p1.TIMER, MINUTE)) > 
                ABS(TIMESTAMP_DIFF((LEAD(p1.TIMER,1) OVER(w)), p1.TIMER, MINUTE))
                )
            -- Posterior glucose occurs within 90 minutes
            AND ABS(TIMESTAMP_DIFF(LEAD(p1.timer,1) OVER(w), p1.timer, MINUTE)) <= 90
            -- Preceding glucose 2 rows above occured up to 90 minutes before
            AND ABS(TIMESTAMP_DIFF((LAG(p1.TIMER,2) OVER(w)), p1.TIMER, MINUTE)) <= 90
            -- Posterior glucose reading is greater or equal to 90 mg/dL
            AND (LEAD(p1.GLC,1) OVER(w)) >= 90
            -- Posterior glucose value is higher than the preceding glucose 2 rows above
            AND LAG(p1.GLC,2) OVER(w) < LEAD(p1.GLC,1) OVER(w)
        -- Return the POSTERIOR glucose measurement that gathers the previous conditions
        THEN 4
        
        -- %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        -- 5TH CLAUSE
        -- When posterior timestamp is for Insulin bolus/infusion but preceding is glucose
        -- and there is a glucose 2 rows below.

        -- Identify an insulin event either bolus or infusion
        WHEN upper(p1.EVENT) IN('BOLUS_INJECTION', 'BOLUS_PUSH', 'INFUSION')
            -- Regular insulin or short-acting
            AND upper(p1.INSULINTYPE) IN('SHORT')
            -- Identify preceding glucose reading 1 row above
            AND (LAG(upper(p1.GLCSOURCE),1) OVER(w)) IN('BLOOD', 'FINGERSTICK')
            -- Identify posterior glucose reading 2 rows below
            AND (LEAD(upper(p1.GLCSOURCE),2) OVER(w)) IN('BLOOD', 'FINGERSTICK')
            -- One row BELOW there is another regular insulin
            AND (LEAD(upper(p1.EVENT),1) OVER(w)) IN('BOLUS_INJECTION','BOLUS_PUSH','INFUSION')
            AND (LEAD(upper(p1.INSULINTYPE),1) OVER(w)) IN('SHORT')
            AND ( -- Preceding glucose has a shorter OR equal time-gap to insulin than the posterior
                ABS(TIMESTAMP_DIFF((LAG(p1.TIMER,1) OVER(w)), p1.TIMER, MINUTE)) <= 
                ABS(TIMESTAMP_DIFF((LEAD(p1.TIMER,2) OVER(w)), p1.TIMER, MINUTE))
                )
            -- Preceding glucose reading is greater or equal to 90 mg/dL
            AND (LAG(p1.GLC,1) OVER(w)) >= 90
            -- Preceding glucose 2 rows above occured up to 90 minutes before
            AND ABS(TIMESTAMP_DIFF((LAG(p1.TIMER,1) OVER(w)), p1.TIMER, MINUTE)) <= 90
            -- Posterior glucose value (2 rows below) is lower than the preceding glucose 1 row above
            AND LAG(p1.GLC,1) OVER(w) >= LEAD(p1.GLC,2) OVER(w)
        -- Return the PRECEDING glucose (1 row above) measurement that gathers the previous conditions
        THEN 4
        
        -- %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        -- 6TH CLAUSE
        -- When posterior glucose reading (2 rows below) is higher than preceding glucose.
        
        -- Identify an insulin event either bolus or infusion
        WHEN upper(p1.EVENT) IN('BOLUS_INJECTION', 'BOLUS_PUSH', 'INFUSION')
            -- Regular insulin or short-acting
            AND upper(p1.INSULINTYPE) IN('Short')
            -- Identify preceding glucose reading 1 row above
            AND (LAG(upper(p1.GLCSOURCE),1) OVER(w)) IN('BLOOD', 'FINGERSTICK')
            -- Identify posterior glucose reading 2 rows below
            AND (LEAD(upper(p1.GLCSOURCE),2) OVER(w)) IN('BLOOD', 'FINGERSTICK')
            -- One row BELOW there is another regular insulin
            AND (LEAD(upper(p1.EVENT),1) OVER(w)) IN('BOLUS_INJECTION','BOLUS_PUSH','INFUSION')
            AND (LEAD(upper(p1.INSULINTYPE),1) OVER(w)) IN('SHORT')
            AND ( -- Preceding glucose has a longer time-gap to insulin than the posterior
                ABS(TIMESTAMP_DIFF((LAG(p1.TIMER,1) OVER(w)), p1.TIMER, MINUTE)) > 
                ABS(TIMESTAMP_DIFF((LEAD(p1.TIMER,2) OVER(w)), p1.TIMER, MINUTE))
                )
            -- Posterior glucose reading is greater or equal to 90 mg/dL
            AND (LEAD(p1.GLC,2) OVER(w)) >= 90
            -- Posterior glucose (2 rows below) occurs within 90 minutes
            AND ABS(TIMESTAMP_DIFF(LEAD(p1.timer,2) OVER(w), p1.timer, MINUTE)) <= 90
            -- Preceding glucose 1 row above occures up to 90 minutes before
            AND ABS(TIMESTAMP_DIFF((LAG(p1.TIMER,1) OVER(w)), p1.TIMER, MINUTE)) <= 90
            -- Posterior glucose value (2 rows below) is higher than the preceding glucose 1 row above
            AND LAG(p1.GLC,1) OVER(w) < LEAD(p1.GLC,2) OVER(w)
        -- Return the POSTERIOR glucose (2 rows below) measurement that gathers the previous conditions
        THEN 4

        -- %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        -- 7TH CLAUSE
        -- When it is the last insulin dose and record in an ICU stay

        -- Identify an insulin event either bolus or infusion
        WHEN upper(p1.EVENT) IN('BOLUS_INJECTION', 'BOLUS_PUSH', 'INFUSION')
            -- Identify preceding glucose reading
            AND (LAG(upper(p1.GLCSOURCE),1) OVER(w)) IN('BLOOD', 'FINGERSTICK')
            -- Time-gap between preceding glucose and insulin, should be equal or less than 90 minutes
            AND (ABS(TIMESTAMP_DIFF((LAG(p1.TIMER,1) OVER(w)), p1.TIMER, MINUTE)) <= 90)
            -- Preceding glucose should be equal or greater than 90 mg/dL
            AND (LAG(p1.GLC,1) OVER(w)) >= 90
        -- Return the PRECEDING glucose measurement that gathers the previous conditions
        THEN 1

        -- %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        -- 8TH CLAUSE
        -- When there is no preceding glucose reading within 90 min, but there is a posterior 
        -- glucose within 90 min

        -- Identify an insulin event either bolus or infusion
        WHEN upper(p1.EVENT) IN('BOLUS_INJECTION', 'BOLUS_PUSH', 'INFUSION')
            -- Regular insulin or short-acting
            AND upper(p1.INSULINTYPE) IN('SHORT')
            -- Identify preceding glucose reading 1 row above
            AND (LAG(upper(p1.GLCSOURCE),1) OVER(w)) IN('BLOOD', 'FINGERSTICK')
            -- Identify posterior glucose reading 1 row below
            AND (LEAD(upper(p1.GLCSOURCE),1) OVER(w)) IN('BLOOD', 'FINGERSTICK')
            -- Time-gap between preceding glucose and insulin is greater than 90 minutes
            AND (ABS(TIMESTAMP_DIFF((LAG(p1.TIMER,1) OVER(w)), p1.TIMER, MINUTE)) > 90)
            -- Time-gap between posterior glucose and insulin is equal or less than 90 minutes
            AND (ABS(TIMESTAMP_DIFF((LEAD(p1.TIMER,1) OVER(w)), p1.TIMER, MINUTE)) <= 90)
            -- Posterior glucose should be equal or greater than 90 mg/dL
            AND (LEAD(p1.GLC,1) OVER(w)) >= 90
        -- Return the Rule number applied
        THEN 2
        
        -- Otherwise, return null value and finish CASE clause
        ELSE null END
    ) AS RULE

    FROM {{ ref('stg_glucose_insulin_icustay') }} AS p1
    WINDOW w AS(PARTITION BY CAST(p1.HADM_ID AS INT64) ORDER BY p1.TIMER)
)

-- Create a colum that identifies the glucose readings were paired and are duplicated in pg
SELECT pg.*
, (CASE
          WHEN pg.GLCSOURCE_AL IS null 
          AND (LEAD(pg.GLCTIMER_AL,1) OVER(x) = pg.GLCTIMER)
          THEN 1 
          WHEN pg.GLCSOURCE_AL IS null 
          AND (LAG(pg.GLCTIMER_AL,1) OVER(x) = pg.GLCTIMER)
          AND LAG(endtime,1) OVER(x) IS NOT null 
          THEN 1
          ELSE null END) AS Repeated
FROM pg
WINDOW x AS(PARTITION BY stay_id ORDER BY pg.timer)
ORDER BY STAY_ID,TIMER