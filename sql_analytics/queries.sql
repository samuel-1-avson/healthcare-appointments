-- ============================================================
-- Week 2: SQL Analytics - Healthcare Appointments KPI Queries
-- ============================================================
-- Database: healthcare.db
-- Table: appointments (110,527 rows)
-- Purpose: Answer 10 stakeholder questions about appointment no-shows
-- ============================================================

-- QUERY 1: Overall No-Show Rate and Appointment Volume
-- Business Question: What is our baseline performance?
-- ============================================================
-- name: overall_metrics
SELECT 
    COUNT(*) as total_appointments,
    SUM(No_Show) as total_no_shows,
    SUM(showed_up) as total_showed_up,
    ROUND(AVG(No_Show) * 100, 2) as no_show_rate_percent,
    ROUND(AVG(showed_up) * 100, 2) as show_up_rate_percent,
    COUNT(DISTINCT PatientId) as unique_patients,
    ROUND(COUNT(*) * 1.0 / COUNT(DISTINCT PatientId), 2) as avg_appointments_per_patient
FROM appointments;


-- QUERY 2: Neighborhood-Level No-Show Analysis
-- Business Question: Which neighborhoods have the highest no-show rates?
-- ============================================================
-- name: neighborhood_analysis
SELECT 
    neighbourhood,
    COUNT(*) as total_appointments,
    SUM(No_Show) as no_shows,
    ROUND(AVG(No_Show) * 100, 2) as no_show_rate_percent,
    COUNT(DISTINCT PatientId) as unique_patients,
    RANK() OVER (ORDER BY AVG(No_Show) DESC) as risk_rank
FROM appointments
GROUP BY neighbourhood
HAVING COUNT(*) >= 100  -- Filter out neighborhoods with very few appointments
ORDER BY no_show_rate_percent DESC
LIMIT 20;


-- QUERY 3: Age Group Attendance Patterns
-- Business Question: How does age affect no-show behavior?
-- ============================================================
-- name: age_group_analysis
SELECT 
    Age_Group,
    COUNT(*) as total_appointments,
    SUM(No_Show) as no_shows,
    SUM(showed_up) as showed_up,
    ROUND(AVG(No_Show) * 100, 2) as no_show_rate_percent,
    ROUND(AVG(Age), 1) as avg_age_in_group,
    MIN(Age) as min_age,
    MAX(Age) as max_age
FROM appointments
WHERE Age_Group IS NOT NULL
GROUP BY Age_Group
ORDER BY 
    CASE Age_Group
        WHEN 'Child' THEN 1
        WHEN 'Teen' THEN 2
        WHEN 'Young Adult' THEN 3
        WHEN 'Adult' THEN 4
        WHEN 'Senior' THEN 5
        WHEN 'Elderly' THEN 6
        ELSE 7
    END;


-- QUERY 4: SMS Reminder Effectiveness Analysis
-- Business Question: Do SMS reminders reduce no-shows?
-- ============================================================
-- name: sms_effectiveness
SELECT 
    CASE WHEN SMS_received = 1 THEN 'SMS Sent' ELSE 'No SMS' END as sms_status,
    COUNT(*) as total_appointments,
    SUM(No_Show) as no_shows,
    SUM(showed_up) as showed_up,
    ROUND(AVG(No_Show) * 100, 2) as no_show_rate_percent,
    ROUND(AVG(showed_up) * 100, 2) as show_up_rate_percent,
    -- Calculate the difference vs baseline
    ROUND(AVG(No_Show) * 100 - (SELECT AVG(No_Show) * 100 FROM appointments), 2) as diff_from_baseline
FROM appointments
GROUP BY SMS_received
ORDER BY SMS_received DESC;


-- QUERY 5: Chronic Condition Impact on Attendance
-- Business Question: Do patients with chronic conditions have different patterns?
-- ============================================================
-- name: chronic_conditions_analysis
SELECT 
    CASE 
        WHEN Hypertension = 1 AND Diabetes = 1 THEN 'Both Conditions'
        WHEN Hypertension = 1 THEN 'Hypertension Only'
        WHEN Diabetes = 1 THEN 'Diabetes Only'
        ELSE 'No Chronic Conditions'
    END as health_status,
    COUNT(*) as total_appointments,
    SUM(No_Show) as no_shows,
    ROUND(AVG(No_Show) * 100, 2) as no_show_rate_percent,
    COUNT(DISTINCT PatientId) as unique_patients,
    ROUND(AVG(Age), 1) as avg_age
FROM appointments
GROUP BY 
    CASE 
        WHEN Hypertension = 1 AND Diabetes = 1 THEN 'Both Conditions'
        WHEN Hypertension = 1 THEN 'Hypertension Only'
        WHEN Diabetes = 1 THEN 'Diabetes Only'
        ELSE 'No Chronic Conditions'
    END
ORDER BY no_show_rate_percent DESC;


-- QUERY 6: Lead Time Correlation Analysis
-- Business Question: How does scheduling advance time affect no-shows?
-- ============================================================
-- name: lead_time_analysis
SELECT 
    CASE 
        WHEN Lead_Days = 0 THEN 'Same Day'
        WHEN Lead_Days BETWEEN 1 AND 3 THEN '1-3 Days'
        WHEN Lead_Days BETWEEN 4 AND 7 THEN '4-7 Days'
        WHEN Lead_Days BETWEEN 8 AND 14 THEN '1-2 Weeks'
        WHEN Lead_Days BETWEEN 15 AND 30 THEN '2-4 Weeks'
        WHEN Lead_Days > 30 THEN 'Over 1 Month'
        ELSE 'Unknown'
    END as lead_time_category,
    COUNT(*) as total_appointments,
    SUM(No_Show) as no_shows,
    ROUND(AVG(No_Show) * 100, 2) as no_show_rate_percent,
    ROUND(AVG(Lead_Days), 1) as avg_lead_days,
    MIN(Lead_Days) as min_lead_days,
    MAX(Lead_Days) as max_lead_days
FROM appointments
WHERE Lead_Days IS NOT NULL
GROUP BY 
    CASE 
        WHEN Lead_Days = 0 THEN 'Same Day'
        WHEN Lead_Days BETWEEN 1 AND 3 THEN '1-3 Days'
        WHEN Lead_Days BETWEEN 4 AND 7 THEN '4-7 Days'
        WHEN Lead_Days BETWEEN 8 AND 14 THEN '1-2 Weeks'
        WHEN Lead_Days BETWEEN 15 AND 30 THEN '2-4 Weeks'
        WHEN Lead_Days > 30 THEN 'Over 1 Month'
        ELSE 'Unknown'
    END
ORDER BY avg_lead_days;


-- QUERY 7: Day of Week Attendance Patterns
-- Business Question: Which days have the highest no-show rates?
-- ============================================================
-- name: weekday_analysis
SELECT 
    Appointment_Weekday,
    COUNT(*) as total_appointments,
    SUM(No_Show) as no_shows,
    ROUND(AVG(No_Show) * 100, 2) as no_show_rate_percent,
    ROUND(COUNT(*) * 100.0 / (SELECT COUNT(*) FROM appointments), 2) as pct_of_total_volume,
    -- Add rank
    RANK() OVER (ORDER BY AVG(No_Show) DESC) as worst_day_rank
FROM appointments
WHERE Appointment_Weekday IS NOT NULL
GROUP BY Appointment_Weekday
ORDER BY 
    CASE Appointment_Weekday
        WHEN 'Monday' THEN 1
        WHEN 'Tuesday' THEN 2
        WHEN 'Wednesday' THEN 3
        WHEN 'Thursday' THEN 4
        WHEN 'Friday' THEN 5
        WHEN 'Saturday' THEN 6
        WHEN 'Sunday' THEN 7
    END;


-- QUERY 8: Scholarship Program Effectiveness
-- Business Question: Do scholarship recipients have different attendance?
-- ============================================================
-- name: scholarship_analysis
SELECT 
    CASE WHEN Scholarship = 1 THEN 'Scholarship Recipient' ELSE 'No Scholarship' END as scholarship_status,
    COUNT(*) as total_appointments,
    SUM(No_Show) as no_shows,
    ROUND(AVG(No_Show) * 100, 2) as no_show_rate_percent,
    COUNT(DISTINCT PatientId) as unique_patients,
    ROUND(AVG(Age), 1) as avg_age,
    -- Check if SMS was sent more to scholarship recipients
    ROUND(AVG(SMS_received) * 100, 2) as pct_received_sms
FROM appointments
GROUP BY Scholarship
ORDER BY Scholarship DESC;


-- QUERY 9: Gender-Based Attendance Analysis
-- Business Question: Does gender correlate with no-show rates?
-- ============================================================
-- name: gender_analysis
SELECT 
    Gender,
    COUNT(*) as total_appointments,
    SUM(No_Show) as no_shows,
    ROUND(AVG(No_Show) * 100, 2) as no_show_rate_percent,
    COUNT(DISTINCT PatientId) as unique_patients,
    ROUND(AVG(Age), 1) as avg_age,
    ROUND(AVG(SMS_received) * 100, 2) as pct_received_sms,
    ROUND(AVG(Scholarship) * 100, 2) as pct_scholarship
FROM appointments
GROUP BY Gender
ORDER BY no_show_rate_percent DESC;


-- QUERY 10: Peak Appointment Periods and Capacity Analysis
-- Business Question: When are appointments concentrated and how effective are they?
-- ============================================================
-- name: temporal_capacity_analysis
SELECT 
    strftime('%Y-%m', AppointmentDay) as appointment_month,
    COUNT(*) as total_appointments,
    SUM(No_Show) as no_shows,
    ROUND(AVG(No_Show) * 100, 2) as no_show_rate_percent,
    COUNT(DISTINCT PatientId) as unique_patients,
    ROUND(COUNT(*) * 1.0 / COUNT(DISTINCT DATE(AppointmentDay)), 1) as avg_appointments_per_day,
    -- Calculate efficiency (showed up appointments)
    SUM(showed_up) as effective_appointments,
    ROUND(SUM(showed_up) * 1.0 / COUNT(DISTINCT DATE(AppointmentDay)), 1) as effective_appointments_per_day
FROM appointments
WHERE AppointmentDay IS NOT NULL
GROUP BY strftime('%Y-%m', AppointmentDay)
ORDER BY appointment_month;


-- ============================================================
-- Additional Advanced Query: Patient Segmentation Analysis
-- Business Question: Can we identify high-risk patient segments?
-- ============================================================
-- name: patient_risk_segments
SELECT 
    CASE 
        WHEN Age < 18 THEN 'Youth'
        WHEN Age >= 60 THEN 'Senior'
        ELSE 'Adult'
    END as age_segment,
    CASE WHEN SMS_received = 1 THEN 'With SMS' ELSE 'No SMS' END as sms_segment,
    CASE WHEN Scholarship = 1 THEN 'Low Income' ELSE 'Regular' END as income_segment,
    COUNT(*) as total_appointments,
    SUM(No_Show) as no_shows,
    ROUND(AVG(No_Show) * 100, 2) as no_show_rate_percent
FROM appointments
GROUP BY age_segment, sms_segment, income_segment
HAVING COUNT(*) >= 50  -- Only show segments with meaningful sample size
ORDER BY no_show_rate_percent DESC
LIMIT 15;
