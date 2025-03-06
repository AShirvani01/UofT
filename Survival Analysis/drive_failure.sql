SELECT serial_number, COUNT(failure) AS num_of_failures
FROM hard_drive.hd_failure
GROUP BY serial_number,failure
HAVING failure = 1;


SELECT 
serial_number, 
model, 
MAX(capacity_bytes) AS capacity_bytes, 
MIN(date) AS start_time, 
MAX(date) AS end_time, 
DATEDIFF(MAX(date),MIN(date)) AS observed_time, 
MAX(failure) AS failure
FROM hard_drive.hd_failure
GROUP BY serial_number,model;

