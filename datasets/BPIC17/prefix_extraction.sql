SELECT caseid, task, end_timestamp, end_timestamp as time_copy, user as role
        FROM original_df;

SELECT a.caseid, min(a.end_timestamp) start_time
        FROM raw a
        GROUP BY a.caseid;

SELECT a.*, b.start_time, ROW_NUMBER() OVER(
            PARTITION BY a.caseid
            ORDER BY end_timestamp asc) task_index
        FROM raw a
        JOIN cases b ON a.caseid = b.caseid;

SELECT DISTINCT o.caseid, o.task_index AS milestone_index, o.task AS milestone, o2.task AS next_activity,
        ROW_NUMBER() OVER(
            PARTITION BY o.caseid
            ORDER BY o.task_index) milestone_id
        FROM traces o JOIN
        traces o2 ON o.task_index = o2.task_index-1 AND o.caseid = o2.caseid;

SELECT CAST(o.caseid AS TEXT)|| '_' ||CAST(m.milestone_id AS TEXT) AS prefix_id, o.caseid, o.task, o.role, o.end_timestamp, o.start_time AS trace_start, ROUND((JULIANDAY(o.end_timestamp) - JULIANDAY(o.start_time))) AS timelapsed, ROUND((JULIANDAY(o.next_time) - JULIANDAY(o.start_time))) AS next_time, o.poac, o.poac_time, m.next_activity, m.milestone, m.milestone_id, o.task_index
        FROM traces o
        JOIN milestones m ON o.caseid = m.caseid AND o.task_index <= m.milestone_index;

