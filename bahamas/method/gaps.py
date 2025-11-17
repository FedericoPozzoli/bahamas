"""
This module generates observation gaps based on scheduled and unscheduled constraints.
It provides a function to create gaps in observation data, considering scheduled periods, unscheduled events, and merging close gaps.
"""
import numpy as np

def generate_gaps(T_obs, scheduled_gap=7*3600, scheduled_period=14*86400, unscheduled_gap=0., exp_scale=10*86400, merge_threshold=1*86400, duty_cycle=0.75):
    """
    Generate observation gaps based on scheduled and unscheduled constraints.
    Args:
        T_obs (int): Total observation time in seconds.
        scheduled_gap (int): Duration of scheduled gaps in seconds.
        scheduled_period (int): Duration of scheduled periods in seconds.
        unscheduled_gap (int): Duration of unscheduled gaps in seconds.
        exp_scale (int): Scale for the exponential distribution for unscheduled gaps.
        merge_threshold (int): Threshold for merging close gaps in seconds.
        duty_cycle (float): Desired duty cycle for observing time.
    Returns:
        tuple: Start and end times of data segments as numpy arrays.
    """
    # Generate scheduled gaps

    tot = scheduled_period + scheduled_gap
  
    seg = np.arange(0, T_obs , tot)
    scheduled_gaps = []
    for i in seg:
        if i + scheduled_gap <= T_obs:
            scheduled_gaps.append((i, i + scheduled_gap))

    data_seg = []

    if unscheduled_gap > 0:
        # Generate unscheduled gaps
        unscheduled_gaps = []
        merged_gaps = []
        t = np.random.exponential(exp_scale)
        while t + unscheduled_gap < T_obs:
            unscheduled_gaps.append((t, t + unscheduled_gap))
            t += np.random.exponential(exp_scale) + unscheduled_gap

        # Combine gaps
        all_gaps = sorted(scheduled_gaps + unscheduled_gaps)

        # Merge overlapping or close gaps
        
        for start, end in all_gaps:
            if merged_gaps and start - merged_gaps[-1][1] < merge_threshold:
                merged_gaps[-1] = (merged_gaps[-1][0], max(merged_gaps[-1][1], end))
            else:
                merged_gaps.append((start, end))

        # Adjust for duty cycle (rescale gaps to ensure 75% observing time)
        total_gap_time = sum(end - start for start, end in merged_gaps)
        required_gap_time = (1 - duty_cycle) * T_obs

        if total_gap_time > required_gap_time:
            excess_time = total_gap_time - required_gap_time
            while excess_time > 0 and merged_gaps:
                start, end = merged_gaps.pop()
                excess_time -= end - start
        
        merged_gaps = np.asarray(merged_gaps)
        #return start and end of data
        for i in range(0, len(merged_gaps.T[0])-1):
            data_start = merged_gaps.T[1][i]
            data_end = merged_gaps.T[0][i+1]    
            if data_end - data_start > 0.90 * scheduled_period:          
                data_seg.append((data_start, data_start +  scheduled_period))

    else:
        for start, end in scheduled_gaps:
            data_start = end
            data_end = end + scheduled_period 
            data_seg.append((data_start, data_end))
            
    
    data_seg = np.asarray(data_seg)
  
    return data_seg.T[0], data_seg.T[1]
