# Corrected matched batch summary

Exploratory 120 s S006–S010/R12 comparison; the three controls use 24/24/34 s. 
Command annotations and physical telemetry metrics are kept in separate columns and per-case JSON reports.

| case | physical_lift_pass | first_sampled_physical_success_time_s | skill_complete | trajectory_complete_time_s | request_active_duration_s | request_switch_count | request_switches_per_minute | rest_command_activation_fraction | rest_moving_fraction | imagery_bouts_missed | mean_imagery_detection_delay_s | mean_target_dispatch_delay_s | mean_physical_action_latency_s | physical_action_latency_unavailable_after_completion | physical_action_latency_not_responding |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| always-hold | False | NA | False | NA | 0 | 0 | 0.000 | NA | NA | NA | NA | NA | NA | 0 | 0 |
| scripted-positive | True | 19.425 | True | 24.000 | 24.000 | 1 | 2.500 | NA | NA | NA | NA | NA | 1.025 | 0 | 0 |
| scripted-pause | True | 22.425 | True | 27.000 | 25.000 | 4 | 7.059 | NA | NA | NA | NA | NA | 0.529 | 0 | 0 |
| S006R12-raw | True | 52.633 | True | 63.000 | 29.700 | 50 | 25.000 | 0.180 | 0.118 | 4 | 1.055 | 0.164 | 0.191 | 8 | 6 |
| S006R12-confidence | False | NA | False | NA | 1.500 | 4 | 2.000 | 0.000 | 0.000 | 13 | 1.400 | 0.500 | 0.525 | 0 | 0 |
| S006R12-temporal | False | NA | False | NA | 0.692 | 2 | 1.000 | 0.000 | 0.000 | 14 | 1.000 | NA | NA | 0 | 1 |
| S007R12-raw | True | 42.358 | True | 73.300 | 43.900 | 65 | 32.500 | 0.307 | 0.129 | 1 | 0.731 | 0.193 | 0.221 | 13 | 6 |
| S007R12-confidence | True | 105.400 | False | NA | 23.908 | 48 | 24.000 | 0.176 | 0.194 | 4 | 1.700 | 0.119 | 0.146 | 0 | 3 |
| S007R12-temporal | False | NA | False | NA | 13.508 | 24 | 12.000 | 0.085 | 0.109 | 7 | 1.957 | 0.083 | 0.110 | 0 | 0 |
| S008R12-raw | True | 23.133 | True | 29.200 | 79.200 | 41 | 20.500 | 0.634 | 0.103 | 0 | 0.787 | 0.250 | 0.275 | 17 | 0 |
| S008R12-confidence | True | 25.417 | True | 31.992 | 67.608 | 48 | 24.000 | 0.545 | 0.131 | 1 | 0.893 | 0.249 | 0.274 | 15 | 3 |
| S008R12-temporal | True | 31.717 | True | 36.292 | 56.808 | 38 | 19.000 | 0.475 | 0.116 | 2 | 0.831 | 0.125 | 0.154 | 11 | 0 |
| S009R12-raw | True | 28.042 | True | 39.700 | 72.900 | 46 | 23.000 | 0.605 | 0.143 | 2 | 0.654 | 0.825 | 0.687 | 15 | 3 |
| S009R12-confidence | True | 57.025 | True | 61.608 | 53.992 | 44 | 22.000 | 0.446 | 0.175 | 3 | 0.683 | 0.092 | 0.116 | 9 | 1 |
| S009R12-temporal | True | 62.633 | True | 69.908 | 43.892 | 28 | 14.000 | 0.363 | 0.152 | 4 | 1.045 | 0.344 | 0.375 | 5 | 2 |
| S010R12-raw | True | 68.933 | True | 92.400 | 29.400 | 41 | 20.500 | 0.210 | 0.181 | 5 | 1.250 | 0.223 | 0.251 | 5 | 3 |
| S010R12-confidence | False | NA | False | NA | 12.300 | 32 | 16.000 | 0.120 | 0.157 | 9 | 1.733 | 0.047 | 0.072 | 0 | 1 |
| S010R12-temporal | False | NA | False | NA | 5.000 | 22 | 11.000 | 0.057 | 0.076 | 11 | 1.875 | 0.091 | 0.116 | 0 | 0 |
