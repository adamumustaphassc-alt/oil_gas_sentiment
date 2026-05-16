import os
import pandas as pd

# Define sample oil & gas technical reports
data = {
    "Report_ID": [101, 102, 103, 104, 105],
    "Timestamp": ["2026-05-16 08:00", "2026-05-16 09:15", "2026-05-16 11:30", "2026-05-16 13:45", "2026-05-16 16:20"],
    "Facility": ["Platform Alpha", "Refinery Zone B", "Pipeline Section 4", "Platform Alpha", "Drilling Rig 7"],
    "Technical_Report": [
        "Routine maintenance completed on the main compressor. All parameters are stable and operating within normal limits.",
        "High pressure alarm triggered on separator vessel V-101. Technicians deployed to investigate a potential blockage.",
        "Scheduled drone inspection completed successfully. No anomalies or structural defects detected along the pipeline corridor.",
        "Minor hydraulic fluid weep identified on crane assembly. Containment measures deployed; replacement parts ordered.",
        "Critical failure reported in the primary mud pump during drilling operations. Operations halted temporarily for emergency repairs."
    ]
}

# Create DataFrame
df = pd.DataFrame(data)

# Ensure the data folder exists relative to this script
output_path = os.path.join(os.path.dirname(__file__), "../data/raw_reports.csv")

# Save to CSV
df.to_csv(output_path, index=False)
print(f"Success! Mock dataset created at: {os.path.abspath(output_path)}")