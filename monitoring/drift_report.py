import pandas as pd
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset


reference = pd.read_csv("monitoring/reference.csv")
current = pd.read_csv("monitoring/current.csv")

report = Report(metrics=[
    DataDriftPreset()
])

report.run(reference_data=reference, current_data=current)

report.save_html("monitoring/drift_report.html")

print("✅ Drift report saved at monitoring/drift_report.html")
