import csv
import json
from collections import defaultdict


nested_data={}

with open("res/input.csv", newline="", encoding="utf-8") as csvfile:
    reader = csv.DictReader(csvfile)
# 	Most Probable Cause	Quick Initial Checks	Troubleshoot Procedure	Tools Required

    for row in reader:
        id = row["id"]

        if id not in nested_data:
            nested_data[id] = {
                "id": id,
                "equipment":{
                    "problem": row["Problem"],
                    "trouble_code": row["Trouble code"],
                    "error": row["Error Message"],
                    "symptom_short": row["Symptom Short"],
                    "sympton": row["Symptom Detailed"],
                    "signs": row["Observable Signs"],
                    "condition": row["Operating Conditions"],
                    "root_cause": row["Possible Root Causes"],
                    "Most_Probable_Cause": row["Most Probable Cause"],
                    "Troubleshoot_Procedure": row["Troubleshoot Procedure"]
                }
            }

    result = list(nested_data.values())

    with open("output.json", "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    print("saved")
