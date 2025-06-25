import os
import json

BASE_PATH = "/home/farzangh/ray_results"
metric_key = "accuracy"  # <-- Change this to your actual metric (e.g., "loss")
maximize = True          # False if you're minimizing a metric like "loss"

best_result = None
best_path = None

for root, dirs, files in os.walk(BASE_PATH):
    if "result.json" in files:
        result_path = os.path.join(root, "result.json")
        try:
            with open(result_path, "r") as f:
                result = json.load(f)
                metric = result.get(metric_key, None)
                if metric is not None:
                    if (best_result is None) or (maximize and metric > best_result) or (not maximize and metric < best_result):
                        best_result = metric
                        best_path = result_path
        except Exception as e:
            print(f"Failed to read {result_path}: {e}")

if best_path:
    print(f"\n✅ Best experiment:")
    print(f"Metric `{metric_key}` = {best_result}")
    print(f"Located at: {best_path}")
else:
    print("\n❌ No valid metric values found. All results are null or missing the key:", metric_key)
