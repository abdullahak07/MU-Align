import json, pandas as pd

with open("results/summary_all_methods.json", "r") as f:
    d = json.load(f)

df = pd.DataFrame(d)
df.to_csv("results/vqa20k_summary16.csv", index=False)
print(df)
