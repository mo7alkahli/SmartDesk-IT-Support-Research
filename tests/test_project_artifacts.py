from pathlib import Path
import pandas as pd
R=Path(__file__).resolve().parents[1]
d=pd.read_csv(R/"data/synthetic_ticket_benchmark.csv")
assert len(d)==1000 and d.category.nunique()==5
assert all(d.category.value_counts()==200)
m=pd.read_csv(R/"results/classification_metrics.csv")
assert len(m)==3 and {"accuracy","macro_f1"}.issubset(m.columns)
print("PASS: dataset balance and metrics checked")
