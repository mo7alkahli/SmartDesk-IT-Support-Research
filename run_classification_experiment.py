from pathlib import Path
import time
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import make_pipeline
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report, confusion_matrix
ROOT=Path(__file__).resolve().parent
df=pd.read_csv(ROOT/"data/synthetic_ticket_benchmark.csv")
Xtr,Xte,ytr,yte=train_test_split(df.text,df.category,test_size=.2,random_state=42,stratify=df.category)
kw={"network":["wifi","website","network","vpn","internet","connection"],"hardware":["laptop","keyboard","monitor","battery","printer","power"],"software":["application","program","software","screen","error","crashes","freezes","update"],"access":["password","account","permission","logging","login","multi factor","shared folder"],"storage":["disk","files","save","storage","drive","backup"]}
def rule(t):
 s=t.lower(); z={c:sum(k in s for k in terms) for c,terms in kw.items()}; return max(z,key=lambda c:(z[c],c))
models=[("Keyword Rules",None),("TF-IDF + Naive Bayes",make_pipeline(TfidfVectorizer(ngram_range=(1,2)),MultinomialNB())),("TF-IDF + Logistic Regression",make_pipeline(TfidfVectorizer(ngram_range=(1,2)),LogisticRegression(max_iter=1000,random_state=42)))]
R=ROOT/"results"; F=ROOT/"documentation/figures"; R.mkdir(exist_ok=True); F.mkdir(parents=True,exist_ok=True); rows=[]; preds={}
for name,model in models:
 t=time.perf_counter()
 if model is None: pred=[rule(x) for x in Xte]
 else: model.fit(Xtr,ytr); pred=model.predict(Xte)
 sec=time.perf_counter()-t; p,r,f,_=precision_recall_fscore_support(yte,pred,average="macro",zero_division=0)
 rows.append({"model":name,"accuracy":accuracy_score(yte,pred),"macro_precision":p,"macro_recall":r,"macro_f1":f,"fit_predict_seconds":sec,"test_samples":len(yte)}); preds[name]=pred
 pd.DataFrame(classification_report(yte,pred,output_dict=True,zero_division=0)).transpose().to_csv(R/(name.lower().replace(" ","_").replace("+","plus")+"_report.csv"))
m=pd.DataFrame(rows); m.to_csv(R/"classification_metrics.csv",index=False); (R/"classification_metrics.json").write_text(m.to_json(orient="records",indent=2))
labels=sorted(df.category.unique()); cm=confusion_matrix(yte,preds["TF-IDF + Logistic Regression"],labels=labels); pd.DataFrame(cm,index=labels,columns=labels).to_csv(R/"confusion_matrix.csv")
plt.figure(figsize=(8,4.5))
for c in ["accuracy","macro_precision","macro_recall","macro_f1"]: plt.plot(m.model,m[c],marker="o",label=c.replace("_"," ").title())
plt.ylim(0,1.05); plt.ylabel("Score"); plt.title("Synthetic Benchmark Classification Metrics"); plt.xticks(rotation=15,ha="right"); plt.legend(); plt.tight_layout(); plt.savefig(F/"classification_metrics.png",dpi=170); plt.close()
plt.figure(figsize=(6,5)); plt.imshow(cm); plt.title("Confusion Matrix: TF-IDF + Logistic Regression"); plt.colorbar(); plt.xticks(range(5),labels,rotation=30,ha="right"); plt.yticks(range(5),labels)
for i in range(5):
 for j in range(5): plt.text(j,i,str(cm[i,j]),ha="center",va="center")
plt.xlabel("Predicted"); plt.ylabel("Actual"); plt.tight_layout(); plt.savefig(F/"confusion_matrix.png",dpi=170); plt.close()
print(m.to_string(index=False))
