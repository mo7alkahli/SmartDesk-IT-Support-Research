# SmartDesk: Intelligent IT Support Ticket Classification
Reproducible proof-of-concept comparing keyword rules, TF-IDF + Naive Bayes, and TF-IDF + Logistic Regression for Network, Hardware, Software, Access, and Storage tickets.

**Important:** the 1,000-record dataset is synthetic and template-generated. Scores show behavior on this generated benchmark only and are not evidence of real-world service desk performance.

## Run
Install Python 3.10+; run `pip install -r requirements.txt`, then `python run_classification_experiment.py`. Metrics are in `results/`; charts are in `documentation/figures/`.

Real-world evaluation requires authorized, de-identified tickets and a separate leakage-resistant test set. This is not a production-ready system.
