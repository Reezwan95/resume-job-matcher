# Resume ↔ Job Matcher (NLP)

Paste a résumé and a job description to get a match score, the skills you share with the job, and the skills it asks for that your résumé doesn't show. It uses spaCy for lemmatization and skill extraction, TF-IDF for text similarity, and Flask for a small web front end.

The [portfolio site](https://reezwan95.github.io/#matcher) has a browser version: a recruiter can paste a job description and see how it lines up with my skills.

## How the score works

```
score = 50% × TF-IDF cosine similarity (lemmatized, stop words removed, 1–2-grams)
      + 50% × skill coverage (share of the job's skills found in the résumé)
```

Skills come from a curated list of about 100 entries (`src/skills.txt`) matched with a spaCy `PhraseMatcher`, so multi-word skills like "power bi" and "a/b testing" are caught as one skill. Coverage is easy to explain: the tool tells you exactly what is missing.

Example (`samples/`): a data-analyst résumé against a junior data-scientist posting scores **28.3**. It matches Python, SQL, scikit-learn, machine learning and Power BI, and flags PyTorch, deep learning, A/B testing, Tableau, AWS and Docker as gaps.

## Benchmark: résumé category classification, and a leakage trap

As a sanity check on the text pipeline, I classified résumés into 25 job categories with TF-IDF and a linear SVM (`src/evaluate.py`), using the popular Kaggle "UpdatedResumeDataSet" (962 rows).

| Setup | Accuracy | Macro-F1 |
|---|---|---|
| Random 80/20 split (what most tutorials report) | 0.990 | 0.989 |
| **Deduplicated, 3-fold stratified CV** | **0.873** | **0.850** |

The dataset has only **166 unique résumés** among its 962 rows, and 99% of the test résumés in a random split also appear in training. The near-perfect score is memorization. Deduplicating first gives the honest number.

## Run it

```bash
pip install flask spacy scikit-learn pandas
python -m spacy download en_core_web_sm
python app.py                                   # web app at http://127.0.0.1:5000
python src/matcher.py samples/resume.txt samples/job.txt
python src/evaluate.py UpdatedResumeDataSet.csv reports   # benchmark
```

## Limitations

- TF-IDF rewards shared wording, not meaning. Sentence embeddings would handle synonyms better ("ML" and "machine learning").
- The skill list is curated by hand and biased toward data and analytics roles.
- A match score supports a human decision. It shouldn't screen candidates on its own.

---
Reezwan-Us Sami · [Portfolio](https://reezwan95.github.io) · [LinkedIn](https://www.linkedin.com/in/reezwans)
