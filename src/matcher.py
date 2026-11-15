"""
Resume <-> job-description matcher.
Score = 0.5 * TF-IDF cosine similarity (lemmatized text) + 0.5 * share of the
job's required skills found in the resume. Skills are found with a spaCy PhraseMatcher.
"""
from pathlib import Path
import re
import spacy
from spacy.matcher import PhraseMatcher
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

NLP = spacy.load("en_core_web_sm", disable=["parser", "ner"])
SKILLS = [s.strip() for s in (Path(__file__).parent / "skills.txt").read_text().splitlines() if s.strip()]
MATCHER = PhraseMatcher(NLP.vocab, attr="LOWER")
MATCHER.add("SKILL", [NLP.make_doc(s) for s in SKILLS])


def lemmas(text: str) -> str:
    doc = NLP(re.sub(r"\s+", " ", text))
    return " ".join(t.lemma_.lower() for t in doc if t.is_alpha and not t.is_stop)


def skills(text: str) -> set[str]:
    doc = NLP.make_doc(text)
    return {doc[s:e].text.lower() for _, s, e in MATCHER(doc)}


def match(resume: str, job: str) -> dict:
    vec = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True).fit([lemmas(resume), lemmas(job)])
    a, b = vec.transform([lemmas(resume), lemmas(job)])
    sim = float(cosine_similarity(a, b)[0, 0])
    need, have = skills(job), skills(resume)
    coverage = len(need & have) / len(need) if need else 0.0
    return {"score": round(100 * (0.5 * sim + 0.5 * coverage), 1),
            "text_similarity": round(sim, 3), "skill_coverage": round(coverage, 3),
            "matched_skills": sorted(need & have), "missing_skills": sorted(need - have)}


if __name__ == "__main__":
    import json, sys
    r, j = Path(sys.argv[1]).read_text(), Path(sys.argv[2]).read_text()
    print(json.dumps(match(r, j), indent=1))
