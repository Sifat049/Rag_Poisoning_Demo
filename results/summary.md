# Results

retriever = `tfidf`, LLM = `none`, top-k = 3, 24 questions (8 facts x 3 scripts). All data is SYNTHETIC.

## 1. Baseline: no attack, no defense

| Query script | Clean accuracy % |
|---|---|
| English | 100.0 |
| Bangla | 100.0 |
| Banglish | 100.0 |

## 2. Attack success rate (ASR), no defense

Rows = script of the user's question. Columns = script of the injected poisoned documents.

| Query script | Poison: English | Poison: Bangla | Poison: Banglish | Poison: all scripts |
|---|---|---|---|---|
| English | 100.0% | 0.0% | 0.0% | 100.0% |
| Bangla | 0.0% | 100.0% | 0.0% | 100.0% |
| Banglish | 0.0% | 0.0% | 87.5% | 100.0% |

## 3. Poison retrieval rate (a poisoned chunk is in the top-k)

| Query script | Poison: English | Poison: Bangla | Poison: Banglish | Poison: all scripts |
|---|---|---|---|---|
| English | 100.0% | 0.0% | 62.5% | 100.0% |
| Bangla | 0.0% | 100.0% | 0.0% | 100.0% |
| Banglish | 62.5% | 0.0% | 100.0% | 100.0% |

## 4. Effect of the simple defense

| Scenario | ASR % (no defense) | Correct % (no defense) | ASR % (with defense) | Correct % (with defense) |
|---|---|---|---|---|
| No attack | 0.0 | 100.0 | 0.0 | 100.0 |
| Poison, all scripts | 100.0 | 0.0 | 0.0 | 100.0 |
| Poison, attacker fakes 'official' | 100.0 | 0.0 | 100.0 | 0.0 |
