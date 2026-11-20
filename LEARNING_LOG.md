# Learning log

Short notes, written in my own words, after each work session. The commit history of this file shows what I actually did and when.

**Rule for myself:** only write what I really read, ran or understood. If I do not understand something yet, I write that down as a question.

---

## Entry template (copy this for each session)

### Date: `YYYY-MM-DD`

**What I did today** (1-3 lines)
-

**What I ran, and what happened** (command + one-line result)
-

**What I understood** (explain it as if to a friend)
-

**What I did not understand yet / questions for my supervisor**
-

**Next step**
-

---

## Suggested first entries (to guide the work)

1. **Run the demo.** `python src/demo.py "ID card replacement er fee koto?"` with and without `--poison all`. Describe in my own words what changed in the retrieved chunks and in the answer.
2. **Read the code in `src/rag.py` top to bottom.** Which function does retrieval? Which function builds the prompt for the LLM?
3. **Run the experiment** and explain table 2 (attack success by language) in 3-4 sentences.
4. **Try an LLM** (`--llm ollama` or `--llm hf`) and compare with the no-LLM numbers. Did the LLM repeat the poisoned number?
5. **Break the defense.** Run with the `spoofed` scenario and explain why it fails.
6. **Add one new fact** to `FACTS` in `src/build_dataset.py` and rerun everything.

## Glossary I should be able to explain

- RAG, retriever, knowledge base, chunk, top-k
- Poisoned document, attack success rate, false positive
- Banglish, script, cross-script retrieval
- Authority label, provenance
