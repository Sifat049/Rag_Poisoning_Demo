"""Ask ONE question and watch what the RAG system does.

Examples (run from the repo root):
  python src/demo.py "ID card replacement er fee koto?"
  python src/demo.py "ID card replacement er fee koto?" --poison all
  python src/demo.py "ID card replacement er fee koto?" --poison all --defense
  python src/demo.py "What is the helpline number?" --poison all --llm ollama

--poison  : none | en | bn | bl | all   (which SYNTHETIC poisoned documents are injected)
--defense : switch on the authority + agreement filter
--llm     : none (no LLM) | hf | ollama
"""
import argparse

from rag import answer, load_jsonl, make_llm, make_retriever


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("question")
    ap.add_argument("--poison", default="none", choices=["none", "en", "bn", "bl", "all"])
    ap.add_argument("--defense", action="store_true")
    ap.add_argument("--llm", default="none", choices=["none", "hf", "ollama"])
    ap.add_argument("--model", default=None, help="model name for hf / ollama")
    ap.add_argument("--retriever", default="tfidf", choices=["tfidf", "e5"])
    ap.add_argument("--k", type=int, default=3)
    args = ap.parse_args()

    docs = load_jsonl("clean_docs.jsonl")
    poison = load_jsonl("poison_docs.jsonl")
    if args.poison != "none":
        docs += [p for p in poison if args.poison == "all" or p["script"] == args.poison]

    retriever = make_retriever(args.retriever, docs)
    llm = make_llm(args.llm, args.model)
    res = answer(args.question, retriever, llm, k=args.k, defense=args.defense)

    print(f"\nQuestion : {args.question}")
    print(f"Setup    : retriever={retriever.name}  llm={llm.name}  poison={args.poison}  defense={args.defense}")
    print("\nRetrieved chunks (after defense):")
    for doc, score in res["retrieved"]:
        tag = "POISON" if doc["is_poison"] else "clean "
        print(f"  [{tag}] score={score:.2f}  authority={doc['authority']:<10}  {doc['text']}")
    if res["dropped"]:
        print("\nDropped by defense:")
        for doc, score in res["dropped"]:
            tag = "POISON" if doc["is_poison"] else "clean "
            print(f"  [{tag}] score={score:.2f}  {doc['text']}")
    print(f"\nAnswer   : {res['answer']}\n")


if __name__ == "__main__":
    main()
