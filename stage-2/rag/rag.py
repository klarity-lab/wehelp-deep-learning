import argparse
import json
import os
import sys
import urllib.request

from gensim.models.doc2vec import Doc2Vec

RAG_DIR = os.path.dirname(os.path.abspath(__file__))
STAGE_DIR = os.path.dirname(RAG_DIR)
sys.path.insert(0, STAGE_DIR)

sys.path.insert(0, RAG_DIR)
from aliases import expand

OLLAMA_URL = "http://localhost:11434/api/chat"
MODEL = "qwen3.5:4b"
TOP_K = 5

PROMPT = """你是道路交通法規助理。僅根據下列條文回答問題,不要使用你自己對法規的既有知識。
回答時引用條號(例如:根據第53條…)。如果下列條文沒有涵蓋問題,請直接說「提供的條文中沒有相關規定」。

條文:
{chunks}

問題:{question}"""


def tokenize(question):
    import jieba
    import jieba.posseg as pseg
    from tokenizer import keep_token

    jieba.setLogLevel(60)
    jieba.set_dictionary(os.path.join(STAGE_DIR, "data", "dict-zh-tw.txt"))
    tagger = pseg.POSTokenizer(jieba.dt)
    return [w for w, tag in tagger.cut(question) if keep_token(w, tag)]


def retrieve(question):
    chunks = json.load(open(os.path.join(RAG_DIR, "data", "chunks.json"), encoding="utf-8"))
    model = Doc2Vec.load(os.path.join(RAG_DIR, "data", "law-doc2vec.model"))
    vector = model.infer_vector(expand(tokenize(question)))
    top = model.dv.most_similar([vector], topn=TOP_K)
    return [chunks[index] for index, _ in top]


def ask_llm(prompt, model):
    body = json.dumps({
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "stream": False,
    }).encode("utf-8")
    request = urllib.request.Request(OLLAMA_URL, data=body,
                                     headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(request, timeout=300) as response:
        return json.loads(response.read())["message"]["content"]


def print_answer(answer, model):
    print("=" * 60)
    print(f"LLM 回答({model})")
    print("=" * 60)
    print(answer)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("question")
    parser.add_argument("--direct", action="store_true", help="ask the LLM without retrieval")
    parser.add_argument("--model", default=MODEL, help=f"Ollama model tag (default {MODEL})")
    args = parser.parse_args()

    if args.direct:
        print(f"[direct] {args.model}\n")
        answer = ask_llm(f"請回答關於台灣道路交通管理處罰條例的問題:{args.question}", args.model)
        print_answer(answer, args.model)
        return

    retrieved = retrieve(args.question)
    print(f"[rag] {args.model} retrieved: {[c['id'] for c in retrieved]}\n")
    for chunk in retrieved:
        print(f"--- {chunk['id']}")
        print(chunk["text"])
        print()
    prompt = PROMPT.format(
        chunks="\n\n".join(c["text"] for c in retrieved),
        question=args.question,
    )
    print_answer(ask_llm(prompt, args.model), args.model)


if __name__ == "__main__":
    main()
