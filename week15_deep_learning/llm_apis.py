"""Week 15 (g): Using LLM APIs - Anthropic, OpenAI, a local Ollama server, or an open model through
Hugging Face. The first backend that is configured is used; nothing is faked when none is.

Configuration (environment variables):
    ANTHROPIC_API_KEY   -> Anthropic Messages API   (model: ANTHROPIC_MODEL, default claude-haiku-5-5)
    OPENAI_API_KEY      -> OpenAI Chat Completions  (model: OPENAI_MODEL, default gpt-4o-mini)
    Ollama on :11434    -> local open model         (model: OLLAMA_MODEL, default llama3.2)
    LLM_BACKEND=hf      -> small open model via transformers (downloads weights on first use)

Author: Parth Denge | PRN: 240705201018
"""
import json
import os
import sys
import urllib.request

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common import banner  # noqa: E402

ARTICLE = ("Transformers replaced recurrent networks in most NLP systems because self-attention lets every token "
           "look at every other token in parallel, which trains faster on GPUs and captures long-range "
           "dependencies better. Large language models are transformers trained to predict the next token on huge "
           "text corpora; they can then be prompted, fine-tuned, or called through an API to summarise, answer "
           "questions and generate text.")

TASKS = {
    "text generation": "Write a two-sentence encouraging message for a student starting a deep learning course.",
    "summarization": f"Summarise the following in one sentence:\n\n{ARTICLE}",
    "question answering": f"Using only this text, answer: why are transformers faster to train than RNNs?\n\n{ARTICLE}",
}


def _ollama_up(host="http://localhost:11434") -> bool:
    try:
        urllib.request.urlopen(f"{host}/api/tags", timeout=1)
        return True
    except OSError:
        return False


def detect_backend() -> str | None:
    forced = os.environ.get("LLM_BACKEND")
    if forced:
        return forced
    if os.environ.get("ANTHROPIC_API_KEY"):
        return "anthropic"
    if os.environ.get("OPENAI_API_KEY"):
        return "openai"
    if _ollama_up():
        return "ollama"
    return None


def ask_anthropic(prompt: str) -> str:
    import anthropic
    msg = anthropic.Anthropic().messages.create(
        model=os.environ.get("ANTHROPIC_MODEL", "claude-haiku-5-5"), max_tokens=300,
        messages=[{"role": "user", "content": prompt}])
    return msg.content[0].text


def ask_openai(prompt: str) -> str:
    from openai import OpenAI
    r = OpenAI().chat.completions.create(model=os.environ.get("OPENAI_MODEL", "gpt-4o-mini"), max_tokens=300,
                                         messages=[{"role": "user", "content": prompt}])
    return r.choices[0].message.content


def ask_ollama(prompt: str) -> str:
    body = json.dumps({"model": os.environ.get("OLLAMA_MODEL", "llama3.2"), "prompt": prompt,
                       "stream": False}).encode()
    req = urllib.request.Request("http://localhost:11434/api/generate", body, {"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=300) as r:
        return json.load(r)["response"]


_hf_pipe = None


def ask_hf(prompt: str) -> str:
    global _hf_pipe
    if _hf_pipe is None:
        from transformers import pipeline
        _hf_pipe = pipeline("text-generation", model=os.environ.get("HF_MODEL", "Qwen/Qwen2.5-0.5B-Instruct"))
    out = _hf_pipe([{"role": "user", "content": prompt}], max_new_tokens=150, do_sample=False)
    return out[0]["generated_text"][-1]["content"]


BACKENDS = {"anthropic": ask_anthropic, "openai": ask_openai, "ollama": ask_ollama, "hf": ask_hf}


def main():
    banner("Week 15g - LLM APIs")
    backend = detect_backend()
    if backend is None:
        print("No LLM backend configured. Set ANTHROPIC_API_KEY or OPENAI_API_KEY, start Ollama "
              "(`ollama serve`), or run with LLM_BACKEND=hf.")
        return
    print(f"Backend: {backend}\n")
    for task, prompt in TASKS.items():
        print(f"--- {task} ---\n{BACKENDS[backend](prompt).strip()}\n")


if __name__ == "__main__":
    main()
