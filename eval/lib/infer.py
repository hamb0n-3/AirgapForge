import time

from mlx_lm import generate, load
from mlx_lm.sample_utils import make_sampler

_CACHE = {}


class Model:
    def __init__(self, repo, adapter_path=None):
        self.repo = repo
        self.adapter_path = adapter_path
        t0 = time.time()
        self.model, self.tokenizer = load(repo, adapter_path=adapter_path)
        self.load_time = time.time() - t0

    def _apply_template(self, messages):
        # suppress reasoning preambles; kwargs vary by model family
        for extra in ({"enable_thinking": False, "reasoning_effort": "low"},
                      {"enable_thinking": False},
                      {"reasoning_effort": "low"},
                      {}):
            try:
                return self.tokenizer.apply_chat_template(
                    messages, add_generation_prompt=True, tokenize=False, **extra)
            except TypeError:
                continue
        return self.tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=False)

    @staticmethod
    def _clean(text):
        # pull the final answer out of gpt-oss harmony output
        marker = "<|channel|>final<|message|>"
        if marker in text:
            text = text.split(marker, 1)[1]
        for stop in ("<|return|>", "<|end|>", "<|channel|>", "<|start|>"):
            text = text.split(stop, 1)[0]
        return text.strip()

    def chat(self, messages, max_tokens=512, temp=0.0, top_p=0.0, top_k=0):
        prompt = self._apply_template(messages)
        sampler = make_sampler(temp=temp, top_p=top_p, top_k=top_k)
        out = generate(self.model, self.tokenizer, prompt,
                       max_tokens=max_tokens, sampler=sampler, verbose=False)
        return self._clean(out)

    def ask(self, user, system=None, **kw):
        msgs = ([{"role": "system", "content": system}] if system else []) + \
               [{"role": "user", "content": user}]
        return self.chat(msgs, **kw)


def is_degenerate(text):
    t = (text or "").strip()
    if len(t) < 5:
        return True
    if t.count("<tool_call>") >= 8 or t.count("</tool_call>") >= 8:
        return True
    toks = t.split()
    if len(toks) > 30 and len(set(toks)) <= 4:
        return True
    if len(t) > 300 and len(set(t)) < 15:
        return True
    return False


def get_model(repo, adapter_path=None):
    key = (repo, adapter_path)
    if key not in _CACHE:
        _CACHE[key] = Model(repo, adapter_path)
    return _CACHE[key]


def free_all():
    _CACHE.clear()
