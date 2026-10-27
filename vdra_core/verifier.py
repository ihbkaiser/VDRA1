"""One math verifier shared by every framework.

Priority:
  1. treetune's grader (the one SPO / VDRA training uses): extract_math_answer
     + eval_math, used when the igpo repo is on PYTHONPATH;
  2. HuggingFace Math-Verify (`pip install math-verify`);
  3. exact match of the last \\boxed{...} after light normalisation.

`score(response, gold, problem="") -> 1.0 / 0.0`.
"""

from __future__ import annotations

import hashlib
import importlib
import importlib.metadata
import inspect
import json
import os
import re
from typing import Optional

_BACKEND = None


def _last_boxed(text: str) -> Optional[str]:
    idx = text.rfind("\\boxed")
    if idx < 0:
        return None
    i = text.find("{", idx)
    if i < 0:
        return None
    depth, j = 0, i
    while j < len(text):
        if text[j] == "{":
            depth += 1
        elif text[j] == "}":
            depth -= 1
            if depth == 0:
                return text[i + 1:j]
        j += 1
    return None


def _norm(s: str) -> str:
    s = s.strip().strip("$").replace(" ", "").replace("\\!", "")
    s = s.replace("\\left", "").replace("\\right", "").replace("dfrac", "frac").replace("tfrac", "frac")
    s = re.sub(r"\\text\{(.*?)\}", r"\1", s)
    return s.rstrip(".")


def _pick_backend():
    global _BACKEND
    if _BACKEND is not None:
        return _BACKEND
    try:
        from treetune.tasks.math_answer_exctraction import extract_math_answer  # type: ignore
        from treetune.tasks.math_grader_minerva import eval_math  # type: ignore

        def _tt(response: str, gold: str, problem: str) -> float:
            pred = extract_math_answer(problem, response)
            item = {"answer": gold, "prediction": pred}
            try:
                return 1.0 if eval_math(item) else 0.0
            except Exception:
                return 0.0

        _BACKEND = ("treetune", _tt)
        return _BACKEND
    except Exception:
        pass
    try:
        from math_verify import parse, verify  # type: ignore

        def _mv(response: str, gold: str, problem: str) -> float:
            try:
                return 1.0 if verify(parse("$" + gold + "$"), parse(response)) else 0.0
            except Exception:
                return 0.0

        _BACKEND = ("math_verify", _mv)
        return _BACKEND
    except Exception:
        pass

    def _exact(response: str, gold: str, problem: str) -> float:
        pred = _last_boxed(response)
        return 1.0 if pred is not None and _norm(pred) == _norm(gold) else 0.0

    _BACKEND = ("boxed_exact", _exact)
    return _BACKEND


def backend_name() -> str:
    return _pick_backend()[0]


def _callable_fingerprint(function) -> dict:
    identity = {
        "module": getattr(function, "__module__", type(function).__module__),
        "qualname": getattr(function, "__qualname__", type(function).__qualname__),
    }
    try:
        source = inspect.getsource(function)
    except (OSError, TypeError):
        code = getattr(function, "__code__", None)
        source = code.co_code.hex() if code is not None else repr(function)
    identity["source_sha256"] = hashlib.sha256(source.encode("utf-8")).hexdigest()

    module_name = identity["module"]
    try:
        module = importlib.import_module(module_name)
    except Exception:
        module = None
    if module is not None:
        module_file = getattr(module, "__file__", None)
        if module_file and os.path.isfile(module_file):
            try:
                with open(module_file, "rb") as source_file:
                    identity["module_sha256"] = hashlib.sha256(source_file.read()).hexdigest()
            except OSError:
                identity["module_sha256"] = "unreadable"
        top_level = module_name.split(".", 1)[0]
        distributions = {
            "treetune": ("treetune",),
            "math_verify": ("math-verify",),
        }.get(top_level, ())
        versions = {}
        for distribution in sorted(distributions):
            try:
                versions[distribution] = importlib.metadata.version(distribution)
            except importlib.metadata.PackageNotFoundError:
                versions[distribution] = None
        if versions:
            identity["distributions"] = versions
    return identity


def backend_fingerprint() -> str:
    """Fingerprint the selected verifier and the scorer implementation it uses."""
    name, scorer = _pick_backend()
    helpers = {}
    try:
        closure = inspect.getclosurevars(scorer)
    except (TypeError, ValueError):
        closure = None
    if closure is not None:
        referenced = {**closure.globals, **closure.nonlocals}
        for helper_name, helper in sorted(referenced.items()):
            if inspect.isfunction(helper) or inspect.isclass(helper) or inspect.isbuiltin(helper):
                helpers[helper_name] = _callable_fingerprint(helper)

    own_file = os.path.abspath(__file__)
    with open(own_file, "rb") as source_file:
        own_sha256 = hashlib.sha256(source_file.read()).hexdigest()
    identity = {
        "backend": name,
        "scorer": _callable_fingerprint(scorer),
        "helpers": helpers,
        "verifier_module_sha256": own_sha256,
    }
    encoded = json.dumps(identity, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def score(response: str, gold: str, problem: str = "") -> float:
    return _pick_backend()[1](response, gold, problem)
