import ast
import os
from pathlib import Path
from types import MethodType, SimpleNamespace


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE = ROOT / "work/TreeRL/openrlhf/trainer/ppo_utils/experience_maker_reinforce.py"
SOURCE = Path(os.environ.get("VDRA_TREERL_EXPERIENCE_MAKER_SOURCE", DEFAULT_SOURCE))


class _SamplingRouteSelected(Exception):
    def __init__(self, route):
        self.route = route


def _load_mcts_dispatch_method():
    tree = ast.parse(SOURCE.read_text(encoding="utf-8"))
    cls = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "NaiveExperienceMakerReinforce")
    method = next(node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name == "_make_experience")
    method.decorator_list = []
    method.returns = None
    for arg in (*method.args.posonlyargs, *method.args.args, *method.args.kwonlyargs):
        arg.annotation = None
    if method.args.vararg:
        method.args.vararg.annotation = None
    if method.args.kwarg:
        method.args.kwarg.annotation = None

    module = ast.fix_missing_locations(ast.Module(body=[method], type_ignores=[]))
    namespace = {"_treerl_debug_print": lambda *args, **kwargs: None}
    exec(compile(module, str(SOURCE), "exec"), namespace)
    return namespace[method.name]


def _selected_sampling_route(use_mcts):
    def raise_route(route):
        def sample(*args, **kwargs):
            raise _SamplingRouteSelected(route)

        return sample

    class Probe:
        def __init__(self):
            self.actor = SimpleNamespace(eval=lambda: None)
            self.strategy = SimpleNamespace(args=SimpleNamespace(num_trace_per_sample=1))
            self.sample_responses_bymcts = raise_route("mcts")
            self.sample_responses = raise_route("regular")

    probe = Probe()
    probe._make_experience = MethodType(_load_mcts_dispatch_method(), probe)
    try:
        probe._make_experience("prompt", use_mcts=use_mcts, probe_marker="kept")
    except _SamplingRouteSelected as selected:
        return selected.route
    raise AssertionError("experience construction returned without selecting a sampler")


def test_tree_rl_use_mcts_flag_routes_to_tree_search():
    assert _selected_sampling_route(True) == "mcts"


def test_tree_rl_disabled_mcts_flag_keeps_regular_sampler():
    assert _selected_sampling_route(False) == "regular"
