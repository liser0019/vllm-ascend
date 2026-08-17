import ast
import os
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SETUP = ROOT / "setup.py"


def _setup_tree() -> ast.Module:
    return ast.parse(SETUP.read_text(encoding="utf-8"), filename=str(SETUP))


def _root_assignment() -> ast.Assign:
    assignments = [
        node
        for node in _setup_tree().body
        if isinstance(node, ast.Assign)
        and any(isinstance(target, ast.Name) and target.id == "ROOT_DIR" for target in node.targets)
    ]
    assert len(assignments) == 1
    return assignments[0]


def _evaluate_root(setup_file: str, cwd: Path) -> Path:
    expression = ast.Expression(body=_root_assignment().value)
    ast.fix_missing_locations(expression)
    previous_cwd = Path.cwd()
    try:
        os.chdir(cwd)
        result = eval(
            compile(expression, str(SETUP), "eval"),
            {"Path": Path, "__file__": setup_file},
        )
    finally:
        os.chdir(previous_cwd)
    return Path(result)


def test_root_is_absolute_when_setup_file_is_relative():
    root = _evaluate_root("setup.py", ROOT)
    assert root.is_absolute()
    assert root == ROOT


def test_root_is_independent_of_caller_working_directory(tmp_path):
    root = _evaluate_root(str(SETUP), tmp_path)
    assert root.is_absolute()
    assert root == ROOT


def test_aclnn_build_uses_absolute_script_and_repository_cwd():
    tree = _setup_tree()
    build_class = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "build_and_install_aclnn"
    )
    run_method = next(
        node
        for node in build_class.body
        if isinstance(node, ast.FunctionDef) and node.name == "run"
    )
    script_assignment = next(
        node
        for node in ast.walk(run_method)
        if isinstance(node, ast.Assign)
        and any(isinstance(target, ast.Name) and target.id == "build_script" for target in node.targets)
    )
    assert ast.unparse(script_assignment.value) == (
        "os.path.join(ROOT_DIR, 'csrc', 'build_aclnn.sh')"
    )

    check_call = next(
        node
        for node in ast.walk(run_method)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "check_call"
    )
    assert ast.unparse(check_call.args[0]) == (
        "['bash', build_script, ROOT_DIR, envs.SOC_VERSION]"
    )
    assert any(
        keyword.arg == "cwd"
        and isinstance(keyword.value, ast.Name)
        and keyword.value.id == "ROOT_DIR"
        for keyword in check_call.keywords
    )
