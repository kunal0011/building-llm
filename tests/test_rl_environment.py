"""
Comprehensive Test Suite for Post-Training Reinforcement Learning (RL / GRPO) Suite:
1. CodeExecutionSandbox (Subprocess isolation, test pass/fail rewards)
2. SymbolicMathVerifier (SymPy algebraic equivalence & step derivation)
3. ProcessRewardAnnotator (Step-level CoT decomposition & error attribution)
4. GRPORewardEngine (Multi-objective rewards & group advantage normalization)
"""

import math
from pipelines.rl.execution_sandbox import CodeExecutionSandbox
from pipelines.rl.symbolic_math_verifier import SymbolicMathVerifier
from pipelines.rl.process_reward_annotator import ProcessRewardAnnotator
from pipelines.rl.grpo_reward_engine import GRPORewardEngine


def test_code_execution_sandbox():
    print("[+] Testing CodeExecutionSandbox...")
    sandbox = CodeExecutionSandbox(default_timeout_sec=5.0)

    # 1. Valid Solution that passes tests
    solution_code = """
def add_numbers(a: int, b: int) -> int:
    return a + b
"""
    test_code = """
assert add_numbers(2, 3) == 5
assert add_numbers(-1, 1) == 0
"""
    res = sandbox.execute_python_code(solution_code, test_code)
    assert res["passed"] is True
    assert res["reward"] == 1.0
    assert res["exit_code"] == 0

    # 2. Flawed Solution that fails test assertion
    buggy_code = """
def add_numbers(a: int, b: int) -> int:
    return a - b
"""
    res_bug = sandbox.execute_python_code(buggy_code, test_code)
    assert res_bug["passed"] is False
    assert res_bug["reward"] == 0.0
    assert res_bug["exit_code"] != 0

    # 3. Code with infinite loop / timeout
    hanging_code = "while True: pass"
    res_timeout = sandbox.execute_python_code(hanging_code, timeout_sec=0.5)
    assert res_timeout["timeout"] is True
    assert res_timeout["reward"] == 0.0

    print("  -> CodeExecutionSandbox verified successfully!")


def test_symbolic_math_verifier():
    print("[+] Testing SymbolicMathVerifier...")
    verifier = SymbolicMathVerifier()

    # 1. Algebraic Expansion Equivalence: (x + 1)^2 == x^2 + 2x + 1
    expr_a = "(x + 1)**2"
    expr_b = "x**2 + 2*x + 1"
    is_eq, details = verifier.verify_equivalence(expr_a, expr_b)
    assert is_eq is True
    assert details["reward"] == 1.0

    # 2. LaTeX Fraction & Delimiters Equivalence
    latex_pred = r"\frac{4}{8}"
    latex_gold = r"\frac{1}{2}"
    is_eq, details = verifier.verify_equivalence(latex_pred, latex_gold)
    assert is_eq is True
    assert details["reward"] == 1.0

    # 3. Divergent Equations
    is_eq_div, details_div = verifier.verify_equivalence("x + 5", "x + 7")
    assert is_eq_div is False
    assert details_div["reward"] == 0.0

    # 4. Step Sequence Verification
    # Step 0: 2*(x + 3)
    # Step 1: 2*x + 6 (Valid)
    # Step 2: 2*x + 7 (Error!)
    steps = ["2*(x + 3)", "2*x + 6", "2*x + 7"]
    all_valid, first_err, step_evals = verifier.verify_step_sequence(steps)
    assert all_valid is False
    assert first_err == 2 # Step 2 is the flawed transition
    assert step_evals[0]["valid"] is True
    assert step_evals[1]["valid"] is False

    print("  -> SymbolicMathVerifier verified successfully!")


def test_process_reward_annotator():
    print("[+] Testing ProcessRewardAnnotator...")
    annotator = ProcessRewardAnnotator()

    cot_text = """
<step>Let the unknown number be x.</step>
<step>We set up the equation: 2*x + 5 = 15.</step>
<step>Subtract 5 from both sides: 2*x = 10.</step>
<step>Divide by 2: x = 5.</step>
"""
    steps = annotator.segment_reasoning_steps(cot_text)
    assert len(steps) == 4
    assert steps[0] == "Let the unknown number be x."
    assert steps[-1] == "Divide by 2: x = 5."

    # Test Step-level annotation with pinpointed error at step 2
    annotated = annotator.annotate_trajectory(cot_text, first_error_index=2)
    assert annotated[0]["process_reward"] == 1.0
    assert annotated[1]["process_reward"] == 1.0
    assert annotated[2]["process_reward"] == -1.0 # First error
    assert annotated[2]["label"] == "first_error"
    assert annotated[3]["process_reward"] == -1.0 # Cascaded error

    print("  -> ProcessRewardAnnotator verified successfully!")


def test_grpo_reward_engine():
    print("[+] Testing GRPORewardEngine...")
    engine = GRPORewardEngine(weight_correctness=1.0, weight_format=0.20, weight_efficiency=0.10)

    # Simulate group of G=4 candidate rollouts for a coding prompt
    rollouts = [
        # Rollout 0: Correct code, valid thought tags, concise
        {
            "output_text": "<thought>\nStep by step logic.\n</thought>\ndef solve(): return 42",
            "token_count": 50,
            "verifiable_correctness": 1.0
        },
        # Rollout 1: Correct code, missing thought closing tag
        {
            "output_text": "<thought>\nStep by step logic.\ndef solve(): return 42",
            "token_count": 50,
            "verifiable_correctness": 1.0
        },
        # Rollout 2: Flawed code (fails unit tests), valid thought tags
        {
            "output_text": "<thought>\nFlawed deduction.\n</thought>\ndef solve(): return -1",
            "token_count": 50,
            "verifiable_correctness": 0.0
        },
        # Rollout 3: Flawed code, extremely long repetitive loop (exceeds max tokens)
        {
            "output_text": "Random output without thought tags",
            "token_count": 4000,
            "verifiable_correctness": 0.0
        }
    ]

    evaluated_rollouts = []
    for r in rollouts:
        reward_dict = engine.compute_composite_reward(
            verifiable_correctness=r["verifiable_correctness"],
            output_text=r["output_text"],
            token_count=r["token_count"]
        )
        combined = {**r, **reward_dict}
        evaluated_rollouts.append(combined)

    # Compute group-normalized advantages
    enriched_group = engine.compute_group_advantages(evaluated_rollouts)
    assert len(enriched_group) == 4

    # The best rollout (Rollout 0) must have the highest positive advantage
    adv_0 = enriched_group[0]["advantage"]
    adv_3 = enriched_group[3]["advantage"]
    assert adv_0 > 0.5, f"Expected positive advantage for top rollout: {adv_0}"
    assert adv_3 < -0.5, f"Expected negative advantage for worst rollout: {adv_3}"

    # Group advantage sum should approximately be zero (normalized)
    adv_sum = sum(item["advantage"] for item in enriched_group)
    assert abs(adv_sum) < 0.01, f"Expected normalized advantages to sum to ~0, got {adv_sum}"

    print(f"  -> GRPO Advantages computed: {[r['advantage'] for r in enriched_group]}")
    print("  -> GRPORewardEngine verified successfully!")


if __name__ == "__main__":
    test_code_execution_sandbox()
    test_symbolic_math_verifier()
    test_process_reward_annotator()
    test_grpo_reward_engine()
    print("\n[SUCCESS] All Post-Training RL / GRPO modules verified successfully!")
