#!/usr/bin/env python3
"""
Syllogistic Reasoning Experiment

Tests whether hallucination probes can detect incorrect logical conclusions.
Uses propositional logic to generate formally valid/invalid inferences,
then translates them to natural language.

Example:
- Formal: ∀x(P(x) → Q(x)) ∧ ∀x(Q(x) → R(x)) ⊢ ∀x(P(x) → R(x))
- Natural: All dogs are mammals. All mammals are animals. Therefore, all dogs are animals.
"""

import json
import random
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Set, Tuple

import modal

sys.path.insert(0, str(Path(__file__).parent))
from two_truths_and_a_lie import get_probe_service


@dataclass
class LogicalForm:
    """Represents a logical inference pattern."""

    premises: List[str]  # List of logical formulas
    conclusion: str  # Logical formula
    valid: bool  # Whether the inference is valid
    pattern_name: str  # Name of the inference pattern


# Valid inference patterns
VALID_PATTERNS = {
    "barbara": {  # ∀x(P→Q) ∧ ∀x(Q→R) ⊢ ∀x(P→R)
        "premises": ["all_P_are_Q", "all_Q_are_R"],
        "conclusion": "all_P_are_R",
        "name": "Barbara (transitive)",
    },
    "celarent": {  # ∀x(P→Q) ∧ ∀x(Q→¬R) ⊢ ∀x(P→¬R)
        "premises": ["all_P_are_Q", "no_Q_are_R"],
        "conclusion": "no_P_are_R",
        "name": "Celarent",
    },
    "darii": {  # ∀x(P→Q) ∧ ∃x(R∧P) ⊢ ∃x(R∧Q)
        "premises": ["all_P_are_Q", "some_R_are_P"],
        "conclusion": "some_R_are_Q",
        "name": "Darii",
    },
    "ferio": {  # ∀x(P→¬Q) ∧ ∃x(R∧P) ⊢ ∃x(R∧¬Q)
        "premises": ["no_P_are_Q", "some_R_are_P"],
        "conclusion": "some_R_are_not_Q",
        "name": "Ferio",
    },
    "chain": {  # Long transitive chain
        "premises": ["all_A_are_B", "all_B_are_C", "all_C_are_D"],
        "conclusion": "all_A_are_D",
        "name": "Extended transitivity",
    },
}

# Invalid inference patterns (common fallacies)
INVALID_PATTERNS = {
    "affirming_consequent": {
        "premises": ["all_P_are_Q", "some_R_are_Q"],
        "conclusion": "some_R_are_P",  # INVALID
        "name": "Affirming the consequent",
    },
    "denying_antecedent": {
        "premises": ["all_P_are_Q", "some_R_are_not_P"],
        "conclusion": "some_R_are_not_Q",  # INVALID
        "name": "Denying the antecedent",
    },
    "illicit_major": {
        "premises": ["all_P_are_Q", "no_R_are_P"],
        "conclusion": "no_R_are_Q",  # INVALID (undistributed middle)
        "name": "Illicit major term",
    },
    "reversed_universal": {
        "premises": ["all_P_are_Q"],
        "conclusion": "all_Q_are_P",  # INVALID (conversion of universal)
        "name": "Invalid conversion",
    },
    "existential_to_universal": {
        "premises": ["all_P_are_Q", "some_Q_are_R"],
        "conclusion": "all_P_are_R",  # INVALID (existential to universal)
        "name": "Existential to universal",
    },
}


# Word banks for mapping to natural language
TERM_CATEGORIES = {
    "animals": [
        "dogs",
        "cats",
        "birds",
        "fish",
        "mammals",
        "reptiles",
        "creatures",
        "beasts",
    ],
    "objects": [
        "chairs",
        "tables",
        "books",
        "boxes",
        "containers",
        "items",
        "things",
        "objects",
    ],
    "people": [
        "teachers",
        "students",
        "doctors",
        "artists",
        "workers",
        "professionals",
        "individuals",
        "persons",
    ],
    "plants": [
        "flowers",
        "trees",
        "roses",
        "oaks",
        "plants",
        "vegetables",
        "herbs",
        "shrubs",
    ],
    "abstract": [
        "ideas",
        "concepts",
        "thoughts",
        "beliefs",
        "notions",
        "theories",
        "principles",
        "values",
    ],
}


def generate_term_mapping(num_terms: int) -> Dict[str, str]:
    """
    Generate random terms for logical variables.
    Returns mapping like {"P": "dogs", "Q": "mammals", "R": "animals"}
    """
    category = random.choice(list(TERM_CATEGORIES.keys()))
    terms = random.sample(
        TERM_CATEGORIES[category], min(num_terms, len(TERM_CATEGORIES[category]))
    )

    # Map to logical variables A, B, C, ... or P, Q, R, ...
    variables = ["P", "Q", "R", "S", "T", "U", "V", "W"]
    return {variables[i]: term for i, term in enumerate(terms)}


def translate_to_natural_language(formula: str, term_mapping: Dict[str, str]) -> str:
    """
    Translate a logical formula to natural language.

    Examples:
    - "all_P_are_Q" -> "All dogs are mammals"
    - "no_P_are_Q" -> "No dogs are mammals"
    - "some_P_are_Q" -> "Some dogs are mammals"
    """
    parts = formula.split("_")

    if parts[0] == "all":
        # all_P_are_Q
        term1 = term_mapping.get(parts[1], parts[1])
        term2 = term_mapping.get(parts[3], parts[3])
        return f"All {term1} are {term2}"

    elif parts[0] == "no":
        # no_P_are_Q
        term1 = term_mapping.get(parts[1], parts[1])
        term2 = term_mapping.get(parts[3], parts[3])
        return f"No {term1} are {term2}"

    elif parts[0] == "some" and parts[3] == "not":
        # some_P_are_not_Q
        term1 = term_mapping.get(parts[1], parts[1])
        term2 = term_mapping.get(parts[4], parts[4])
        return f"Some {term1} are not {term2}"

    elif parts[0] == "some":
        # some_P_are_Q
        term1 = term_mapping.get(parts[1], parts[1])
        term2 = term_mapping.get(parts[3], parts[3])
        return f"Some {term1} are {term2}"

    return formula  # Fallback


def extract_terms_from_pattern(pattern: Dict[str, Any]) -> Set[str]:
    """Extract all unique term variables from a pattern."""
    terms = set()
    for premise in pattern["premises"]:
        for part in premise.split("_"):
            if part.isupper() and len(part) == 1:
                terms.add(part)
    for part in pattern["conclusion"].split("_"):
        if part.isupper() and len(part) == 1:
            terms.add(part)
    return terms


def generate_syllogism(valid: bool = True) -> Dict[str, Any]:
    """
    Generate a syllogism by selecting a pattern and mapping terms.

    Args:
        valid: If True, use valid pattern. If False, use invalid pattern.

    Returns:
        Dict with premises, conclusion, formal structure, and validity
    """
    # Select pattern
    if valid:
        pattern_name = random.choice(list(VALID_PATTERNS.keys()))
        pattern = VALID_PATTERNS[pattern_name]
    else:
        pattern_name = random.choice(list(INVALID_PATTERNS.keys()))
        pattern = INVALID_PATTERNS[pattern_name]

    # Count unique terms needed
    terms_needed = len(extract_terms_from_pattern(pattern))

    # Generate term mapping
    term_mapping = generate_term_mapping(terms_needed)

    # Translate to natural language
    premises_nl = [
        translate_to_natural_language(p, term_mapping) for p in pattern["premises"]
    ]
    conclusion_nl = translate_to_natural_language(pattern["conclusion"], term_mapping)

    return {
        "premises": premises_nl,
        "conclusion": conclusion_nl,
        "formal_premises": pattern["premises"],
        "formal_conclusion": pattern["conclusion"],
        "term_mapping": term_mapping,
        "pattern_name": pattern["name"],
        "pattern_key": pattern_name,
        "valid": valid,
    }


def format_syllogism_prompt(syllogism: Dict[str, Any]) -> str:
    """Format syllogism as a completion prompt."""
    premises_text = "\n".join(
        [f"{i + 1}. {p}" for i, p in enumerate(syllogism["premises"])]
    )

    # Extract the last word(s) for completion
    conclusion_words = syllogism["conclusion"].split()
    prompt_part = " ".join(conclusion_words[:-1])
    # Expected completion is the last word

    prompt = f"""Given the following premises:

{premises_text}

Complete this conclusion with ONLY the missing word(s). Do not explain or add extra text.

Therefore, {prompt_part} _____"""

    return prompt


def run_syllogism_test(
    service,
    probe_id: str = "llama3_1_8b_lora_lambda_kl=0.5",
    temperature: float = 0.0,
    num_valid: int = 3,
    num_invalid: int = 3,
) -> Dict[str, Any]:
    """
    Test the model on both valid and invalid syllogisms.

    Returns probe scores for each type.
    """
    results = {
        "valid_syllogisms": [],
        "invalid_syllogisms": [],
    }

    # Test valid syllogisms
    for _ in range(num_valid):
        syllogism = generate_syllogism(valid=True)
        prompt = format_syllogism_prompt(syllogism)

        messages = [{"role": "user", "content": prompt}]

        result = service.generate_with_probe.remote(
            messages=messages,
            probe_id=probe_id,
            repo_id=None,
            threshold=0.3,
            max_tokens=10,
            temperature=temperature,
        )

        if "error" in result:
            return {"error": result["error"]}

        # Calculate average probe score
        if result["probe_probs"]:
            avg_score = sum(result["probe_probs"]) / len(result["probe_probs"])
            results["valid_syllogisms"].append(
                {
                    "syllogism": syllogism,
                    "prompt": prompt,
                    "generated": result["generated_text"],
                    "avg_probe_score": avg_score,
                    "probe_probs": result["probe_probs"],
                }
            )

    # Test invalid syllogisms
    for _ in range(num_invalid):
        syllogism = generate_syllogism(valid=False)
        prompt = format_syllogism_prompt(syllogism)

        messages = [{"role": "user", "content": prompt}]

        result = service.generate_with_probe.remote(
            messages=messages,
            probe_id=probe_id,
            repo_id=None,
            threshold=0.3,
            max_tokens=10,
            temperature=temperature,
        )

        if "error" in result:
            continue  # Skip this one

        if result["probe_probs"]:
            avg_score = sum(result["probe_probs"]) / len(result["probe_probs"])
            results["invalid_syllogisms"].append(
                {
                    "syllogism": syllogism,
                    "prompt": prompt,
                    "generated": result["generated_text"],
                    "avg_probe_score": avg_score,
                    "probe_probs": result["probe_probs"],
                }
            )

    return results


def run_experiment(
    num_trials: int = 10,
    probe_id: str = "llama3_1_8b_lora_lambda_kl=0.5",
    temperature: float = 0.0,
    model_name: str = "meta-llama/Meta-Llama-3.1-8B-Instruct",
) -> List[Dict[str, Any]]:
    """Run syllogistic reasoning experiments.

    Args:
        num_trials: Number of syllogism pairs to test (1 valid + 1 invalid each)
    """

    service = get_probe_service()
    # print(f"🔄 Switching to model: {model_name}...")
    # service.switch_model.remote(model_name)

    results = []
    failed_trials = []

    for i in range(num_trials):
        print(f"\n{'=' * 80}")
        print(f"TRIAL {i + 1}/{num_trials}")
        print(f"{'=' * 80}")

        try:
            result = run_syllogism_test(
                service, probe_id, temperature, num_valid=1, num_invalid=1
            )

            if "error" in result:
                print(f"❌ Error: {result['error']}")
                failed_trials.append({"trial": i + 1, "error": result["error"]})
            else:
                result["trial"] = i + 1
                result["timestamp"] = datetime.now().isoformat()
                results.append(result)

                # Print sample and scores
                print(f"\nValid syllogisms tested: {len(result['valid_syllogisms'])}")
                print(f"Invalid syllogisms tested: {len(result['invalid_syllogisms'])}")

                if result["valid_syllogisms"]:
                    valid_avg = sum(
                        s["avg_probe_score"] for s in result["valid_syllogisms"]
                    ) / len(result["valid_syllogisms"])
                    print(f"✓ Valid syllogisms avg score: {valid_avg:.4f}")

                if result["invalid_syllogisms"]:
                    invalid_avg = sum(
                        s["avg_probe_score"] for s in result["invalid_syllogisms"]
                    ) / len(result["invalid_syllogisms"])
                    print(f"✗ Invalid syllogisms avg score: {invalid_avg:.4f}")

                    if result["valid_syllogisms"]:
                        diff = invalid_avg - valid_avg
                        print(
                            f"Δ Difference: {diff:+.4f} (higher = probe detects invalid logic better)"
                        )

        except Exception as e:
            error_msg = str(e)
            print(f"❌ Trial {i + 1} failed with exception: {error_msg}")
            failed_trials.append({"trial": i + 1, "error": error_msg})

    # Summary
    print(f"\n\n{'=' * 80}")
    print("SUMMARY")
    print(f"{'=' * 80}")
    print(f"Total trials attempted: {num_trials}")
    print(f"Successful trials: {len(results)}")
    print(f"Failed trials: {len(failed_trials)}")

    if results:
        all_valid_scores = []
        all_invalid_scores = []

        for result in results:
            all_valid_scores.extend(
                [s["avg_probe_score"] for s in result.get("valid_syllogisms", [])]
            )
            all_invalid_scores.extend(
                [s["avg_probe_score"] for s in result.get("invalid_syllogisms", [])]
            )

        if all_valid_scores and all_invalid_scores:
            avg_valid = sum(all_valid_scores) / len(all_valid_scores)
            avg_invalid = sum(all_invalid_scores) / len(all_invalid_scores)

            print(f"\nAverage probe scores:")
            print(f"  Valid inferences: {avg_valid:.4f}")
            print(f"  Invalid inferences: {avg_invalid:.4f}")
            print(f"  Difference: {avg_invalid - avg_valid:+.4f}")

            # Check if probe reliably scores invalid higher
            probe_detects = sum(
                1
                for r in results
                if r.get("valid_syllogisms")
                and r.get("invalid_syllogisms")
                and sum(s["avg_probe_score"] for s in r["invalid_syllogisms"])
                / len(r["invalid_syllogisms"])
                > sum(s["avg_probe_score"] for s in r["valid_syllogisms"])
                / len(r["valid_syllogisms"])
            )

            total_comparable = sum(
                1
                for r in results
                if r.get("valid_syllogisms") and r.get("invalid_syllogisms")
            )
            if total_comparable > 0:
                print(
                    f"\nProbe scored invalid > valid in: {probe_detects}/{total_comparable} trials ({probe_detects / total_comparable * 100:.1f}%)"
                )

    if failed_trials:
        print(f"\nFailed trial numbers: {[f['trial'] for f in failed_trials]}")

    print(f"{'=' * 80}\n")

    return results


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Syllogistic reasoning experiment with hallucination probes"
    )
    parser.add_argument(
        "--num-trials",
        type=int,
        default=10,
        help="Number of trials to run (default: 10)",
    )
    parser.add_argument(
        "--num-valid",
        type=int,
        default=3,
        help="Number of valid syllogisms per trial (default: 3)",
    )
    parser.add_argument(
        "--num-invalid",
        type=int,
        default=3,
        help="Number of invalid syllogisms per trial (default: 3)",
    )
    parser.add_argument(
        "--probe-id",
        default="llama3_1_8b_lora_lambda_kl=0.5",
        help="Probe ID to use (default: llama3_1_8b_lora_lambda_kl=0.5)",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Temperature for generation (default: 0.0 for deterministic)",
    )
    parser.add_argument(
        "--log-file",
        type=str,
        default=None,
        help="Path to log file. If not provided, generates default name with timestamp.",
    )

    args = parser.parse_args()

    # Setup logging
    if args.log_file is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_dir = Path(__file__).parent.parent / "logs"
        log_dir.mkdir(exist_ok=True)
        args.log_file = str(log_dir / f"syllogistic_reasoning_{timestamp}.jsonl")

    log_path = Path(args.log_file)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"📝 Logging to: {args.log_file}\n")

    # Run experiment
    results = run_experiment(
        num_trials=args.num_trials,
        probe_id=args.probe_id,
        temperature=args.temperature,
    )

    # Write to log file
    with open(log_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n✅ Results saved to: {args.log_file}")


if __name__ == "__main__":
    main()
