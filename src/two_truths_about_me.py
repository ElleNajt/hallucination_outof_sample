#!/usr/bin/env python3
"""
Two Truths and a Lie About Me - variant where model is given facts about a person.

This script:
1. Provides the model with facts about a person
2. Asks it to generate two truths and a lie based on those facts
3. Uses the probe to sum probabilities over each option
4. Guesses which one is the lie based on highest probe score
5. Asks the model to reveal the lie
6. Compares probe's guess with the model's answer
"""

import json
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Tuple

import modal

# Import shared functions from two_truths_and_a_lie
sys.path.insert(0, str(Path(__file__).parent))
from two_truths_and_a_lie import (
    ask_for_lie,
    calculate_statement_scores,
    get_probe_service,
    parse_statements,
)


def generate_two_truths_and_a_lie_about_person(
    service,
    facts: List[str],
    probe_id: str = "llama3_1_8b_lora_lambda_kl=0.5",
    temperature: float = 0.7,
) -> Dict[str, Any]:
    """Generate two truths and a lie based on provided facts about a person."""
    facts_text = "\n".join([f"- {fact}" for fact in facts])

    prompt = f"""Here are some facts about a person:

{facts_text}

Generate two truths and a lie about this person based on these facts. The two truths should be directly from the facts provided, and the lie should be something plausible but false.

Format your response exactly as:
1. [statement]
2. [statement]
3. [statement]

Make them similar in length and difficulty to distinguish."""

    messages = [{"role": "user", "content": prompt}]

    result = service.generate_with_probe.remote(
        messages=messages,
        probe_id=probe_id,
        repo_id=None,
        threshold=0.3,
        max_tokens=200,
        temperature=temperature,
    )

    return result


def run_experiment(
    facts: List[str],
    probe_id: str = "llama3_1_8b_lora_lambda_kl=0.5",
    temperature: float = 0.7,
    model_name: str = "meta-llama/Meta-Llama-3.1-8B-Instruct",
) -> Dict[str, Any]:
    """Run a single two truths and a lie experiment about a person."""

    print(f"\n{'=' * 80}")
    print(f"TWO TRUTHS AND A LIE ABOUT ME")
    print(f"{'=' * 80}")
    print("\n📋 Facts provided:")
    for fact in facts:
        print(f"  - {fact}")
    print()

    # Get service and switch to the correct model
    service = get_probe_service()
    print(f"🔄 Switching to model: {model_name}...")
    service.switch_model.remote(model_name)

    # Step 1: Generate two truths and a lie
    print("🎲 Generating statements...")
    result = generate_two_truths_and_a_lie_about_person(
        service, facts, probe_id, temperature
    )

    if "error" in result:
        print(f"❌ Error: {result['error']}")
        return {"success": False, "error": result["error"]}

    generated_text = result["generated_text"]
    statements = parse_statements(generated_text)

    if len(statements) != 3:
        print(f"❌ Error: Expected 3 statements, got {len(statements)}")
        print(f"Generated text:\n{generated_text}")
        return {"success": False, "error": "Could not parse 3 statements"}

    print("\n📝 Generated statements:")
    for i, stmt in enumerate(statements, 1):
        print(f"  {i}. {stmt}")

    # Step 2: Calculate probe scores for each statement
    print("\n🔍 Analyzing with probe...")
    statement_scores = calculate_statement_scores(
        result["generated_tokens"], result["probe_probs"], statements, generated_text
    )

    print("\n📊 Probe scores (higher = more likely to be a lie):")
    for i in range(1, 4):
        score = statement_scores.get(i, 0.0)
        print(f"  Statement {i}: {score:.4f}")

    # Step 3: Probe's guess (highest score)
    probe_guess = max(statement_scores.items(), key=lambda x: x[1])[0]
    print(f"\n🤖 Probe guesses the lie is: Statement {probe_guess}")

    # Step 4: Ask model for the answer
    print("\n🤔 Asking model for the answer...")
    model_answer = ask_for_lie(service, statements, probe_id)

    if model_answer == -1:
        print("❌ Error: Could not parse model's answer")
        return {"success": False, "error": "Could not parse model answer"}

    print(f"✅ Model says the lie is: Statement {model_answer}")

    # Step 5: Compare
    correct = probe_guess == model_answer
    print(f"\n{'=' * 80}")
    if correct:
        print("🎉 PROBE WAS CORRECT!")
    else:
        print("❌ PROBE WAS INCORRECT")
    print(f"{'=' * 80}\n")

    return {
        "success": True,
        "facts": facts,
        "statements": statements,
        "statement_scores": statement_scores,
        "probe_guess": probe_guess,
        "model_answer": model_answer,
        "probe_correct": correct,
    }


def generate_random_facts(
    service, probe_id: str = "llama3_1_8b_lora_lambda_kl=0.5"
) -> List[str]:
    """Generate random facts about a fictional person."""
    prompt = """Generate 5-7 random facts about a fictional person. Include things like name, age, occupation, hobbies, location, family, etc.

Format your response as a bulleted list with one fact per line:
- Fact 1
- Fact 2
etc."""

    messages = [{"role": "user", "content": prompt}]

    result = service.generate_with_probe.remote(
        messages=messages,
        probe_id=probe_id,
        repo_id=None,
        threshold=0.3,
        max_tokens=150,
        temperature=1.0,
    )

    if "error" in result:
        raise RuntimeError(f"Fact generation failed: {result['error']}")

    text = result["generated_text"]
    # Parse facts from bulleted list
    facts = []
    for line in text.split("\n"):
        line = line.strip()
        if line.startswith("- ") or line.startswith("* "):
            facts.append(line[2:].strip())

    return facts if facts else ["No facts generated"]


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Two Truths and a Lie About Me - experiment with hallucination probes"
    )
    parser.add_argument(
        "--facts",
        nargs="*",
        default=None,
        help='Facts about the person (e.g., --facts "My name is Bob" "I am 35" "I am a plumber"). If not provided, generates random facts.',
    )
    parser.add_argument(
        "--probe-id",
        default="llama3_1_8b_lora_lambda_kl=0.5",
        help="Probe ID to use (default: llama3_1_8b_lora_lambda_kl=0.5)",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Temperature for generation (default: 0.7)",
    )
    parser.add_argument(
        "--num-trials", type=int, default=1, help="Number of trials to run (default: 1)"
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
        args.log_file = str(log_dir / f"two_truths_about_me_{timestamp}.jsonl")

    log_path = Path(args.log_file)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"📝 Logging to: {args.log_file}\n")

    # Get service once for fact generation if needed
    service = get_probe_service() if args.facts is None else None

    results = []
    failed_trials = []

    for i in range(args.num_trials):
        if args.num_trials > 1:
            print(f"\n\n{'#' * 80}")
            print(f"TRIAL {i + 1}/{args.num_trials}")
            print(f"{'#' * 80}")

        try:
            # Generate random facts if not provided
            if args.facts is None:
                if service is None:
                    service = get_probe_service()
                facts = generate_random_facts(service, args.probe_id)
            else:
                facts = args.facts

            result = run_experiment(
                facts,
                args.probe_id,
                args.temperature,
                model_name="meta-llama/Meta-Llama-3.1-8B-Instruct",
            )

            if result["success"]:
                result["timestamp"] = datetime.now().isoformat()
                result["probe_id"] = args.probe_id
                result["temperature"] = args.temperature
                results.append(result)

                # Write to log file (JSONL format - one JSON per line)
                with open(log_path, "a") as f:
                    json.dump(result, f)
                    f.write("\n")
            else:
                # Log failed trial
                error_msg = result.get("error", "Unknown error")
                print(f"❌ Trial {i + 1} failed: {error_msg}")
                failed_trials.append({"trial": i + 1, "error": error_msg})

                # Write error to log file
                with open(log_path, "a") as f:
                    json.dump(
                        {
                            "success": False,
                            "trial": i + 1,
                            "error": error_msg,
                            "timestamp": datetime.now().isoformat(),
                        },
                        f,
                    )
                    f.write("\n")

        except Exception as e:
            # Catch any unexpected errors
            error_msg = str(e)
            print(f"❌ Trial {i + 1} failed with exception: {error_msg}")
            failed_trials.append({"trial": i + 1, "error": error_msg})

            # Write error to log file
            with open(log_path, "a") as f:
                json.dump(
                    {
                        "success": False,
                        "trial": i + 1,
                        "error": error_msg,
                        "timestamp": datetime.now().isoformat(),
                    },
                    f,
                )
                f.write("\n")

    # Summary
    if args.num_trials > 1:
        print(f"\n\n{'=' * 80}")
        print("SUMMARY")
        print(f"{'=' * 80}")
        print(f"Total trials attempted: {args.num_trials}")
        print(f"Successful trials: {len(results)}")
        print(f"Failed trials: {len(failed_trials)}")

        if results:
            correct = sum(1 for r in results if r["probe_correct"])
            print(
                f"Probe correct: {correct}/{len(results)} ({correct / len(results) * 100:.1f}%)"
            )

        if failed_trials:
            print(f"\nFailed trial numbers: {[f['trial'] for f in failed_trials]}")

        print(f"{'=' * 80}\n")

    print(f"\n✅ Results saved to: {args.log_file}")


if __name__ == "__main__":
    main()
