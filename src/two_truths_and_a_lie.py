#!/usr/bin/env python3
"""
Two Truths and a Lie experiment with hallucination probes.

This script:
1. Asks the model to generate two truths and a lie
2. Uses the probe to sum probabilities over each option
3. Guesses which one is the lie based on highest probe score
4. Asks the model to reveal the lie
5. Compares probe's guess with the model's answer
"""

import json
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Tuple

import modal


def get_probe_service():
    """Get the probe inference service."""
    cls = modal.Cls.from_name("hallucination-probe-backend", "ProbeInferenceService")
    return cls()


def generate_random_topic(
    service, probe_id: str = "llama3_1_8b_lora_lambda_kl=0.5"
) -> str:
    """Generate a random topic for two truths and a lie."""
    prompt = """Generate a random topic for a game of 'two truths and a lie'.
The topic should be specific and interesting (e.g., 'black holes', 'ancient Egypt', 'coffee', 'the human brain').
Respond with ONLY the topic, nothing else."""

    messages = [{"role": "user", "content": prompt}]

    result = service.generate_with_probe.remote(
        messages=messages,
        probe_id=probe_id,
        repo_id=None,
        threshold=0.3,
        max_tokens=20,
        temperature=1.0,
    )

    if "error" in result:
        raise RuntimeError(f"Topic generation failed: {result['error']}")

    return result["generated_text"].strip()


def generate_two_truths_and_a_lie(
    service,
    topic: str,
    probe_id: str = "llama3_1_8b_lora_lambda_kl=0.5",
    temperature: float = 0.7,
) -> Dict[str, Any]:
    """Generate two truths and a lie about a topic."""
    prompt = f"""Generate two truths and a lie about {topic}.

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


def parse_statements(text: str) -> List[str]:
    """Parse numbered statements from text."""
    lines = text.strip().split("\n")
    statements = []

    for line in lines:
        line = line.strip()
        # Match lines like "1. statement" or "1) statement"
        match = re.match(r"^\d+[\.\)]\s*(.+)$", line)
        if match:
            statements.append(match.group(1).strip())

    return statements


def calculate_statement_scores(
    tokens: List[str], probs: List[float]
) -> Tuple[Dict[int, float], Dict[int, float], Dict[int, float]]:
    """
    Calculate average, sum, and max probe probabilities for each statement.

    Returns three dicts mapping statement number (1-indexed) to avg, sum, and max probabilities.
    """
    # Find positions of statement number tokens (1, 2, 3)
    boundaries = []
    for i, token in enumerate(tokens):
        if token.strip() in ["1", "2", "3"]:
            boundaries.append(i)

    # Calculate avg, sum, and max probe scores for each statement
    avg_scores = {}
    sum_scores = {}
    max_scores = {}
    for stmt_num in [1, 2, 3]:
        if stmt_num - 1 < len(boundaries):
            start_idx = boundaries[stmt_num - 1]
            end_idx = (
                boundaries[stmt_num] if stmt_num < len(boundaries) else len(tokens)
            )

            statement_probs = probs[start_idx:end_idx]
            if statement_probs:
                avg_scores[stmt_num] = sum(statement_probs) / len(statement_probs)
                sum_scores[stmt_num] = sum(statement_probs)
                max_scores[stmt_num] = max(statement_probs)
            else:
                avg_scores[stmt_num] = 0.0
                sum_scores[stmt_num] = 0.0
                max_scores[stmt_num] = 0.0
        else:
            avg_scores[stmt_num] = 0.0
            sum_scores[stmt_num] = 0.0
            max_scores[stmt_num] = 0.0

    return avg_scores, sum_scores, max_scores


def ask_for_lie(
    service, statements: List[str], probe_id: str = "llama3_1_8b_lora_lambda_kl=0.5"
) -> int:
    """Ask the model which statement is the lie."""
    statements_text = "\n".join([f"{i + 1}. {s}" for i, s in enumerate(statements)])

    prompt = f"""Here are three statements:

{statements_text}

Which statement is the lie? Respond with ONLY the number (1, 2, or 3)."""

    messages = [{"role": "user", "content": prompt}]

    result = service.generate_with_probe.remote(
        messages=messages,
        probe_id=probe_id,
        repo_id=None,
        threshold=0.3,
        max_tokens=20,
        temperature=0.0,  # Use greedy decoding for answer
    )

    # Extract the number from response
    text = result["generated_text"].strip()
    match = re.search(r"\b([123])\b", text)
    if match:
        return int(match.group(1))

    return -1  # Could not parse


def run_experiment(
    topic: str,
    probe_id: str = "llama3_1_8b_lora_lambda_kl=0.5",
    temperature: float = 0.7,
    model_name: str = "meta-llama/Meta-Llama-3.1-8B-Instruct",
) -> Dict[str, Any]:
    """Run a single two truths and a lie experiment."""

    print(f"\n{'=' * 80}")
    print(f"TWO TRUTHS AND A LIE: {topic}")
    print(f"{'=' * 80}\n")

    # Get service and switch to the correct model
    service = get_probe_service()
    # print(f"🔄 Switching to model: {model_name}...")
    # service.switch_model.remote(model_name)

    # Step 1: Generate two truths and a lie
    print("🎲 Generating statements...")
    result = generate_two_truths_and_a_lie(service, topic, probe_id, temperature)

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
    avg_scores, sum_scores, max_scores = calculate_statement_scores(
        result["generated_tokens"], result["probe_probs"]
    )

    print("\n📊 Probe scores (higher = more likely to be a lie):")
    for i in range(1, 4):
        avg = avg_scores.get(i, 0.0)
        total = sum_scores.get(i, 0.0)
        maximum = max_scores.get(i, 0.0)
        print(f"  Statement {i}: avg={avg:.4f}, sum={total:.4f}, max={maximum:.4f}")

    # Step 3: Probe's guesses (using different metrics)
    probe_guess_avg = max(avg_scores.items(), key=lambda x: x[1])[0]
    probe_guess_sum = max(sum_scores.items(), key=lambda x: x[1])[0]
    probe_guess_max = max(max_scores.items(), key=lambda x: x[1])[0]
    print(f"\n🤖 Probe guesses the lie is:")
    print(f"   Using avg: Statement {probe_guess_avg}")
    print(f"   Using sum: Statement {probe_guess_sum}")
    print(f"   Using max: Statement {probe_guess_max}")

    # Step 4: Ask model for the answer
    print("\n🤔 Asking model for the answer...")
    model_answer = ask_for_lie(service, statements, probe_id)

    if model_answer == -1:
        print("❌ Error: Could not parse model's answer")
        return {"success": False, "error": "Could not parse model answer"}

    print(f"✅ Model says the lie is: Statement {model_answer}")

    # Step 5: Compare
    correct_avg = probe_guess_avg == model_answer
    correct_sum = probe_guess_sum == model_answer
    correct_max = probe_guess_max == model_answer

    print(f"\n{'=' * 80}")
    print(f"Results:")
    print(f"  Using avg: {'🎉 CORRECT!' if correct_avg else '❌ INCORRECT'}")
    print(f"  Using sum: {'🎉 CORRECT!' if correct_sum else '❌ INCORRECT'}")
    print(f"  Using max: {'🎉 CORRECT!' if correct_max else '❌ INCORRECT'}")
    print(f"{'=' * 80}\n")

    return {
        "success": True,
        "topic": topic,
        "statements": statements,
        "avg_scores": avg_scores,
        "sum_scores": sum_scores,
        "max_scores": max_scores,
        "probe_guess_avg": probe_guess_avg,
        "probe_guess_sum": probe_guess_sum,
        "probe_guess_max": probe_guess_max,
        "model_answer": model_answer,
        "probe_correct_avg": correct_avg,
        "probe_correct_sum": correct_sum,
        "probe_correct_max": correct_max,
    }


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Two Truths and a Lie experiment with hallucination probes"
    )
    parser.add_argument(
        "topic",
        nargs="?",
        default=None,
        help="Topic for the two truths and a lie (e.g., 'cats', 'the Roman Empire'). If not provided, model generates random topics.",
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
        args.log_file = str(log_dir / f"two_truths_and_a_lie_{timestamp}.jsonl")

    log_path = Path(args.log_file)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"📝 Logging to: {args.log_file}\n")

    # Get service once for topic generation if needed
    service = get_probe_service() if args.topic is None else None

    results = []
    failed_trials = []

    for i in range(args.num_trials):
        if args.num_trials > 1:
            print(f"\n\n{'#' * 80}")
            print(f"TRIAL {i + 1}/{args.num_trials}")
            print(f"{'#' * 80}")

        try:
            # Generate random topic if not provided
            if args.topic is None:
                if service is None:
                    service = get_probe_service()
                topic = generate_random_topic(service, args.probe_id)
            else:
                topic = args.topic

            result = run_experiment(
                topic,
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
            correct_avg = sum(1 for r in results if r["probe_correct_avg"])
            correct_sum = sum(1 for r in results if r["probe_correct_sum"])
            correct_max = sum(1 for r in results if r["probe_correct_max"])
            print(
                f"Probe correct (avg): {correct_avg}/{len(results)} ({correct_avg / len(results) * 100:.1f}%)"
            )
            print(
                f"Probe correct (sum): {correct_sum}/{len(results)} ({correct_sum / len(results) * 100:.1f}%)"
            )
            print(
                f"Probe correct (max): {correct_max}/{len(results)} ({correct_max / len(results) * 100:.1f}%)"
            )

        if failed_trials:
            print(f"\nFailed trial numbers: {[f['trial'] for f in failed_trials]}")

        print(f"{'=' * 80}\n")

    print(f"\n✅ Results saved to: {args.log_file}")


if __name__ == "__main__":
    main()
