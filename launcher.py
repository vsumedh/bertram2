"""Launcher for coordinating TextWorld green and white agents."""

from __future__ import annotations

import asyncio
import json
import multiprocessing
from contextlib import suppress
from typing import Any, Dict, Optional

from a2a.utils import get_text_parts

from src.utils.a2a_client import A2AMessenger
from src.green_agent.agent import start_green_agent
from src.white_agent.agent import start_white_agent
from src.white_agent.agent_hardcoded import start_hardcoded_agent
from src.green_agent.green_assessor import (
    TrajectoryEval,
    aggregate_evals,
    print_batch_summary,
)
from src.utils.messaging import parse_tags


async def _ensure_agent_ready(
    messenger: A2AMessenger, url: str, timeout: int = 15
) -> None:
    """Ensure agent is ready, raise error if timeout."""
    if not await messenger.wait_agent_ready(url, timeout=timeout):
        raise RuntimeError(f"Agent at {url} did not become ready in time")


async def launch_evaluation(
    *,
    task_config: Optional[Dict[str, Any]] = None,
    green_host: str = "127.0.0.1",
    green_port: int = 9001,
    white_host: str = "127.0.0.1",
    white_port: int = 9002,
    fast_white: bool = False,
    use_hardcoded: bool = False,
) -> None:
    """Launch the end-to-end evaluation loop for TextWorld."""
    task_config = task_config or {
        "task_index": 0,
        "max_steps": 50,
    }

    # Calculate timeout based on max_steps: ~10 seconds per step + buffer
    max_steps = task_config.get("max_steps", 50)
    timeout = max(600.0, max_steps * 12.0)  # Minimum 10 min, or 12 sec per step
    print(f"Using timeout: {timeout:.1f} seconds for {max_steps} steps")
    messenger = A2AMessenger(timeout=timeout)
    processes: list[multiprocessing.Process] = []

    green_url = f"http://{green_host}:{green_port}"
    white_url = f"http://{white_host}:{white_port}"

    try:
        # Start green agent
        green_process = multiprocessing.Process(
            target=start_green_agent,
            kwargs={"agent_name": "agent_card", "host": green_host, "port": green_port},
            daemon=True,
        )
        green_process.start()
        processes.append(green_process)
        await _ensure_agent_ready(messenger, green_url)
        print("Green agent is ready.")

        if not fast_white:
            if use_hardcoded:
                # Start hardcoded white agent
                white_process = multiprocessing.Process(
                    target=start_hardcoded_agent,
                    kwargs={
                        "agent_name": "agent_card_hardcoded",
                        "host": white_host,
                        "port": white_port,
                    },
                    daemon=True,
                )
                white_process.start()
                processes.append(white_process)
                await _ensure_agent_ready(messenger, white_url)
                print("Hardcoded white agent is ready.")
            else:
                # Start regular LLM white agent
                white_process = multiprocessing.Process(
                    target=start_white_agent,
                    kwargs={
                        "agent_name": "agent_card",
                        "host": white_host,
                        "port": white_port,
                    },
                    daemon=True,
                )
                white_process.start()
                processes.append(white_process)
                await _ensure_agent_ready(messenger, white_url)
                print("White agent is ready.")
        else:
            print("Fast white mode: skipping LLM white agent startup.")

        # Send task configuration to green agent
        task_config_json = json.dumps(task_config, indent=2)
        payload_parts = [
            "<task_config>",
            task_config_json,
            "</task_config>",
            "<fast_white>",
            "1" if fast_white else "0",
            "</fast_white>",
        ]
        if not fast_white:
            payload_parts.insert(0, "</white_agent_url>")
            payload_parts.insert(0, f"{white_url}")
            payload_parts.insert(0, "<white_agent_url>")
        payload = "\n".join(payload_parts)

        print("Sending task to green agent...")
        response = await messenger.send_text(
            green_url,
            payload,
        )

        # Print the response (skip in demo mode; green prints its own demo output)
        import os as _os

        if _os.environ.get("DEMO_MODE", "0") != "1":
            response_texts = get_text_parts(response.parts)
            for text in response_texts:
                print(text)

    finally:
        # Cleanup
        for process in processes:
            if process.is_alive():
                process.terminate()
            with suppress(Exception):
                process.join(timeout=5)
        await messenger.aclose()


def _parse_episode_summary(text: str) -> Dict[str, Any]:
    """Parse success, steps, quick and overall rating from green report text."""
    result: Dict[str, Any] = {
        "success": None,
        "steps": None,
        "quick": None,
        "overall": None,
    }
    try:
        for line in text.splitlines():
            line_stripped = line.strip()
            if line_stripped.lower().startswith("success:"):
                # Success: True/False
                value = line_stripped.split(":", 1)[1].strip()
                result["success"] = value.lower() in ("true", "1", "yes")
            elif line_stripped.lower().startswith("steps:"):
                # Steps: 12 / 50
                right = line_stripped.split(":", 1)[1].strip()
                num = right.split("/", 1)[0].strip()
                result["steps"] = int(num)
            elif line_stripped.lower().startswith("quick rating:"):
                # Quick Rating: 6.5/10
                right = line_stripped.split(":", 1)[1].strip()
                num = right.split("/", 1)[0].strip()
                result["quick"] = float(num)
            elif line_stripped.lower().startswith(
                "overall rating (weighted):"
            ) or line_stripped.lower().startswith("overall rating:"):
                right = line_stripped.split(":", 1)[1].strip()
                num = right.split("/", 1)[0].strip()
                try:
                    result["overall"] = float(num)
                except Exception:
                    pass
    except Exception:
        pass
    return result


async def benchmark_evaluation(
    *,
    task_indices: list[int],
    max_steps: int = 50,
    variants: list[Dict[str, Any]] | None = None,
    green_host: str = "127.0.0.1",
    green_port: int = 8722,
    white_host: str = "127.0.0.1",
    fast_white: bool = False,
) -> None:
    """Run multiple tasks across white-agent variants and print accuracy table."""
    variants = variants or [{"name": "default", "model": "Qwen/Qwen2.5-7B-Instruct"}]
    messenger = A2AMessenger(timeout=max(600.0, max_steps * 12.0))
    processes: list[multiprocessing.Process] = []

    green_url = f"http://{green_host}:{green_port}"

    try:
        # Start green once
        green_process = multiprocessing.Process(
            target=start_green_agent,
            kwargs={"agent_name": "agent_card", "host": green_host, "port": green_port},
            daemon=True,
        )
        green_process.start()
        processes.append(green_process)
        await _ensure_agent_ready(messenger, green_url)
        print("Green agent is ready.")

        # Run each variant sequentially (or fast mode without white agent)
        results_by_variant: Dict[str, list[Dict[str, Any]]] = {}
        base_white_port = 8800

        if fast_white:
            variant_name = "fast_white"
            results_by_variant[variant_name] = []
            for task_index in task_indices:
                task_config = {
                    "task_index": int(task_index),
                    "max_steps": int(max_steps),
                }
                payload = (
                    "<task_config>\n"
                    f"{json.dumps(task_config, indent=2)}\n"
                    "</task_config>\n"
                    "<benchmark_mode>1</benchmark_mode>\n"
                    "<fast_white>1</fast_white>"
                )
                print(f"→ [fast_white] Task {task_index}: sending to green agent...")
                response = await messenger.send_text(green_url, payload)
                response_texts = get_text_parts(response.parts)
                for text in response_texts:
                    print(text)
                final_text = response_texts[-1] if response_texts else ""
                summary = _parse_episode_summary(final_text)
                success = bool(summary.get("success") is True)
                steps = summary.get("steps")
                print(f"  Result: success={success}, steps={steps}")
                results_by_variant[variant_name].append(summary)
        else:
            for idx, variant in enumerate(variants):
                name = variant.get("name", f"variant_{idx + 1}")
                model = variant.get("model", "Qwen/Qwen2.5-7B-Instruct")
                temperature = float(variant.get("temperature", 0.0))
                prompt_profile = variant.get("prompt_profile", "standard")

                # Use dedicated port per variant
                white_port = base_white_port + idx
                white_url = f"http://{white_host}:{white_port}"

                print(
                    f"\nStarting white agent '{name}' (model={model}, profile={prompt_profile}) on {white_url} ..."
                )
                white_process = multiprocessing.Process(
                    target=start_white_agent,
                    kwargs={
                        "agent_name": "agent_card",
                        "host": white_host,
                        "port": white_port,
                        "model": model,
                        "temperature": temperature,
                        "prompt_profile": prompt_profile,
                    },
                    daemon=True,
                )
                white_process.start()
                processes.append(white_process)
                await _ensure_agent_ready(messenger, white_url)
                print(f"White agent '{name}' is ready.")

                variant_results: list[Dict[str, Any]] = []

                # Run tasks for this variant
                for task_index in task_indices:
                    task_config = {
                        "task_index": int(task_index),
                        "max_steps": int(max_steps),
                    }
                    payload = (
                        "<white_agent_url>\n"
                        f"{white_url}\n"
                        "</white_agent_url>\n"
                        "<benchmark_mode>1</benchmark_mode>\n"
                        "<task_config>\n"
                        f"{json.dumps(task_config, indent=2)}\n"
                        "</task_config>\n"
                        "<fast_white>0</fast_white>"
                    )
                    print(f"→ [{name}] Task {task_index}: sending to green agent...")
                    response = await messenger.send_text(green_url, payload)
                    response_texts = get_text_parts(response.parts)
                    # Print per-task report returned by green (includes pre/post and preview)
                    for text in response_texts:
                        print(text)
                    # Parse success and steps from the last text
                    final_text = response_texts[-1] if response_texts else ""
                    summary = _parse_episode_summary(final_text)
                    success = bool(summary.get("success") is True)
                    steps = summary.get("steps")
                    print(f"  Result: success={success}, steps={steps}")
                    variant_results.append(summary)

                results_by_variant[name] = variant_results

                # Stop this white agent before next
                if white_process.is_alive():
                    white_process.terminate()
                with suppress(Exception):
                    white_process.join(timeout=5)
                processes.remove(white_process)
                print(f"White agent '{name}' stopped.")

        # Aggregate and print summary
        print("\nBenchmark Summary")
        print("=================")
        for name, summaries in results_by_variant.items():
            total = len(summaries)
            successes = sum(1 for s in summaries if s.get("success") is True)
            acc = (successes / total) if total > 0 else 0.0
            mean_steps = (
                sum(
                    s.get("steps", 0)
                    for s in summaries
                    if isinstance(s.get("steps"), int)
                )
                / total
                if total > 0
                else None
            )
            print(
                f"- {name}: accuracy={acc:.2%}, mean_steps={'-' if mean_steps is None else f'{mean_steps:.1f}'}"
            )

    finally:
        # Cleanup residual processes and client
        for process in processes:
            if process.is_alive():
                process.terminate()
            with suppress(Exception):
                process.join(timeout=5)
        await messenger.aclose()


def _extract_eval_from_parts(parts: list[str]) -> TrajectoryEval | None:
    for text in parts:
        tags = parse_tags(text)
        if "eval_json" in tags:
            try:
                data = json.loads(tags["eval_json"])
                return TrajectoryEval(**data)
            except Exception:
                continue
    return None


async def run_assessment_batch(
    *,
    task_indices: list[int],
    max_steps: int = 50,
    green_host: str = "127.0.0.1",
    green_port: int = 9001,
) -> None:
    """Run fixed 5-task assessment using fast white mode and aggregate results."""
    messenger = A2AMessenger(timeout=max(600.0, max_steps * 12.0))
    processes: list[multiprocessing.Process] = []
    green_url = f"http://{green_host}:{green_port}"
    evals: list[TrajectoryEval] = []
    white_label = "fast white"

    try:
        # Start green agent (fast white mode skips LLM white)
        green_process = multiprocessing.Process(
            target=start_green_agent,
            kwargs={"agent_name": "agent_card", "host": green_host, "port": green_port},
            daemon=True,
        )
        green_process.start()
        processes.append(green_process)
        await _ensure_agent_ready(messenger, green_url)
        print("Green agent is ready (fast white).")

        for task_index in task_indices:
            task_config = {"task_index": int(task_index), "max_steps": int(max_steps)}
            payload = (
                "<task_config>\n"
                f"{json.dumps(task_config, indent=2)}\n"
                "</task_config>\n"
                "<fast_white>1</fast_white>"
            )
            print(f"\n→ Task {task_index}: sending to green agent...")
            response = await messenger.send_text(green_url, payload)
            response_texts = get_text_parts(response.parts)
            eval_obj = _extract_eval_from_parts(response_texts)
            # Print per-task report (skip eval_json block)
            for text in response_texts:
                if "<eval_json>" in text:
                    continue
                print(text)
            if eval_obj:
                evals.append(eval_obj)
            else:
                print("Warning: no eval_json returned for this task.")

        if evals:
            summary = print_batch_summary(
                evals=evals,
                white_label=white_label,
                step_budget=max_steps,
            )
            print("\n" + summary)
        else:
            print("No evaluations collected; skipping batch summary.")

    finally:
        for process in processes:
            if process.is_alive():
                process.terminate()
            with suppress(Exception):
                process.join(timeout=5)
        await messenger.aclose()


async def run_evaluation(
    *,
    task_indices: list[int],
    max_steps: int = 50,
    agent_mode: str = "llm",  # "llm", "expert", or "fast"
    expert_profile: str = "expert",  # Profile for expert mode
    verbose: bool = False,
    green_host: str = "127.0.0.1",
    green_port: int = 8722,
    white_host: str = "127.0.0.1",
    white_port: int = 8724,
) -> None:
    """Unified evaluation function for single or multiple tasks.

    Args:
        task_indices: List of task indices to evaluate
        max_steps: Maximum steps per task
        agent_mode: "llm" (LLM white agent), "expert" (hardcoded + reasoning), or "fast" (internal)
        expert_profile: Profile for expert mode (expert, competent, novice, lucky_guesser, overthinker)
        verbose: Show step-by-step execution details
        green_host: Host for green agent
        green_port: Port for green agent
        white_host: Host for white agent
        white_port: Port for white agent
    """
    messenger = A2AMessenger(timeout=max(600.0, max_steps * 12.0))
    processes: list[multiprocessing.Process] = []
    evals: list[TrajectoryEval] = []

    green_url = f"http://{green_host}:{green_port}"
    white_url = f"http://{white_host}:{white_port}"

    # Determine mode settings
    use_fast_white = agent_mode == "fast"
    use_hardcoded = agent_mode == "hardcoded"

    # Agent mode label for summary
    if use_hardcoded:
        agent_label = f"Hardcoded ({expert_profile})"
    else:
        agent_labels = {"llm": "LLM (vLLM)", "fast": "Fast (internal)"}
        agent_label = agent_labels.get(agent_mode, agent_mode)

    print("=== Evaluation Configuration ===")
    print(f"Tasks: {task_indices}")
    print(f"Agent mode: {agent_label}")
    if use_hardcoded:
        print(f"Profile: {expert_profile}")
    print(f"Max steps: {max_steps}")
    print(f"Verbose: {verbose}")
    print()

    try:
        # Start green agent
        green_process = multiprocessing.Process(
            target=start_green_agent,
            kwargs={"agent_name": "agent_card", "host": green_host, "port": green_port},
            daemon=True,
        )
        green_process.start()
        processes.append(green_process)
        await _ensure_agent_ready(messenger, green_url)
        print("Green agent is ready.")

        # Start white agent if not using fast mode
        if not use_fast_white:
            if use_hardcoded:
                white_process = multiprocessing.Process(
                    target=start_hardcoded_agent,
                    kwargs={
                        "agent_name": "agent_card_hardcoded",
                        "host": white_host,
                        "port": white_port,
                        "profile": expert_profile,
                    },
                    daemon=True,
                )
                white_process.start()
                processes.append(white_process)
                await _ensure_agent_ready(messenger, white_url)
                print(f"Hardcoded white agent is ready (profile={expert_profile}).")
            else:
                white_process = multiprocessing.Process(
                    target=start_white_agent,
                    kwargs={
                        "agent_name": "agent_card",
                        "host": white_host,
                        "port": white_port,
                    },
                    daemon=True,
                )
                white_process.start()
                processes.append(white_process)
                await _ensure_agent_ready(messenger, white_url)
                print("LLM white agent is ready.")
        else:
            print("Fast mode: using internal expert plan.")

        print()

        # Run each task
        for i, task_index in enumerate(task_indices):
            print(f"=== Task {task_index} ({i + 1}/{len(task_indices)}) ===")

            task_config = {"task_index": int(task_index), "max_steps": int(max_steps)}

            # Build payload
            if use_fast_white:
                payload = (
                    "<task_config>\n"
                    f"{json.dumps(task_config, indent=2)}\n"
                    "</task_config>\n"
                    "<fast_white>1</fast_white>"
                )
            else:
                payload = (
                    "<white_agent_url>\n"
                    f"{white_url}\n"
                    "</white_agent_url>\n"
                    "<task_config>\n"
                    f"{json.dumps(task_config, indent=2)}\n"
                    "</task_config>\n"
                    "<fast_white>0</fast_white>"
                )

            # Send to green agent
            response = await messenger.send_text(green_url, payload)
            response_texts = get_text_parts(response.parts)

            # Extract evaluation
            eval_obj = _extract_eval_from_parts(response_texts)

            if verbose:
                # Print full response (excluding raw eval_json)
                for text in response_texts:
                    # Skip the raw eval_json block but keep everything else
                    lines = []
                    skip_eval_json = False
                    for line in text.split("\n"):
                        if "<eval_json>" in line:
                            skip_eval_json = True
                        if not skip_eval_json:
                            lines.append(line)
                        if "</eval_json>" in line:
                            skip_eval_json = False
                    filtered_text = "\n".join(lines).strip()
                    if filtered_text:
                        print(filtered_text)
            else:
                # Print brief summary
                if eval_obj:
                    status = "YES" if eval_obj.success else "NO"
                    print(f"Success: {status}")
                    print(f"Steps: {eval_obj.steps} / {max_steps}")
                    print(f"Overall: {eval_obj.overall:.1f} / 10")
                else:
                    print("Warning: no evaluation returned for this task.")

            if eval_obj:
                evals.append(eval_obj)

            print()

        # Print aggregate summary
        if evals:
            print("=== AGGREGATE SUMMARY ===")
            summary_data = aggregate_evals(evals, step_budget=max_steps)

            success_count = summary_data["success_count"]
            total = summary_data["total"]
            success_rate = summary_data["success_rate"]
            mean_steps = summary_data["mean_steps"]
            mean_overall = summary_data["mean_overall"]
            mean_strategy = summary_data["mean_strategy"]
            mean_reasoning = summary_data["mean_reasoning"]

            print(f"Agent: {agent_label}")
            print(
                f"Tasks: {success_count}/{total} successful ({success_rate * 100:.0f}%)"
            )
            print(f"Mean steps: {mean_steps:.1f} / {max_steps}")
            print(f"Mean overall: {mean_overall:.1f} / 10")
            print()
            print("Per-criterion averages:")
            print(f"  Correctness: {summary_data['correctness_batch']:.1f} / 10")
            print(f"  Efficiency: {summary_data['efficiency_batch']:.1f} / 10")
            print(f"  Strategy: {mean_strategy:.1f} / 10")
            print(f"  Reasoning: {mean_reasoning:.1f} / 10")
        else:
            print("No evaluations collected.")

    finally:
        # Cleanup
        for process in processes:
            if process.is_alive():
                process.terminate()
            with suppress(Exception):
                process.join(timeout=5)
        await messenger.aclose()


if __name__ == "__main__":
    asyncio.run(launch_evaluation())
