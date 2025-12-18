"""Launcher for coordinating TextWorld green and white agents."""

from __future__ import annotations

import asyncio
import json
import multiprocessing
import os
from contextlib import suppress
from typing import Any, Dict

# Enable unbuffered output in demo mode for real-time subprocess prints
if os.environ.get("DEMO_MODE", "0") == "1":
    os.environ["PYTHONUNBUFFERED"] = "1"

from a2a.utils import get_text_parts

from src.utils.a2a_client import A2AMessenger
from src.green_agent.agent import start_green_agent
from src.white_agent.agent import start_white_agent
from src.white_agent.agent_hardcoded import start_hardcoded_agent
from src.green_agent.green_assessor import (
    TrajectoryEval,
    aggregate_evals,
)
from src.green_agent.output_formatter import TerminalColors, _score_color
from src.utils.messaging import parse_tags


async def _ensure_agent_ready(
    messenger: A2AMessenger, url: str, timeout: int = 15
) -> None:
    """Ensure agent is ready, raise error if timeout."""
    if not await messenger.wait_agent_ready(url, timeout=timeout):
        raise RuntimeError(f"Agent at {url} did not become ready in time")


def _format_aggregate_summary(
    agent_label: str,
    summary_data: Dict[str, Any],
    max_steps: int,
) -> str:
    """Format the aggregate summary with ANSI colors matching evaluation output.

    Uses the same color conventions as DemoFormatter.format_evaluation():
    - DIV_DOUBLE for major section boundaries
    - BOLD for headers and key values
    - _score_color() for score values (green >= 7, yellow >= 4, red < 4)
    - Success rate color-coded (green >= 80%, yellow >= 50%, red < 50%)
    """
    c = TerminalColors

    success_count = summary_data["success_count"]
    total = summary_data["total"]
    success_rate = summary_data["success_rate"]
    mean_steps = summary_data["mean_steps"]
    mean_overall = summary_data["mean_overall"]

    # Color-code success rate
    if success_rate >= 0.8:
        rate_color = c.GREEN_FG
    elif success_rate >= 0.5:
        rate_color = c.YELLOW_FG
    else:
        rate_color = c.RED_FG

    # Get scores for color-coding
    correctness = summary_data["correctness_batch"]
    efficiency = summary_data["efficiency_batch"]
    strategy = summary_data["mean_strategy"]
    reasoning = summary_data["mean_reasoning"]

    lines = [
        c.DIV_DOUBLE,
        f"{c.BOLD}AGGREGATE SUMMARY{c.RESET}",
        c.DIV_DOUBLE,
        "",
        f"Agent:        {c.BOLD}{agent_label}{c.RESET}",
        f"Tasks:        {rate_color}{c.BOLD}{success_count}/{total}{c.RESET} successful ({rate_color}{success_rate * 100:.0f}%{c.RESET})",
        f"Mean steps:   {c.BOLD}{mean_steps:.1f}{c.RESET} / {max_steps}",
        f"Mean overall: {_score_color(mean_overall)}{c.BOLD}{mean_overall:.1f}{c.RESET} / 10",
        "",
        "Per-criterion averages:",
        f"  Correctness: {_score_color(correctness)}{correctness:.1f}{c.RESET} / 10",
        f"  Efficiency:  {_score_color(efficiency)}{efficiency:.1f}{c.RESET} / 10",
        f"  Strategy:    {_score_color(strategy)}{strategy:.1f}{c.RESET} / 10",
        f"  Reasoning:   {_score_color(reasoning)}{reasoning:.1f}{c.RESET} / 10",
        "",
        c.DIV_DOUBLE,
    ]
    return "\n".join(lines)


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


async def run_evaluation(
    *,
    task_indices: list[int],
    max_steps: int = 50,
    agent_mode: str = "llm",  # "llm" or "expert"
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
        agent_mode: "llm" (LLM white agent), "expert" (hardcoded + reasoning)
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
    use_hardcoded = agent_mode == "hardcoded"

    # Agent mode label for summary
    if use_hardcoded:
        agent_label = f"Hardcoded ({expert_profile})"
    else:
        agent_labels = {"llm": "LLM (vLLM)"}
        agent_label = agent_labels.get(agent_mode, agent_mode)

    print("=== Evaluation Configuration ===")
    print(f"Tasks: {task_indices}")
    print(f"Agent mode: {agent_label}")
    if use_hardcoded:
        print(f"Profile: {expert_profile}")
    print(f"Max steps: {max_steps}")
    print(f"Verbose: {verbose}")
    print()

    # Set verbose environment variable for child processes
    import os

    if verbose:
        os.environ["GREEN_VERBOSE"] = "1"
    else:
        os.environ.pop("GREEN_VERBOSE", None)

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
        print("Green agent is ready.", flush=True)

        # Start white agent
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
            print(
                f"Hardcoded white agent is ready (profile={expert_profile}).",
                flush=True,
            )
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
            print("LLM white agent is ready.", flush=True)

        print(flush=True)

        # Run each task
        for i, task_index in enumerate(task_indices):
            print(
                f"=== Task {task_index} ({i + 1}/{len(task_indices)}) ===", flush=True
            )

            task_config = {"task_index": int(task_index), "max_steps": int(max_steps)}

            # Build payload
            payload = (
                "<white_agent_url>\n"
                f"{white_url}\n"
                "</white_agent_url>\n"
                "<task_config>\n"
                f"{json.dumps(task_config, indent=2)}\n"
                "</task_config>\n"
            )

            # Send to green agent
            response = await messenger.send_text(green_url, payload)
            response_texts = get_text_parts(response.parts)

            # Extract evaluation
            eval_obj = _extract_eval_from_parts(response_texts)

            # Check if demo mode is active
            demo_mode = os.environ.get("DEMO_MODE", "0") == "1"

            if demo_mode:
                # Demo mode: detailed EVALUATION already printed by subprocess
                # Skip brief summary here to avoid redundancy
                pass
            elif verbose:
                # Verbose non-demo: print full response (excluding raw eval_json)
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
                # Non-verbose non-demo: print brief summary only
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

        # Print aggregate summary (skip for single-task demo mode - already shown in EVALUATION)
        demo_mode_active = os.environ.get("DEMO_MODE", "0") == "1"
        if evals and not (len(evals) == 1 and demo_mode_active):
            summary_data = aggregate_evals(evals, step_budget=max_steps)
            print(_format_aggregate_summary(agent_label, summary_data, max_steps))
        elif not evals:
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
    asyncio.run(run_evaluation(task_indices=[0], max_steps=50))
