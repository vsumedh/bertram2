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

        # Start white agent
        white_process = multiprocessing.Process(
            target=start_white_agent,
            kwargs={"agent_name": "agent_card", "host": white_host, "port": white_port},
            daemon=True,
        )
        white_process.start()
        processes.append(white_process)
        await _ensure_agent_ready(messenger, white_url)
        print("White agent is ready.")

        # Send task configuration to green agent
        task_config_json = json.dumps(task_config, indent=2)
        payload = (
            "<white_agent_url>\n"
            f"{white_url}\n"
            "</white_agent_url>\n"
            "<task_config>\n"
            f"{task_config_json}\n"
            "</task_config>"
        )

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
    result: Dict[str, Any] = {"success": None, "steps": None, "quick": None, "overall": None}
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
            elif line_stripped.lower().startswith("overall rating (weighted):") or line_stripped.lower().startswith("overall rating:"):
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
) -> None:
    """Run multiple tasks across white-agent variants and print accuracy table."""
    variants = variants or [{"name": "default", "model": "openai/gpt-4o"}]
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

        # Run each variant sequentially
        results_by_variant: Dict[str, list[Dict[str, Any]]] = {}
        base_white_port = 8800

        for idx, variant in enumerate(variants):
            name = variant.get("name", f"variant_{idx+1}")
            model = variant.get("model", "openai/gpt-4o")
            temperature = float(variant.get("temperature", 0.0))
            prompt_profile = variant.get("prompt_profile", "standard")

            # Use dedicated port per variant
            white_port = base_white_port + idx
            white_url = f"http://{white_host}:{white_port}"

            print(f"\nStarting white agent '{name}' (model={model}, profile={prompt_profile}) on {white_url} ...")
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
                task_config = {"task_index": int(task_index), "max_steps": int(max_steps)}
                payload = (
                    "<white_agent_url>\n"
                    f"{white_url}\n"
                    "</white_agent_url>\n"
                    "<benchmark_mode>1</benchmark_mode>\n"
                    "<task_config>\n"
                    f"{json.dumps(task_config, indent=2)}\n"
                    "</task_config>"
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
                sum(s.get("steps", 0) for s in summaries if isinstance(s.get("steps"), int)) / total
                if total > 0
                else None
            )
            print(f"- {name}: accuracy={acc:.2%}, mean_steps={'-' if mean_steps is None else f'{mean_steps:.1f}'}")

    finally:
        # Cleanup residual processes and client
        for process in processes:
            if process.is_alive():
                process.terminate()
            with suppress(Exception):
                process.join(timeout=5)
        await messenger.aclose()


if __name__ == "__main__":
    asyncio.run(launch_evaluation())
