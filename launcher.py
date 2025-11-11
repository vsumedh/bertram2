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

        # Print the response
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


if __name__ == "__main__":
    asyncio.run(launch_evaluation())
