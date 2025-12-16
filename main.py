"""CLI entry point for TextWorld agentify evaluation."""

import asyncio
import os
import typer

from typing_extensions import Annotated  # For Python 3.8+
from src.green_agent.agent import start_green_agent
from src.white_agent.agent import start_white_agent
from launcher import (
    launch_evaluation,
    benchmark_evaluation,
    run_assessment_batch,
    run_evaluation,
)

app = typer.Typer(help="TextWorld Agentify - Agent evaluation framework")


@app.command()
def green(
    agent_name: str = "agent_card",
    host: Annotated[str, typer.Argument(envvar="HOST")] = "0.0.0.0",
    port: Annotated[int, typer.Argument(envvar="AGENT_PORT")] = 8722,
):
    """Start the green agent (evaluation manager)."""
    start_green_agent(agent_name=agent_name, host=host, port=port)


@app.command()
def white(
    agent_name: str = "agent_card",
    host: Annotated[str, typer.Argument(envvar="HOST")] = "0.0.0.0",
    port: Annotated[int, typer.Argument(envvar="AGENT_PORT")] = 8723,
    model: str = typer.Option(
        "Qwen/Qwen2.5-7B-Instruct", help="LLM model for the white agent"
    ),
    temperature: float = typer.Option(0.0, help="Sampling temperature"),
    prompt_profile: str = typer.Option(
        "standard", help="Prompt profile: standard|concise"
    ),
    vllm_url: str = typer.Option("http://localhost:8000/v1", help="vLLM server URL"),
):
    """Start the white agent (agent under test)."""
    start_white_agent(
        agent_name=agent_name,
        host=host,
        port=port,
        model=model,
        temperature=temperature,
        prompt_profile=prompt_profile,
        vllm_base_url=vllm_url,
    )


def _parse_task_indices(tasks: str) -> list[int]:
    """Parse task indices from string: '4', '0,1,2', or 'all'."""
    tasks = tasks.strip().lower()
    if tasks == "all":
        return list(range(20))  # All 20 tasks
    parts = [p.strip() for p in tasks.split(",") if p.strip()]
    return [int(p) for p in parts]


@app.command()
def evaluate(
    tasks: str = typer.Option("0", help='Task indices: "4", "0,1,2,3", or "all"'),
    agent: str = typer.Option("llm", help='Agent mode: "llm", "hardcoded", or "fast"'),
    profile: str = typer.Option(
        "expert",
        help="Agent profile (hardcoded mode only): expert/competent/novice/lucky_guesser/overthinker",
    ),
    max_steps: int = typer.Option(50, help="Max steps per task"),
    verbose: bool = typer.Option(
        False, "--verbose", "-v", help="Show step-by-step details"
    ),
    green_host: str = typer.Option("127.0.0.1", help="Green agent host"),
    green_port: int = typer.Option(8722, help="Green agent port"),
    white_host: str = typer.Option("127.0.0.1", help="White agent host"),
    white_port: int = typer.Option(8724, help="White agent port"),
):
    """Run evaluation on one or more tasks.

    Agent modes:
      - llm: LLM-based white agent (Qwen2.5-7B via vLLM) - variable reasoning quality
      - hardcoded: Pre-recorded trajectories with configurable profile - see below
      - fast: Internal expert plan, minimal reasoning - fastest (~4/10)

    Hardcoded profiles (--profile, only for --agent hardcoded):
      - expert: High reasoning + optimal strategy (~9/10)
      - competent: Medium reasoning + suboptimal strategy (~7.5/10)
      - novice: Low reasoning + poor strategy (~5/10)
      - lucky_guesser: Low reasoning + optimal strategy (~7/10)
      - overthinker: High reasoning + suboptimal strategy (~8/10)

    Examples:
      python main.py evaluate --tasks 4 --agent llm
      python main.py evaluate --tasks 4 --agent hardcoded --profile expert
      python main.py evaluate --tasks 4 --agent hardcoded --profile novice
      python main.py evaluate --tasks all --agent fast
    """
    # Validate agent mode
    valid_modes = ["llm", "hardcoded", "fast"]
    if agent.lower() not in valid_modes:
        raise typer.BadParameter(
            f"Invalid agent mode '{agent}'. Must be one of: {valid_modes}"
        )

    # Validate profile
    valid_profiles = ["expert", "competent", "novice", "lucky_guesser", "overthinker"]
    if profile.lower() not in valid_profiles:
        raise typer.BadParameter(
            f"Invalid profile '{profile}'. Must be one of: {valid_profiles}"
        )

    task_indices = _parse_task_indices(tasks)

    asyncio.run(
        run_evaluation(
            task_indices=task_indices,
            max_steps=max_steps,
            agent_mode=agent.lower(),
            expert_profile=profile.lower(),
            verbose=verbose,
            green_host=green_host,
            green_port=green_port,
            white_host=white_host,
            white_port=white_port,
        )
    )


@app.command(deprecated=True)
def launch(
    task_index: int = 0,
    max_steps: int = 50,
    green_host: str = "127.0.0.1",
    green_port: int = 8722,
    white_host: str = "127.0.0.1",
    white_port: int = 8724,
    use_hardcoded: bool = typer.Option(
        False, "--hardcoded", help="Use hardcoded white agent instead of LLM"
    ),
):
    """[Deprecated] Use 'evaluate' instead. Launch single-task evaluation."""
    fast_white = os.environ.get("FAST_WHITE", "0") == "1"
    task_config = {
        "task_index": task_index,
        "max_steps": max_steps,
    }
    asyncio.run(
        launch_evaluation(
            task_config=task_config,
            green_host=green_host,
            green_port=green_port,
            white_host=white_host,
            white_port=white_port,
            fast_white=fast_white,
            use_hardcoded=use_hardcoded,
        )
    )


@app.command(deprecated=True)
def benchmark(
    tasks: str = typer.Option(
        "4,6,5,7,1", help='Comma-separated task indices, e.g. "4,6,5,7,1"'
    ),
    max_steps: int = typer.Option(50, help="Max steps per task"),
    green_host: str = "127.0.0.1",
    green_port: int = 8722,
    white_host: str = "127.0.0.1",
):
    """[Deprecated] Use 'evaluate' instead. Run benchmark over multiple tasks."""
    fast_white = os.environ.get("FAST_WHITE", "0") == "1"
    # Parse comma-separated indices
    parts = [p.strip() for p in tasks.split(",") if p.strip()]
    task_indices = [int(p) for p in parts]
    asyncio.run(
        benchmark_evaluation(
            task_indices=task_indices,
            max_steps=max_steps,
            green_host=green_host,
            green_port=green_port,
            white_host=white_host,
            fast_white=fast_white,
        )
    )


@app.command(deprecated=True)
def assess(
    tasks: str = typer.Option(
        "4,6,5,7,1", help="Comma-separated task indices (5 tasks, default 4,6,5,7,1)"
    ),
    max_steps: int = typer.Option(
        50, help="Max steps per task (step budget fixed at 50 for scoring)"
    ),
    green_host: str = "127.0.0.1",
    green_port: int = 8722,
):
    """[Deprecated] Use 'evaluate --agent fast' instead. Run assessment with fast mode."""
    parts = [p.strip() for p in tasks.split(",") if p.strip()]
    task_indices = [int(p) for p in parts]
    asyncio.run(
        run_assessment_batch(
            task_indices=task_indices,
            max_steps=max_steps,
            green_host=green_host,
            green_port=green_port,
        )
    )


if __name__ == "__main__":
    app()
