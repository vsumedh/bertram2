"""CLI entry point for TextWorld agentify evaluation."""

import asyncio
import typer

from typing_extensions import Annotated  # For Python 3.8+
from src.green_agent.agent import start_green_agent
from src.white_agent.agent import start_white_agent
from launcher import run_evaluation

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
        "Qwen/Qwen2.5-14B-Instruct-GPTQ-Int4", help="LLM model for the white agent"
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
    agent: str = typer.Option("llm", help='Agent mode: "llm" or "hardcoded"'),
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
      - llm: LLM-based white agent (Qwen2.5-14B via vLLM, FP8 quantized) - variable reasoning quality
      - hardcoded: Pre-recorded trajectories with configurable profile - see below

    Hardcoded profiles (--profile, only for --agent hardcoded):
      Profile calibration targets (relative to ground truth optimal steps):
      - oracle:      1.0x  (uses ground truth actions - ceiling performance)
      - expert:      1.5-2.0x  (optimal search priorities, high reasoning)
      - competent:   2.5-3.5x  (good priorities, medium reasoning)
      - overthinker: 3.0-4.0x  (good priorities, extra verification, high reasoning)
      - lucky_guesser: 1.5-2.0x steps, low reasoning
      - novice:      4.0-6.0x  (random priorities, many inefficiencies, low reasoning)

    Examples:
      python main.py evaluate --tasks 4 --agent llm
      python main.py evaluate --tasks 4 --agent hardcoded --profile expert
      python main.py evaluate --tasks 4 --agent hardcoded --profile oracle
    """
    # Validate agent mode
    valid_modes = ["llm", "hardcoded"]
    if agent.lower() not in valid_modes:
        raise typer.BadParameter(
            f"Invalid agent mode '{agent}'. Must be one of: {valid_modes}"
        )

    # Validate profile
    valid_profiles = ["oracle", "expert", "competent", "novice", "lucky_guesser", "overthinker"]
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


if __name__ == "__main__":
    app()
