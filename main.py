"""CLI entry point for TextWorld agentify evaluation."""

import asyncio
import typer

from typing_extensions import Annotated # For Python 3.8+
from src.green_agent.agent import start_green_agent
from src.white_agent.agent import start_white_agent
from launcher import launch_evaluation, benchmark_evaluation

app = typer.Typer(help="TextWorld Agentify - Agent evaluation framework")


@app.command()
def green(
    agent_name: str = "agent_card",
    host: Annotated[str, typer.Argument(envvar="HOST")] = "0.0.0.0",
    port: Annotated[int, typer.Argument(envvar="AGENT_PORT")] = 8722
):
    """Start the green agent (evaluation manager)."""
    start_green_agent(agent_name=agent_name, host=host, port=port)


@app.command()
def white(
    agent_name: str = "agent_card",
    host: Annotated[str, typer.Argument(envvar="HOST")] = "0.0.0.0",
    port: Annotated[int, typer.Argument(envvar="AGENT_PORT")] = 8723,
    model: str = typer.Option("openai/gpt-4o", help="LLM model for the white agent"),
    temperature: float = typer.Option(0.0, help="Sampling temperature"),
    prompt_profile: str = typer.Option("standard", help="Prompt profile: standard|concise"),
):
    """Start the white agent (agent under test)."""
    start_white_agent(
        agent_name=agent_name,
        host=host,
        port=port,
        model=model,
        temperature=temperature,
        prompt_profile=prompt_profile,
    )


@app.command()
def launch(
    task_index: int = 0,
    max_steps: int = 50,
    green_host: str = "127.0.0.1",
    green_port: int = 8722,
    white_host: str = "127.0.0.1",
    white_port: int = 8724,
):
    """Launch the complete evaluation workflow."""
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
        )
    )

@app.command()
def benchmark(
    tasks: str = typer.Option("4,6,5,7,1", help='Comma-separated task indices, e.g. "4,6,5,7,1"'),
    max_steps: int = typer.Option(50, help="Max steps per task"),
    green_host: str = "127.0.0.1",
    green_port: int = 8722,
    white_host: str = "127.0.0.1",
):
    """Run a small benchmark over multiple tasks and white-agent variants."""
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
        )
    )


if __name__ == "__main__":
    app()
