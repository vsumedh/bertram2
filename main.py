"""CLI entry point for TextWorld agentify evaluation."""

import asyncio
import typer

from typing_extensions import Annotated # For Python 3.8+
from src.green_agent.agent import start_green_agent
from src.white_agent.agent import start_white_agent
from launcher import launch_evaluation

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
    host: str = "0.0.0.0",
    port: int = 8724,
):
    """Start the white agent (agent under test)."""
    start_white_agent(agent_name=agent_name, host=host, port=port)


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


if __name__ == "__main__":
    app()
