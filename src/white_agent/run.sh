#!/bin/bash
# Start white agent (LLM) for agentbeats integration
# AGENT_PORT is automatically set by `agentbeats run_ctrl` and read by the CLI

cd "$(dirname "$0")/../.."
exec python main.py white
