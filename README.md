# TextWorld Agentify Evaluation Framework

A lean evaluation framework for assessing agent performance in TextWorld using the A2A (Agent-to-Agent) protocol.

## Overview

This framework evaluates white agents' ability to complete household tasks in TextWorld. The green agent manages the environment and evaluates performance, while the white agent makes decisions using either an LLM or hardcoded trajectories.

## Architecture

- **Green Agent**: Manages TextWorld environment, tracks trajectory, rates performance using LLM-as-a-judge
- **White Agent**: Generates commands based on observations. Two modes available:
  - **LLM mode**: Uses Qwen2.5-14B-Instruct via vLLM server (FP8 quantized)
  - **Hardcoded mode**: Pre-recorded trajectories with configurable reasoning profiles

## Installation

```bash
pip install -r requirements.txt
```

Or using `uv`:
```bash
uv pip install -r requirements.txt
```

### Prerequisites

1. **ALFWorld data**: Game files, PDDL logic, and expert plans must be available:
   ```bash
   export ALFWORLD_DATA=~/.cache/alfworld  # Default location
   ```

2. **vLLM server** (required for LLM white agent mode):
   ```bash
   # Start vLLM server with Qwen2.5-14B-GPTQ-Int4 (pre-quantized, ~9GB VRAM)
   ./scripts/start_vllm_server.sh
   # Or manually:
   python -m vllm.entrypoints.openai.api_server \
       --model Qwen/Qwen2.5-14B-Instruct-GPTQ-Int4 \
       --port 8000 \
       --quantization gptq \
       --enable-prefix-caching
   ```
   
   **Note**: Uses pre-quantized GPTQ-Int4 model (~9GB VRAM, ~93% quality).
   For larger GPUs (>28GB), you can use FP8: `--model Qwen/Qwen2.5-14B-Instruct --quantization fp8`

The evaluation judge also uses the same vLLM server - no separate OpenAI API key required.

## Usage

### Quick Start: Run Evaluation

The primary command is `evaluate`, which handles all agent orchestration:

```bash
# Run single task with LLM white agent (requires vLLM server)
python main.py evaluate --tasks 0 --agent llm

# Run single task with hardcoded agent
python main.py evaluate --tasks 0 --agent hardcoded --profile expert

# Run multiple tasks
python main.py evaluate --tasks 0,1,2,3 --agent hardcoded

# Run all 20 tasks
python main.py evaluate --tasks all --agent hardcoded --profile expert
```

### Agent Modes

| Mode | Description |
|------|-------------|
| `--agent llm` | LLM-based agent using Qwen2.5-14B via vLLM (FP8 quantized) |
| `--agent hardcoded` | Pre-recorded trajectories with synthetic reasoning |

**Note**: Both modes require a vLLM server running because the evaluation judge uses it for scoring.

### Hardcoded Agent Profiles

When using `--agent hardcoded`, you can select different reasoning quality profiles:

| Profile | Strategy | Reasoning | Expected Score |
|---------|----------|-----------|----------------|
| `expert` | Optimal | High quality | ~9/10 |
| `competent` | Suboptimal | Medium quality | ~7.5/10 |
| `novice` | Poor | Low quality | ~5/10 |
| `lucky_guesser` | Optimal | Low quality | ~7/10 |
| `overthinker` | Suboptimal | High quality | ~8/10 |

```bash
# Example: Test with novice profile
python main.py evaluate --tasks 4 --agent hardcoded --profile novice
```

### Full Options

```bash
python main.py evaluate --help
```

Key options:
- `--tasks`: Task indices (`"0"`, `"0,1,2"`, or `"all"`)
- `--agent`: Agent mode (`"llm"` or `"hardcoded"`)
- `--profile`: Hardcoded profile (expert/competent/novice/lucky_guesser/overthinker)
- `--max-steps`: Maximum steps per task (default: 50)
- `--verbose` / `-v`: Show step-by-step details
- `--green-port`: Green agent port (default: 8722)
- `--white-port`: White agent port (default: 8724)

### Demo Mode

For formatted terminal output with colors:
```bash
DEMO_MODE=1 python main.py evaluate --tasks 0 --agent hardcoded --profile expert
```

### Start Agents Separately

For manual orchestration or debugging:

Green agent (evaluation manager):
```bash
python main.py green --host 0.0.0.0 --port 8722
```

White agent (LLM-based):
```bash
python main.py white --host 0.0.0.0 --port 8723
```

### Using Launcher Directly

```bash
python launcher.py  # Runs task 0 with default LLM agent
```

## Evaluation Metrics

The framework uses an LLM-as-a-judge approach to evaluate agent performance. After each episode, the LLM judge provides structured assessments across multiple dimensions.

### Quantitative Metrics

- **Success**: Whether task was completed (0 or 1)
- **Step Count**: Number of steps taken vs budget
- **Quick Rating**: Lightweight LLM rating (1-10) without detailed explanation
- **Overall Rating**: Weighted composite score (1-10) based on multiple criteria

### Per-Criterion Scores

| Criterion | Weight | Description |
|-----------|--------|-------------|
| Task Completion | 30% | Whether goal was achieved correctly |
| Efficiency | 20% | Steps used relative to budget, wasted actions |
| Strategy Quality | 25% | Sensible action sequence, prioritization, adaptation |
| Reasoning Quality | 25% | Goal-awareness, observation grounding, coherence |

### Qualitative Assessments

The LLM judge also provides:
- **Strengths**: What the agent did well
- **Weaknesses**: Areas for improvement
- **Notable Behaviors**: Interesting or unexpected patterns
- **Recommendations**: Suggestions for improvement
- **Reasoning Trace Analysis**: Assessment of goal-awareness, observation grounding, and adaptation

### Rubric Configuration

The evaluation rubric is configurable via `src/green_agent/evaluation_rubric.json`. You can:
- Adjust criterion weights
- Modify scoring guidelines
- Customize prompt templates
- Enable/disable qualitative assessments

The framework looks for the rubric in this order:
1. Path provided to `TextWorldGreenAgentExecutor.__init__()`
2. `TEXTWORLD_EVALUATION_RUBRIC_PATH` environment variable
3. Default: `src/green_agent/evaluation_rubric.json`

## Project Structure

```
textworld-agentify/
├── main.py                # CLI entry point (green, white, evaluate commands)
├── launcher.py            # Evaluation orchestration logic
├── requirements.txt       # Dependencies
├── scripts/
│   └── start_vllm_server.sh  # vLLM server startup script
├── src/
│   ├── green_agent/       # Evaluation manager
│   │   ├── agent.py       # Green agent executor
│   │   ├── episode_runner.py  # Episode execution
│   │   ├── evaluator.py   # LLM judge implementation
│   │   ├── green_assessor.py  # Trajectory assessment
│   │   ├── evaluation_rubric.json  # Scoring criteria
│   │   └── task_list.txt  # Task definitions
│   ├── white_agent/       # Agent under test
│   │   ├── agent.py       # LLM-based white agent
│   │   └── agent_hardcoded.py  # Hardcoded trajectory agent
│   └── utils/             # Shared utilities
│       ├── a2a_client.py  # A2A protocol client
│       ├── textworld_env.py  # ALFWorld environment wrapper
│       └── vllm_client.py # vLLM API client
└── config/
    └── vllm_server.yaml   # vLLM server configuration (Qwen2.5-14B GPTQ-Int4)
```

## Task Configuration

Tasks are defined in `src/green_agent/task_list.txt` with format:
```
game_path|split|task_type|difficulty|description
```

Example:
```
pick_clean_then_place_in_recep-Tomato-None-SinkBasin-8/trial_T20190909_024521_815140|valid_seen|pick_clean_then_place|medium|Clean tomato and place in sink
```

## Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `ALFWORLD_DATA` | Path to ALFWorld data | `~/.cache/alfworld` |
| `VLLM_MODEL` | Model name for vLLM | `Qwen/Qwen2.5-14B-Instruct-GPTQ-Int4` |
| `VLLM_BASE_URL` | vLLM server URL | `http://localhost:8000/v1` |
| `VLLM_QUANTIZATION` | Quantization format (gptq/awq/fp8/none) | `gptq` |
| `VLLM_KV_CACHE_DTYPE` | KV cache data type | `auto` |
| `VLLM_TIMEOUT` | Request timeout in seconds | `180.0` |
| `DEMO_MODE` | Enable formatted output | `0` |
| `GREEN_VERBOSE` | Verbose green agent logging | Not set |

## A2A Protocol Integration

This framework uses the A2A (Agent-to-Agent) protocol for communication between green and white agents. Both agents expose HTTP endpoints and can be registered with A2A-compatible platforms.

For detailed instructions on integrating with the AgentBeats platform, see [AGENTBEATS_INTEGRATION.md](AGENTBEATS_INTEGRATION.md).

For future integration plans, see [NEXT_STEPS.md](NEXT_STEPS.md).
