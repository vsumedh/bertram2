# TextWorld Agentify Evaluation Framework

A lean evaluation framework for assessing agent performance in TextWorld using the A2A (Agent-to-Agent) protocol.

## Overview

This framework evaluates white agents' ability to complete household tasks in TextWorld. The green agent manages the environment and evaluates performance, while the white agent uses an LLM to make decisions.

## Architecture

- **Green Agent**: Manages TextWorld environment, tracks trajectory, rates performance
- **White Agent**: Uses LLM (GPT-4o) to generate commands based on observations

## Installation

```bash
pip install -r requirements.txt
```

Or using `uv`:
```bash
uv pip install -r requirements.txt
```

## Configuration

1. Set environment variables:
   ```bash
   export OPENAI_API_KEY=your-api-key
   export ALFWORLD_DATA=~/.cache/alfworld  # Optional, defaults to ~/.cache/alfworld
   ```

2. Ensure ALFWorld data is available (game files, PDDL, etc.)

3. Create/edit `src/green_agent/task_list.txt` with your tasks:
   ```
   game_path|split|task_type|difficulty|description
   ```

## Usage

### Full Evaluation (Recommended)

Launch both agents and run evaluation:
```bash
python main.py launch --task-index 0 --max-steps 30
```

### Start Agents Separately

Green agent (evaluation manager):
```bash
python main.py green --host 0.0.0.0 --port 9001
```

White agent (agent under test):
```bash
python main.py white --host 0.0.0.0 --port 9002
```

### Using Launcher Directly

```bash
python launcher.py
```

### Integrating with agentbeats

#### Run Green agent

```bash
conda activate earthshaker2
cd ~/dev/textworld-agentify/src/green_agent
HOST=ab.veenasumedh.com PORT=8011 agentbeats run_ctrl
```

This will run the green agent, with a controller url of http://ab.veenasumedh.com:8011/ which can be registered on v2.agentbeats.org

#### Run Green agent

```bash
conda activate earthshaker2
cd ~/dev/textworld-agentify/src/white_agent
HOST=ab.veenasumedh.com PORT=8012 agentbeats run_ctrl
```

This will run the white agent, with a controller url of http://ab.veenasumedh.com:8012/ which can be registered on v2.agentbeats.org


## Evaluation Metrics

The framework uses an LLM-as-a-judge approach to evaluate agent performance. After each episode, the LLM judge provides structured assessments across multiple dimensions.

### Quantitative Metrics

- **Success**: Whether task was completed (0 or 1)
- **Step Count**: Number of steps taken
- **Quick Rating**: Fast LLM rating (1-10) without detailed explanation
- **Overall Rating**: Weighted composite score (1-10) based on multiple criteria
- **Per-Criterion Scores**: Individual scores for:
  - Task Completion (40% weight)
  - Efficiency (30% weight)
  - Strategy Quality (20% weight)
  - Execution Quality (5% weight)
  - Reasoning Quality (5% weight)

### Qualitative Assessments

The LLM judge also provides:
- **Strengths**: What the agent did well
- **Weaknesses**: Areas for improvement
- **Notable Behaviors**: Interesting or unexpected patterns
- **Recommendations**: Suggestions for improvement
- **Reasoning Trace Analysis**: Assessment of reasoning quality, planning evidence, and error handling

### Detailed Documentation

For comprehensive information about the evaluation methodology, see [EVALUATION_METHODOLOGY.md](EVALUATION_METHODOLOGY.md).

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
├── src/
│   ├── green_agent/      # Evaluation manager
│   ├── white_agent/       # Agent under test
│   └── utils/             # Shared utilities
├── launcher.py            # Evaluation launcher
├── main.py                # CLI entry point
└── requirements.txt       # Dependencies
```

