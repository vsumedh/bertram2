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

Also make sure agentbeats is installed correctly.

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

### Running green and white agents evaluations locally using A2A protocols

We have full support for running agent evaluations locally, i.e. without using the agentbeats online platform. Green and white agents communicate correctly via the A2A protocol. 

Steps:
1. Ensure agentbeats is installed (e.g. via `pip install earthshaker`) and available locally
2. All requirements in `requirements.txt` are installed, including alfworld and the datasets
3. Run the assessments locally using the command `python3 launcher.py`

You should see the evaluation run as in the following screenshot:

<img width="1197" height="686" alt="Screenshot 2025-11-20 at 8 38 58 PM" src="https://github.com/user-attachments/assets/1d24ad75-531d-4e0b-a3b8-3e057e2ef008" />

Finally, evaluations are performed as shown in the screenshot below:

<img width="1115" height="63" alt="Screenshot 2025-11-20 at 8 45 30 PM" src="https://github.com/user-attachments/assets/2dbb7022-79d8-4ad6-bd45-0f73290375af" />


### Integrating with agentbeats

#### Run Green agent

```bash
conda activate earthshaker2
cd ~/dev/textworld-agentify/src/green_agent
HOST=ab.veenasumedh.com PORT=8011 agentbeats run_ctrl
```
<img width="1213" height="117" alt="run_green_agent" src="https://github.com/user-attachments/assets/df3dbbef-b24c-4cdc-86ed-628057bbcc00" />

This will run the green agent, with a controller url of http://ab.veenasumedh.com:8011/ which can be registered on v2.agentbeats.org

<img width="614" height="526" alt="register_green_agent" src="https://github.com/user-attachments/assets/246ab13a-acc2-4525-8a8d-15f5d49c9c6d" />

The agent card is visible from the status page, showing it is correctly registered:

<img width="1169" height="814" alt="green_agent_card" src="https://github.com/user-attachments/assets/554a89d6-ee24-42a4-a23a-f805177fd2e3" />


#### Run White agent

```bash
conda activate earthshaker2
cd ~/dev/textworld-agentify/src/white_agent
HOST=ab.veenasumedh.com PORT=8012 agentbeats run_ctrl
```
<img width="1211" height="113" alt="run_white_agent" src="https://github.com/user-attachments/assets/7eb846d2-3f2d-443c-bca4-b66811edf583" />

This will run the white agent, with a controller url of http://ab.veenasumedh.com:8012/ which can be registered on v2.agentbeats.org

<img width="614" height="518" alt="register_white_agent" src="https://github.com/user-attachments/assets/b0370dbd-fdc4-4646-804c-008475cf2702" />

The agent card is visible from the status page, showing it is correctly registered:

<img width="1186" height="828" alt="white_agent_card" src="https://github.com/user-attachments/assets/947bd88a-5201-4f43-a0be-01a724938d57" />


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

