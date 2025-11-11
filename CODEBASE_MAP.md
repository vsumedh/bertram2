# TextWorld Agentify Codebase Map

Heard u guys like maps? 

## Overview

The TextWorld agentify framework is an evaluation system for assessing AI agent performance in TextWorld household tasks. It uses the A2A (Agent-to-Agent) protocol to coordinate two agents:
- **Green Agent**: Manages the TextWorld environment and evaluates performance
- **White Agent**: The agent under test that uses an LLM to make decisions

## Directory Structure

```
textworld-agentify/
├── main.py                    # CLI entry point
├── launcher.py                # Evaluation workflow coordinator
├── requirements.txt           # Python dependencies
├── pyproject.toml             # Project configuration
├── README.md                  # Composer made dis but it looks good
├── EVALUATION_METHODOLOGY.md  # Composer made dis idk y
├── src/
│   ├── green_agent/          # Evaluation manager
│   ├── white_agent/          # Agent under test
│   └── utils/                # Shared utilities
└── __pycache__/              # Python bytecode cache
```

---

## Root-Level Files

### `main.py`
**Purpose**: CLI entry point providing three commands for running the evaluation system.

**Functionality**:
- **`green` command**: Starts the *green agent* (evaluation manager) as an HTTP server
  - Default port: 8722
  - Loads agent card configuration
  - Initializes the green agent executor
  
- **`white` command**: Starts the *white agent* (agent under test) as an HTTP server
  - Default port: 8724
  - Loads agent card configuration
  - Initializes the white agent executor

- **`launch` command**: Orchestrates the complete evaluation workflow
  - Accepts task configuration (task_index, max_steps)
  - Coordinates both agents via the launcher
  - Runs end-to-end evaluation

**Role in workflow**: Primary user interface for running evaluations. Provides flexible deployment options (agents separately or together).

---

### `launcher.py`
**Purpose**: Coordinates the complete evaluation workflow by managing agent processes and communication.

**Functionality**:
- **Process Management**: Spawns green and white agents as separate processes using the `multiprocessing` library.
- **Agent Readiness**: Checks to make sure that both agents are ready before starting evaluation.
- **Task Configuration**: Packages task configuration and white agent URL into structured payload
- **Message Dispatch**: Sends initial task payload to green agent via A2A protocol
- **Cleanup**: Terminates agent processes and closes connections after evaluation completes

**Key Components**:
- `launch_evaluation()`: Main async function that orchestrates the workflow
- `_ensure_agent_ready()`: Helper to wait for agent to become available
- Timeout calculation based on max_steps (12 seconds per step, minimum 10 minutes)

**Role in workflow**: Central coordinator that manages the evaluation lifecycle from agent startup to completion.

---

### `requirements.txt` & `pyproject.toml`
**Purpose**: Dependency management for the project.

**Key Dependencies**:
- `a2a-sdk[http-server]`: A2A protocol SDK with HTTP server support
- `alfworld`: ALFWorld environment for TextWorld tasks
- `litellm`: LLM API abstraction (supports OpenAI, etc.)
- `uvicorn`: ASGI server for running HTTP agents
- `typer`: CLI framework
- `python-dotenv`: Environment variable management
- `httpx`: Async HTTP client

---

## Source Directory (`src/`)

### `src/green_agent/` - Evaluation Manager

The green agent is responsible for:
1. Managing the TextWorld environment
2. Coordinating with the white agent
3. Collecting trajectory data (sequence of white agent's observations, actions, and reasoning)
4. Evaluating performance using LLM-as-a-judge

#### `agent.py`
**Purpose**: Core implementation of the green agent executor.

**Key Components**:

- **`TextWorldGreenAgentExecutor`**: Main executor class implementing the A2A agent interface
  - **Initialization**: Loads evaluation rubric and initializes LLM judge evaluator
  - **`execute()`**: Main evaluation loop that:
    1. Parses task configuration from incoming request
    2. Sets up TextWorld environment
    3. Runs interactive episode loop:
       - Sends observations/goals to white agent
       - Receives commands with reasoning
       - Executes commands in environment
       - Tracks trajectory with rich metadata
    4. Performs post-episode evaluation using LLM judge
    5. Generates comprehensive evaluation report

- **Message Formatting**:
  - `_format_initial_message()`: Formats first message with goal and initial observation
  - `_format_observation_message()`: Formats subsequent observation updates
  - Both include available command templates

- **Command Parsing**:
  - `_parse_command()`: Extracts reasoning and command from white agent responses
  - Uses XML-style tag parsing (`<reasoning>`, `<command>`)

- **Evaluation Methods** (fallback if rubric not loaded):
  - `_rate_trajectory_quick()`: Fast numeric rating (1-10)
  - `_rate_trajectory_detailed()`: Detailed rating with reasoning

- **Report Generation**:
  - `_generate_enhanced_report()`: Creates structured evaluation report with:
    - Success status and step counts
    - Quick and overall ratings
    - Per-criterion scores
    - Qualitative assessments (strengths, weaknesses, recommendations)
    - Reasoning trace analysis

- **`start_green_agent()`**: Factory function that:
  - Loads agent card from TOML
  - Creates HTTP server using uvicorn
  - Registers executor with A2A application

**Role in workflow**: Orchestrates the evaluation episode, manages environment state, and coordinates with white agent.

---

#### `evaluator.py`
**Purpose**: LLM-as-a-judge evaluator using configurable rubric.

**Key Components**:

- **`LLMJudgeEvaluator`**: Main evaluator class
  - **`rate_trajectory_quick()`**: Fast numeric rating (1-10) without detailed reasoning
    - Uses temperature=0.0 for consistency
    - Parses numeric rating from LLM response
    - Falls back to heuristic if LLM call fails
    
  - **`rate_trajectory_detailed()`**: Comprehensive structured evaluation
    - Builds prompt from rubric criteria
    - Requests structured output with category ratings
    - Parses overall rating, per-criterion scores, qualitative assessments
    - Returns `ScoreBreakdown` with weighted composite score
    
  - **`assess_reasoning_traces()`**: Analyzes reasoning quality
    - Evaluates reasoning coherence and usefulness
    - Assesses planning evidence and strategic thinking
    - Examines error handling patterns
    - Returns `ReasoningTraceAssessment`

- **Data Classes**:
  - **`CategoryRating`**: Single criterion score with weight
  - **`ScoreBreakdown`**: Comprehensive evaluation results
    - Overall rating and weighted composite
    - Per-criterion ratings
    - Qualitative assessments (strengths, weaknesses, etc.)
    - Detailed reasoning
  - **`ReasoningTraceAssessment`**: Analysis of reasoning traces

- **Trajectory Formatting**:
  - `_format_trajectory()`: Formats trajectory for LLM evaluation
  - `_format_trajectory_with_reasoning()`: Emphasizes reasoning traces

- **Response Parsing**:
  - `_parse_overall_rating()`: Extracts overall score
  - `_parse_category_ratings()`: Extracts per-criterion scores
  - `_parse_qualitative_assessments()`: Extracts bullet lists (strengths, weaknesses, etc.)
  - `_parse_detailed_reasoning()`: Extracts detailed analysis text

**Role in workflow**: Provides sophisticated LLM-based evaluation of agent trajectories with structured scoring.

---

#### `rubric.py`
**Purpose**: Evaluation rubric configuration and loading.

**Key Components**:

- **`EvaluationRubric`**: Main rubric dataclass
  - Contains criteria, scoring scale, qualitative assessment settings
  - Model settings for different evaluation types
  - Validation logic to ensure rubric is well-formed
  
- **`CriterionConfig`**: Configuration for individual evaluation criteria
  - Name, weight, description
  - Scoring guidelines (high/medium/low)
  - Examples and prompt templates
  
- **`ScoringScale`**: Configurable scoring range (default: 1-10)
  
- **`ModelSettings`**: LLM model and temperature settings per evaluation type

- **Loading Functions**:
  - `load_rubric()`: Main entry point with fallback logic:
    1. Explicit path if provided
    2. `TEXTWORLD_EVALUATION_RUBRIC_PATH` environment variable
    3. Default: `src/green_agent/evaluation_rubric.json`
  - `load_rubric_from_json()`: Loads JSON-format rubric
  - `load_rubric_from_toml()`: Loads TOML-format rubric
  - Validation ensures weights sum to ~1.0 and all required fields present

**Role in workflow**: Provides configurable evaluation standards that guide LLM judge assessments.

---

#### `evaluation_rubric.json`
**Purpose**: Default evaluation rubric configuration.

**Structure**:
- Scoring scale: 1-10 (configurable)
- Five evaluation criteria with weights:
  1. **Task Completion** (40%): Success in achieving goal
  2. **Efficiency** (30%): Minimal unnecessary actions
  3. **Strategy Quality** (20%): Logical planning and subgoal decomposition
  4. **Execution Quality** (5%): Valid commands and syntax
  5. **Reasoning Quality** (5%): Coherent reasoning traces
- Qualitative assessments: Enabled with strengths, weaknesses, notable behaviors, recommendations
- Reasoning trace analysis: Enabled with focus on reasoning quality, planning evidence, error handling
- Model settings: Configures LLM models and temperatures for different evaluation types

**Role in workflow**: Default configuration that can be customized per project or task type.

---

#### `task_list.txt`
**Purpose**: Index of available TextWorld tasks for evaluation.

**Format**: Pipe-delimited lines:
```
game_file_path|split|task_type|difficulty|description
```

**Fields**:
- `game_file_path`: Relative path from ALFWorld data directory
- `split`: Dataset split (valid_unseen, valid_seen, train, etc.)
- `task_type`: ALFWorld task category (1-6)
- `difficulty`: Estimated difficulty level
- `description`: Human-readable task description

**Role in workflow**: Task registry that green agent uses to load specific evaluation scenarios.

---

#### `agent_card.toml`
**Purpose**: A2A agent card configuration for green agent.

**Contains**: Agent metadata (name, description, URL, capabilities, skills) for A2A protocol registration.

---

### `src/white_agent/` - Agent Under Test

The white agent is the agent being evaluated. It uses an LLM to make decisions in TextWorld.

#### `agent.py`
**Purpose**: Core implementation of the white agent executor.

**Key Components**:

- **`TextWorldWhiteAgentExecutor`**: Main executor class
  - **Conversation Management**: Maintains per-context message history
  - **`execute()`**: Processes incoming messages:
    - Detects if message contains gameplay (observation/goal) or generic content
    - Routes to appropriate handler
    - Generates response using LLM
    - Returns response via A2A protocol
    
  - **`_generate_command()`**: Generates TextWorld commands
    - Adds system message on first turn with task instructions
    - Maintains conversation history with LLM
    - Ensures response includes `<reasoning>` and `<command>` tags
    - Fallback formatting if tags missing
    
- **`start_white_agent()`**: Factory function that:
  - Loads or generates agent card
  - Creates HTTP server using uvicorn
  - Registers executor with A2A application

**LLM Integration**:
- Uses `litellm` with `openai/gpt-4o` model
- Temperature: 0.0 for consistent command generation
- System prompt provides TextWorld action guidelines
- Ensures responses are properly formatted with reasoning and command tags

**Role in workflow**: Implements the agent under test that receives observations and generates actions.

---

#### `agent_card.toml`
**Purpose**: A2A agent card configuration for white agent.

---

### `src/utils/` - Shared Utilities

Shared utilities used by both green and white agents.

#### `textworld_env.py`
**Purpose**: TextWorld environment wrapper around ALFWorld.

**Key Components**:

- **`TextWorldEnvironment`**: Main environment wrapper class
  - **Initialization**: Takes `TaskConfig` with task index and max steps
  - **`setup()`**: 
    - Loads task metadata from task list
    - Constructs game file path from ALFWorld data
    - Initializes ALFWorld environment
    - Extracts goal and initial observation
    - Returns setup payload
    
  - **`step()`**: Executes action in environment
    - Sanitizes action string
    - Calls ALFWorld environment step
    - Updates internal state (observation, reward, done flag)
    - Returns `StepResult` with observation, reward, done, metadata
    
  - **`metrics()`**: Returns episode metrics
    - Success status, step count, cumulative reward
    
  - **`reset()`**: Cleans up environment resources

- **`TaskConfig`**: Configuration dataclass
  - Task index, max steps, optional task list path
  - Parses from JSON payload
  
- **`StepResult`**: Result dataclass for environment steps

- **`load_task_from_list()`**: Loads task metadata from task list file
  - Parses pipe-delimited format
  - Validates task index and format

- **`TEXTWORLD_ENV_CFG`**: Default ALFWorld configuration
  - Specifies data paths, task types, PDDL domain
  - Configures environment settings

**Role in workflow**: Abstraction layer over ALFWorld that provides clean interface for green agent to manage TextWorld episodes.

---

#### `a2a_client.py`
**Purpose**: A2A protocol client utilities for inter-agent communication.

**Key Components**:

- **`A2AMessenger`**: Utility wrapper for A2A client operations
  - **Caching**: Caches agent cards and clients to avoid repeated resolution
  - **`send_text()`**: Sends text message to agent
    - Resolves agent card if needed
    - Creates message with proper A2A structure
    - Handles errors and returns response
    
  - **`stream_text()`**: Streams messages to agent (for future use)
  
  - **`wait_agent_ready()`**: Polls agent card endpoint to check readiness
    - Used by launcher to ensure agents are available before starting

- **HTTP Client Management**:
  - Uses `httpx.AsyncClient` for async HTTP requests
  - Configurable timeout
  - Proper cleanup via `aclose()`

**Role in workflow**: Provides reliable communication between green and white agents using A2A protocol.

---

#### `messaging.py`
**Purpose**: Message parsing and sanitization utilities.

**Functions**:

- **`parse_tags()`**: Parses XML-style tags from text
  - Extracts `<tag>content</tag>` pairs
  - Returns dictionary mapping tag names to content
  - Used to extract reasoning, commands, and other structured data
  
- **`sanitize_action()`**: Normalizes LLM responses to TextWorld commands
  - Removes markdown code blocks
  - Extracts first valid command line
  - Filters out comments and formatting
  - Ensures single-line command suitable for TextWorld

**Role in workflow**: Utilities for parsing structured messages and cleaning agent outputs.

---

## Evaluation Workflow

### Complete Evaluation Flow

1. **Launch Phase** (`launcher.py`):
   - Spawn green and white agents as separate processes
   - Wait for both to become ready
   - Package task configuration

2. **Task Initialization** (`green_agent/agent.py`):
   - Green agent receives task configuration
   - Loads task metadata from task list
   - Sets up TextWorld environment
   - Extracts goal and initial observation

3. **Episode Loop** (`green_agent/agent.py`):
   - Green agent formats message with observation/goal
   - Sends to white agent via A2A protocol
   - White agent generates reasoning and command using LLM
   - Green agent parses response and executes command
   - Environment returns observation and reward
   - Trajectory data collected with rich metadata
   - Loop continues until done or max steps reached

4. **Evaluation Phase** (`green_agent/evaluator.py`):
   - LLM judge performs quick rating
   - LLM judge performs detailed evaluation:
     - Overall rating
     - Per-criterion scores (weighted)
     - Qualitative assessments
     - Reasoning trace analysis (if enabled)

5. **Report Generation** (`green_agent/agent.py`):
   - Generates structured evaluation report
   - Includes all metrics and assessments
   - Returns to user via A2A protocol

6. **Cleanup** (`launcher.py`):
   - Terminates agent processes
   - Closes connections
   - Resets environment

---

## Key Design Patterns

### Agent-to-Agent Communication
- Both agents expose HTTP endpoints via A2A protocol
- Green agent acts as client when communicating with white agent
- Uses context IDs for conversation continuity

### Error Handling
- White agent communication failures tracked with consecutive failure counter
- Episode terminates early if white agent consistently unresponsive
- Environment errors propagate with clear error messages
- LLM evaluation failures fall back to heuristic scoring

### Trajectory Tracking
- Rich metadata collected per step:
  - Reasoning traces for analysis
  - Observation changes to detect progress
  - Action validity and errors
  - Step timing for performance analysis

### Configuration Management
- Rubric configurable via JSON/TOML with fallback chain
- Task list specifies available evaluation scenarios
- Environment variables for data paths and API keys

---

## Extension Points

### Custom Evaluation Rubrics
Modify `evaluation_rubric.json` to:
- Adjust criterion weights
- Add/remove criteria
- Customize scoring guidelines
- Change LLM models or temperatures

### Custom Task Lists
Modify `task_list.txt` to:
- Add new evaluation scenarios
- Filter by task type or difficulty
- Organize by dataset split

### Custom White Agents
Replace `white_agent/agent.py` to:
- Use different LLM models
- Implement custom decision-making logic
- Add memory or planning capabilities
- Integrate external tools or APIs

---

## Dependencies and Data Requirements

### Required Dependencies
- ALFWorld data: Game files, PDDL domain, grammar files
- LLM API access: OpenAI API key (or other via litellm)
- Python packages: See `requirements.txt`

### Environment Variables
- `OPENAI_API_KEY`: LLM API key for evaluation and agent decisions
- `ALFWORLD_DATA`: Path to ALFWorld data directory (default: `~/.cache/alfworld`)
- `TEXTWORLD_EVALUATION_RUBRIC_PATH`: Optional custom rubric path

---

This codebase provides a complete, extensible framework for evaluating AI agents in TextWorld using standardized protocols and LLM-based assessment.
