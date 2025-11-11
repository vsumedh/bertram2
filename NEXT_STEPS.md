# Next Steps

This document outlines the rationale and context for key development priorities in the TextWorld agentify evaluation framework. Each item addresses gaps or opportunities to enhance the framework's capabilities, integration, and evaluation quality.

---

## AgentBeats Integration

### Context

The current TextWorld agentify framework operates as a standalone evaluation system. While it provides comprehensive LLM-as-a-judge evaluation for TextWorld tasks, it lacks integration with broader evaluation platforms that could enable standardized benchmarking, community sharing, and comparative analysis across different agent implementations.

AgentBeats is a platform for standardized, reproducible multi-agent evaluation that provides:
- **Agent Registry**: Centralized registration and discovery of agents
- **Battle Orchestration**: Infrastructure for coordinating multi-agent scenarios
- **Metrics Tracking**: Multi-level interaction tracking and performance analysis
- **Web-Based Visualization**: Frontend/backend for viewing evaluations and results
- **Standardized Reporting**: Consistent evaluation formats for cross-agent comparison

### Rationale

Integration with AgentBeats would enable several important capabilities:

1. **Standardized Benchmarking**: The framework could become part of a larger ecosystem where agents are evaluated using consistent protocols and metrics, making it easier to compare different approaches and track progress over time.

2. **Community Sharing**: Agents evaluated through this framework could be registered and shared with the research community, facilitating collaboration and reproducibility.

3. **Scalable Evaluation**: AgentBeats infrastructure could support batch evaluations, distributed testing, and large-scale comparative studies that would be difficult to orchestrate manually.

4. **Rich Visualization**: The web-based interface would provide better visualization of trajectories, metrics, and comparative analysis than command-line outputs.

5. **Platform Integration**: As a green agent in the AgentBeats ecosystem, this framework could evaluate various white agents registered on the platform, expanding the scope of agents that can be tested.

6. **Standardized Reporting**: Evaluation results would be formatted according to AgentBeats standards, making them compatible with other tools and platforms in the ecosystem.

### Current State

The framework currently:
- Uses A2A protocol (already compatible with AgentBeats)
- Exposes agents as HTTP servers (compatible with AgentBeats architecture)
- Generates evaluation reports (but in custom format)
- Operates independently without external registration or visualization

### Integration Points

The integration would need to bridge:
- Agent registration and discovery mechanisms
- Evaluation result formatting to AgentBeats standards
- Trajectory recording and storage for visualization
- Battle/orchestration lifecycle management
- Metrics aggregation and reporting pipelines

---

## Refined End-of-Episode Evaluation

### Context

The current evaluation framework uses LLM-as-a-judge to assess agent performance after each episode completes. While this provides rich qualitative and quantitative assessments, there are opportunities to refine the evaluation methodology to address limitations and enhance the reliability and usefulness of assessments.

### Current Limitations

The evaluation methodology documentation identifies several limitations:
- **LLM Variability**: Non-deterministic scoring due to temperature settings can lead to inconsistent results for identical trajectories
- **Subjectivity**: Criteria like strategy quality involve subjective judgment that may vary between evaluations
- **Prompt Sensitivity**: Evaluation quality depends heavily on prompt design and rubric clarity
- **Cost**: Each evaluation requires LLM API calls, which can be expensive for large-scale evaluation
- **Single-Episode Focus**: Current evaluation is per-episode; aggregation and statistical analysis across multiple episodes is manual

### Rationale

Refining the end-of-episode evaluation would address several needs:

1. **Reliability**: Implement mechanisms to reduce variability and increase consistency in scoring, such as multi-judge consensus, temperature calibration, or structured output validation.

2. **Statistical Rigor**: Add support for aggregating results across multiple episodes to compute meaningful statistics (mean, variance, confidence intervals) rather than relying on single-episode assessments.

3. **Cost Efficiency**: Develop strategies to reduce evaluation costs while maintaining quality, such as caching, sampling strategies, or using more efficient models for certain evaluation types.

4. **Comprehensive Metrics**: Expand beyond LLM-as-a-judge to include objective metrics that can be computed deterministically from trajectory data (e.g., path efficiency, action success rates, condition-level progress tracking).

5. **Evaluation Feedback Loops**: Enable real-time or incremental evaluation during episodes to provide feedback earlier, potentially improving agent performance mid-episode.

6. **Cross-Episode Analysis**: Support comparative analysis across episodes, agents, or task types to identify patterns, systematic issues, or improvement trends.

7. **Calibration**: Implement mechanisms to calibrate LLM judgments against ground truth or expert evaluations where available, improving reliability.

### Current State

The framework currently:
- Performs quick and detailed LLM-based evaluations post-episode
- Uses configurable rubrics with weighted criteria
- Provides qualitative assessments (strengths, weaknesses, recommendations)
- Analyzes reasoning traces when enabled
- Generates comprehensive reports per episode
- Has fallback mechanisms for LLM failures

### Areas for Refinement

Key areas that could benefit from refinement:
- Multi-judge consensus or averaging
- Statistical aggregation across episodes
- Objective metrics computation from trajectory data
- Evaluation caching and cost optimization
- Real-time or incremental evaluation during episodes
- Calibration against known baselines or expert evaluations
- Cross-episode comparative analysis

---

## Refined White Agent Scaffolding

### Context

The current white agent implementation is a minimal scaffold that uses an LLM (GPT-4o) to generate commands based on observations. While functional, it provides a basic foundation that could be enhanced with more sophisticated capabilities to enable more advanced agent architectures and evaluation scenarios.

### Current State

The white agent currently:
- Maintains simple conversation history per context
- Uses a fixed system prompt with TextWorld action guidelines
- Generates commands with optional reasoning traces
- Has basic error handling (fallback to "look" command)
- Uses a single LLM model (GPT-4o) with fixed temperature
- Has no explicit planning, memory, or tool-use capabilities beyond basic LLM prompting

### Rationale

Refining the white agent scaffolding would enable:

1. **Diverse Agent Architectures**: Support for different agent types (reactive, planning-based, memory-augmented) would allow evaluation of various approaches rather than just one simple LLM-based policy.

2. **Advanced Capabilities**: Enable agents with:
   - **Planning**: Multi-step planning and subgoal decomposition
   - **Memory**: Long-term memory across episodes or within-episode state tracking
   - **Tool Use**: Integration with external tools, APIs, or specialized modules
   - **Reflection**: Self-assessment and error correction capabilities

3. **Flexible Configuration**: Allow agents to be configured with different:
   - LLM models and parameters
   - Prompting strategies
   - Memory mechanisms
   - Planning approaches
   - Tool integrations

4. **Better Evaluation**: More sophisticated agents would provide richer trajectories for evaluation, including planning traces, memory states, and tool usage patterns, enabling more nuanced assessment of agent capabilities.

5. **Research Enablement**: A flexible scaffolding would allow researchers to easily experiment with different agent architectures without rebuilding the entire communication and evaluation infrastructure.

6. **Baseline Comparison**: Enhanced scaffolding would enable comparison between simple reactive agents and more sophisticated approaches, helping identify which capabilities contribute most to performance.

7. **Real-World Applicability**: Agents with planning, memory, and tool-use capabilities better reflect real-world agent requirements and would make evaluations more relevant to practical applications.

### Current Limitations

The minimal scaffolding currently constrains:
- Agent architecture to simple prompt-based LLM interaction
- Limited configurability (single model, fixed approach)
- No support for advanced capabilities (planning, memory, tools)
- Limited error recovery and adaptation mechanisms
- No hooks for integrating external systems or modules

### Refinement Areas

Key areas for scaffolding enhancement:
- Modular architecture supporting different agent types
- Configurable LLM models and parameters
- Planning and reasoning frameworks
- Memory and state management systems
- Tool integration interfaces
- Error handling and recovery mechanisms
- Reflection and self-assessment capabilities

---

## ALFWorld Integration

### Context

The framework currently uses ALFWorld's TextWorld environment (`AlfredTWEnv`) through a thin wrapper. While functional for basic text-based evaluation, there are opportunities to leverage more of ALFWorld's capabilities and better align with ALFWorld's design principles and evaluation methodologies.

### ALFWorld Capabilities

ALFWorld provides:
- **Dual Environments**: Both TextWorld (text-based) and THOR (embodied visual) environments, enabling evaluation of both text-only and embodied agents
- **PDDL Planning**: Built-in PDDL domain definitions and planning infrastructure for goal specification and verification
- **Task Specifications**: Structured task definitions with goal conditions that can be evaluated deterministically
- **Rich State Information**: Access to object attributes, relations, and world state that can be used for detailed metrics
- **Expert Agents**: Reference implementations for baseline comparison
- **Task Types**: Six distinct task categories (pick_and_place_simple, look_at_obj_in_light, pick_clean_then_place_in_recep, pick_heat_then_place_in_recep, pick_cool_then_place_in_recep, pick_two_obj_and_place)

### Current Integration State

The framework currently:
- Uses ALFWorld's `AlfredTWEnv` environment class (text-only)
- Loads task files from ALFWorld data directory structure
- Extracts basic goal descriptions from environment info
- Uses simple success/failure metrics from environment
- Has hardcoded paths and configuration that may not align with ALFWorld best practices
- Only supports text-based observations and commands

### Rationale

Deeper ALFWorld integration would enable:

1. **Visual Embodied Agent Support**: ALFWorld's THOR environment (`AlfredThorEnv`) provides visual observations as RGB images representing the agent's egocentric view of the scene. Enhanced integration would enable evaluation of embodied agents that:
   - Receive visual input (RGB images) representing their current perspective
   - Generate navigation actions (e.g., "MoveAhead", "RotateLeft", "RotateRight", "LookUp", "LookDown") to navigate the 3D environment
   - Interact with objects through visual grounding rather than textual descriptions
   - This would expand evaluation beyond text-only agents to vision-based embodied agents

2. **Goal Condition Evaluation**: ALFWorld tasks have structured PDDL goal specifications that can be evaluated deterministically. Currently, the framework relies on environment-provided success flags, but could leverage goal condition checking to:
   - Track partial progress toward goals
   - Identify which specific conditions were met/unmet
   - Provide more granular success metrics

3. **Objective Metrics**: ALFWorld's state information enables computation of standard metrics like:
   - **Success Rate (SR)**: Task completion percentage
   - **Goal Condition Progress (GC)**: Percentage of goal conditions satisfied
   - **SPL (Success weighted by Path Length)**: Efficiency metric
   - **Action Correctness Score (ACS)**: Valid action percentage
   - **Visual Grounding Accuracy (VGA)**: For embodied agents
   - **Hallucination Detection (H@O)**: Object reference accuracy

4. **Task Type Alignment**: Better integration with ALFWorld's six task types would enable:
   - Task-type-specific evaluation criteria
   - Task-type-specific metrics and analysis
   - Understanding performance variations across task categories

5. **PDDL State Tracking**: Leveraging ALFWorld's PDDL state representation would enable:
   - Deterministic goal condition evaluation
   - State-based progress tracking
   - Condition-level breakdown of agent performance

6. **Baseline Comparison**: Integration with ALFWorld's expert agents and evaluation infrastructure would enable comparison with established baselines and standardized evaluation protocols.

7. **Data Consistency**: Aligning with ALFWorld's data structures and conventions would ensure compatibility with ALFWorld tooling, datasets, and research that uses the same environment.

### Visual Embodied Agent Requirements

For embodied visual agents using ALFWorld's THOR environment:

- **Input Format**: The agent receives RGB images (typically 224×224 or 300×300 pixels) representing its egocentric first-person view of the 3D scene. These images capture the visual state of the environment, including objects, receptacles, and spatial relationships.

- **Output Format**: The agent generates navigation and interaction actions that describe how to navigate and manipulate the environment. Actions include:
  - **Navigation**: "MoveAhead", "RotateLeft", "RotateRight", "MoveBack", "LookUp", "LookDown"
  - **Interaction**: "PickupObject", "PutObject", "OpenObject", "CloseObject", "ToggleObject", "SliceObject"
  - **Examination**: "ExamineObject", "UseObject"
  - Actions may include parameters such as object identifiers, coordinates, or interaction types

- **Evaluation Challenges**: Visual agents require different evaluation considerations:
  - Visual grounding: Can the agent correctly identify objects from visual input?
  - Spatial reasoning: Can the agent navigate based on visual landmarks?
  - Action precision: Can the agent execute precise low-level actions vs. high-level commands?
  - Partial observability: Visual field of view limitations require exploration and memory

### Current Gaps

The integration currently misses:
- Direct goal condition evaluation (relies on environment success flags)
- ALFWorld-standard metrics computation
- Support for embodied/visual agents (THOR environment)
- Task-type-specific handling
- PDDL state tracking and condition-level analysis
- Alignment with ALFWorld evaluation conventions
- Visual observation handling (image input) and navigation action support

### Integration Opportunities

Key areas for deeper integration:
- Goal specification parsing and condition evaluation
- ALFWorld-standard metrics computation
- Support for both text (`AlfredTWEnv`) and visual (`AlfredThorEnv`) ALFWorld environments
- Visual observation processing (RGB images) and navigation action generation
- Task-type-specific evaluation and analysis
- PDDL state tracking and progress measurement
- Alignment with ALFWorld evaluation protocols and data formats
