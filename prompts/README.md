# Agent Prompts Directory

This directory contains YAML configuration files that define the behavior of all agents in the system.

## Directory Structure

```
prompts/
├── supervisor/          # Supervisor agent prompts
├── specialized/         # Specialized agent prompts
├── shared/             # Shared instructions and templates
└── schemas/            # JSON schemas for validation
```

## Prompt File Format

All prompt files follow this structure (validated by `schemas/prompt_schema.json`):

```yaml
agent:
  name: "Agent Name"
  version: "1.0.0"
  description: "What this agent does"

system_prompt:
  role: |
    The agent's role description...

  capabilities:
    - "Capability 1"
    - "Capability 2"

  constraints:
    - "Limitation 1"
    - "Rule 2"

  tone: "professional and helpful"

  instructions: |
    Step-by-step instructions for the agent...

tools:
  - name: "tool_name"
    description: "What the tool does"
    requires_approval: true

interaction_flow:
  step_1:
    message: "Message to user"
    action: "What the agent does"

approval_messages:
  tool_name: "Approval request template"

error_messages:
  error_type: "Error message template"
```

## Creating a New Agent

1. Copy `prompts/template_agent.yaml` to `prompts/specialized/your_agent.yaml`
2. Fill in all required fields
3. Validate: `python scripts/validate_prompts.py`
4. Create corresponding agent class extending `BaseAgent`

## Validation

All prompts are validated against the JSON schema on startup. Use the validation script:

```bash
python scripts/validate_prompts.py
```

## Best Practices

- Keep instructions clear and concise
- Use examples for complex interactions
- Always specify which tools require approval
- Include error messages for common failure cases
- Use `|` for multi-line strings in YAML
