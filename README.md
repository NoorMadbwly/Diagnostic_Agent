# Diagnostic Agent 


A tool-calling clinical decision support agent that turns structured OPG detection output (FDI-numbered findings from the YOLOv11x pipeline) into a validated diagnosis and treatment plan.

Unlike a single prompt-to-JSON call, this agent decides at runtime which tools it needs and calls them:

- `calculate_dmft` — WHO DMFT (Decayed-Missing-Filled Teeth) index from the detections
- `get_treatment_protocol` — clinic treatment protocol lookup per disease
- `assess_triage_priority` — overall case urgency/priority

## Why an agent instead of a prompt template

The model reasons over the findings and pulls exactly the data it needs from tools instead of having every number pre-computed and stuffed into the prompt. This keeps clinical logic (protocols, DMFT methodology) in code — auditable and testable — while letting the model handle synthesis, prioritization, and natural-language clinical notes.

## Reliability

- Input and output are both validated against Pydantic schemas (`DetectionInput`, `Diagnosis`)
- Malformed model output triggers an automatic retry with the error fed back to the model, up to a configurable limit
- Failures raise a clear `DiagnosticAgentError` instead of failing silently

## Usage

```python
from diagnostic_agent import run_diagnostic_agent

diagnosis = run_diagnostic_agent(detections_json)
print(diagnosis.model_dump_json(indent=2))
```

`detections_json` matches the `/analyze/opg` FastAPI response shape: `teeth_detected`, `missing_teeth`, `diseases`, `blockers`, `modifiers`, `orthodontic_ready`, `reason`, `quadrants`.

## Stack

Claude API (tool use), Pydantic, Python
