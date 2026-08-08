"""
Dental AI — Diagnostic Agent (Claude API)
==========================================

Replaces the single-shot Gemini prompt-to-JSON call with a real
tool-calling agent:

    FastAPI /analyze/opg output  ->  Diagnostic Agent (Claude, with tools)
                                        |
                                        v
                             validated structured diagnosis
                                        |
                                        v
                              generate_pdf_report() (existing)

Design notes / assumptions (confirm these against your real dental_api.py):
  - Input matches the /analyze/opg response shape:
        teeth_detected, missing_teeth, diseases, blockers, modifiers,
        orthodontic_ready, reason, quadrants
  - Each item in `diseases` looks like:
        {"fdi": int, "disease": str, "confidence": float}
  - Model: claude-sonnet-5 (swap if you want haiku for cost, opus for depth)

If your actual schema differs, only INPUT PARSING (DetectionInput) and the
tool implementations need to change — the agent loop and validation layer
stay the same.
"""

from __future__ import annotations

import json
import time
from typing import Literal

import anthropic
from pydantic import BaseModel, Field, ValidationError

# --------------------------------------------------------------------------
# 1. Input schema — matches dental_api.py /analyze/opg response
# --------------------------------------------------------------------------

class Disease(BaseModel):
    fdi: int
    disease: Literal["Caries", "Deep Caries", "Periapical Lesion", "Impacted"]
    confidence: float


class DetectionInput(BaseModel):
    teeth_detected: list[int] = Field(default_factory=list)
    missing_teeth: list[int] = Field(default_factory=list)
    diseases: list[Disease] = Field(default_factory=list)
    blockers: list[str] = Field(default_factory=list)
    modifiers: list[str] = Field(default_factory=list)
    orthodontic_ready: bool = False
    reason: str = ""
    quadrants: dict = Field(default_factory=dict)


# --------------------------------------------------------------------------
# 2. Output schema — what the agent must ultimately produce
# --------------------------------------------------------------------------

class TreatmentPhase(BaseModel):
    phase: int
    title: str
    items: list[str]


class Diagnosis(BaseModel):
    health_status: str
    urgency: Literal["URGENT", "SOON", "ROUTINE"]
    total_findings: int
    dmft_index: float
    triage_priority: str
    summary: dict
    clinical_notes: str
    treatment_plan: list[TreatmentPhase]
    orthodontic_assessment: str


# --------------------------------------------------------------------------
# 3. Tools — the agent decides when to call these, it doesn't just get
#    handed pre-computed numbers in the prompt
# --------------------------------------------------------------------------

TREATMENT_PROTOCOLS = {
    "Deep Caries": "RCT + Crown",
    "Periapical Lesion": "RCT + Antibiotics",
    "Caries": "Composite restoration",
    "Impacted": "Surgical assessment — extraction or monitoring",
}

# WHO DMFT weighting: each affected tooth counts once, missing teeth count too
DMFT_DISEASE_WEIGHT = 1.0


def tool_calculate_dmft(detections: DetectionInput) -> dict:
    """WHO DMFT index: Decayed + Missing + Filled Teeth."""
    decayed_fdi = {d.fdi for d in detections.diseases if "Caries" in d.disease}
    missing = len(detections.missing_teeth)
    decayed = len(decayed_fdi)
    # "Filled" isn't detectable from X-ray findings alone in this pipeline —
    # flagged explicitly rather than silently assumed as 0.
    filled = 0
    dmft = decayed + missing + filled
    return {
        "dmft_index": dmft,
        "decayed": decayed,
        "missing": missing,
        "filled": filled,
        "note": "filled=0 is a placeholder — this pipeline has no filling-detection class yet",
    }


def tool_get_treatment_protocol(disease: str) -> dict:
    protocol = TREATMENT_PROTOCOLS.get(disease)
    if protocol is None:
        return {"disease": disease, "protocol": "Consult specialist — unrecognized finding"}
    return {"disease": disease, "protocol": protocol}


def tool_assess_triage_priority(detections: DetectionInput) -> dict:
    """Rank overall urgency from the raw findings."""
    diseases = [d.disease for d in detections.diseases]
    if "Periapical Lesion" in diseases or "Deep Caries" in diseases:
        priority, urgency = "HIGH", "URGENT"
    elif "Caries" in diseases or "Impacted" in diseases:
        priority, urgency = "MEDIUM", "SOON"
    else:
        priority, urgency = "LOW", "ROUTINE"
    return {"triage_priority": priority, "urgency": urgency}


TOOLS = [
    {
        "name": "calculate_dmft",
        "description": "Calculate the WHO DMFT (Decayed-Missing-Filled Teeth) index from the detection data.",
        "input_schema": {"type": "object", "properties": {}, "required": []},
    },
    {
        "name": "get_treatment_protocol",
        "description": "Look up the standard treatment protocol for a specific disease finding.",
        "input_schema": {
            "type": "object",
            "properties": {"disease": {"type": "string", "description": "One of: Caries, Deep Caries, Periapical Lesion, Impacted"}},
            "required": ["disease"],
        },
    },
    {
        "name": "assess_triage_priority",
        "description": "Assess overall case triage priority and urgency level from the findings.",
        "input_schema": {"type": "object", "properties": {}, "required": []},
    },
]

TOOL_IMPL = {
    "calculate_dmft": lambda args, det: tool_calculate_dmft(det),
    "get_treatment_protocol": lambda args, det: tool_get_treatment_protocol(args["disease"]),
    "assess_triage_priority": lambda args, det: tool_assess_triage_priority(det),
}


# --------------------------------------------------------------------------
# 4. System prompt
# --------------------------------------------------------------------------

SYSTEM_PROMPT = """You are a senior dental radiologist and clinical decision \
support AI reviewing structured findings from an automated panoramic X-ray \
(OPG) analysis pipeline (YOLOv11x quadrant + tooth enumeration + disease \
detection, FDI notation).

Use the available tools to compute the DMFT index, look up treatment \
protocols for each distinct disease found, and assess triage priority. Do \
NOT compute these yourself from memory — call the tools, since they reflect \
this clinic's actual protocols and WHO methodology.

Once you have gathered what you need from tools, respond with ONLY a single \
JSON object (no markdown fences, no commentary) matching this schema:

{
  "health_status": "string",
  "urgency": "URGENT | SOON | ROUTINE",
  "total_findings": integer,
  "dmft_index": number,
  "triage_priority": "string",
  "summary": {
    "caries_count": integer,
    "deep_caries_count": integer,
    "impacted_count": integer,
    "infection_count": integer
  },
  "clinical_notes": "string",
  "treatment_plan": [
    {"phase": integer, "title": "string", "items": ["string"]}
  ],
  "orthodontic_assessment": "string"
}
"""


# --------------------------------------------------------------------------
# 5. Agent loop — real tool-use turns, with retry + validation
# --------------------------------------------------------------------------

class DiagnosticAgentError(Exception):
    pass


def run_diagnostic_agent(
    detections: dict,
    api_key: str | None = None,
    model: str = "claude-sonnet-5",
    max_tool_turns: int = 6,
    max_retries: int = 2,
) -> Diagnosis:
    """
    Runs the full agent loop: Claude decides which tools to call, we execute
    them locally, feed results back, and repeat until Claude returns the
    final JSON diagnosis. Validated against the Diagnosis schema before
    returning — raises DiagnosticAgentError instead of handing back garbage.
    """
    try:
        det = DetectionInput(**detections)
    except ValidationError as e:
        raise DiagnosticAgentError(f"Input detections don't match expected schema: {e}") from e

    client = anthropic.Anthropic(api_key=api_key)  # falls back to ANTHROPIC_API_KEY env var

    findings_text = "\n".join(
        f"- Tooth {d.fdi}: {d.disease} (confidence {d.confidence:.1%})" for d in det.diseases
    ) or "No disease findings."

    user_content = (
        f"Findings:\n{findings_text}\n\n"
        f"Teeth detected: {det.teeth_detected}\n"
        f"Missing teeth: {det.missing_teeth}\n"
        f"Blockers: {det.blockers}\n"
        f"Modifiers: {det.modifiers}\n"
        f"Orthodontic ready (pre-agent flag): {det.orthodontic_ready} — reason: {det.reason}\n"
    )

    messages = [{"role": "user", "content": user_content}]

    last_error = None
    for attempt in range(max_retries + 1):
        try:
            for _ in range(max_tool_turns):
                response = client.messages.create(
                    model=model,
                    max_tokens=2000,
                    system=SYSTEM_PROMPT,
                    tools=TOOLS,
                    messages=messages,
                )

                if response.stop_reason == "tool_use":
                    tool_results = []
                    assistant_blocks = []
                    for block in response.content:
                        assistant_blocks.append(block)
                        if block.type == "tool_use":
                            fn = TOOL_IMPL.get(block.name)
                            if fn is None:
                                result = {"error": f"unknown tool {block.name}"}
                            else:
                                result = fn(block.input, det)
                            tool_results.append(
                                {
                                    "type": "tool_result",
                                    "tool_use_id": block.id,
                                    "content": json.dumps(result),
                                }
                            )
                    messages.append({"role": "assistant", "content": assistant_blocks})
                    messages.append({"role": "user", "content": tool_results})
                    continue

                # Final turn — expect JSON text
                text_blocks = [b.text for b in response.content if b.type == "text"]
                raw = "\n".join(text_blocks).strip()
                if raw.startswith("```"):
                    raw = raw.split("\n", 1)[1].rsplit("```", 1)[0]

                parsed = json.loads(raw)  # raises json.JSONDecodeError on malformed output
                return Diagnosis(**parsed)  # raises ValidationError if schema mismatch

            raise DiagnosticAgentError(f"Agent did not converge within {max_tool_turns} tool turns")

        except (json.JSONDecodeError, ValidationError, anthropic.APIError) as e:
            last_error = e
            if attempt < max_retries:
                time.sleep(1.5 * (attempt + 1))  # backoff before retry
                messages.append(
                    {
                        "role": "user",
                        "content": f"Your previous response was invalid ({e}). "
                        "Respond again with ONLY the corrected JSON object, no other text.",
                    }
                )
                continue
            raise DiagnosticAgentError(
                f"Diagnostic agent failed after {max_retries + 1} attempts: {last_error}"
            ) from last_error

    raise DiagnosticAgentError(f"Unreachable — last error: {last_error}")


# --------------------------------------------------------------------------
# 6. Example usage
# --------------------------------------------------------------------------

if __name__ == "__main__":
    sample_detections = {
        "teeth_detected": [11, 12, 13, 21, 22, 23, 46, 47],
        "missing_teeth": [16, 26],
        "diseases": [
            {"fdi": 46, "disease": "Deep Caries", "confidence": 0.91},
            {"fdi": 47, "disease": "Periapical Lesion", "confidence": 0.87},
            {"fdi": 12, "disease": "Caries", "confidence": 0.76},
        ],
        "blockers": [],
        "modifiers": [],
        "orthodontic_ready": False,
        "reason": "active infection present",
        "quadrants": {"1": "detected", "2": "detected", "3": "partial", "4": "detected"},
    }

    diagnosis = run_diagnostic_agent(sample_detections)
    print(diagnosis.model_dump_json(indent=2))
