"""
PlantSense — Synthetic Maintenance Log Generator
"""

import json
import time
import getpass
import os
from pydantic import BaseModel, Field
from google import genai
from google.genai import types
from google.genai.errors import ClientError
from seed_logs import seed_logs

api_key = getpass.getpass("Enter your Gemini API key: ")
client = genai.Client(api_key=api_key)

MODEL = "gemini-3.1-flash-lite"  # more generous free tier (15 RPM) than gemini-3.5-flash
                                   # (5 RPM) — plenty for structured, low-reasoning generation
                                   # like this. Check ai.google.dev/gemini-api/docs/rate-limits
                                   # if quotas shift again.

GENERATION_PLAN = {
    "bearing_lubrication": 22,
    "fan": 20,
    "seals_other": 16,
    "hpc": 12,
    "lpt_hpt": 12,
    "edge_case": 4,
}

# Free tier: 15 RPM for gemini-3.1-flash-lite (vs only 5 RPM on 3.5-flash).
# 7s gives comfortable margin, accounting for the SDK's own internal retries.
SECONDS_BETWEEN_CALLS = 7
CHECKPOINT_PATH = "maintenance_logs_checkpoint.json"


class MaintenanceLog(BaseModel):
    log_id: str = Field(description="Unique log ID following the ML-XXXX numbering pattern")
    unit: str = Field(description="Engine unit number between 01 and 100")
    date: str = Field(description="Date string across 2026")
    reported_issue: str
    root_cause: str = Field(description="Specific, physically plausible mechanical/thermal mechanism")
    action_taken: str = Field(description="Concrete corrective action with an implied outcome")
    resolution_time: str = Field(description="Includes downtime figure and disposition")
    subsystem: str


def build_prompt(subsystem: str, n: int, examples: list) -> str:
    examples_text = "\n\n".join(json.dumps(ex, indent=2) for ex in examples)
    return f"""You are generating SYNTHETIC turbofan engine maintenance log entries.
These logs will be used to build a RAG knowledge base alongside a CMAPSS-based predictive model.

Below are real example logs focused on the "{subsystem}" subsystem. Study their tone and structure.

EXAMPLES:
{examples_text}

Generate exactly {n} NEW, DISTINCT log entries focused on the "{subsystem}" subsystem.
Requirements:
- Each log must describe a GRADUAL, sensor-detectable degradation pattern (except edge_case)
- Vary sensor names (T2, T24, T30, Nf/N1, Nc/N2, etc.)
- Use unit numbers between 01 and 100 (do not repeat example units)
- Vary dates across 2026
- Starting log_id number for this batch must begin from {{START_ID}}
- Map your output to the requested array of objects schema."""


def load_checkpoint() -> list:
    """Resume from a previous run if a checkpoint file exists."""
    if os.path.exists(CHECKPOINT_PATH):
        with open(CHECKPOINT_PATH) as f:
            data = json.load(f)
        print(f"Found checkpoint with {len(data)} already-generated logs. Resuming.")
        return data
    return []


def save_checkpoint(generated_logs: list):
    with open(CHECKPOINT_PATH, "w") as f:
        json.dump(generated_logs, f, indent=2)


def generate_batch(subsystem: str, n: int, start_id: int, max_retries: int = 4) -> list:
    examples = [log for log in seed_logs if log["subsystem"] == subsystem]
    if not examples:
        examples = seed_logs[:3]

    prompt = build_prompt(subsystem, n, examples).replace("{START_ID}", f"{start_id:04d}")

    for attempt in range(max_retries):
        try:
            response = client.models.generate_content(
                model=MODEL,
                contents=prompt,
                config=types.GenerateContentConfig(
                    max_output_tokens=8000,
                    temperature=0.9,
                    response_mime_type="application/json",
                    response_schema=list[MaintenanceLog],
                    thinking_config=types.ThinkingConfig(
                        thinking_level=types.ThinkingLevel.MINIMAL
                    ),
                ),
            )
            break
        except ClientError as e:
            if "RESOURCE_EXHAUSTED" in str(e) or "429" in str(e):
                wait_time = 60 * (attempt + 1)  # 60s, 120s, 180s, 240s — generous
                print(f"  [RATE LIMIT] Hit quota. Waiting {wait_time}s before retry "
                      f"({attempt + 1}/{max_retries})...")
                time.sleep(wait_time)
                continue
            else:
                print(f"  [ERROR] Unexpected API error: {e}")
                return []
    else:
        print(f"  [FAIL] Gave up on this batch after {max_retries} retries.")
        return []

    raw_text = response.text.strip() if response.text else ""

    if not raw_text:
        finish_reason = (
            response.candidates[0].finish_reason if response.candidates else "unknown"
        )
        print(f"  [WARN] Empty response for '{subsystem}' (finish_reason: {finish_reason})")
        return []

    try:
        batch = json.loads(raw_text)
        return batch
    except json.JSONDecodeError as e:
        print(f"  [WARN] Failed to parse batch for '{subsystem}': {e}")
        with open("failed_batch.txt", "w") as f:
            f.write(raw_text)
        return []


def main():
    all_generated = load_checkpoint()
    already_done_ids = {log["log_id"] for log in all_generated}
    done_subsystem_counts = {}
    for log in all_generated:
        done_subsystem_counts[log["subsystem"]] = done_subsystem_counts.get(log["subsystem"], 0) + 1

    next_id = 15 + len(all_generated)
    MAX_PER_CALL = 8

    for subsystem, total_count in GENERATION_PLAN.items():
        already_have = done_subsystem_counts.get(subsystem, 0)
        remaining = total_count - already_have

        if remaining <= 0:
            print(f"Skipping {subsystem} — already have {already_have}/{total_count} from checkpoint.")
            continue

        print(f"Generating {remaining} more logs for subsystem: {subsystem} "
              f"(already have {already_have}/{total_count})...")

        while remaining > 0:
            batch_n = min(MAX_PER_CALL, remaining)
            batch = generate_batch(subsystem, batch_n, next_id)
            print(f"  -> Got {len(batch)}/{batch_n} valid logs in this sub-batch")

            all_generated.extend(batch)
            next_id += len(batch)
            remaining -= len(batch) if len(batch) > 0 else batch_n

            save_checkpoint(all_generated)  # persist progress after every sub-batch
            time.sleep(SECONDS_BETWEEN_CALLS)

    full_dataset = seed_logs + all_generated
    output_path = "maintenance_logs_full.json"
    with open(output_path, "w") as f:
        json.dump(full_dataset, f, indent=2)

    print(f"\nDone. Total logs: {len(full_dataset)} (14 seed + {len(all_generated)} generated)")
    print(f"Saved to {output_path}")

    from collections import Counter
    counts = Counter(log["subsystem"] for log in full_dataset)
    print("Final subsystem distribution:", dict(counts))


if __name__ == "__main__":
    main()