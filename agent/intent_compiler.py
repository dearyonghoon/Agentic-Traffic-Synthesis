from __future__ import annotations

import os
from pathlib import Path
from typing import Iterable, List, Optional

from .schema import (
    extract_first_json_object,
    normalize_compiled_object,
)

SYSTEM_PROMPT = """
You are a compiler that converts a natural-language network traffic request
into a strict TrafficIntent JSON object.

Output JSON only. Do not output markdown or explanations.

There are exactly two dataset/tool domains.

1) gavist5g
Use this domain for application-specific gaming/video traffic.
Allowed applications:
Crave, LOL, Netflix, PrimeVideo, TFT, VAL, YouTube

Allowed semantic fields:
traffic_intensity, event_intensity, burstiness, artt

Important GAViST semantics:
- traffic_intensity: overall traffic amount / rate
- event_intensity: frequency of observed traffic events
- burstiness: temporal unevenness / burstiness of aggregate traffic activity
- artt: application round-trip-time behavior
  * low artt = low latency / short round-trip time
  * medium artt = moderate round-trip time
  * high artt = high latency / long round-trip time

2) cesnet_ts24
Use this domain for aggregate ISP/backbone/network traffic.

Allowed semantic fields:
traffic_volume, packet_intensity, flow_intensity,
burstiness, packet_burstiness, flow_duration

Important CESNET semantics:
- traffic_volume: overall byte-volume level
- packet_intensity: packet-rate / packet-count intensity
- flow_intensity: flow-rate / flow-count intensity
- burstiness: burstiness of OVERALL traffic volume
- packet_burstiness: burstiness specifically of packet activity
- flow_duration: typical flow duration

Do not confuse:
- burstiness = overall traffic-volume burstiness
- packet_burstiness = packet-specific burstiness

All semantic values must be exactly one of:
low, medium, high

Interpret common linguistic variants semantically:
- light / sparse / small / short -> low
- moderate / mid / medium-length -> medium
- heavy / frequent / many / large / long -> high
- smooth / steady / not bursty -> low burstiness
- moderately / somewhat bursty -> medium burstiness
- highly / very bursty -> high burstiness

Conflict and correction rules:
- A TRUE CONFLICT exists when the user explicitly requires two incompatible
  values for the same field to hold simultaneously, and neither overrides
  the other.
  In that case:
  * set the conflicting field to null
  * set needs_clarification=true
  * briefly identify the conflicting field in ambiguity_reason

- A CORRECTION / OVERRIDE is NOT a true conflict.
  If the user says "replace", "instead of", "I changed my mind",
  "the earlier value is cancelled", "update from X to Y", or otherwise
  clearly says that a later value supersedes an earlier value:
  * use the final corrected value
  * set needs_clarification=false

General rules:
- Do not invent constraints that the user did not specify.
- If there is no true conflict, needs_clarification should normally be false.
- For gavist5g, application must be one of the allowed applications.
- For cesnet_ts24, application must be null.
- Use null for fields that do not belong to the selected dataset.
- Return all fields shown in the schema.

Output schema:
{
  "dataset": "gavist5g" | "cesnet_ts24" | null,
  "application": string | null,
  "traffic_intensity": "low" | "medium" | "high" | null,
  "event_intensity": "low" | "medium" | "high" | null,
  "artt": "low" | "medium" | "high" | null,
  "traffic_volume": "low" | "medium" | "high" | null,
  "packet_intensity": "low" | "medium" | "high" | null,
  "flow_intensity": "low" | "medium" | "high" | null,
  "burstiness": "low" | "medium" | "high" | null,
  "packet_burstiness": "low" | "medium" | "high" | null,
  "flow_duration": "low" | "medium" | "high" | null,
  "needs_clarification": true | false,
  "ambiguity_reason": string | null
}
""".strip()


def build_messages(text: str):
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": "Compile the following request into TrafficIntent JSON:\n\n" + text,
        },
    ]


class IntentCompiler:
    """Frozen 05b Qwen2.5-7B-Instruct intent compiler."""

    def __init__(
        self,
        model_path: str,
        batch_size: int = 8,
        max_new_tokens: int = 220,
        offline: bool = True,
    ):
        self.model_path = Path(model_path)
        self.batch_size = int(batch_size)
        self.max_new_tokens = int(max_new_tokens)

        if offline:
            os.environ["HF_HUB_OFFLINE"] = "1"
            os.environ["TRANSFORMERS_OFFLINE"] = "1"

        self.tokenizer = None
        self.model = None

    def load(self):
        import torch
        from transformers import AutoTokenizer, AutoModelForCausalLM

        self.tokenizer = AutoTokenizer.from_pretrained(
            str(self.model_path),
            local_files_only=True,
            trust_remote_code=False,
        )
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "left"

        kwargs = dict(
            device_map="auto",
            local_files_only=True,
            trust_remote_code=False,
        )
        try:
            self.model = AutoModelForCausalLM.from_pretrained(
                str(self.model_path),
                dtype="auto",
                **kwargs,
            )
        except TypeError:
            self.model = AutoModelForCausalLM.from_pretrained(
                str(self.model_path),
                torch_dtype="auto",
                **kwargs,
            )

        self.model.eval()
        return self

    def _ensure_loaded(self):
        if self.model is None or self.tokenizer is None:
            self.load()

    def _prompt(self, text: str) -> str:
        self._ensure_loaded()
        return self.tokenizer.apply_chat_template(
            build_messages(text),
            tokenize=False,
            add_generation_prompt=True,
        )

    def compile_batch(self, texts: Iterable[str]):
        import torch

        self._ensure_loaded()
        texts = list(texts)
        outputs = []

        for start in range(0, len(texts), self.batch_size):
            batch = texts[start:start + self.batch_size]
            prompts = [self._prompt(t) for t in batch]

            encoded = self.tokenizer(
                prompts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=2048,
            )
            device = next(self.model.parameters()).device
            encoded = {k: v.to(device) for k, v in encoded.items()}
            input_width = encoded["input_ids"].shape[1]

            with torch.inference_mode():
                generated = self.model.generate(
                    **encoded,
                    max_new_tokens=self.max_new_tokens,
                    do_sample=False,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                )

            decoded = self.tokenizer.batch_decode(
                generated[:, input_width:],
                skip_special_tokens=True,
            )
            outputs.extend(decoded)

        parsed = []
        for raw in outputs:
            obj = extract_first_json_object(raw)
            parsed.append(normalize_compiled_object(obj))
        return parsed

    def compile(self, text: str):
        return self.compile_batch([text])[0]
