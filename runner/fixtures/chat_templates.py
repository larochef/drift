# /// script
# requires-python = ">=3.11"
# dependencies = ["transformers", "jinja2", "tokenizers", "sentencepiece", "protobuf"]
# ///
"""Golden renders of the models' chat templates (specs/42, step 5).

For each model, transformers' own apply_chat_template renders every
conversation of conversations.json with each set of flags; the runner's Jinja
interpreter must produce the same text. Run by hand; the templates and the
renders are committed, so the tests need no Python.

    uv run runner/fixtures/chat_templates.py
"""

import datetime
import json
from pathlib import Path

from transformers import AutoTokenizer

here = Path(__file__).resolve().parent
out = here.parent / "test" / "resources" / "chat-templates"
models = {
    "qwen3": "Qwen/Qwen3-4B",
    "qwen3.6": "Qwen/Qwen3.6-35B-A3B",
    "qwen3.8": "Qwen/Qwen3.8-Flash-Next",
    "gemma4": "google/gemma-4-26b-a4b-it",
    "gpt-oss": "openai/gpt-oss-20b",
}
conversations = json.loads((here / "conversations.json").read_text())
out.mkdir(parents=True, exist_ok=True)
(out / "conversations.json").write_text((here / "conversations.json").read_text())  # the tests read this copy
flag_sets = [
    {"add_generation_prompt": True},
    {"add_generation_prompt": False},
    {"add_generation_prompt": True, "enable_thinking": False},
]

for name, repository in models.items():
    tokenizer = AutoTokenizer.from_pretrained(repository)
    folder = out / name
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "template.jinja").write_text(tokenizer.chat_template)
    renders = []
    for conversation in conversations:
        for flags in flag_sets:
            try:
                text = tokenizer.apply_chat_template(
                    conversation["messages"], tools=conversation.get("tools"), tokenize=False, **flags
                )
                renders.append({"conversation": conversation["name"], "flags": flags, "text": text})
            except Exception as error:  # a template may refuse a conversation: that is part of the golden
                renders.append({"conversation": conversation["name"], "flags": flags, "error": str(error)})
    # what apply_chat_template passes besides the conversation: the special
    # tokens as variables, and the date strftime_now reads
    golden = {
        "variables": {k: v for k, v in tokenizer.special_tokens_map.items() if isinstance(v, str)},
        "date": datetime.date.today().isoformat(),
        "renders": renders,
    }
    (folder / "renders.json").write_text(json.dumps(golden, ensure_ascii=False, indent=1))
    print(f"{name}: {len(renders)} renders")
