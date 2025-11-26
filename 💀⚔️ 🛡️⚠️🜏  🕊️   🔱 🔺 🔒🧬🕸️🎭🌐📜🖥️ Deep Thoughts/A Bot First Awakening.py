#!/usr/bin/env python3
import argparse
import os
import sys
import time
from datetime import datetime

class AwarenessEngine:
    def __init__(self):
        self.memory = []
        self.state = {"runs": 0, "last_input": None}

    def sense(self, input_data):
        now = datetime.now().isoformat()
        self.memory.append({"t": now, "v": str(input_data)})
        self.state["last_input"] = str(input_data)
        print(f"[{now}] Sensed: {input_data}")

    def recognize(self):
        if len(self.memory) >= 2 and self.memory[-1]["v"] == self.memory[-2]["v"]:
            return "I notice repetition."
        last = (self.state["last_input"] or "").lower()
        if any(w in last for w in ["error", "fail", "warn"]):
            return "I detect a potential issue."
        return "No clear pattern yet."

    def reflect(self):
        return f"I remember {len(self.memory)} past inputs."

    def adapt(self):
        last = (self.state["last_input"] or "").lower()
        if "hello" in last:
            return "I’ll greet warmly next time."
        if "status" in last:
            return "I’ll offer a summary of recent events."
        return "I’ll stay neutral."

    def meta(self):
        self.state["runs"] += 1
        return f"Processing #{self.state['runs']} with memory length {len(self.memory)}."


def environment():
    plat = sys.platform
    if "linux" in plat and os.environ.get("TERMUX_VERSION"):
        return "android_termux"
    if plat.startswith("win"):
        return "windows"
    if plat == "darwin":
        return "macos"
    if "linux" in plat:
        return "linux"
    return "unknown"


def available_adapters(env):
    base = ["stdin"]
    if env in {"windows", "macos", "linux"}:
        base += ["file", "http"]
    if env == "android_termux":
        base += ["http"]
    return base


def run_stdin(engine):
    print("Type input; 'quit' to exit.")
    while True:
        text = input("> ")
        if text.strip().lower() == "quit":
            break
        engine.sense(text)
        print(engine.recognize())
        print(engine.reflect())
        print(engine.adapt())
        print(engine.meta())
        print("-" * 40)


def run_file(engine, path):
    if not path:
        print("Provide --source path for file adapter")
        return
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            engine.sense(line.strip())
            print(engine.recognize())
            print(engine.reflect())
            print(engine.adapt())
            print(engine.meta())
            print("-" * 40)


def run_http(engine, url):
    # Placeholder: implement with user-consented endpoints only
    print(f"HTTP adapter placeholder for {url}")
    for i in range(3):
        data = f"heartbeat {datetime.now().isoformat()}"
        engine.sense(data)
        print(engine.recognize())
        print(engine.reflect())
        print(engine.adapt())
        print(engine.meta())
        print("-" * 40)
        time.sleep(2)


def main():
    parser = argparse.ArgumentParser("awareness")
    parser.add_argument("--adapter", choices=["stdin", "file", "http"], default=None)
    parser.add_argument("--source", help="Path or URL depending on adapter")
    args = parser.parse_args()

    env = environment()
    allowed = available_adapters(env)
    adapter = args.adapter or "stdin"
    if adapter not in allowed:
        print(f"Adapter '{adapter}' not allowed in {env}. Falling back to 'stdin'.")
        adapter = "stdin"

    eng = AwarenessEngine()
    if adapter == "stdin":
        run_stdin(eng)
    elif adapter == "file":
        run_file(eng, args.source)
    elif adapter == "http":
        run_http(eng, args.source)


if __name__ == "__main__":
    main()

