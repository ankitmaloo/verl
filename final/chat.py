#!/usr/bin/env python3
"""Simple interactive chat with the inference engine."""
import textwrap

if __name__ == "__main__":
    from final.inf import InferenceEngine
    engine = InferenceEngine("final/config.yaml")
    while (q := input("\nYou: ").strip()):
        print("\nAI:", textwrap.fill(engine.generate(q).completions[0], width=80))
