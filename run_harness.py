#!/usr/bin/env python3
"""CLI entrypoint for the reasoning harness."""

from __future__ import annotations

import argparse
import json

from reasoning_harness import ReasoningHarness


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the reasoning harness")
    parser.add_argument("question", help="Question or task to reason about")
    parser.add_argument(
        "--json",
        action="store_true",
        help="Emit machine-readable JSON instead of plain text",
    )
    args = parser.parse_args()

    harness = ReasoningHarness()
    result = harness.run(args.question)

    if args.json:
        print(
            json.dumps(
                {
                    "question": result.question,
                    "answer": result.answer,
                    "confidence": result.confidence,
                    "created_at": result.created_at,
                    "trace": [
                        {
                            "name": step.name,
                            "content": step.content,
                            "score": step.score,
                        }
                        for step in result.trace
                    ],
                },
                indent=2,
            )
        )
    else:
        print(f"Question: {result.question}")
        print(f"Confidence: {result.confidence}")
        print("Trace:")
        for step in result.trace:
            print(f"- [{step.name}] ({step.score:.2f}) {step.content}")
        print("\nAnswer:\n" + result.answer)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
