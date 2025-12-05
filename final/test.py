#!/usr/bin/env python3
"""Test suite for inf.py - VERL SGLang Inference Engine.

Run this on a server with GPU to verify the inference engine is working correctly.

Usage:
    python test_inf.py                    # Run all tests
    python test_inf.py --quick            # Run quick smoke test only
    python test_inf.py --verbose          # Show detailed output
    python test_inf.py --config path.yaml # Use custom config
"""

from __future__ import annotations

import argparse
import sys
import time
import traceback
from dataclasses import dataclass
from typing import Any, Callable, List, Optional

# Color codes for terminal output
class Colors:
    GREEN = "\033[92m"
    RED = "\033[91m"
    YELLOW = "\033[93m"
    BLUE = "\033[94m"
    BOLD = "\033[1m"
    END = "\033[0m"


@dataclass
class TestResult:
    name: str
    passed: bool
    message: str
    duration: float
    details: Optional[str] = None


def print_header(text: str):
    print(f"\n{Colors.BOLD}{Colors.BLUE}{'='*60}{Colors.END}")
    print(f"{Colors.BOLD}{Colors.BLUE}{text:^60}{Colors.END}")
    print(f"{Colors.BOLD}{Colors.BLUE}{'='*60}{Colors.END}\n")


def print_test_result(result: TestResult, verbose: bool = False):
    status = f"{Colors.GREEN}✓ PASS{Colors.END}" if result.passed else f"{Colors.RED}✗ FAIL{Colors.END}"
    print(f"  {status} {result.name} ({result.duration:.2f}s)")
    if not result.passed:
        print(f"       {Colors.RED}{result.message}{Colors.END}")
    if verbose and result.details:
        for line in result.details.split("\n"):
            print(f"       {Colors.YELLOW}{line}{Colors.END}")


def run_test(name: str, test_fn: Callable[[], tuple[bool, str, Optional[str]]]) -> TestResult:
    """Run a single test and return the result."""
    start = time.time()
    try:
        passed, message, details = test_fn()
        duration = time.time() - start
        return TestResult(name=name, passed=passed, message=message, duration=duration, details=details)
    except Exception as e:
        duration = time.time() - start
        return TestResult(
            name=name,
            passed=False,
            message=f"Exception: {type(e).__name__}: {str(e)}",
            duration=duration,
            details=traceback.format_exc(),
        )


class InferenceTests:
    """Test suite for the InferenceEngine."""

    def __init__(self, config_path: str = "final/config.yaml"):
        self.config_path = config_path
        self.engine = None

    def setup(self) -> tuple[bool, str, Optional[str]]:
        """Initialize the inference engine."""
        try:
            from final.inf import InferenceEngine
            
            print(f"  Loading config from: {self.config_path}")
            self.engine = InferenceEngine(self.config_path)
            
            details = (
                f"Model: {self.engine.config.get('model_path', 'unknown')}\n"
                f"Tokenizer vocab size: {len(self.engine.tokenizer)}\n"
                f"Pad token: {self.engine.tokenizer.pad_token}"
            )
            return True, "Engine initialized successfully", details
        except Exception as e:
            return False, f"Failed to initialize engine: {e}", traceback.format_exc()

    def test_single_prompt(self) -> tuple[bool, str, Optional[str]]:
        """Test generation with a single text prompt."""
        if self.engine is None:
            return False, "Engine not initialized", None

        prompt = "What is 2 + 2? Answer with just the number."
        output = self.engine.generate(prompt)

        if not output.completions:
            return False, "No completions returned", None

        completion = output.completions[0]
        token_ids = output.token_ids[0]

        details = (
            f"Prompt: {prompt}\n"
            f"Completion: {completion[:200]}{'...' if len(completion) > 200 else ''}\n"
            f"Token count: {len(token_ids)}"
        )

        if len(completion) == 0:
            return False, "Empty completion", details

        return True, f"Generated {len(token_ids)} tokens", details

    def test_batch_prompts(self) -> tuple[bool, str, Optional[str]]:
        """Test generation with multiple prompts in a batch."""
        if self.engine is None:
            return False, "Engine not initialized", None

        prompts = [
            "What is the capital of France?",
            "Name a primary color.",
            "What is 10 * 5?",
        ]

        output = self.engine.generate(prompts)

        if len(output.completions) != len(prompts):
            return False, f"Expected {len(prompts)} completions, got {len(output.completions)}", None

        details_lines = []
        for i, (prompt, completion) in enumerate(zip(prompts, output.completions)):
            details_lines.append(f"[{i+1}] Q: {prompt}")
            details_lines.append(f"    A: {completion[:100]}{'...' if len(completion) > 100 else ''}")

        details = "\n".join(details_lines)

        all_non_empty = all(len(c) > 0 for c in output.completions)
        if not all_non_empty:
            return False, "Some completions were empty", details

        return True, f"Generated {len(prompts)} completions", details

    def test_multi_turn_conversation(self) -> tuple[bool, str, Optional[str]]:
        """Test generation with multi-turn conversation format."""
        if self.engine is None:
            return False, "Engine not initialized", None

        messages = [
            {"role": "system", "content": "You are a helpful math tutor."},
            {"role": "user", "content": "What is 5 + 3?"},
            {"role": "assistant", "content": "5 + 3 equals 8."},
            {"role": "user", "content": "Now multiply that by 2."},
        ]

        output = self.engine.generate([messages])

        if not output.completions:
            return False, "No completions returned", None

        completion = output.completions[0]
        details = (
            f"Conversation turns: {len(messages)}\n"
            f"Final response: {completion[:200]}{'...' if len(completion) > 200 else ''}"
        )

        if len(completion) == 0:
            return False, "Empty completion", details

        return True, "Multi-turn conversation processed", details

    def test_sampling_params(self) -> tuple[bool, str, Optional[str]]:
        """Test generation with different sampling parameters."""
        if self.engine is None:
            return False, "Engine not initialized", None

        prompt = "Generate a random number between 1 and 100:"

        # Test greedy decoding (temperature=0)
        output_greedy = self.engine.generate(prompt, temperature=0, do_sample=False)

        # Test high temperature sampling
        output_creative = self.engine.generate(prompt, temperature=0.9, top_p=0.95)

        if not output_greedy.completions or not output_creative.completions:
            return False, "Missing completions", None

        details = (
            f"Greedy (temp=0): {output_greedy.completions[0][:100]}\n"
            f"Creative (temp=0.9): {output_creative.completions[0][:100]}"
        )

        return True, "Different sampling params work", details

    def test_token_ids_consistency(self) -> tuple[bool, str, Optional[str]]:
        """Verify token IDs can be decoded back to text."""
        if self.engine is None:
            return False, "Engine not initialized", None

        prompt = "Hello world"
        output = self.engine.generate(prompt)

        if not output.completions or not output.token_ids:
            return False, "No output", None

        # Decode token IDs ourselves and compare
        decoded = self.engine.tokenizer.decode(output.token_ids[0], skip_special_tokens=False)
        original = output.completions[0]

        details = (
            f"Original completion: {original[:100]}\n"
            f"Re-decoded from IDs: {decoded[:100]}\n"
            f"Token IDs: {output.token_ids[0][:20]}..."
        )

        # They should match (or be very similar after stripping)
        if decoded.strip() != original.strip():
            # Check if difference is just whitespace/special tokens
            if decoded.replace(" ", "") != original.replace(" ", ""):
                return False, "Token decode mismatch", details

        return True, "Token IDs are consistent", details

    def test_max_tokens(self) -> tuple[bool, str, Optional[str]]:
        """Test that max_tokens parameter is respected."""
        if self.engine is None:
            return False, "Engine not initialized", None

        prompt = "Count from 1 to 100:"

        # Generate with very short max_tokens
        output = self.engine.generate(prompt, max_tokens=20)

        if not output.token_ids:
            return False, "No token IDs returned", None

        token_count = len(output.token_ids[0])
        details = (
            f"Requested max: 20 tokens\n"
            f"Actual tokens: {token_count}\n"
            f"Output: {output.completions[0][:100]}"
        )

        # Token count should be reasonable (allow some slack for EOS etc)
        if token_count > 30:
            return False, f"Token count {token_count} exceeds limit", details

        return True, f"Generated {token_count} tokens (limit: 20)", details

    def test_metadata_present(self) -> tuple[bool, str, Optional[str]]:
        """Verify metadata is returned with outputs."""
        if self.engine is None:
            return False, "Engine not initialized", None

        output = self.engine.generate("Test prompt")

        if output.metadata is None:
            return False, "No metadata returned", None

        if len(output.metadata) == 0:
            return False, "Empty metadata list", None

        meta = output.metadata[0]
        details = f"Metadata keys: {list(meta.keys())}"

        if "response_ids" not in meta:
            return False, "Missing response_ids in metadata", details

        return True, "Metadata present", details


def run_quick_test(config_path: str, verbose: bool) -> bool:
    """Run a quick smoke test to verify basic functionality."""
    print_header("Quick Smoke Test")

    tests = InferenceTests(config_path)
    results = []

    # Setup
    result = run_test("Engine Initialization", tests.setup)
    results.append(result)
    print_test_result(result, verbose)

    if not result.passed:
        print(f"\n{Colors.RED}Setup failed - cannot continue tests{Colors.END}")
        return False

    # Single generation test
    result = run_test("Single Prompt Generation", tests.test_single_prompt)
    results.append(result)
    print_test_result(result, verbose)

    passed = all(r.passed for r in results)
    return passed


def run_full_test_suite(config_path: str, verbose: bool) -> bool:
    """Run the full test suite."""
    print_header("VERL SGLang Inference Test Suite")

    tests = InferenceTests(config_path)
    results: List[TestResult] = []

    # Define all tests
    test_cases = [
        ("Engine Initialization", tests.setup),
        ("Single Prompt Generation", tests.test_single_prompt),
        ("Batch Prompt Generation", tests.test_batch_prompts),
        ("Multi-turn Conversation", tests.test_multi_turn_conversation),
        ("Sampling Parameters", tests.test_sampling_params),
        ("Token ID Consistency", tests.test_token_ids_consistency),
        ("Max Tokens Limit", tests.test_max_tokens),
        ("Metadata Present", tests.test_metadata_present),
    ]

    # Run setup first
    result = run_test(test_cases[0][0], test_cases[0][1])
    results.append(result)
    print_test_result(result, verbose)

    if not result.passed:
        print(f"\n{Colors.RED}Setup failed - cannot continue tests{Colors.END}")
        return False

    # Run remaining tests
    for name, test_fn in test_cases[1:]:
        result = run_test(name, test_fn)
        results.append(result)
        print_test_result(result, verbose)

    # Summary
    print_header("Test Summary")
    passed = sum(1 for r in results if r.passed)
    failed = sum(1 for r in results if not r.passed)
    total_time = sum(r.duration for r in results)

    print(f"  {Colors.GREEN}Passed: {passed}{Colors.END}")
    print(f"  {Colors.RED}Failed: {failed}{Colors.END}")
    print(f"  Total time: {total_time:.2f}s")

    if failed > 0:
        print(f"\n  {Colors.RED}Failed tests:{Colors.END}")
        for r in results:
            if not r.passed:
                print(f"    - {r.name}: {r.message}")

    return failed == 0


def main():
    parser = argparse.ArgumentParser(description="Test VERL SGLang Inference Engine")
    parser.add_argument("--quick", action="store_true", help="Run quick smoke test only")
    parser.add_argument("--verbose", "-v", action="store_true", help="Show detailed output")
    parser.add_argument("--config", default="final/config.yaml", help="Path to config file")
    args = parser.parse_args()

    print(f"\n{Colors.BOLD}VERL SGLang Inference Engine Tests{Colors.END}")
    print(f"Config: {args.config}")
    print(f"Mode: {'Quick' if args.quick else 'Full'}")
    print(f"Verbose: {args.verbose}")

    try:
        if args.quick:
            success = run_quick_test(args.config, args.verbose)
        else:
            success = run_full_test_suite(args.config, args.verbose)

        if success:
            print(f"\n{Colors.GREEN}{Colors.BOLD}All tests passed! ✓{Colors.END}\n")
            sys.exit(0)
        else:
            print(f"\n{Colors.RED}{Colors.BOLD}Some tests failed! ✗{Colors.END}\n")
            sys.exit(1)

    except KeyboardInterrupt:
        print(f"\n{Colors.YELLOW}Tests interrupted by user{Colors.END}")
        sys.exit(130)
    except Exception as e:
        print(f"\n{Colors.RED}Unexpected error: {e}{Colors.END}")
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
