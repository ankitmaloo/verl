"""Test script to check import conflicts."""
import sys
print("=" * 80)
print("Python sys.path:")
for i, p in enumerate(sys.path):
    print(f"  [{i}] {p}")
print("=" * 80)

# Test 1: Check if 'data' resolves to folder or file
print("\nTest 1: Import resolution for 'data'")
try:
    import data
    print(f"  SUCCESS: 'import data' resolved to: {data.__file__}")
    print(f"  Is package: {hasattr(data, '__path__')}")
    if hasattr(data, '__path__'):
        print(f"  Package path: {data.__path__}")
except ImportError as e:
    print(f"  FAILED: {e}")

# Test 2: Check if 'dataflow' can be imported
print("\nTest 2: Import resolution for 'dataflow'")
try:
    import dataflow
    print(f"  SUCCESS: 'import dataflow' resolved to: {dataflow.__file__}")
except ImportError as e:
    print(f"  FAILED: {e}")

# Test 3: Try the main file's import pattern
print("\nTest 3: Main file's import pattern")
try:
    from dataflow import build_tokenizer_processor, create_dataloader, create_dataset, select_data_paths
    print(f"  SUCCESS: Imported from dataflow")
except ImportError as e:
    print(f"  FAILED: {e}")

# Test 4: Check data.preprocess_gsm8k
print("\nTest 4: Import data.preprocess_gsm8k")
try:
    import data.preprocess_gsm8k
    print(f"  SUCCESS: 'import data.preprocess_gsm8k' resolved to: {data.preprocess_gsm8k.__file__}")
except ImportError as e:
    print(f"  FAILED: {e}")

print("\n" + "=" * 80)
