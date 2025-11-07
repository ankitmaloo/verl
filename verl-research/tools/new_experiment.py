#!/usr/bin/env python3
"""
Create a new experiment from template.

Usage:
    python new_experiment.py "experiment_name"
    python new_experiment.py "paper_dapo" --description "Test DAPO advantage"
"""

import argparse
import os
import shutil
from pathlib import Path
from datetime import datetime


def find_next_experiment_number(experiments_dir: Path) -> int:
    """Find the next available experiment number"""
    existing = [
        int(d.name.split('_')[0])
        for d in experiments_dir.iterdir()
        if d.is_dir() and d.name[0].isdigit()
    ]
    return max(existing, default=-1) + 1


def create_experiment(name: str, description: str = "", experiments_dir: Path = None):
    """
    Create a new experiment from template.

    Args:
        name: Experiment name (e.g., "paper_dapo")
        description: Short description
        experiments_dir: Where to create experiment (default: ../experiments)
    """

    # Default to ../experiments relative to this script
    if experiments_dir is None:
        script_dir = Path(__file__).parent
        experiments_dir = script_dir.parent / "experiments"

    # Ensure experiments directory exists
    experiments_dir.mkdir(exist_ok=True)

    # Find next experiment number
    exp_number = find_next_experiment_number(experiments_dir)

    # Create experiment directory name: NN_name
    exp_dir_name = f"{exp_number:02d}_{name}"
    exp_dir = experiments_dir / exp_dir_name

    if exp_dir.exists():
        print(f"❌ Error: {exp_dir} already exists!")
        return False

    # Copy template
    template_dir = experiments_dir / "template"
    if not template_dir.exists():
        print(f"❌ Error: Template directory not found at {template_dir}")
        return False

    print(f"Creating experiment: {exp_dir_name}")
    shutil.copytree(template_dir, exp_dir)

    # Update config.yaml
    config_path = exp_dir / "config.yaml"
    with open(config_path, 'r') as f:
        config = f.read()

    # Replace placeholders
    config = config.replace('template_experiment', name)
    config = config.replace('Description of your idea', description or f"Experiment {name}")

    with open(config_path, 'w') as f:
        f.write(config)

    # Update README.md
    readme_path = exp_dir / "README.md"
    with open(readme_path, 'r') as f:
        readme = f.read()

    readme = readme.replace('[Experiment Name]', name.replace('_', ' ').title())
    readme = readme.replace('[Describe what you\'re testing and why]', description or "TODO: Add hypothesis")

    # Add metadata
    metadata = f"""
## Metadata

- **Created**: {datetime.now().strftime('%Y-%m-%d %H:%M')}
- **Experiment ID**: {exp_dir_name}
- **Status**: 🔵 Not started

"""
    readme = readme + metadata

    with open(readme_path, 'w') as f:
        f.write(readme)

    # Create additional directories
    (exp_dir / "checkpoints").mkdir(exist_ok=True)
    (exp_dir / "plots").mkdir(exist_ok=True)

    # Create placeholder results.json
    results_path = exp_dir / "results.json"
    with open(results_path, 'w') as f:
        f.write("""{{
  "experiment": "{name}",
  "status": "not_started",
  "results": {{
    "quick_test": null,
    "small_train": null,
    "full_train": null
  }}
}}
""".format(name=name))

    print(f"✅ Created experiment: {exp_dir}")
    print(f"\nNext steps:")
    print(f"  1. cd {exp_dir}")
    print(f"  2. Edit config.yaml with your settings")
    print(f"  3. Edit variant.py to implement your idea")
    print(f"  4. Run: python ../../tools/quick_test.py .")
    print(f"\nGood luck! 🚀")

    return True


def main():
    parser = argparse.ArgumentParser(
        description="Create a new experiment from template"
    )
    parser.add_argument(
        "name",
        help="Experiment name (e.g., 'paper_dapo_advantage')"
    )
    parser.add_argument(
        "-d", "--description",
        default="",
        help="Short description of the experiment"
    )
    parser.add_argument(
        "--dir",
        type=Path,
        default=None,
        help="Experiments directory (default: ../experiments)"
    )

    args = parser.parse_args()

    # Clean up name (replace spaces with underscores, lowercase)
    name = args.name.lower().replace(' ', '_').replace('-', '_')

    success = create_experiment(
        name=name,
        description=args.description,
        experiments_dir=args.dir
    )

    return 0 if success else 1


if __name__ == "__main__":
    exit(main())
