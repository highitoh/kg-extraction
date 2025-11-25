"""Strategy Planner - Creates improvement plans by analyzing categorized issues using LLM."""

import asyncio
import json
import os
import sys
import argparse
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional
import glob

from openai import APIConnectionError, RateLimitError, APIError

# Add src directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.llm_factory import LLMFactory


class StrategyPlanner:
    """Creates improvement plans by analyzing categorized issues using LLM."""

    def __init__(
        self,
        project: str = "default",
        model: str = "gpt-5-mini",
        progress: bool = True,
        max_retries: int = 3,
    ):
        """
        Initialize the StrategyPlanner.

        Args:
            project: Project name (resolves to /workspace/app/output/{project}/)
            model: LLM model name
            progress: Whether to show progress messages
            max_retries: Maximum number of retries for LLM API calls
        """
        self.project = project
        self.model = model
        self.progress = progress
        self.max_retries = max_retries
        self.output_base_dir = "/workspace/app/output/optimizer"

        # Resolve issue category file path from project name
        self.issue_category_file = self._resolve_issue_category_path(project)

        # Load issue category data
        self.issue_category_data = self._load_json(self.issue_category_file)

        # Create LLM instance with high reasoning effort for strategic planning
        self.llm = LLMFactory.create_llm(
            model=model,
            temperature=0.3,
            reasoning_effort="high"
        )

        # Load schema (path: /workspace/app/schemas/strategy-planner/)
        self.schemas_dir = Path(__file__).parent.parent.parent / "schemas" / "strategy-planner"
        self.llm_schema = self._load_schema("llm.schema.json")

        # Bind LLM with JSON schema
        self.llm_bound = self.llm.bind(
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": "strategy_planner_response",
                    "schema": self.llm_schema,
                    "strict": True
                }
            }
        )

        # Load prompt template (path: /workspace/app/prompts/)
        self.prompt_template = self._load_prompt("strategy_planner.txt")

    def _load_schema(self, schema_name: str) -> Dict[str, Any]:
        """Load a JSON schema file."""
        schema_path = self.schemas_dir / schema_name
        with open(schema_path, 'r', encoding='utf-8') as f:
            return json.load(f)

    def _load_prompt(self, prompt_filename: str) -> str:
        """Load a prompt template."""
        prompt_path = Path(__file__).parent.parent.parent / "prompts" / prompt_filename
        with open(prompt_path, 'r', encoding='utf-8') as f:
            return f.read()

    def _load_json(self, file_path: str) -> Dict[str, Any]:
        """Load a JSON file."""
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)

    def _resolve_issue_category_path(self, project: str) -> str:
        """
        Resolve issue_category file path from project name.

        Args:
            project: Project name

        Returns:
            Path to the most recent issue_category_*.json file

        Raises:
            FileNotFoundError: If required files don't exist
        """
        project_dir = Path(self.output_base_dir) / project

        # Check if project directory exists
        if not project_dir.exists():
            raise FileNotFoundError(
                f"Project directory not found: {project_dir}\n"
                f"Available projects: {', '.join([d.name for d in Path(self.output_base_dir).iterdir() if d.is_dir()])}"
            )

        # Find most recent issue_category file
        issue_category_pattern = str(project_dir / "issue_category_*.json")
        issue_category_files = sorted(glob.glob(issue_category_pattern), reverse=True)

        if not issue_category_files:
            raise FileNotFoundError(
                f"No issue_category_*.json files found in project directory: {project_dir}\n"
                f"Pattern searched: {issue_category_pattern}\n"
                f"Please run IssueCategorizer first."
            )

        # Use the most recent issue category
        issue_category_file = issue_category_files[0]

        if self.progress:
            print(f"[StrategyPlanner] Resolved issue category file: {issue_category_file}")

        return str(issue_category_file)

    def _create_planning_prompt(self) -> str:
        """Create planning prompt with issue categories context."""
        # Format issue categories as JSON
        categories = self.issue_category_data.get('categories', [])
        issue_categories_json = json.dumps(categories, indent=2, ensure_ascii=False)

        # Replace placeholder in template
        prompt = self.prompt_template.replace("{issue_categories_json}", issue_categories_json)

        return prompt

    async def _invoke_with_retry(self, prompt: str):
        """Invoke LLM with retry logic and exponential backoff."""
        delay = 1.0
        max_delay = 60.0
        last_exception = None

        for attempt in range(self.max_retries + 1):
            try:
                return await self.llm_bound.ainvoke(prompt)
            except (APIConnectionError, RateLimitError, APIError) as e:
                last_exception = e

                if attempt == self.max_retries:
                    raise

                wait_time = min(delay, max_delay)
                if self.progress:
                    print(f"[StrategyPlanner Retry {attempt + 1}/{self.max_retries}] API error: {type(e).__name__}. Retrying in {wait_time:.1f}s...")
                await asyncio.sleep(wait_time)
                delay *= 2

        if last_exception:
            raise last_exception

    async def analyze_and_plan_async(self) -> Dict[str, Any]:
        """Analyze issues and create improvement plan (async)."""
        categories = self.issue_category_data.get('categories', [])

        if self.progress:
            print(f"[StrategyPlanner] Analyzing {len(categories)} issue categories...")

        if len(categories) == 0:
            if self.progress:
                print("[StrategyPlanner] No issue categories found. Nothing to plan.")
            return {
                "timestamp": datetime.now().strftime("%Y%m%d_%H%M%S"),
                "issue_category_file": self.issue_category_file,
                "recommended_actions": []
            }

        # Create planning prompt
        prompt = self._create_planning_prompt()

        if self.progress:
            print("[StrategyPlanner] Invoking LLM for strategy planning...")

        try:
            # Invoke LLM
            response = await self._invoke_with_retry(prompt)

            # Handle response content
            if isinstance(response.content, str):
                result = json.loads(response.content)
            elif isinstance(response.content, list) and len(response.content) > 0:
                text_record = next((item for item in response.content if item.get('type') == 'text'), None)
                if text_record is None:
                    raise ValueError(f"No text record found in response.content: {response.content}")
                result = json.loads(text_record.get('text', '{}'))
            else:
                raise ValueError(f"Unexpected response format: {type(response.content)}")

            if 'recommended_actions' not in result:
                raise ValueError(f"Expected 'recommended_actions' in response, got: {result.keys()}")

            recommended_actions = result['recommended_actions']

            if self.progress:
                print(f"[StrategyPlanner] LLM generated {len(recommended_actions)} recommended actions")
                for i, action in enumerate(recommended_actions, 1):
                    priority = action.get('priority', 'unknown')
                    target_type = action.get('target_type', 'unknown')
                    target = action.get('target', 'unknown')
                    print(f"  {i}. [{priority.upper()}] {target_type}:{target}")

            # Build output
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output = {
                "timestamp": timestamp,
                "issue_category_file": self.issue_category_file,
                "recommended_actions": recommended_actions
            }

            if self.progress:
                print("[StrategyPlanner] Strategy planning completed!")

            return output

        except Exception as e:
            if self.progress:
                print(f"[StrategyPlanner] Error during planning: {e}")
            raise

    def analyze_and_plan(self) -> Dict[str, Any]:
        """Analyze issues and create improvement plan (synchronous wrapper)."""
        return asyncio.run(self.analyze_and_plan_async())

    def save_result(self, result: Dict[str, Any], output_dir: Optional[str] = None) -> str:
        """Save strategy plan to file."""
        if output_dir is None:
            # Default: save to project directory
            output_dir = Path(self.output_base_dir) / self.project
        else:
            output_dir = Path(output_dir)

        output_dir.mkdir(parents=True, exist_ok=True)

        timestamp = result['timestamp']
        output_file = output_dir / f"strategy_plan_{timestamp}.json"

        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(result, f, indent=2, ensure_ascii=False)

        return str(output_file)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Strategy Planner - Creates improvement plans by analyzing categorized issues")
    parser.add_argument(
        "--project",
        required=True,
        help="Project name (resolves to /workspace/app/output/{project}/)"
    )
    parser.add_argument(
        "--model",
        default="gpt-5-mini",
        help="LLM model name (default: gpt-5-mini)"
    )
    parser.add_argument(
        "--no-progress",
        action="store_true",
        help="Disable progress messages"
    )

    args = parser.parse_args()

    try:
        # Create planner
        print(f"[StrategyPlanner] Initializing StrategyPlanner with model: {args.model}")
        planner = StrategyPlanner(
            project=args.project,
            model=args.model,
            progress=not args.no_progress
        )

        # Run planning
        print("[StrategyPlanner] Starting strategy planning process...")
        result = planner.analyze_and_plan()

        # Save result
        output_file = planner.save_result(result)
        print(f"\n[StrategyPlanner] Planning completed!")
        print(f"[StrategyPlanner] Result saved to: {output_file}")

        # Print summary
        actions = result.get('recommended_actions', [])
        if actions:
            print(f"\n[StrategyPlanner] Summary: {len(actions)} recommended actions")
            for i, action in enumerate(actions, 1):
                target_type = action.get('target_type', '?')
                target = action.get('target', '?')
                print(f"  {i}. [{action.get('priority', '?').upper()}] {target_type}:{target}")
        else:
            print("\n[StrategyPlanner] No recommended actions generated.")

    except FileNotFoundError as e:
        print(f"\n[StrategyPlanner] Error: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\n[StrategyPlanner] Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
