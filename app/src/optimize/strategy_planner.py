"""Strategy Planner - Creates improvement plans using issue statistics, strategy history, and reward history."""

import argparse
import asyncio
import glob
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

# Add src directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.utils.llm_factory import DEFAULT_LLM_MODEL, LLMFactory
from src.utils.llm_retry import call_with_retry
from src.utils.prompt_utils import fill_prompt_template


class StrategyPlanner:
    """Creates improvement plans using issue statistics, strategy history, and reward history."""

    def __init__(
        self,
        model: str | None = None,
        progress: bool = True,
        verbose: bool = False,
        issue_statistics: Optional[List[Dict[str, Any]]] = None,
        previous_strategies: Optional[List[Dict[str, Any]]] = None,
        previous_rewards: Optional[List[float]] = None,
        run_id: Optional[str] = None,
        output_base_dir: str = "/workspace/app/output",
    ):
        """
        Initialize the StrategyPlanner.

        Args:
            model: LLM model name
            progress: Whether to show progress messages
            verbose: If True, LLM outputs reason field for each action.
                     If False (default), reason is omitted to reduce token consumption.
            issue_statistics: Issue category counts [{category, count}], ranked by frequency
            previous_strategies: Compact strategy history from recent iterations
            previous_rewards: Combined total quality scores from recent iterations
            run_id: Iteration run ID for scoping strategy history log.
                    When set, auto-loads history and saves to iterations/{run_id}/strategy_planner.json.
                    When None, history logging is disabled.
            output_base_dir: Base output directory
        """
        self.model = model
        self.progress = progress
        self.verbose = verbose
        self.output_base_dir = output_base_dir
        self.optimize_dir = str(Path(output_base_dir) / "optimize")
        self.run_id = run_id

        # Set input data
        self.issue_statistics = issue_statistics or []
        self.previous_rewards = previous_rewards or []

        # Load strategy history: use explicit param if provided, else auto-load from log
        if previous_strategies is not None:
            self.previous_strategies = previous_strategies
        elif run_id is not None:
            self.previous_strategies = self._load_strategy_history()
        else:
            self.previous_strategies = []

        self.llm = LLMFactory.create_llm(model=model, temperature=0.0, timeout=300.0)

        # Load schema (path: /workspace/app/schemas/strategy-planner/)
        self.schemas_dir = Path(__file__).parent.parent.parent / "schemas" / "strategy-planner"
        llm_schema_file = "llm.schema.json" if verbose else "llm-compact.schema.json"
        self.llm_schema = self._load_schema(llm_schema_file)

        # Bind LLM with JSON schema
        self.llm_bound = self.llm.bind(
            response_format={
                "type": "json_schema",
                "json_schema": {"name": "strategy_planner_response", "schema": self.llm_schema, "strict": True},
            }
        )

        # Load prompt template (path: /workspace/app/prompts/)
        self.prompt_template = self._load_prompt("strategy_planner.txt")

        # Load metamodel for dynamic prompt injection
        self.metamodel = self._load_metamodel()

    def _load_schema(self, schema_name: str) -> Dict[str, Any]:
        """Load a JSON schema file."""
        schema_path = self.schemas_dir / schema_name
        with open(schema_path, "r", encoding="utf-8") as f:
            return json.load(f)

    def _load_prompt(self, prompt_filename: str) -> str:
        """Load a prompt template."""
        prompt_path = Path(__file__).parent.parent.parent / "prompts" / prompt_filename
        with open(prompt_path, "r", encoding="utf-8") as f:
            return f.read()

    def _load_metamodel(self) -> Dict[str, Any]:
        """Load metamodel from metamodel.json."""
        metamodel_path = Path(__file__).parent.parent.parent / "metamodel" / "metamodel.json"
        with open(metamodel_path, "r", encoding="utf-8") as f:
            return json.load(f)

    def _create_classes_list(self) -> str:
        """Create a simple list of metamodel class names."""
        classes = self.metamodel.get("classes", [])

        # Extract class names and format as comma-separated list
        class_names = [cls.get("name", "") for cls in classes]

        # Format as a readable list
        return ", ".join(f'"{name}"' for name in class_names)

    def _create_properties_list(self) -> str:
        """Create a simple list of metamodel property names."""
        properties = self.metamodel.get("properties", [])

        # Extract property names and format as comma-separated list
        property_names = [prop.get("name", "") for prop in properties]

        # Format as a readable list
        return ", ".join(f'"{name}"' for name in property_names)

    # ---- Strategy history log (scoped by run_id) ----

    def _get_strategy_history_path(self) -> Optional[Path]:
        """Get the path to the strategy history log file. Returns None if run_id is not set."""
        if self.run_id is None:
            return None
        return Path(self.output_base_dir) / "iterations" / self.run_id / "strategy_planner.json"

    def _load_strategy_history(self) -> List[Dict[str, Any]]:
        """Load strategy history from the run-scoped log file.

        Returns:
            List of strategy history entries, or [] if not found or run_id is None.
        """
        history_path = self._get_strategy_history_path()
        if history_path is None or not history_path.exists():
            return []

        try:
            with open(history_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            return data.get("history", [])
        except json.JSONDecodeError, IOError:
            return []

    def _save_strategy_history(self, recommended_actions: List[Dict[str, Any]]) -> None:
        """Append a compact strategy entry to the run-scoped log file.

        No-op if run_id is None.

        Args:
            recommended_actions: The full recommended_actions list from the strategy plan
        """
        history_path = self._get_strategy_history_path()
        if history_path is None:
            return

        # Ensure directory exists
        history_path.parent.mkdir(parents=True, exist_ok=True)

        # Load existing history
        history = self._load_strategy_history()

        # Create compact entry
        compact_actions = [
            {
                "target_type": a.get("target_type", ""),
                "target": a.get("target", ""),
                "priority": a.get("priority", ""),
            }
            for a in recommended_actions
        ]

        entry = {
            "timestamp": datetime.now().strftime("%Y%m%d%H%M%S"),
            "actions": compact_actions,
        }
        history.append(entry)

        # Save
        data = {
            "run_id": self.run_id,
            "history": history,
        }
        with open(history_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

        if self.progress:
            print(f"[StrategyPlanner] Strategy history saved to: {history_path}")

    def get_recent_strategies(self, last_n: int = 5) -> List[Dict[str, Any]]:
        """Return the last N strategy history entries.

        Args:
            last_n: Number of recent entries to return

        Returns:
            List of strategy history entries (most recent last)
        """
        history = self._load_strategy_history()
        return history[-last_n:] if history else []

    # ---- Issue statistics computation ----

    @staticmethod
    def compute_issue_statistics(
        issue_category_data: Dict[str, Any],
        metamodel: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        """Compute category counts with class/property breakdowns, sorted by frequency.

        Args:
            issue_category_data: Loaded issue_category_*.json data
            metamodel: Optional metamodel dict for IRI-to-name mapping

        Returns:
            List of {category, count, class_breakdown?, property_breakdown?} dicts,
            sorted by count descending
        """
        # Build IRI -> name lookup from metamodel
        class_iri_to_name: Dict[str, str] = {}
        prop_iri_to_name: Dict[str, str] = {}
        if metamodel:
            for cls in metamodel.get("classes", []):
                class_iri_to_name[cls.get("iri", "")] = cls.get("name", "")
            for prop in metamodel.get("properties", []):
                prop_iri_to_name[prop.get("iri", "")] = prop.get("name", "")

        def _resolve_class_name(iri: str) -> str:
            if iri in class_iri_to_name:
                return class_iri_to_name[iri]
            return iri.rsplit("#", 1)[-1] if "#" in iri else iri

        def _resolve_prop_name(iri: str) -> str:
            if iri in prop_iri_to_name:
                return prop_iri_to_name[iri]
            return iri.rsplit("#", 1)[-1] if "#" in iri else iri

        categories = issue_category_data.get("categories", [])
        stats = []
        for cat in categories:
            validations = cat.get("validations", {})
            class_vals = validations.get("class_validations", [])
            prop_vals = validations.get("property_validations", [])
            count = len(class_vals) + len(prop_vals)

            entry: Dict[str, Any] = {
                "category": cat.get("category", ""),
                "count": count,
            }

            # Class breakdown by metamodel class type
            if class_vals:
                class_counts: Dict[str, int] = {}
                for cv in class_vals:
                    name = _resolve_class_name(cv.get("class_iri", ""))
                    class_counts[name] = class_counts.get(name, 0) + 1
                entry["class_breakdown"] = dict(sorted(class_counts.items(), key=lambda x: x[1], reverse=True))

            # Property breakdown by metamodel property type
            if prop_vals:
                prop_counts: Dict[str, int] = {}
                for pv in prop_vals:
                    name = _resolve_prop_name(pv.get("property_iri", ""))
                    prop_counts[name] = prop_counts.get(name, 0) + 1
                entry["property_breakdown"] = dict(sorted(prop_counts.items(), key=lambda x: x[1], reverse=True))

            stats.append(entry)

        stats.sort(key=lambda x: x["count"], reverse=True)
        return stats

    # ---- Prompt construction ----

    def _create_planning_prompt(self) -> str:
        """Create planning prompt with issue statistics, strategy history, reward history, and metamodel context."""
        issue_statistics_json = json.dumps(self.issue_statistics, indent=2, ensure_ascii=False)
        previous_strategies_json = json.dumps(self.previous_strategies, indent=2, ensure_ascii=False)
        previous_rewards_json = json.dumps(self.previous_rewards, ensure_ascii=False)

        # Generate metamodel name lists
        classes_list = self._create_classes_list()
        properties_list = self._create_properties_list()

        # Generate reason instruction based on verbose flag
        if self.verbose:
            reason_instruction = (
                "- **reason**: Why this edit will address the quality issues "
                "(reference specific issue categories/labels)"
            )
            reason_example = (
                '\n      "reason": "Multiple instances of RELATIONSHIP_STRENGTH_OVERESTIMATION suggest the ranker '
                "is assigning strength=3 to weak dependencies. Need clearer criteria for distinguishing strong vs "
                'weak relationships.",'
            )
        else:
            reason_instruction = ""
            reason_example = ""

        # Replace all placeholders in template
        prompt = fill_prompt_template(
            self.prompt_template,
            issue_statistics_json=issue_statistics_json,
            previous_strategies_json=previous_strategies_json,
            previous_rewards_json=previous_rewards_json,
            metamodel_classes_list=classes_list,
            metamodel_properties_list=properties_list,
            reason_instruction=reason_instruction,
            reason_example=reason_example,
        )

        return prompt

    # ---- Main execution ----

    async def analyze_and_plan_async(self) -> Dict[str, Any]:
        """Analyze issues and create improvement plan (async)."""
        if self.progress:
            print(f"[StrategyPlanner] Analyzing {len(self.issue_statistics)} issue categories...")
            print(f"[StrategyPlanner] Previous strategies: {len(self.previous_strategies)} iterations")
            print(f"[StrategyPlanner] Previous rewards: {self.previous_rewards}")

        if len(self.issue_statistics) == 0:
            if self.progress:
                print("[StrategyPlanner] No issue statistics provided. Nothing to plan.")
            return {
                "timestamp": datetime.now().strftime("%Y%m%d%H%M%S"),
                "input_summary": {
                    "issue_statistics_count": 0,
                    "previous_strategies_count": len(self.previous_strategies),
                    "previous_rewards": self.previous_rewards,
                },
                "recommended_actions": [],
            }

        # Create planning prompt
        prompt = self._create_planning_prompt()

        if self.progress:
            print("[StrategyPlanner] Invoking LLM for strategy planning...")

        try:
            # Invoke LLM
            response = await call_with_retry(
                lambda: self.llm_bound.ainvoke(prompt),
                timeout=300.0,
                label="StrategyPlanner",
            )

            # Handle response content
            if isinstance(response.content, str):
                result = json.loads(response.content)
            elif isinstance(response.content, list) and len(response.content) > 0:
                text_record = next((item for item in response.content if item.get("type") == "text"), None)
                if text_record is None:
                    raise ValueError(f"No text record found in response.content: {response.content}")
                result = json.loads(text_record.get("text", "{}"))
            else:
                raise ValueError(f"Unexpected response format: {type(response.content)}")

            if "recommended_actions" not in result:
                raise ValueError(f"Expected 'recommended_actions' in response, got: {result.keys()}")

            recommended_actions = result["recommended_actions"]

            # Normalize compact format: inject empty reason for downstream compatibility
            if not self.verbose:
                for action in recommended_actions:
                    if "reason" not in action:
                        action["reason"] = ""

            if self.progress:
                print(f"[StrategyPlanner] LLM generated {len(recommended_actions)} recommended actions")
                for i, action in enumerate(recommended_actions, 1):
                    priority = action.get("priority", "unknown")
                    target_type = action.get("target_type", "unknown")
                    target = action.get("target", "unknown")
                    print(f"  {i}. [{priority.upper()}] {target_type}:{target}")

            # Build output
            timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
            output = {
                "timestamp": timestamp,
                "input_summary": {
                    "issue_statistics_count": len(self.issue_statistics),
                    "previous_strategies_count": len(self.previous_strategies),
                    "previous_rewards": self.previous_rewards,
                },
                "recommended_actions": recommended_actions,
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
        """Save strategy plan to file and append to strategy history log."""
        if output_dir is None:
            # Default: save to optimize directory
            output_dir = Path(self.optimize_dir)
        else:
            output_dir = Path(output_dir)

        output_dir.mkdir(parents=True, exist_ok=True)

        timestamp = result["timestamp"]
        output_file = output_dir / f"strategy_plan_{timestamp}.json"

        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)

        # Append to strategy history log (scoped by run_id)
        self._save_strategy_history(result.get("recommended_actions", []))

        return str(output_file)


def _find_latest_issue_category(optimize_dir: str = "/workspace/app/output/optimize") -> str:
    """Find the most recent issue_category file for standalone CLI usage.

    Args:
        optimize_dir: Directory to search in

    Returns:
        Path to the most recent issue_category_*.json file

    Raises:
        FileNotFoundError: If no issue_category files exist
    """
    pattern = str(Path(optimize_dir) / "issue_category_*.json")
    files = sorted(glob.glob(pattern), reverse=True)

    if not files:
        raise FileNotFoundError(
            f"No issue_category_*.json files found in: {optimize_dir}\nPlease run IssueCategorizer first."
        )

    return files[0]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Strategy Planner - Creates improvement plans by analyzing categorized issues"
    )
    parser.add_argument("--model", default=DEFAULT_LLM_MODEL, help="LLM model name (default: %(default)s)")
    parser.add_argument("--no-progress", action="store_true", help="Disable progress messages")
    parser.add_argument(
        "--verbose",
        action="store_true",
        default=False,
        help="Output reason field for each action (default: omit to reduce tokens)",
    )

    args = parser.parse_args()

    try:
        # Load issue_category file and compute statistics for standalone mode
        issue_category_file = _find_latest_issue_category()
        print(f"[StrategyPlanner] Loading issue categories from: {issue_category_file}")
        with open(issue_category_file, "r", encoding="utf-8") as f:
            issue_category_data = json.load(f)
        # Load metamodel for IRI-to-name resolution in statistics
        metamodel_path = Path(__file__).parent.parent.parent / "metamodel" / "metamodel.json"
        with open(metamodel_path, "r", encoding="utf-8") as f:
            metamodel_data = json.load(f)
        issue_stats = StrategyPlanner.compute_issue_statistics(issue_category_data, metamodel=metamodel_data)
        print(f"[StrategyPlanner] Computed {len(issue_stats)} issue category statistics")

        # Create planner (standalone mode: no run_id, no history)
        print(f"[StrategyPlanner] Initializing StrategyPlanner with model: {args.model}")
        planner = StrategyPlanner(
            model=args.model,
            progress=not args.no_progress,
            verbose=args.verbose,
            issue_statistics=issue_stats,
            previous_strategies=[],
            previous_rewards=[],
        )

        # Run planning
        print("[StrategyPlanner] Starting strategy planning process...")
        result = planner.analyze_and_plan()

        # Save result
        output_file = planner.save_result(result)
        print("\n[StrategyPlanner] Planning completed!")
        print(f"[StrategyPlanner] Result saved to: {output_file}")

        # Print summary
        actions = result.get("recommended_actions", [])
        if actions:
            print(f"\n[StrategyPlanner] Summary: {len(actions)} recommended actions")
            for i, action in enumerate(actions, 1):
                target_type = action.get("target_type", "?")
                target = action.get("target", "?")
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
