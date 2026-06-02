"""Action Planner - Generates concrete fix proposals from strategy plans using LLM."""

import asyncio
import json
import sys
import argparse
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
import glob
from collections import defaultdict


# Add src directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.llm_factory import LLMFactory
from utils.llm_retry import call_with_retry
from utils.prompt_utils import fill_prompt_template


class ActionPlanner:
    """Generates concrete fix proposals from strategy plans using LLM."""

    # Change-volume cap per iteration (fixed).
    MAX_TARGETS_PER_ITER = 8
    # Compression constraints for custom prompt proposals.
    MAX_COMPRESSED_PROMPT_BYTES = 2000
    MAX_COMPRESSION_ATTEMPTS = 3

    def __init__(
        self,
        model: str = "gemini-3.1-flash-lite",
        temperature: float = 0.0,
        max_concurrency: int = 8,
        progress: bool = True,
    ):
        """
        Initialize the ActionPlanner.

        Args:
            model: LLM model name
            temperature: LLM temperature for consistency
            max_concurrency: Maximum concurrent LLM calls
            progress: Whether to show progress messages
        """
        self.model = model
        self.temperature = temperature
        self.max_concurrency = max_concurrency
        self.progress = progress
        self.output_base_dir = "/workspace/app/output/optimize"

        # Resolve strategy plan file path
        self.strategy_plan_file = self._resolve_strategy_plan_path()

        # Load strategy plan data
        self.strategy_plan_data = self._load_json(self.strategy_plan_file)

        # Load metamodel
        self.metamodel = self._load_metamodel()

        # Create LLM instance with medium reasoning effort
        self.llm = LLMFactory.create_llm(model=model, temperature=temperature, reasoning_effort="low")

        # Load schemas (path: /workspace/app/schemas/action-planner/)
        self.schemas_dir = Path(__file__).parent.parent.parent / "schemas" / "action-planner"
        self.llm_schema = self._load_schema("llm.schema.json")

        # Bind LLM with JSON schema
        self.llm_bound = self.llm.bind(
            response_format={
                "type": "json_schema",
                "json_schema": {"name": "action_planner_response", "schema": self.llm_schema, "strict": True},
            }
        )

        # Load prompt templates (path: /workspace/app/prompts/)
        self.custom_prompt_template = self._load_prompt("action_planner_custom_prompt.txt")
        self.compress_prompt_template = self._load_prompt("action_planner_compress_prompt.txt")
        self.metamodel_template = self._load_prompt("action_planner_metamodel.txt")

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

    def _load_json(self, file_path: str) -> Dict[str, Any]:
        """Load a JSON file."""
        with open(file_path, "r", encoding="utf-8") as f:
            return json.load(f)

    def _load_metamodel(self) -> Dict[str, Any]:
        """Load metamodel from metamodel.json."""
        metamodel_path = Path(__file__).parent.parent.parent / "metamodel" / "metamodel.json"
        with open(metamodel_path, "r", encoding="utf-8") as f:
            return json.load(f)

    def _resolve_strategy_plan_path(self) -> str:
        """
        Resolve strategy_plan file path.

        Returns:
            Path to the most recent strategy_plan_*.json file

        Raises:
            FileNotFoundError: If required files don't exist
        """
        optimize_dir = Path(self.output_base_dir)

        # Check if optimize directory exists
        if not optimize_dir.exists():
            raise FileNotFoundError(f"Optimize directory not found: {optimize_dir}")

        # Find most recent strategy_plan file
        strategy_plan_pattern = str(optimize_dir / "strategy_plan_*.json")
        strategy_plan_files = sorted(glob.glob(strategy_plan_pattern), reverse=True)

        if not strategy_plan_files:
            raise FileNotFoundError(
                f"No strategy_plan_*.json files found in optimize directory: {optimize_dir}\n"
                f"Pattern searched: {strategy_plan_pattern}\n"
                f"Please run StrategyPlanner first."
            )

        # Use the most recent strategy plan
        strategy_plan_file = strategy_plan_files[0]

        if self.progress:
            print(f"[ActionPlanner] Resolved strategy plan file: {strategy_plan_file}")

        return str(strategy_plan_file)

    def _filter_actions(self, actions: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Filter actions to only include high/medium priority with suggested_changes.

        Args:
            actions: List of recommended actions from strategy plan

        Returns:
            Filtered list of valid actions
        """
        valid_actions = [
            action
            for action in actions
            if action.get("priority", "").lower() in ["high", "medium"] and action.get("suggested_changes", "").strip()
        ]

        if self.progress:
            print(
                f"[ActionPlanner] Filtered actions: {len(valid_actions)}/{len(actions)} (high/medium priority with suggestions)"
            )

        return valid_actions

    def _group_actions_by_target(self, actions: List[Dict[str, Any]]) -> Dict[Tuple[str, str], List[Dict[str, Any]]]:
        """
        Group actions by (target_type, target).

        Args:
            actions: List of filtered actions

        Returns:
            Dictionary mapping (target_type, target) to list of actions
        """
        grouped = defaultdict(list)
        for action in actions:
            target_type = action.get("target_type", "")
            target = action.get("target", "")
            key = (target_type, target)
            grouped[key].append(action)

        if self.progress:
            print(f"[ActionPlanner] Grouped into {len(grouped)} unique targets")

        return dict(grouped)

    @staticmethod
    def _priority_rank(priority: str) -> int:
        """Return a sortable rank for priority (lower means higher priority)."""
        p = (priority or "").lower()
        if p == "high":
            return 0
        if p == "medium":
            return 1
        return 2

    def _select_target_keys(
        self, grouped_actions: Dict[Tuple[str, str], List[Dict[str, Any]]]
    ) -> List[Tuple[str, str]]:
        """
        Select target keys under the fixed per-iteration cap.

        Selection rule:
        1) Priority order: high -> medium
        2) Stable tie-break: first appearance order
        """
        if len(grouped_actions) <= self.MAX_TARGETS_PER_ITER:
            return list(grouped_actions.keys())

        ranked = []
        for idx, (key, target_actions) in enumerate(grouped_actions.items()):
            best_rank = min(self._priority_rank(a.get("priority", "")) for a in target_actions)
            ranked.append((best_rank, idx, key))

        ranked.sort(key=lambda x: (x[0], x[1]))
        return [key for _, _, key in ranked[: self.MAX_TARGETS_PER_ITER]]

    def _load_custom_prompt_file(self, filename: str) -> str:
        """
        Load current content of a custom prompt file.

        Args:
            filename: Name of the custom prompt file

        Returns:
            Current file content (empty string if file doesn't exist or is empty)
        """
        custom_path = Path(__file__).parent.parent.parent / "prompts" / "custom" / filename

        if not custom_path.exists():
            return ""

        with open(custom_path, "r", encoding="utf-8") as f:
            content = f.read()
            return content if content else ""

    def _get_metamodel_rules(self, target: str, rule_type: str) -> Tuple[List[str], str, str]:
        """
        Get current rules from metamodel for a class or property.

        Args:
            target: Class or property name
            rule_type: Either "extract_rules" or "prohibit_rules"

        Returns:
            Tuple of (rules_list, target_description, entity_type)

        Raises:
            ValueError: If target not found in metamodel
        """
        # Search in classes
        for cls in self.metamodel.get("classes", []):
            if cls.get("name") == target:
                rules = cls.get(rule_type, [])
                description = cls.get("description", "")
                return rules, description, "class"

        # Search in properties
        for prop in self.metamodel.get("properties", []):
            if prop.get("name") == target:
                rules = prop.get(rule_type, [])
                description = prop.get("description", "")
                return rules, description, "property"

        raise ValueError(f"Target '{target}' not found in metamodel")

    def _create_custom_prompt_plan_prompt(self, target: str, actions: List[Dict[str, Any]]) -> str:
        """Create prompt for custom_prompt target."""
        current_content = self._load_custom_prompt_file(target)
        file_purpose = self._get_custom_prompt_purpose(target)
        suggested_changes_list = self._format_suggested_changes(actions)

        # Replace placeholders
        prompt = fill_prompt_template(
            self.custom_prompt_template,
            target_file=target,
            file_purpose=file_purpose,
            current_content=current_content if current_content else "(empty file)",
            suggested_changes_list=suggested_changes_list,
        )

        return prompt

    def _create_custom_prompt_compress_prompt(
        self,
        target: str,
        actions: List[Dict[str, Any]],
        original_content: str,
        proposed_content: str,
        max_bytes: int,
        current_bytes: int,
        attempt: int,
    ) -> str:
        """Create compression prompt for custom_prompt target."""
        file_purpose = self._get_custom_prompt_purpose(target)
        suggested_changes_list = self._format_suggested_changes(actions)

        return fill_prompt_template(
            self.compress_prompt_template,
            target_file=target,
            file_purpose=file_purpose,
            original_content=original_content if original_content else "(empty file)",
            proposed_content=proposed_content if proposed_content else "(empty content)",
            suggested_changes_list=suggested_changes_list,
            target_max_bytes=max_bytes,
            current_proposed_bytes=current_bytes,
            compression_attempt=attempt,
        )

    @staticmethod
    def _content_size_bytes(content: str) -> int:
        """Return UTF-8 byte size for content."""
        return len((content or "").encode("utf-8"))

    @staticmethod
    def _get_custom_prompt_purpose(target: str) -> str:
        """Return human-readable purpose for custom prompt file."""
        purpose_map = {
            "class_extractor_custom.txt": "Guidance for extracting class instances from text",
            "class_reviewer_custom.txt": "Validation criteria for class extraction decisions",
            "class_consolidator_custom.txt": "Criteria for selecting canonical labels when merging duplicate classes",
            "property_extractor_custom.txt": "Guides which property (relationship) candidates to extract from class pairs",
            "property_ranker_custom.txt": "Guidelines for evaluating relationship strength (0-2 textual explicitness scale)",
            "property_reviewer_custom.txt": "Validation criteria for property extraction decisions",
            "property_consolidator_custom.txt": "Criteria for grouping duplicate/similar properties",
        }
        return purpose_map.get(target, "Custom rules for the extraction pipeline")

    @staticmethod
    def _format_suggested_changes(actions: List[Dict[str, Any]]) -> str:
        """Format suggested changes for prompt templates."""
        return "\n\n".join(
            [
                f"**Change {i + 1}** (Priority: {action.get('priority', 'unknown')})\n"
                f"Reason: {action.get('reason', '')}\n"
                f"Suggested: {action.get('suggested_changes', '')}"
                for i, action in enumerate(actions)
            ]
        )

    @staticmethod
    def _parse_llm_json_response(response: Any) -> Dict[str, Any]:
        """Parse LLM JSON response content."""
        if isinstance(response.content, str):
            return json.loads(response.content)

        if isinstance(response.content, list) and len(response.content) > 0:
            text_record = next((item for item in response.content if item.get("type") == "text"), None)
            if text_record is None:
                raise ValueError(f"No text record found in response.content: {response.content}")
            return json.loads(text_record.get("text", "{}"))

        raise ValueError(f"Unexpected response format: {type(response.content)}")

    async def _compress_custom_prompt_async(
        self,
        target: str,
        actions: List[Dict[str, Any]],
        original_content: str,
        proposed_content: str,
        max_bytes: int,
        current_bytes: int,
        attempt: int,
    ) -> Tuple[str, str]:
        """
        Compress proposed custom prompt while preserving intent.

        Returns:
            Tuple of (compressed_content, compression_summary)
        """
        compress_prompt = self._create_custom_prompt_compress_prompt(
            target=target,
            actions=actions,
            original_content=original_content,
            proposed_content=proposed_content,
            max_bytes=max_bytes,
            current_bytes=current_bytes,
            attempt=attempt,
        )

        response = await call_with_retry(
            lambda: self.llm_bound.ainvoke(compress_prompt),
            timeout=90.0,
            label="ActionPlanner-Compress",
        )
        compressed_result = self._parse_llm_json_response(response)
        compressed_content = compressed_result.get("updated_content_text", "")
        compression_summary = compressed_result.get("changes_summary", "")

        return compressed_content, compression_summary

    async def _compress_custom_prompt_until_limit_async(
        self, target: str, actions: List[Dict[str, Any]], original_content: str, proposed_content: str
    ) -> Tuple[str, str, int, bool]:
        """
        Compress content repeatedly until byte-size limit is satisfied or attempts are exhausted.

        Returns:
            Tuple of (final_content, merged_summary, attempts_used, within_limit)
        """
        candidate_content = proposed_content
        summaries: List[str] = []
        attempts_used = 0

        for attempt in range(1, self.MAX_COMPRESSION_ATTEMPTS + 1):
            current_bytes = self._content_size_bytes(candidate_content)
            if current_bytes <= self.MAX_COMPRESSED_PROMPT_BYTES:
                break

            if self.progress:
                print(
                    f"[ActionPlanner] Compress attempt {attempt}/{self.MAX_COMPRESSION_ATTEMPTS} "
                    f"for {target} ({current_bytes} bytes -> <= {self.MAX_COMPRESSED_PROMPT_BYTES} bytes)"
                )

            compressed_content, compression_summary = await self._compress_custom_prompt_async(
                target=target,
                actions=actions,
                original_content=original_content,
                proposed_content=candidate_content,
                max_bytes=self.MAX_COMPRESSED_PROMPT_BYTES,
                current_bytes=current_bytes,
                attempt=attempt,
            )
            attempts_used = attempt

            if not compressed_content.strip():
                if self.progress:
                    print(
                        f"[ActionPlanner] Compression attempt {attempt} returned empty content for {target}; "
                        "stopping retries and keeping last non-empty content."
                    )
                break

            candidate_content = compressed_content
            if compression_summary:
                summaries.append(f"attempt {attempt}: {compression_summary}")

        final_bytes = self._content_size_bytes(candidate_content)
        within_limit = final_bytes <= self.MAX_COMPRESSED_PROMPT_BYTES
        merged_summary = " | ".join(summaries)

        if self.progress:
            if within_limit:
                print(
                    f"[ActionPlanner] Compression result for {target}: "
                    f"{final_bytes} bytes (limit {self.MAX_COMPRESSED_PROMPT_BYTES})"
                )
            else:
                print(
                    f"[ActionPlanner] Compression limit not reached for {target}: "
                    f"{final_bytes} bytes after {attempts_used} attempts "
                    f"(limit {self.MAX_COMPRESSED_PROMPT_BYTES})"
                )

        return candidate_content, merged_summary, attempts_used, within_limit

    def _create_metamodel_rules_plan_prompt(self, target: str, target_type: str, actions: List[Dict[str, Any]]) -> str:
        """Create prompt for extract_rules or prohibit_rules target."""
        rule_type = target_type  # "extract_rules" or "prohibit_rules"

        try:
            current_rules, description, entity_type = self._get_metamodel_rules(target, rule_type)
        except ValueError as e:
            raise ValueError(f"Cannot create prompt for {target_type}:{target} - {e}")

        # Format current rules
        if current_rules:
            current_rules_list = "\n".join([f"{i + 1}. {rule}" for i, rule in enumerate(current_rules)])
        else:
            current_rules_list = "(no rules currently defined)"

        # Format suggested changes
        suggested_changes_list = "\n\n".join(
            [
                f"**Change {i + 1}** (Priority: {action.get('priority', 'unknown')})\n"
                f"Reason: {action.get('reason', '')}\n"
                f"Suggested: {action.get('suggested_changes', '')}"
                for i, action in enumerate(actions)
            ]
        )

        # Replace placeholders
        prompt = fill_prompt_template(
            self.metamodel_template,
            rule_type=rule_type,
            target_name=target,
            target_description=description,
            entity_type=entity_type,
            current_rules_list=current_rules_list,
            suggested_changes_list=suggested_changes_list,
        )

        return prompt

    async def _generate_proposal_async(
        self, target_type: str, target: str, actions: List[Dict[str, Any]], semaphore: asyncio.Semaphore
    ) -> Dict[str, Any]:
        """
        Generate a proposal for a single target (with concurrency control).

        Args:
            target_type: Type of target (custom_prompt, extract_rules, prohibit_rules)
            target: Target name (filename or class/property name)
            actions: List of actions for this target
            semaphore: Semaphore for concurrency control

        Returns:
            Proposal dictionary
        """
        async with semaphore:
            if self.progress:
                print(f"[ActionPlanner] Generating proposal for {target_type}:{target}...")

            try:
                # Create appropriate prompt
                if target_type == "custom_prompt":
                    prompt = self._create_custom_prompt_plan_prompt(target, actions)
                    current_content = self._load_custom_prompt_file(target)
                elif target_type in ["extract_rules", "prohibit_rules"]:
                    prompt = self._create_metamodel_rules_plan_prompt(target, target_type, actions)
                    current_rules, _, _ = self._get_metamodel_rules(target, target_type)
                    current_content = current_rules
                else:
                    raise ValueError(f"Unknown target_type: {target_type}")

                # Invoke LLM
                response = await call_with_retry(
                    lambda: self.llm_bound.ainvoke(prompt),
                    timeout=90.0,
                    label="ActionPlanner",
                )

                # Parse response
                result = self._parse_llm_json_response(response)

                # Extract updated_content based on target_type
                if target_type == "custom_prompt":
                    updated_content = result.get("updated_content_text", "")
                else:  # extract_rules or prohibit_rules
                    updated_content = result.get("updated_content_array", [])

                changes_summary = result.get("changes_summary", "")

                # Compression step for custom prompt proposals.
                if target_type == "custom_prompt" and isinstance(updated_content, str) and updated_content.strip():
                    if self.progress:
                        initial_bytes = self._content_size_bytes(updated_content)
                        print(
                            f"[ActionPlanner] Compressing proposal for {target} "
                            f"(initial {initial_bytes} bytes, limit {self.MAX_COMPRESSED_PROMPT_BYTES})..."
                        )
                    try:
                        (
                            compressed_content,
                            compression_summary,
                            _,
                            _,
                        ) = await self._compress_custom_prompt_until_limit_async(
                            target=target,
                            actions=actions,
                            original_content=current_content if isinstance(current_content, str) else "",
                            proposed_content=updated_content,
                        )
                        if compressed_content.strip():
                            updated_content = compressed_content
                        if compression_summary:
                            if changes_summary:
                                changes_summary = f"{changes_summary} | Compression: {compression_summary}"
                            else:
                                changes_summary = compression_summary
                    except Exception as compress_error:
                        if self.progress:
                            print(
                                f"[ActionPlanner] - Compression skipped for {target} "
                                f"(using uncompressed proposal): {compress_error}"
                            )

                if self.progress:
                    print(f"[ActionPlanner] ✓ {target_type}:{target} - {changes_summary}")

                # Build proposal
                proposal = {
                    "target_type": target_type,
                    "target": target,
                    "current_content": current_content,
                    "updated_content": updated_content,
                    "actions_addressed": [
                        {
                            "reason": action.get("reason", ""),
                            "suggested_changes": action.get("suggested_changes", ""),
                            "priority": action.get("priority", ""),
                        }
                        for action in actions
                    ],
                }

                return proposal

            except Exception as e:
                if self.progress:
                    print(f"[ActionPlanner] ✗ {target_type}:{target} - Error: {e}")
                raise

    async def generate_proposals_async(self) -> Dict[str, Any]:
        """Generate proposals for all valid actions (async)."""
        actions = self.strategy_plan_data.get("recommended_actions", [])
        total_actions = len(actions)

        if self.progress:
            print(f"[ActionPlanner] Processing {total_actions} actions from strategy plan...")

        # Filter actions (high/medium priority + has suggestions)
        valid_actions = self._filter_actions(actions)
        filtered_count = len(valid_actions)

        if filtered_count == 0:
            if self.progress:
                print("[ActionPlanner] No valid actions to process. Exiting.")
            return {
                "timestamp": datetime.now().strftime("%Y%m%d%H%M%S"),
                "strategy_plan_file": self.strategy_plan_file,
                "total_actions": total_actions,
                "filtered_actions": 0,
                "proposals": [],
            }

        # Group by target
        grouped_actions = self._group_actions_by_target(valid_actions)
        selected_target_keys = self._select_target_keys(grouped_actions)
        selected_actions_by_target = {key: grouped_actions[key] for key in selected_target_keys}

        grouped_count = len(grouped_actions)
        selected_count = len(selected_actions_by_target)
        dropped_count = grouped_count - selected_count
        processed_actions = sum(len(v) for v in selected_actions_by_target.values())
        dropped_actions = filtered_count - processed_actions

        if self.progress and dropped_count > 0:
            print(
                "[ActionPlanner] Change-volume cap applied: "
                f"selected {selected_count}/{grouped_count} targets "
                f"(max={self.MAX_TARGETS_PER_ITER}, dropped={dropped_count})"
            )

        # Create semaphore for concurrency control
        semaphore = asyncio.Semaphore(self.max_concurrency)

        # Generate proposals in parallel
        if self.progress:
            print(f"[ActionPlanner] Generating proposals (max concurrency: {self.max_concurrency})...")

        tasks = [
            self._generate_proposal_async(target_type, target, target_actions, semaphore)
            for (target_type, target), target_actions in selected_actions_by_target.items()
        ]

        proposals = await asyncio.gather(*tasks)

        # Build output
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        output = {
            "timestamp": timestamp,
            "strategy_plan_file": self.strategy_plan_file,
            "total_actions": total_actions,
            "filtered_actions": filtered_count,
            "grouped_targets": grouped_count,
            "selected_targets": selected_count,
            "dropped_targets": dropped_count,
            "processed_actions": processed_actions,
            "dropped_actions": dropped_actions,
            "max_targets_per_iter": self.MAX_TARGETS_PER_ITER,
            "proposals": proposals,
        }

        if self.progress:
            print(f"[ActionPlanner] ✓ Generated {len(proposals)} proposals!")

        return output

    def generate_proposals(self) -> Dict[str, Any]:
        """Generate proposals for all valid actions (synchronous wrapper)."""
        return asyncio.run(self.generate_proposals_async())

    def save_result(self, result: Dict[str, Any], output_dir: Optional[str] = None) -> str:
        """Save action proposals to file."""
        if output_dir is None:
            # Default: save to optimize directory
            output_dir = Path(self.output_base_dir)
        else:
            output_dir = Path(output_dir)

        output_dir.mkdir(parents=True, exist_ok=True)

        timestamp = result["timestamp"]
        output_file = output_dir / f"action_proposals_{timestamp}.json"

        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)

        return str(output_file)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Action Planner - Generates concrete fix proposals from strategy plans"
    )
    parser.add_argument(
        "--model", default="gemini-3.1-flash-lite", help="LLM model name (default: gemini-3.1-flash-lite)"
    )
    parser.add_argument("--temperature", type=float, default=0.0, help="LLM temperature (default: 0.0)")
    parser.add_argument("--max-concurrency", type=int, default=8, help="Maximum concurrent LLM calls (default: 8)")
    parser.add_argument("--no-progress", action="store_true", help="Disable progress messages")

    args = parser.parse_args()

    try:
        # Create planner
        print(f"[ActionPlanner] Initializing ActionPlanner with model: {args.model}")
        planner = ActionPlanner(
            model=args.model,
            temperature=args.temperature,
            max_concurrency=args.max_concurrency,
            progress=not args.no_progress,
        )

        # Generate proposals
        print("[ActionPlanner] Starting proposal generation...")
        result = planner.generate_proposals()

        # Save result
        output_file = planner.save_result(result)
        print("\n[ActionPlanner] Proposal generation completed!")
        print(f"[ActionPlanner] Result saved to: {output_file}")

        # Print summary
        proposals = result.get("proposals", [])
        total_actions = result.get("total_actions", 0)
        filtered_actions = result.get("filtered_actions", 0)

        print("\n[ActionPlanner] Summary:")
        print(f"  Total actions in strategy plan: {total_actions}")
        print(f"  Actions processed (high/medium with suggestions): {filtered_actions}")
        print(f"  Proposals generated: {len(proposals)}")

        if proposals:
            print("\n[ActionPlanner] Proposals:")
            for i, proposal in enumerate(proposals, 1):
                target_type = proposal.get("target_type", "?")
                target = proposal.get("target", "?")
                num_actions = len(proposal.get("actions_addressed", []))
                print(f"  {i}. {target_type}:{target} ({num_actions} actions)")

    except FileNotFoundError as e:
        print(f"\n[ActionPlanner] Error: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\n[ActionPlanner] Unexpected error: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)
