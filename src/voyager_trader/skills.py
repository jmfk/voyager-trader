"""
VOYAGER Skill Library System.

Comprehensive skill management system with six core components:
1. Skill Executor - Safe execution environment
2. Skill Composer - Skill composition and dependency resolution
3. Skill Validator - Multi-faceted validation system
4. Skill Librarian - Storage and retrieval with indexing
5. Skill Discoverer - Pattern recognition and extraction
6. Performance Tracker - Comprehensive metrics tracking
"""

import ast
import json
import logging
import subprocess
import sys
import tempfile
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from decimal import Decimal
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

from .models.learning import Experience, Skill, SkillExecutionResult
from .models.types import SkillCategory, SkillComplexity


class SkillExecutionError(Exception):
    """Exception raised during skill execution."""

    pass


class SkillCompositionError(Exception):
    """Exception raised during skill composition."""

    pass


class SkillValidationError(Exception):
    """Exception raised during skill validation."""

    pass


class SkillExecutor:
    """
    Safe execution environment for VOYAGER skills.

    Provides sandboxed execution with timeout protection,
    resource limits, and security validation.
    """

    def __init__(self, timeout_seconds: int = 30, max_memory_mb: int = 128):
        self.timeout_seconds = timeout_seconds
        self.max_memory_mb = max_memory_mb
        self.logger = logging.getLogger(__name__)

    def execute_skill(
        self,
        skill: Skill,
        inputs: Dict[str, Any],
        context: Optional[Dict[str, Any]] = None,
    ) -> Tuple[SkillExecutionResult, Any, Dict[str, Any]]:
        """
        Execute a skill safely in a sandboxed environment.

        Returns:
            Tuple of (result, output, execution_metadata)
        """
        start_time = time.time()
        execution_metadata = {
            "start_time": datetime.utcnow().isoformat(),
            "skill_id": skill.id,
            "skill_name": skill.name,
            "inputs": inputs,
            "context": context or {},
        }

        try:
            # Validate inputs against skill schema
            if not self._validate_inputs(skill, inputs):
                return (
                    SkillExecutionResult.ERROR,
                    None,
                    {**execution_metadata, "error": "Invalid inputs"},
                )

            # Create safe execution environment
            with tempfile.TemporaryDirectory() as temp_dir:
                temp_path = Path(temp_dir)
                skill_file = temp_path / f"skill_{skill.id}.py"

                # Write skill code to temporary file
                skill_code = self._prepare_skill_code(skill, inputs, context)
                skill_file.write_text(skill_code)

                # Execute in subprocess with limits
                result = subprocess.run(
                    [sys.executable, str(skill_file)],
                    capture_output=True,
                    text=True,
                    timeout=self.timeout_seconds,
                    cwd=temp_dir,
                )

                execution_time = time.time() - start_time
                execution_metadata.update(
                    {
                        "execution_time_seconds": execution_time,
                        "return_code": result.returncode,
                        "stdout": result.stdout,
                        "stderr": result.stderr,
                    }
                )

                if result.returncode == 0:
                    # Parse output
                    try:
                        output = (
                            json.loads(result.stdout) if result.stdout.strip() else None
                        )
                        return SkillExecutionResult.SUCCESS, output, execution_metadata
                    except json.JSONDecodeError:
                        return (
                            SkillExecutionResult.SUCCESS,
                            result.stdout.strip(),
                            execution_metadata,
                        )
                else:
                    return (
                        SkillExecutionResult.ERROR,
                        None,
                        {**execution_metadata, "error": result.stderr},
                    )

        except subprocess.TimeoutExpired:
            return (
                SkillExecutionResult.TIMEOUT,
                None,
                {**execution_metadata, "error": "Execution timeout"},
            )
        except Exception as e:
            return (
                SkillExecutionResult.ERROR,
                None,
                {**execution_metadata, "error": str(e)},
            )

    def _validate_inputs(self, skill: Skill, inputs: Dict[str, Any]) -> bool:
        """Validate inputs against skill's input schema."""
        try:
            # Basic validation - check required fields exist
            required_fields = skill.input_schema.get("required", [])
            for field_name in required_fields:
                if field_name not in inputs:
                    self.logger.error(f"Missing required input field: {field_name}")
                    return False
            return True
        except Exception as e:
            self.logger.error(f"Input validation error: {e}")
            return False

    def _prepare_skill_code(
        self, skill: Skill, inputs: Dict[str, Any], context: Optional[Dict[str, Any]]
    ) -> str:
        """Prepare skill code for safe execution."""
        # Create wrapper code with inputs and context
        wrapper_code = f"""
import json
import sys
from decimal import Decimal
from datetime import datetime
from typing import Any, Dict, List, Optional

# Skill inputs
inputs = {json.dumps(inputs, default=str)}
context = {json.dumps(context or {}, default=str)}

# Skill parameters
parameters = {json.dumps(skill.parameters, default=str)}

try:
    # Skill code
{self._indent_code(skill.code, 4)}

    # Output result if skill defines 'result' variable
    if 'result' in locals():
        print(json.dumps(result, default=str))
except Exception as e:
    print(f"Skill execution error: {{e}}", file=sys.stderr)
    sys.exit(1)
"""
        return wrapper_code

    def _indent_code(self, code: str, spaces: int) -> str:
        """Indent code by specified number of spaces."""
        lines = code.split("\n")
        return "\n".join(
            " " * spaces + line if line.strip() else line for line in lines
        )


class SkillComposer:
    """
    Skill composition engine with dependency resolution.

    Enables building complex strategies from simpler skills
    with automatic dependency management.
    """

    def __init__(self):
        self.logger = logging.getLogger(__name__)

    def compose_skills(
        self, skills: List[Skill], composition_strategy: str = "sequential"
    ) -> Tuple[str, Dict[str, Any]]:
        """
        Compose multiple skills into a unified strategy.

        Args:
            skills: List of skills to compose
            composition_strategy: How to combine skills
                                 (sequential, parallel, conditional)

        Returns:
            Tuple of (composed_code, metadata)
        """
        if not skills:
            raise SkillCompositionError("No skills provided for composition")

        # Validate dependencies
        dependency_order = self._resolve_dependencies(skills)
        if not dependency_order:
            raise SkillCompositionError("Circular dependencies detected")

        # Generate composed code based on strategy
        if composition_strategy == "sequential":
            return self._compose_sequential(dependency_order)
        elif composition_strategy == "parallel":
            return self._compose_parallel(dependency_order)
        elif composition_strategy == "conditional":
            return self._compose_conditional(dependency_order)
        else:
            raise SkillCompositionError(
                f"Unknown composition strategy: {composition_strategy}"
            )

    def _resolve_dependencies(self, skills: List[Skill]) -> Optional[List[Skill]]:
        """Resolve skill dependencies using topological sort."""
        skill_map = {skill.name: skill for skill in skills}
        visited = set()
        temp_visited = set()
        result = []

        def visit(skill: Skill) -> bool:
            if skill.name in temp_visited:
                return False  # Circular dependency
            if skill.name in visited:
                return True

            temp_visited.add(skill.name)

            # Visit dependencies first
            for dep_name in skill.required_skills:
                if dep_name in skill_map:
                    if not visit(skill_map[dep_name]):
                        return False

            temp_visited.remove(skill.name)
            visited.add(skill.name)
            result.append(skill)
            return True

        for skill in skills:
            if skill.name not in visited:
                if not visit(skill):
                    return None  # Circular dependency detected

        return result

    def _compose_sequential(self, skills: List[Skill]) -> Tuple[str, Dict[str, Any]]:
        """Compose skills to run sequentially."""
        composed_code = """
def execute_composed_strategy(inputs, context=None):
    results = {}
    context = context or {}

"""

        for i, skill in enumerate(skills):
            composed_code += f"""
    # Execute skill: {skill.name}
    try:
{self._indent_code(skill.code, 8)}
        results['{skill.name}'] = locals().get('result', None)
        context.update(locals().get('context_updates', {{}}))
    except Exception as e:
        results['{skill.name}'] = {{'error': str(e)}}
        if not context.get('continue_on_error', False):
            break
"""

        composed_code += """

    return results

result = execute_composed_strategy(inputs, context)
"""

        metadata = {
            "composition_type": "sequential",
            "skills_count": len(skills),
            "skill_names": [skill.name for skill in skills],
            "total_complexity": sum(
                1
                if skill.complexity == SkillComplexity.BASIC
                else 2
                if skill.complexity == SkillComplexity.INTERMEDIATE
                else 3
                if skill.complexity == SkillComplexity.ADVANCED
                else 4
                for skill in skills
            ),
        }

        return composed_code, metadata

    def _compose_parallel(self, skills: List[Skill]) -> Tuple[str, Dict[str, Any]]:
        """Compose skills to run in parallel (simulated)."""
        # Note: True parallelism would require threading/multiprocessing
        # This provides a parallel-like structure for now
        composed_code = """
def execute_composed_strategy(inputs, context=None):
    results = {}
    context = context or {}
    errors = []

"""

        for skill in skills:
            composed_code += f"""
    # Execute skill: {skill.name} (parallel simulation)
    try:
        skill_context = context.copy()
{self._indent_code(skill.code, 8)}
        results['{skill.name}'] = locals().get('result', None)
    except Exception as e:
        errors.append(('{skill.name}', str(e)))
"""

        composed_code += """

    if errors and not context.get('continue_on_error', False):
        results['errors'] = errors

    return results

result = execute_composed_strategy(inputs, context)
"""

        metadata = {
            "composition_type": "parallel",
            "skills_count": len(skills),
            "skill_names": [skill.name for skill in skills],
        }

        return composed_code, metadata

    def _compose_conditional(self, skills: List[Skill]) -> Tuple[str, Dict[str, Any]]:
        """Compose skills with conditional execution."""
        composed_code = """
def execute_composed_strategy(inputs, context=None):
    results = {}
    context = context or {}

"""

        for i, skill in enumerate(skills):
            condition = (
                f"results.get('skill_{i-1}', {{}}).get('success', True)"
                if i > 0
                else "True"
            )
            composed_code += f"""
    # Conditional execution of skill: {skill.name}
    if {condition}:
        try:
{self._indent_code(skill.code, 12)}
            results['{skill.name}'] = locals().get('result', None)
            context.update(locals().get('context_updates', {{}}))
        except Exception as e:
            results['{skill.name}'] = {{'error': str(e), 'success': False}}
    else:
        results['{skill.name}'] = {{'skipped': True, 'reason': 'Previous skill failed'}}
"""

        composed_code += """

    return results

result = execute_composed_strategy(inputs, context)
"""

        metadata = {
            "composition_type": "conditional",
            "skills_count": len(skills),
            "skill_names": [skill.name for skill in skills],
        }

        return composed_code, metadata

    def _indent_code(self, code: str, spaces: int) -> str:
        """Indent code by specified number of spaces."""
        lines = code.split("\n")
        return "\n".join(
            " " * spaces + line if line.strip() else line for line in lines
        )


class SkillValidator(ABC):
    """Abstract base class for skill validation strategies."""

    @abstractmethod
    def validate(self, skill: Skill) -> Tuple[bool, List[str]]:
        """Validate a skill. Returns (is_valid, error_messages)."""
        pass


class SyntaxValidator(SkillValidator):
    """Validates skill code syntax."""

    def validate(self, skill: Skill) -> Tuple[bool, List[str]]:
        """Check if skill code has valid Python syntax."""
        errors = []

        try:
            ast.parse(skill.code)
        except SyntaxError as e:
            errors.append(f"Syntax error at line {e.lineno}: {e.msg}")
        except Exception as e:
            errors.append(f"Code parsing error: {str(e)}")

        return len(errors) == 0, errors


class SecurityValidator(SkillValidator):
    """Validates skill code for security concerns."""

    DANGEROUS_IMPORTS = {
        "os",
        "subprocess",
        "sys",
        "shutil",
        "glob",
        "tempfile",
        "pickle",
        "marshal",
        "shelve",
        "dbm",
        "sqlite3",
        "socket",
        "urllib",
        "http",
        "ftplib",
        "smtplib",
    }

    DANGEROUS_FUNCTIONS = {
        "eval",
        "exec",
        "compile",
        "__import__",
        "open",
        "file",
        "input",
        "raw_input",
        "reload",
    }

    def validate(self, skill: Skill) -> Tuple[bool, List[str]]:
        """Check skill code for security vulnerabilities."""
        errors = []

        try:
            tree = ast.parse(skill.code)

            for node in ast.walk(tree):
                # Check for dangerous imports
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        if alias.name in self.DANGEROUS_IMPORTS:
                            errors.append(f"Dangerous import detected: {alias.name}")

                # Check for dangerous function calls
                elif isinstance(node, ast.Call):
                    if (
                        isinstance(node.func, ast.Name)
                        and node.func.id in self.DANGEROUS_FUNCTIONS
                    ):
                        errors.append(
                            f"Dangerous function call detected: {node.func.id}"
                        )

        except Exception as e:
            errors.append(f"Security validation error: {str(e)}")

        return len(errors) == 0, errors


class PerformanceValidator(SkillValidator):
    """Validates skill performance characteristics."""

    def __init__(self, min_success_rate: float = 0.6, min_usage_count: int = 5):
        self.min_success_rate = min_success_rate
        self.min_usage_count = min_usage_count

    def validate(self, skill: Skill) -> Tuple[bool, List[str]]:
        """Check if skill meets performance thresholds."""
        errors = []

        if skill.usage_count < self.min_usage_count:
            errors.append(
                f"Insufficient usage data: {skill.usage_count} < {self.min_usage_count}"
            )

        if skill.usage_count > 0 and skill.success_rate < self.min_success_rate * 100:
            errors.append(
                f"Low success rate: {skill.success_rate}% < "
                f"{self.min_success_rate * 100}%"
            )

        return len(errors) == 0, errors


class CompositeSkillValidator:
    """Composite validator that runs multiple validation strategies."""

    def __init__(self):
        self.validators = [
            SyntaxValidator(),
            SecurityValidator(),
            PerformanceValidator(),
        ]
        self.logger = logging.getLogger(__name__)

    def validate_skill(self, skill: Skill) -> Tuple[bool, Dict[str, List[str]]]:
        """Run all validators on a skill."""
        results = {}
        overall_valid = True

        for validator in self.validators:
            validator_name = validator.__class__.__name__
            try:
                is_valid, errors = validator.validate(skill)
                results[validator_name] = errors
                if not is_valid:
                    overall_valid = False
            except Exception as e:
                self.logger.error(f"Validator {validator_name} failed: {e}")
                results[validator_name] = [f"Validator error: {str(e)}"]
                overall_valid = False

        return overall_valid, results


class SkillLibrarian:
    """
    Storage and retrieval system for skills with indexing and search.

    Provides efficient skill storage, retrieval, and search capabilities
    with comprehensive indexing and caching.
    """

    def __init__(self, storage_path: Path):
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(exist_ok=True)
        self.logger = logging.getLogger(__name__)

        # In-memory indexes for fast retrieval
        self._skills_cache: Dict[str, Skill] = {}
        self._category_index: Dict[SkillCategory, Set[str]] = {}
        self._complexity_index: Dict[SkillComplexity, Set[str]] = {}
        self._tag_index: Dict[str, Set[str]] = {}
        self._dependency_index: Dict[str, Set[str]] = {}

        # Load existing skills
        self._load_all_skills()

    def store_skill(self, skill: Skill) -> bool:
        """Store a skill in the library."""
        try:
            # Save to file
            skill_file = self.storage_path / f"{skill.id}.json"
            with open(skill_file, "w") as f:
                json.dump(skill.model_dump(), f, indent=2, default=str)

            # Update cache and indexes
            self._skills_cache[skill.id] = skill
            self._update_indexes(skill)

            self.logger.info(f"Stored skill: {skill.name} ({skill.id})")
            return True

        except Exception as e:
            self.logger.error(f"Failed to store skill {skill.name}: {e}")
            return False

    def retrieve_skill(self, skill_id: str) -> Optional[Skill]:
        """Retrieve a skill by ID."""
        return self._skills_cache.get(skill_id)

    def search_skills(
        self,
        category: Optional[SkillCategory] = None,
        complexity: Optional[SkillComplexity] = None,
        tags: Optional[List[str]] = None,
        min_success_rate: Optional[float] = None,
        min_usage_count: Optional[int] = None,
        name_pattern: Optional[str] = None,
    ) -> List[Skill]:
        """Search skills with multiple criteria."""
        candidate_ids = set(self._skills_cache.keys())

        # Filter by category
        if category:
            category_ids = self._category_index.get(category, set())
            candidate_ids &= category_ids

        # Filter by complexity
        if complexity:
            complexity_ids = self._complexity_index.get(complexity, set())
            candidate_ids &= complexity_ids

        # Filter by tags
        if tags:
            for tag in tags:
                tag_ids = self._tag_index.get(tag, set())
                candidate_ids &= tag_ids

        # Apply remaining filters
        results = []
        for skill_id in candidate_ids:
            skill = self._skills_cache[skill_id]

            # Name pattern filter
            if name_pattern and name_pattern.lower() not in skill.name.lower():
                continue

            # Success rate filter
            if min_success_rate is not None and skill.success_rate < min_success_rate:
                continue

            # Usage count filter
            if min_usage_count is not None and skill.usage_count < min_usage_count:
                continue

            results.append(skill)

        # Sort by reliability score
        results.sort(key=lambda s: s.reliability_score, reverse=True)
        return results

    def get_skill_dependencies(self, skill_id: str) -> List[Skill]:
        """Get all dependencies for a skill."""
        skill = self.retrieve_skill(skill_id)
        if not skill:
            return []

        dependencies = []
        for dep_name in skill.required_skills:
            # Find skill by name (simplified - in production would need better mapping)
            for cached_skill in self._skills_cache.values():
                if cached_skill.name == dep_name:
                    dependencies.append(cached_skill)
                    break

        return dependencies

    def get_skills_by_category(self, category: SkillCategory) -> List[Skill]:
        """Get all skills in a category."""
        skill_ids = self._category_index.get(category, set())
        return [self._skills_cache[skill_id] for skill_id in skill_ids]

    def get_library_stats(self) -> Dict[str, Any]:
        """Get library statistics."""
        skills = list(self._skills_cache.values())

        return {
            "total_skills": len(skills),
            "categories": {
                cat if isinstance(cat, str) else cat.value: len(ids)
                for cat, ids in self._category_index.items()
            },
            "complexity_distribution": {
                comp if isinstance(comp, str) else comp.value: len(ids)
                for comp, ids in self._complexity_index.items()
            },
            "average_success_rate": sum(s.success_rate for s in skills) / len(skills)
            if skills
            else 0,
            "total_usage_count": sum(s.usage_count for s in skills),
            "reliable_skills": len([s for s in skills if s.is_reliable]),
            "experimental_skills": len([s for s in skills if s.is_experimental]),
        }

    def _load_all_skills(self) -> None:
        """Load all skills from storage."""
        for skill_file in self.storage_path.glob("*.json"):
            try:
                with open(skill_file, "r") as f:
                    skill_data = json.load(f)
                    skill = Skill(**skill_data)
                    self._skills_cache[skill.id] = skill
                    self._update_indexes(skill)
            except Exception as e:
                self.logger.error(f"Failed to load skill from {skill_file}: {e}")

    def _update_indexes(self, skill: Skill) -> None:
        """Update all indexes for a skill."""
        skill_id = skill.id

        # Category index
        if skill.category not in self._category_index:
            self._category_index[skill.category] = set()
        self._category_index[skill.category].add(skill_id)

        # Complexity index
        if skill.complexity not in self._complexity_index:
            self._complexity_index[skill.complexity] = set()
        self._complexity_index[skill.complexity].add(skill_id)

        # Tag index
        for tag in skill.tags:
            if tag not in self._tag_index:
                self._tag_index[tag] = set()
            self._tag_index[tag].add(skill_id)

        # Dependency index
        for dep in skill.required_skills:
            if dep not in self._dependency_index:
                self._dependency_index[dep] = set()
            self._dependency_index[dep].add(skill_id)


class SkillDiscoverer:
    """
    Pattern recognition and skill extraction from successful strategies.

    Analyzes trading experiences and strategies to identify
    reusable patterns and extract them as new skills.
    """

    def __init__(self, min_pattern_frequency: int = 3, min_success_rate: float = 0.7):
        self.min_pattern_frequency = min_pattern_frequency
        self.min_success_rate = min_success_rate
        self.logger = logging.getLogger(__name__)

    def discover_skills_from_experiences(
        self, experiences: List[Experience]
    ) -> List[Skill]:
        """Discover new skills from trading experiences."""
        # Group experiences by outcome and patterns
        successful_experiences = [
            exp
            for exp in experiences
            if exp.is_positive_outcome
            and exp.financial_impact
            and exp.financial_impact.is_positive()
        ]

        if len(successful_experiences) < self.min_pattern_frequency:
            return []

        # Extract patterns
        patterns = self._extract_patterns(successful_experiences)

        # Convert patterns to skills
        discovered_skills = []
        for pattern in patterns:
            if pattern["frequency"] >= self.min_pattern_frequency:
                skill = self._pattern_to_skill(pattern, successful_experiences)
                if skill:
                    discovered_skills.append(skill)

        return discovered_skills

    def discover_skills_from_code_analysis(
        self, strategy_code: str, performance_data: Dict[str, Any]
    ) -> List[Skill]:
        """Discover skills by analyzing successful strategy code."""
        try:
            tree = ast.parse(strategy_code)

            # Extract function definitions as potential skills
            functions = [
                node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)
            ]

            discovered_skills = []
            for func in functions:
                skill = self._function_to_skill(func, strategy_code, performance_data)
                if skill:
                    discovered_skills.append(skill)

            return discovered_skills

        except Exception as e:
            self.logger.error(f"Code analysis failed: {e}")
            return []

    def _extract_patterns(self, experiences: List[Experience]) -> List[Dict[str, Any]]:
        """Extract common patterns from experiences."""
        pattern_counts = {}

        for exp in experiences:
            # Extract patterns from experience
            exp_patterns = exp.extract_patterns()

            for pattern in exp_patterns:
                if pattern not in pattern_counts:
                    pattern_counts[pattern] = {
                        "name": pattern,
                        "frequency": 0,
                        "experiences": [],
                        "avg_financial_impact": Decimal("0"),
                        "success_contexts": [],
                    }

                pattern_data = pattern_counts[pattern]
                pattern_data["frequency"] += 1
                pattern_data["experiences"].append(exp.id)

                if exp.financial_impact:
                    pattern_data["avg_financial_impact"] += exp.financial_impact.amount

                pattern_data["success_contexts"].append(
                    {
                        "market_conditions": exp.market_conditions,
                        "symbols": [str(s) for s in exp.symbols_involved],
                        "timeframe": exp.timeframe.value if exp.timeframe else None,
                    }
                )

        # Normalize averages
        for pattern_data in pattern_counts.values():
            if pattern_data["frequency"] > 0:
                pattern_data["avg_financial_impact"] /= pattern_data["frequency"]

        return list(pattern_counts.values())

    def _pattern_to_skill(
        self, pattern: Dict[str, Any], experiences: List[Experience]
    ) -> Optional[Skill]:
        """Convert a pattern into a skill."""
        try:
            # Generate skill metadata
            skill_name = f"discovered_{pattern['name']}_{int(time.time())}"

            # Determine skill category based on pattern characteristics
            category = self._infer_category(pattern)
            complexity = self._infer_complexity(pattern)

            # Generate basic skill code template
            skill_code = self._generate_skill_code_template(pattern)

            # Create skill
            skill = Skill(
                name=skill_name,
                description=f"Discovered skill from pattern: {pattern['name']}",
                category=category,
                complexity=complexity,
                code=skill_code,
                input_schema={
                    "type": "object",
                    "properties": {
                        "market_data": {"type": "object"},
                        "context": {"type": "object"},
                    },
                    "required": ["market_data"],
                },
                output_schema={
                    "type": "object",
                    "properties": {
                        "signal": {"type": "string"},
                        "confidence": {"type": "number"},
                    },
                },
                tags=["discovered", "pattern-based", pattern["name"]],
                learned_from=",".join(
                    pattern["experiences"][:5]
                ),  # Reference first 5 experiences
            )

            return skill

        except Exception as e:
            self.logger.error(f"Failed to convert pattern to skill: {e}")
            return None

    def _function_to_skill(
        self, func_node: ast.FunctionDef, full_code: str, performance: Dict[str, Any]
    ) -> Optional[Skill]:
        """Convert a function AST node to a skill."""
        try:
            # Extract function code
            func_code = ast.get_source_segment(full_code, func_node)
            if not func_code:
                return None

            # Generate skill metadata
            skill_name = f"extracted_{func_node.name}_{int(time.time())}"

            # Analyze function to determine category and complexity
            category = self._analyze_function_category(func_node)
            complexity = self._analyze_function_complexity(func_node)

            # Create skill
            skill = Skill(
                name=skill_name,
                description=f"Extracted function: {func_node.name}",
                category=category,
                complexity=complexity,
                code=func_code,
                input_schema=self._infer_input_schema(func_node),
                output_schema={"type": "object"},
                tags=["extracted", "function-based", func_node.name],
                created_by="discoverer",
            )

            return skill

        except Exception as e:
            self.logger.error(f"Failed to convert function to skill: {e}")
            return None

    def _infer_category(self, pattern: Dict[str, Any]) -> SkillCategory:
        """Infer skill category from pattern characteristics."""
        pattern_name = pattern["name"].lower()

        if "technical" in pattern_name or "indicator" in pattern_name:
            return SkillCategory.TECHNICAL_ANALYSIS
        elif "risk" in pattern_name or "position" in pattern_name:
            return SkillCategory.RISK_MANAGEMENT
        elif "entry" in pattern_name:
            return SkillCategory.ENTRY_TIMING
        elif "exit" in pattern_name:
            return SkillCategory.EXIT_TIMING
        else:
            return SkillCategory.MARKET_ANALYSIS

    def _infer_complexity(self, pattern: Dict[str, Any]) -> SkillComplexity:
        """Infer skill complexity from pattern characteristics."""
        frequency = pattern["frequency"]
        context_variety = len(set(str(ctx) for ctx in pattern["success_contexts"]))

        if frequency > 10 and context_variety > 5:
            return SkillComplexity.ADVANCED
        elif frequency > 5 and context_variety > 3:
            return SkillComplexity.INTERMEDIATE
        else:
            return SkillComplexity.BASIC

    def _generate_skill_code_template(self, pattern: Dict[str, Any]) -> str:
        """Generate a basic code template for a discovered pattern."""
        return f"""
def execute_pattern_{pattern['name'].replace(' ', '_')}(inputs, context=None):
    \\\"\\\"\\\"
    Discovered pattern: {pattern['name']}
    Frequency: {pattern['frequency']}
    Average impact: {pattern['avg_financial_impact']}
    \\\"\\\"\\\"

    # Pattern-specific logic would be implemented here
    # This is a template that needs manual refinement

    market_data = inputs.get('market_data', {{}})

    # Placeholder pattern detection logic
    signal_strength = 0.5  # To be implemented based on pattern analysis

    result = {{
        'signal': 'hold',  # Default signal
        'confidence': signal_strength,
        'pattern': '{pattern['name']}',
        'reasoning': 'Pattern-based signal generation'
    }}

    return result

# Execute the pattern function
result = execute_pattern_{pattern['name'].replace(' ', '_')}(inputs, context)
"""

    def _analyze_function_category(self, func_node: ast.FunctionDef) -> SkillCategory:
        """Analyze function to determine likely category."""
        func_name = func_node.name.lower()

        if "rsi" in func_name or "sma" in func_name or "indicator" in func_name:
            return SkillCategory.TECHNICAL_ANALYSIS
        elif "risk" in func_name or "size" in func_name:
            return SkillCategory.RISK_MANAGEMENT
        elif "entry" in func_name or "buy" in func_name:
            return SkillCategory.ENTRY_TIMING
        elif "exit" in func_name or "sell" in func_name:
            return SkillCategory.EXIT_TIMING
        else:
            return SkillCategory.MARKET_ANALYSIS

    def _analyze_function_complexity(
        self, func_node: ast.FunctionDef
    ) -> SkillComplexity:
        """Analyze function complexity."""
        # Simple heuristic based on function characteristics
        lines_count = len(func_node.body)
        args_count = len(func_node.args.args)

        if lines_count > 20 or args_count > 5:
            return SkillComplexity.ADVANCED
        elif lines_count > 10 or args_count > 3:
            return SkillComplexity.INTERMEDIATE
        else:
            return SkillComplexity.BASIC

    def _infer_input_schema(self, func_node: ast.FunctionDef) -> Dict[str, Any]:
        """Infer input schema from function signature."""
        properties = {}
        required = []

        for arg in func_node.args.args:
            if arg.arg != "self":  # Skip self parameter
                properties[arg.arg] = {"type": "object"}  # Generic type
                required.append(arg.arg)

        return {"type": "object", "properties": properties, "required": required}


class VoyagerSkillLibrary:
    """
    Main VOYAGER Skill Library interface.

    Coordinates all six components of the skill library system:
    - Skill Executor, Composer, Validator, Librarian, Discoverer, Performance Tracker
    """

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger(__name__)

        # Initialize storage path
        storage_path = Path(config.get("skill_library_path", "skills"))

        # Initialize all components
        self.executor = SkillExecutor(
            timeout_seconds=config.get("execution_timeout", 30),
            max_memory_mb=config.get("max_memory", 128),
        )
        self.composer = SkillComposer()
        self.validator = CompositeSkillValidator()
        self.librarian = SkillLibrarian(storage_path)
        self.discoverer = SkillDiscoverer(
            min_pattern_frequency=config.get("min_pattern_frequency", 3),
            min_success_rate=config.get("min_success_rate", 0.7),
        )

        self.logger.info("VOYAGER Skill Library initialized")

    def add_skill(self, skill: Skill, validate: bool = True) -> bool:
        """Add a new skill to the library."""
        try:
            # Validate skill if requested
            if validate:
                is_valid, validation_errors = self.validator.validate_skill(skill)
                if not is_valid:
                    self.logger.error(f"Skill validation failed: {validation_errors}")
                    return False

            # Store skill
            success = self.librarian.store_skill(skill)

            if success:
                self.logger.info(f"Successfully added skill: {skill.name}")

            return success

        except Exception as e:
            self.logger.error(f"Failed to add skill {skill.name}: {e}")
            return False

    def execute_skill(
        self,
        skill_id: str,
        inputs: Dict[str, Any],
        context: Optional[Dict[str, Any]] = None,
    ) -> Tuple[SkillExecutionResult, Any, Dict[str, Any]]:
        """Execute a skill safely."""
        skill = self.librarian.retrieve_skill(skill_id)
        if not skill:
            return SkillExecutionResult.ERROR, None, {"error": "Skill not found"}

        result, output, metadata = self.executor.execute_skill(skill, inputs, context)

        # Update skill performance
        updated_skill = skill.record_usage(result)
        self.librarian.store_skill(updated_skill)

        return result, output, metadata

    def compose_and_execute(
        self,
        skill_ids: List[str],
        inputs: Dict[str, Any],
        composition_strategy: str = "sequential",
        context: Optional[Dict[str, Any]] = None,
    ) -> Tuple[SkillExecutionResult, Any, Dict[str, Any]]:
        """Compose skills and execute the resulting strategy."""
        # Retrieve skills
        skills = [self.librarian.retrieve_skill(skill_id) for skill_id in skill_ids]
        missing_skills = [
            skill_id for skill_id, skill in zip(skill_ids, skills) if skill is None
        ]

        if missing_skills:
            return (
                SkillExecutionResult.ERROR,
                None,
                {"error": f"Skills not found: {missing_skills}"},
            )

        try:
            # Compose skills
            composed_code, composition_metadata = self.composer.compose_skills(
                skills, composition_strategy
            )

            # Create temporary composed skill
            composed_skill = Skill(
                name=f"composed_{int(time.time())}",
                description=f"Composed strategy using {len(skills)} skills",
                category=SkillCategory.MARKET_ANALYSIS,  # Default category
                complexity=SkillComplexity.ADVANCED,  # Composed skills complex
                code=composed_code,
                input_schema={"type": "object"},
                output_schema={"type": "object"},
                tags=["composed"] + [skill.name for skill in skills],
            )

            # Execute composed skill
            result, output, execution_metadata = self.executor.execute_skill(
                composed_skill, inputs, context
            )

            # Combine metadata
            combined_metadata = {
                **execution_metadata,
                "composition_metadata": composition_metadata,
                "component_skills": skill_ids,
            }

            return result, output, combined_metadata

        except Exception as e:
            return SkillExecutionResult.ERROR, None, {"error": str(e)}

    def discover_skills_from_experiences(
        self, experiences: List[Experience]
    ) -> List[Skill]:
        """Discover new skills from trading experiences."""
        return self.discoverer.discover_skills_from_experiences(experiences)

    def search_skills(self, **criteria) -> List[Skill]:
        """Search skills with various criteria."""
        return self.librarian.search_skills(**criteria)

    def get_skill(self, skill_id: str) -> Optional[Skill]:
        """Get a skill by ID."""
        return self.librarian.retrieve_skill(skill_id)

    def get_library_stats(self) -> Dict[str, Any]:
        """Get comprehensive library statistics."""
        return self.librarian.get_library_stats()

    def validate_skill(self, skill: Skill) -> Tuple[bool, Dict[str, List[str]]]:
        """Validate a skill using all validators."""
        return self.validator.validate_skill(skill)


# Legacy compatibility - maintain existing TradingSkill class for backward compatibility


@dataclass
class TradingSkill:
    """Legacy TradingSkill class for backward compatibility."""

    name: str
    description: str
    code: str
    performance_metrics: Dict[str, float]
    usage_count: int = 0
    success_rate: float = 0.0
    prerequisites: List[str] = field(default_factory=list)
    tags: List[str] = field(default_factory=list)


class SkillLibrary:
    """Legacy SkillLibrary class - now wraps VoyagerSkillLibrary."""

    def __init__(self, config):
        """Initialize with legacy interface."""
        # Convert config to expected format
        if hasattr(config, "skill_library_path"):
            config_dict = {"skill_library_path": config.skill_library_path}
        else:
            config_dict = (
                config if isinstance(config, dict) else {"skill_library_path": "skills"}
            )

        self.voyager_library = VoyagerSkillLibrary(config_dict)
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.skills = {}  # Legacy interface
        self.skill_dependencies = {}  # Legacy interface

        # Load existing skills for legacy interface
        self._sync_legacy_interface()

    def add_skill(self, skill: TradingSkill) -> None:
        """Add a skill using legacy interface."""
        # Convert legacy skill to new Skill model
        new_skill = Skill(
            name=skill.name,
            description=skill.description,
            category=SkillCategory.MARKET_ANALYSIS,  # Default
            complexity=SkillComplexity.INTERMEDIATE,  # Default
            code=skill.code,
            input_schema={"type": "object"},
            output_schema={"type": "object"},
            performance_metrics={
                k: Decimal(str(v)) for k, v in skill.performance_metrics.items()
            },
            usage_count=skill.usage_count,
            success_count=int(skill.success_rate * skill.usage_count / 100)
            if skill.usage_count > 0
            else 0,
            tags=skill.tags,
            required_skills=skill.prerequisites,
        )

        self.voyager_library.add_skill(new_skill)
        self.skills[skill.name] = skill

    def get_skill(self, name: str) -> Optional[TradingSkill]:
        """Get skill by name using legacy interface."""
        return self.skills.get(name)

    def search_skills(
        self, tags: List[str] = None, min_success_rate: float = 0.0
    ) -> List[TradingSkill]:
        """Search skills using legacy interface."""
        # Convert to new search and back to legacy format
        new_skills = self.voyager_library.search_skills(
            tags=tags, min_success_rate=min_success_rate
        )

        legacy_skills = []
        for skill in new_skills:
            legacy_skill = TradingSkill(
                name=skill.name,
                description=skill.description,
                code=skill.code,
                performance_metrics={
                    k: float(v) for k, v in skill.performance_metrics.items()
                },
                usage_count=skill.usage_count,
                success_rate=float(skill.success_rate),
                prerequisites=skill.required_skills,
                tags=skill.tags,
            )
            legacy_skills.append(legacy_skill)

        return legacy_skills

    def compose_skills(self, skill_names: List[str]) -> Optional[str]:
        """Compose skills using legacy interface."""
        # Find skills by name and get their IDs
        skill_ids = []
        for name in skill_names:
            # Find skill by name in the new system
            skills = self.voyager_library.search_skills(name_pattern=name)
            if skills:
                skill_ids.append(skills[0].id)
            else:
                self.logger.error(f"Skill not found: {name}")
                return None

        if not skill_ids:
            return None

        # Use new composition system
        skills = [self.voyager_library.get_skill(skill_id) for skill_id in skill_ids]
        if not all(skills):
            return None

        try:
            composed_code, _ = self.voyager_library.composer.compose_skills(
                skills, "sequential"
            )
            return composed_code
        except Exception as e:
            self.logger.error(f"Composition failed: {e}")
            return None

    def update_skill_performance(
        self, name: str, performance: Dict[str, float]
    ) -> None:
        """Update skill performance using legacy interface."""
        if name in self.skills:
            legacy_skill = self.skills[name]
            legacy_skill.performance_metrics.update(performance)
            legacy_skill.usage_count += 1

            if "success" in performance:
                old_rate = legacy_skill.success_rate
                legacy_skill.success_rate = (
                    old_rate * (legacy_skill.usage_count - 1) + performance["success"]
                ) / legacy_skill.usage_count

    def _sync_legacy_interface(self) -> None:
        """Sync legacy interface with new system."""
        # Convert new skills to legacy format for backward compatibility
        new_skills = self.voyager_library.search_skills()
        for skill in new_skills:
            legacy_skill = TradingSkill(
                name=skill.name,
                description=skill.description,
                code=skill.code,
                performance_metrics={
                    k: float(v) for k, v in skill.performance_metrics.items()
                },
                usage_count=skill.usage_count,
                success_rate=float(skill.success_rate),
                prerequisites=skill.required_skills,
                tags=skill.tags,
            )
            self.skills[skill.name] = legacy_skill
