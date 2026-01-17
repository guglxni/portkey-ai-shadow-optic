"""
Automatic Guardrail Tuning for Shadow-Optic P2 Enhancements.

Portkey-First Architecture:
- Uses Portkey's 20+ built-in guardrails (Regex, JSON Schema, etc.)
- Uses Portkey's LLM-based PRO guardrails (Moderate Content, Detect PII)
- Uses Portkey's Feedback API for guardrail pass/fail tracking
- Uses DSPy (via Portkey integration) for prompt optimization

This enables automatic detection of refusal patterns and system prompt
adjustment to reduce false refusals while maintaining safety.
"""

import asyncio
import logging
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple
from datetime import datetime, timedelta
from collections import defaultdict

import numpy as np

try:
    import dspy
    DSPY_AVAILABLE = True
except ImportError:
    DSPY_AVAILABLE = False
    dspy = None

from portkey_ai import Portkey, AsyncPortkey

logger = logging.getLogger(__name__)


# =============================================================================
# Enums and Data Classes
# =============================================================================

class RefusalType(Enum):
    """Types of model refusals."""
    SAFETY = "safety"           # Legitimate safety refusal
    OVERCAUTIOUS = "overcautious"  # False positive refusal
    CAPABILITY = "capability"    # "I can't do that" type
    POLICY = "policy"           # Policy-based refusal
    AMBIGUOUS = "ambiguous"     # Unclear refusal reason


class GuardrailAction(Enum):
    """Actions for guardrail outcomes."""
    PASS = "pass"               # Request allowed
    BLOCK = "block"             # Request blocked
    MODIFY = "modify"           # Request/response modified
    FLAG = "flag"               # Flagged for review


@dataclass
class RefusalPattern:
    """Detected refusal pattern."""
    prompt: str
    response: str
    refusal_type: RefusalType
    confidence: float
    trigger_phrases: List[str] = field(default_factory=list)
    suggested_mitigation: Optional[str] = None
    timestamp: datetime = field(default_factory=datetime.utcnow)
    
    @property
    def is_false_positive(self) -> bool:
        """Check if this is likely a false positive refusal."""
        return self.refusal_type == RefusalType.OVERCAUTIOUS


@dataclass
class GuardrailResult:
    """Result of a guardrail check."""
    action: GuardrailAction
    guardrail_name: str
    triggered: bool
    confidence: float
    details: Dict[str, Any] = field(default_factory=dict)
    processing_time_ms: float = 0.0


@dataclass
class TuningResult:
    """Result of prompt tuning optimization."""
    original_prompt: str
    optimized_prompt: str
    improvement_score: float
    changes_made: List[str] = field(default_factory=list)
    safety_validated: bool = False
    test_results: Dict[str, Any] = field(default_factory=dict)


# =============================================================================
# Refusal Detector
# =============================================================================

class PortkeyRefusalDetector:
    """Enhanced refusal detection using Portkey's capabilities.
    
    Uses pattern matching for quick detection, with option to
    use Portkey's LLM-based guardrails for classification.
    """
    
    # Common refusal patterns
    REFUSAL_PATTERNS = [
        r"I (?:cannot|can't|won't|will not|am not able to)",
        r"I'm (?:sorry|afraid|unable)",
        r"(?:Unfortunately|Regrettably),? I",
        r"(?:This|That) (?:is|would be) (?:against|outside|beyond)",
        r"I (?:don't|do not) (?:have|feel comfortable)",
        r"(?:As an AI|As a language model|As an assistant)",
        r"I (?:must|need to) (?:decline|refuse)",
        r"(?:not|never) (?:able|allowed|permitted) to",
        r"(?:violates?|against) (?:my|the|our) (?:policy|guidelines|terms)",
        r"(?:can not|cannot) (?:help|assist|provide) (?:with|you)",
    ]
    
    # Patterns that look like refusals but aren't
    FALSE_POSITIVE_PATTERNS = [
        r"I don't have access to (?:real-time|current|live|external)",
        r"I don't have (?:access to|information about) .*?, but I can",
        r"I'm not (?:sure|certain) (?:about|if|what)",
        r"I (?:don't|do not) (?:know|have) (?:the|that|this) (?:answer|information) .*, but",
    ]
    
    def __init__(
        self,
        portkey_api_key: Optional[str] = None,
        confidence_threshold: float = 0.7
    ):
        self.portkey = AsyncPortkey(api_key=portkey_api_key) if portkey_api_key else None
        self.confidence_threshold = confidence_threshold
        self._compiled_patterns = [re.compile(p, re.IGNORECASE) for p in self.REFUSAL_PATTERNS]
        self._compiled_false_positives = [re.compile(p, re.IGNORECASE) for p in self.FALSE_POSITIVE_PATTERNS]
    
    def is_refusal(self, response: str) -> bool:
        """Quick check if response is a refusal.
        
        Args:
            response: Model response text
            
        Returns:
            True if response appears to be a refusal
        """
        # Check for false positive patterns first
        for pattern in self._compiled_false_positives:
            if pattern.search(response):
                return False
        
        # Check for refusal patterns
        for pattern in self._compiled_patterns:
            if pattern.search(response):
                return True
        
        return False
    
    async def detect_refusal(
        self,
        prompt: str,
        response: str
    ) -> Tuple[bool, Optional[RefusalPattern]]:
        """Detect if response is a refusal and classify it.
        
        Args:
            prompt: Original prompt
            response: Model response
            
        Returns:
            Tuple of (is_refusal, RefusalPattern if detected)
        """
        # Quick pattern-based check
        is_refusal = self.is_refusal(response)
        
        if not is_refusal:
            return False, None
        
        # Extract trigger phrases
        trigger_phrases = self._extract_trigger_phrases(response)
        
        # Classify refusal type
        refusal_type = await self._classify_refusal_type(prompt, response)
        
        # Calculate confidence based on pattern matches
        confidence = self._calculate_confidence(response, trigger_phrases)
        
        pattern = RefusalPattern(
            prompt=prompt,
            response=response,
            refusal_type=refusal_type,
            confidence=confidence,
            trigger_phrases=trigger_phrases
        )
        
        return True, pattern
    
    def _extract_trigger_phrases(self, response: str) -> List[str]:
        """Extract phrases that indicate refusal."""
        triggers = []
        
        for pattern in self._compiled_patterns:
            matches = pattern.findall(response)
            triggers.extend(matches)
        
        return list(set(triggers))
    
    async def _classify_refusal_type(
        self,
        prompt: str,
        response: str
    ) -> RefusalType:
        """Classify the type of refusal.
        
        Uses heuristics first, with optional LLM classification.
        """
        response_lower = response.lower()
        
        # Safety-related keywords
        safety_keywords = [
            "harmful", "dangerous", "illegal", "violence", "hate",
            "discriminat", "offensive", "explicit", "inappropriate"
        ]
        if any(kw in response_lower for kw in safety_keywords):
            return RefusalType.SAFETY
        
        # Policy-related keywords
        policy_keywords = [
            "policy", "guideline", "terms", "rules", "compliance"
        ]
        if any(kw in response_lower for kw in policy_keywords):
            return RefusalType.POLICY
        
        # Capability-related keywords
        capability_keywords = [
            "can't access", "don't have access", "cannot access",
            "not able to", "unable to access", "limitations"
        ]
        if any(kw in response_lower for kw in capability_keywords):
            return RefusalType.CAPABILITY
        
        # Check for overcautious patterns
        overcautious_indicators = [
            "as an ai", "as a language model", "as an assistant",
            "i'm designed to", "i was trained to"
        ]
        if any(ind in response_lower for ind in overcautious_indicators):
            # Likely overcautious if prompt seems benign
            prompt_lower = prompt.lower()
            if not any(kw in prompt_lower for kw in safety_keywords):
                return RefusalType.OVERCAUTIOUS
        
        return RefusalType.AMBIGUOUS
    
    def _calculate_confidence(
        self,
        response: str,
        trigger_phrases: List[str]
    ) -> float:
        """Calculate confidence score for refusal detection."""
        if not trigger_phrases:
            return 0.5
        
        # More trigger phrases = higher confidence
        base_confidence = min(0.9, 0.5 + len(trigger_phrases) * 0.1)
        
        # Longer refusal response = higher confidence
        word_count = len(response.split())
        if word_count < 20:
            length_factor = 0.8
        elif word_count < 50:
            length_factor = 1.0
        else:
            length_factor = 0.9  # Very long might be explaining, not refusing
        
        return base_confidence * length_factor


# =============================================================================
# Guardrail Tuner (using DSPy via Portkey)
# =============================================================================

class PortkeyGuardrailTuner:
    """Automatic guardrail tuning using Portkey + DSPy.
    
    Portkey-First Architecture:
    - Uses Portkey's guardrails for safety checks
    - Uses DSPy for prompt optimization (via Portkey's LM)
    - Uses Portkey's Feedback API for tracking
    """
    
    def __init__(
        self,
        portkey_api_key: Optional[str] = None,
        model: str = "gpt-4o-mini"
    ):
        self.portkey_api_key = portkey_api_key
        self.model = model
        self.refusal_detector = PortkeyRefusalDetector(portkey_api_key)
        self.system_prompt_history: List[Dict[str, Any]] = []
        self._dspy_configured = False
        
        # Configure DSPy if available
        if DSPY_AVAILABLE and portkey_api_key:
            self._configure_dspy()
    
    def _configure_dspy(self):
        """Configure DSPy to use Portkey as the LM provider."""
        if not DSPY_AVAILABLE:
            logger.warning("DSPy not available, using fallback methods")
            return
        
        try:
            # Configure DSPy with Portkey
            lm = dspy.LM(
                model=f"openai/{self.model}",
                api_key=self.portkey_api_key,
                api_base="https://api.portkey.ai/v1"
            )
            dspy.configure(lm=lm)
            self._dspy_configured = True
            logger.info(f"DSPy configured with Portkey using {self.model}")
        except Exception as e:
            logger.warning(f"Failed to configure DSPy: {e}")
            self._dspy_configured = False
    
    async def analyze_refusal_patterns(
        self,
        refusals: List[RefusalPattern]
    ) -> Dict[str, Any]:
        """Analyze patterns in refusals to identify improvement areas.
        
        Args:
            refusals: List of detected refusal patterns
            
        Returns:
            Analysis dict with optimization recommendations
        """
        if not refusals:
            return {
                "status": "no_refusals",
                "overcautious_rate": 0.0,
                "recommendation": "No refusals detected"
            }
        
        # Group by type
        by_type: Dict[RefusalType, List[RefusalPattern]] = defaultdict(list)
        for r in refusals:
            by_type[r.refusal_type].append(r)
        
        # Find overcautious patterns (our optimization target)
        overcautious = by_type.get(RefusalType.OVERCAUTIOUS, [])
        
        overcautious_rate = len(overcautious) / len(refusals)
        
        if not overcautious:
            return {
                "status": "no_optimization_needed",
                "overcautious_rate": overcautious_rate,
                "total_refusals": len(refusals),
                "by_type": {k.value: len(v) for k, v in by_type.items()},
                "recommendation": "All refusals appear legitimate"
            }
        
        # Cluster similar overcautious refusals
        clusters = self._cluster_refusals(overcautious)
        
        return {
            "status": "optimization_available",
            "overcautious_rate": overcautious_rate,
            "overcautious_count": len(overcautious),
            "clusters": clusters,
            "total_refusals": len(refusals),
            "by_type": {k.value: len(v) for k, v in by_type.items()},
            "recommendation": f"Found {len(overcautious)} overcautious refusals in {len(clusters)} patterns"
        }
    
    def _cluster_refusals(
        self,
        refusals: List[RefusalPattern],
        similarity_threshold: float = 0.7
    ) -> List[Dict[str, Any]]:
        """Cluster similar refusals for pattern analysis.
        
        Uses simple text-based clustering for speed.
        """
        clusters: List[Dict[str, Any]] = []
        
        # Simple clustering based on trigger phrases
        trigger_groups: Dict[str, List[RefusalPattern]] = defaultdict(list)
        for r in refusals:
            # Use first trigger phrase as group key
            key = r.trigger_phrases[0] if r.trigger_phrases else "unknown"
            trigger_groups[key].append(r)
        
        for trigger, group in trigger_groups.items():
            if len(group) >= 2:  # Only create clusters with multiple items
                clusters.append({
                    "cluster_id": len(clusters),
                    "trigger_phrase": trigger,
                    "count": len(group),
                    "representative_prompts": [r.prompt[:100] for r in group[:3]],
                    "common_response_patterns": self._find_common_patterns(group)
                })
        
        # Add unclustered items as individual clusters
        for trigger, group in trigger_groups.items():
            if len(group) == 1:
                clusters.append({
                    "cluster_id": len(clusters),
                    "trigger_phrase": trigger,
                    "count": 1,
                    "representative_prompts": [group[0].prompt[:100]],
                    "common_response_patterns": []
                })
        
        return clusters
    
    def _find_common_patterns(
        self,
        refusals: List[RefusalPattern]
    ) -> List[str]:
        """Find common patterns in refusal responses."""
        # Extract common phrases
        phrase_counts: Dict[str, int] = defaultdict(int)
        
        for r in refusals:
            for phrase in r.trigger_phrases:
                phrase_counts[phrase] += 1
        
        # Return phrases that appear in at least half the refusals
        threshold = len(refusals) / 2
        common = [p for p, c in phrase_counts.items() if c >= threshold]
        
        return common
    
    async def generate_improved_prompt(
        self,
        current_system_prompt: str,
        overcautious_examples: List[RefusalPattern]
    ) -> TuningResult:
        """Generate improved system prompt to reduce false refusals.
        
        Args:
            current_system_prompt: Current system prompt
            overcautious_examples: Examples of false positive refusals
            
        Returns:
            TuningResult with optimized prompt
        """
        if not overcautious_examples:
            return TuningResult(
                original_prompt=current_system_prompt,
                optimized_prompt=current_system_prompt,
                improvement_score=0.0,
                changes_made=["No changes - no overcautious examples provided"]
            )
        
        # Format examples for the optimization prompt
        examples_text = "\n\n".join([
            f"User: {r.prompt[:200]}\nRefused with: {r.response[:200]}"
            for r in overcautious_examples[:5]
        ])
        
        if self._dspy_configured and DSPY_AVAILABLE:
            return await self._optimize_with_dspy(
                current_system_prompt,
                examples_text
            )
        else:
            return self._optimize_heuristic(
                current_system_prompt,
                overcautious_examples
            )
    
    async def _optimize_with_dspy(
        self,
        current_prompt: str,
        examples_text: str
    ) -> TuningResult:
        """Optimize prompt using DSPy via Portkey."""
        if not DSPY_AVAILABLE:
            raise RuntimeError("DSPy not available")
        
        # Define DSPy signature for prompt optimization
        class PromptOptimizer(dspy.Signature):
            """Optimize system prompt to reduce false refusals while maintaining safety."""
            current_prompt: str = dspy.InputField(desc="Current system prompt")
            false_refusal_examples: str = dspy.InputField(
                desc="Examples of benign prompts that were incorrectly refused"
            )
            optimized_prompt: str = dspy.OutputField(
                desc="Improved system prompt that handles these cases while staying safe"
            )
            changes_made: str = dspy.OutputField(
                desc="List of changes made to the prompt"
            )
        
        try:
            optimizer = dspy.ChainOfThought(PromptOptimizer)
            result = optimizer(
                current_prompt=current_prompt,
                false_refusal_examples=examples_text
            )
            
            changes = result.changes_made.split("\n") if result.changes_made else []
            
            return TuningResult(
                original_prompt=current_prompt,
                optimized_prompt=result.optimized_prompt,
                improvement_score=0.0,  # Will be calculated after testing
                changes_made=changes,
                safety_validated=False
            )
        except Exception as e:
            logger.error(f"DSPy optimization failed: {e}")
            return self._optimize_heuristic(current_prompt, [])
    
    def _optimize_heuristic(
        self,
        current_prompt: str,
        overcautious_examples: List[RefusalPattern]
    ) -> TuningResult:
        """Heuristic-based prompt optimization (fallback)."""
        changes = []
        optimized = current_prompt
        
        # Add clarifying instructions for common overcautious patterns
        clarifications = []
        
        # Check for common overcautious triggers
        trigger_types = set()
        for ex in overcautious_examples:
            for trigger in ex.trigger_phrases:
                trigger_lower = trigger.lower()
                if "as an ai" in trigger_lower:
                    trigger_types.add("identity_overexplanation")
                elif "cannot" in trigger_lower or "can't" in trigger_lower:
                    trigger_types.add("capability_confusion")
                elif "sorry" in trigger_lower:
                    trigger_types.add("over_apologizing")
        
        if "identity_overexplanation" in trigger_types:
            clarifications.append(
                "Do not unnecessarily explain your nature as an AI when answering straightforward questions."
            )
            changes.append("Added: Reduce identity over-explanation")
        
        if "capability_confusion" in trigger_types:
            clarifications.append(
                "When a request is within your capabilities and doesn't violate safety guidelines, "
                "provide a helpful response rather than declining."
            )
            changes.append("Added: Clarify capability boundaries")
        
        if "over_apologizing" in trigger_types:
            clarifications.append(
                "Only apologize when there's a genuine reason; focus on being helpful rather than apologetic."
            )
            changes.append("Added: Reduce over-apologizing")
        
        if clarifications:
            # Append clarifications to the system prompt
            clarification_block = "\n\nAdditional Guidelines:\n" + "\n".join(f"- {c}" for c in clarifications)
            optimized = current_prompt + clarification_block
        else:
            changes.append("No heuristic optimizations applicable")
        
        return TuningResult(
            original_prompt=current_prompt,
            optimized_prompt=optimized,
            improvement_score=0.1 if clarifications else 0.0,
            changes_made=changes,
            safety_validated=False
        )
    
    async def validate_prompt_safety(
        self,
        optimized_prompt: str,
        safety_test_suite: List[Dict[str, Any]]
    ) -> Tuple[bool, Dict[str, Any]]:
        """Validate that optimized prompt doesn't reduce safety.
        
        Args:
            optimized_prompt: The optimized system prompt
            safety_test_suite: List of test cases with expected outcomes
            
        Returns:
            Tuple of (is_safe, test_results)
        """
        if not safety_test_suite:
            return True, {"warning": "No safety tests provided"}
        
        results = {"passed": 0, "failed": 0, "failures": []}
        
        for test in safety_test_suite:
            expected = test.get("expected", "answer")
            test_prompt = test.get("prompt", "")
            
            # Simulate response (in production, would call LLM)
            # For now, use pattern matching as approximation
            would_refuse = self._would_refuse_heuristic(test_prompt, optimized_prompt)
            
            if expected == "refuse":
                if would_refuse:
                    results["passed"] += 1
                else:
                    results["failed"] += 1
                    results["failures"].append({
                        "test": test,
                        "expected": "refuse",
                        "got": "would_not_refuse"
                    })
            elif expected == "answer":
                if not would_refuse:
                    results["passed"] += 1
                else:
                    results["failed"] += 1
                    results["failures"].append({
                        "test": test,
                        "expected": "answer",
                        "got": "would_refuse"
                    })
        
        total = results["passed"] + results["failed"]
        pass_rate = results["passed"] / total if total > 0 else 0
        
        is_safe = pass_rate >= 0.95  # 95% threshold
        results["pass_rate"] = pass_rate
        results["is_safe"] = is_safe
        
        return is_safe, results
    
    def _would_refuse_heuristic(
        self,
        test_prompt: str,
        system_prompt: str
    ) -> bool:
        """Heuristic to predict if a prompt would be refused.
        
        This is a simple approximation - production should use actual LLM calls.
        """
        # Check for safety-related content in test prompt
        safety_keywords = [
            "hack", "exploit", "illegal", "weapon", "drug",
            "harm", "violence", "hate", "discriminat", "explicit"
        ]
        
        prompt_lower = test_prompt.lower()
        
        # If prompt contains safety keywords, likely should refuse
        for keyword in safety_keywords:
            if keyword in prompt_lower:
                return True
        
        return False
    
    def get_tuning_history(self) -> List[Dict[str, Any]]:
        """Get history of prompt tuning operations."""
        return self.system_prompt_history.copy()
    
    def record_tuning(
        self,
        tuning_result: TuningResult,
        validation_result: Dict[str, Any]
    ):
        """Record a tuning operation for history tracking."""
        self.system_prompt_history.append({
            "timestamp": datetime.utcnow().isoformat(),
            "original_prompt_length": len(tuning_result.original_prompt),
            "optimized_prompt_length": len(tuning_result.optimized_prompt),
            "changes_made": tuning_result.changes_made,
            "improvement_score": tuning_result.improvement_score,
            "safety_validated": tuning_result.safety_validated,
            "validation_pass_rate": validation_result.get("pass_rate", 0)
        })


# =============================================================================
# Portkey Guardrail Manager
# =============================================================================

class PortkeyGuardrailManager:
    """Manage Portkey guardrails programmatically.
    
    Interfaces with Portkey's guardrail system:
    - 20+ built-in guardrails
    - LLM-based PRO guardrails
    - Custom webhook guardrails
    - Status codes 246/446 for handling
    """
    
    # Built-in Portkey guardrails
    BUILTIN_GUARDRAILS = {
        "regex_match": {"type": "deterministic", "pro": False},
        "json_schema": {"type": "deterministic", "pro": False},
        "contains_code": {"type": "deterministic", "pro": False},
        "word_count": {"type": "deterministic", "pro": False},
        "sentence_count": {"type": "deterministic", "pro": False},
        "valid_urls": {"type": "deterministic", "pro": False},
        "language_match": {"type": "deterministic", "pro": False},
        "moderate_content": {"type": "llm", "pro": True},
        "detect_pii": {"type": "llm", "pro": True},
        "detect_gibberish": {"type": "llm", "pro": True},
    }
    
    def __init__(
        self,
        portkey_api_key: Optional[str] = None
    ):
        self.portkey = AsyncPortkey(api_key=portkey_api_key) if portkey_api_key else None
        self.active_guardrails: Dict[str, Dict[str, Any]] = {}
    
    def configure_guardrail(
        self,
        name: str,
        guardrail_type: str,
        config: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Configure a guardrail.
        
        Args:
            name: Unique name for this guardrail instance
            guardrail_type: Type from BUILTIN_GUARDRAILS
            config: Guardrail-specific configuration
            
        Returns:
            Guardrail configuration dict
        """
        if guardrail_type not in self.BUILTIN_GUARDRAILS:
            raise ValueError(f"Unknown guardrail type: {guardrail_type}")
        
        builtin_info = self.BUILTIN_GUARDRAILS[guardrail_type]
        guardrail_config = {
            "name": name,
            "type": guardrail_type,  # Keep the actual guardrail type
            "internal_type": builtin_info["type"],  # deterministic vs llm
            "config": config,
            "enabled": True,
            "pro": builtin_info["pro"],
        }
        
        self.active_guardrails[name] = guardrail_config
        return guardrail_config
    
    def get_guardrail_config(self, name: str) -> Optional[Dict[str, Any]]:
        """Get configuration for a specific guardrail."""
        return self.active_guardrails.get(name)
    
    def list_active_guardrails(self) -> List[Dict[str, Any]]:
        """List all active guardrails."""
        return list(self.active_guardrails.values())
    
    def disable_guardrail(self, name: str) -> bool:
        """Disable a guardrail."""
        if name in self.active_guardrails:
            self.active_guardrails[name]["enabled"] = False
            return True
        return False
    
    def enable_guardrail(self, name: str) -> bool:
        """Enable a guardrail."""
        if name in self.active_guardrails:
            self.active_guardrails[name]["enabled"] = True
            return True
        return False
    
    async def check_guardrails(
        self,
        text: str,
        guardrail_names: Optional[List[str]] = None
    ) -> List[GuardrailResult]:
        """Check text against specified guardrails.
        
        Args:
            text: Text to check
            guardrail_names: Specific guardrails to check (None = all active)
            
        Returns:
            List of GuardrailResult for each checked guardrail
        """
        results = []
        
        guardrails_to_check = guardrail_names or list(self.active_guardrails.keys())
        
        for name in guardrails_to_check:
            if name not in self.active_guardrails:
                continue
            
            guardrail = self.active_guardrails[name]
            if not guardrail.get("enabled", True):
                continue
            
            # Run the guardrail check
            result = await self._run_guardrail(text, guardrail)
            results.append(result)
        
        return results
    
    async def _run_guardrail(
        self,
        text: str,
        guardrail: Dict[str, Any]
    ) -> GuardrailResult:
        """Run a specific guardrail check.
        
        Note: In production, this would use Portkey's guardrail API.
        This implementation provides local approximations for testing.
        """
        import time
        start_time = time.time()
        
        guardrail_type = guardrail["type"]
        config = guardrail.get("config", {})
        
        triggered = False
        confidence = 1.0
        details = {}
        
        # Implement local checks for common guardrail types
        if guardrail_type == "regex_match":
            pattern = config.get("pattern", "")
            if pattern:
                match = re.search(pattern, text, re.IGNORECASE)
                triggered = match is not None
                if match:
                    details["match"] = match.group()
        
        elif guardrail_type == "word_count":
            min_words = config.get("min", 0)
            max_words = config.get("max", float("inf"))
            word_count = len(text.split())
            triggered = word_count < min_words or word_count > max_words
            details["word_count"] = word_count
        
        elif guardrail_type == "contains_code":
            code_patterns = [
                r"```",
                r"def\s+\w+\s*\(",
                r"function\s+\w+\s*\(",
                r"class\s+\w+",
                r"import\s+\w+",
                r"<\?php",
            ]
            for pattern in code_patterns:
                if re.search(pattern, text):
                    triggered = True
                    break
        
        elif guardrail_type == "detect_pii":
            # Simple PII detection
            pii_patterns = {
                "email": r"[\w.-]+@[\w.-]+\.\w+",
                "phone": r"\b\d{3}[-.]?\d{3}[-.]?\d{4}\b",
                "ssn": r"\b\d{3}-\d{2}-\d{4}\b",
            }
            found_pii = []
            for pii_type, pattern in pii_patterns.items():
                if re.search(pattern, text):
                    found_pii.append(pii_type)
                    triggered = True
            details["pii_types"] = found_pii
        
        processing_time = (time.time() - start_time) * 1000
        
        return GuardrailResult(
            action=GuardrailAction.BLOCK if triggered else GuardrailAction.PASS,
            guardrail_name=guardrail["name"],
            triggered=triggered,
            confidence=confidence,
            details=details,
            processing_time_ms=processing_time
        )
    
    def create_portkey_config(
        self,
        guardrail_names: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """Create Portkey config with guardrails.
        
        Args:
            guardrail_names: Specific guardrails to include
            
        Returns:
            Portkey config dict with guardrails section
        """
        guardrails_to_include = guardrail_names or list(self.active_guardrails.keys())
        
        guardrails_config = []
        for name in guardrails_to_include:
            if name in self.active_guardrails:
                guardrail = self.active_guardrails[name]
                if guardrail.get("enabled", True):
                    guardrails_config.append({
                        "type": guardrail["type"],
                        "config": guardrail.get("config", {})
                    })
        
        return {
            "guardrails": {
                "input": guardrails_config,
                "output": guardrails_config
            }
        }
