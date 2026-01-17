"""
Prompt Template Extraction and Clustering.

Extracts templates from raw prompts to identify structural patterns
and reduce redundant testing of semantically identical prompts with
different variable values.
"""

import re
import random
from collections import defaultdict
from dataclasses import dataclass
from difflib import SequenceMatcher
from typing import Dict, List, Tuple

from shadow_optic.models import GoldenPrompt, ProductionTrace, SampleType


@dataclass
class PromptInstance:
    """A specific instance of a prompt template."""
    prompt: str
    variables: List[str]
    trace: ProductionTrace


class PromptTemplateExtractor:
    """
    Extracts prompt templates from production logs.
    
    Example:
    Input prompts:
    - "Write a poem about cats"
    - "Write a poem about dogs"
    
    Extracted template:
    - "Write a poem about {var_0}"
    - Variables: ["cats", "dogs"]
    """
    
    VARIABLE_PATTERNS = [
        r'\b\d{4}-\d{2}-\d{2}\b',  # Dates
        r'\b\d+\b',  # Numbers
        r'"[^"]*"',  # Quoted strings
        r"'[^']*'",  # Single-quoted strings
        r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b',  # Proper nouns (simple heuristic)
    ]
    
    def __init__(self, similarity_threshold: float = 0.8):
        self.similarity_threshold = similarity_threshold
    
    def extract_template(self, prompt: str) -> Tuple[str, List[str]]:
        """
        Extract template and variables from a prompt.
        
        Returns: (template_string, list_of_variable_values)
        """
        variables = []
        template = prompt
        
        # Replace variable patterns with placeholders
        for i, pattern in enumerate(self.VARIABLE_PATTERNS):
            matches = re.findall(pattern, template)
            for match in matches:
                # Avoid re-replacing already replaced placeholders
                if match in variables: 
                    continue
                    
                variables.append(match)
                # Escape the match for regex replacement to handle special chars
                escaped_match = re.escape(match)
                template = re.sub(escaped_match, f"{{var_{len(variables)-1}}}", template, count=1)
        
        return template, variables
    
    def cluster_into_template_families(
        self, 
        traces: List[ProductionTrace]
    ) -> Dict[str, List[PromptInstance]]:
        """
        Cluster traces into template families.
        
        Algorithm:
        1. Extract template for each prompt
        2. Group templates by similarity
        3. Merge similar templates into families
        """
        families = defaultdict(list)
        
        for trace in traces:
            template, variables = self.extract_template(trace.prompt)
            
            # Find matching family
            matched = False
            for family_template in list(families.keys()):
                # Quick length check optimization
                if abs(len(template) - len(family_template)) / max(len(template), len(family_template)) > (1 - self.similarity_threshold):
                    continue
                    
                similarity = SequenceMatcher(
                    None, template, family_template
                ).ratio()
                
                if similarity >= self.similarity_threshold:
                    families[family_template].append(
                        PromptInstance(prompt=trace.prompt, variables=variables, trace=trace)
                    )
                    matched = True
                    break
            
            if not matched:
                families[template].append(
                    PromptInstance(prompt=trace.prompt, variables=variables, trace=trace)
                )
        
        return families
    
    def select_test_cases(
        self, 
        family: List[PromptInstance],
        n_cases: int = 3
    ) -> List[PromptInstance]:
        """
        Select diverse test cases from a template family.
        
        Strategy:
        1. Include shortest and longest variable combinations
        2. Random sample for diversity
        """
        if len(family) <= n_cases:
            return family
        
        selected = []
        
        # Shortest prompt (often simplest case)
        selected.append(min(family, key=lambda x: len(x.prompt)))
        
        # Longest prompt (often most complex case)
        longest = max(family, key=lambda x: len(x.prompt))
        if longest not in selected:
            selected.append(longest)
        
        # Fill remaining with random samples
        remaining = [p for p in family if p not in selected]
        if remaining:
            needed = n_cases - len(selected)
            if needed > 0:
                selected.extend(random.sample(remaining, min(needed, len(remaining))))
        
        return selected

    def sample_with_templates(
        self,
        traces: List[ProductionTrace],
        samples_per_template: int = 3
    ) -> List[GoldenPrompt]:
        """
        Main entry point: Sample golden prompts using template extraction.
        """
        families = self.cluster_into_template_families(traces)
        golden_prompts = []
        
        for template, instances in families.items():
            test_cases = self.select_test_cases(instances, n_cases=samples_per_template)
            
            for case in test_cases:
                golden_prompts.append(GoldenPrompt(
                    original_trace_id=case.trace.trace_id,
                    prompt=case.trace.prompt,
                    production_response=case.trace.response,
                    sample_type=SampleType.CENTROID, # Using CENTROID as generic type here
                    template=template,
                    variable_values=case.variables
                ))
                
        return golden_prompts
