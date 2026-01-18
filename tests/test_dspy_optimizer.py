"""
Tests for DSPyOptimizer components.

Tests the DSPy integration including:
- DSPyOptimizerConfig model
- SignatureBuilder
- DSPySignature generation
- TrainingExample dataclass
"""

import pytest
from unittest.mock import MagicMock, patch, AsyncMock
from typing import Dict, Any

from shadow_optic.dspy_optimizer import (
    DSPyOptimizerConfig,
    OptimizationStrategy,
    DSPyField,
    DSPySignature,
    SignatureBuilder,
    TrainingExample,
)


# =============================================================================
# DSPyOptimizerConfig Tests
# =============================================================================

class TestDSPyOptimizerConfig:
    """Tests for DSPyOptimizerConfig model."""
    
    def test_default_values(self):
        """Test default configuration values."""
        config = DSPyOptimizerConfig()
        
        assert config.strategy == OptimizationStrategy.MIPRO_V2
        assert config.teacher_model == "gpt-5"
        assert config.student_model == "gpt-5-mini"
        assert config.max_bootstrapped_demos == 4
        assert config.max_labeled_demos == 8
        assert config.num_candidates == 5
        assert config.num_trials == 10
        assert config.min_quality_improvement == 0.05
        assert config.target_quality_score == 0.90
        assert config.use_portkey is True
        assert config.auto_deploy is False
    
    def test_custom_values(self):
        """Test custom configuration values."""
        config = DSPyOptimizerConfig(
            teacher_model="gpt-5",
            student_model="gpt-4.1-nano",
            strategy=OptimizationStrategy.BOOTSTRAP,
            num_trials=25,
            auto_deploy=True
        )
        
        assert config.teacher_model == "gpt-5"
        assert config.student_model == "gpt-4.1-nano"
        assert config.strategy == OptimizationStrategy.BOOTSTRAP
        assert config.num_trials == 25
        assert config.auto_deploy is True
    
    def test_validation_constraints(self):
        """Test validation constraints on config fields."""
        # max_bootstrapped_demos must be 1-10
        with pytest.raises(ValueError):
            DSPyOptimizerConfig(max_bootstrapped_demos=0)
        
        with pytest.raises(ValueError):
            DSPyOptimizerConfig(max_bootstrapped_demos=15)
        
        # num_trials must be 1-50
        with pytest.raises(ValueError):
            DSPyOptimizerConfig(num_trials=100)
    
    def test_serialization(self):
        """Test config serialization."""
        config = DSPyOptimizerConfig(
            teacher_model="gpt-5",
            student_model="gpt-5-mini"
        )
        data = config.model_dump()
        
        assert data["teacher_model"] == "gpt-5"
        assert data["student_model"] == "gpt-5-mini"
        assert data["strategy"] == OptimizationStrategy.MIPRO_V2


# =============================================================================
# OptimizationStrategy Tests
# =============================================================================

class TestOptimizationStrategy:
    """Tests for OptimizationStrategy enum."""
    
    def test_all_strategies_exist(self):
        """Verify all expected strategies exist."""
        assert OptimizationStrategy.MIPRO_V2.value == "mipro_v2"
        assert OptimizationStrategy.BOOTSTRAP.value == "bootstrap"
        assert OptimizationStrategy.COPRO.value == "copro"
        assert OptimizationStrategy.SIGNATURE_OPT.value == "signature_opt"
    
    def test_strategy_from_string(self):
        """Test creating strategy from string value."""
        assert OptimizationStrategy("mipro_v2") == OptimizationStrategy.MIPRO_V2
        assert OptimizationStrategy("bootstrap") == OptimizationStrategy.BOOTSTRAP


# =============================================================================
# DSPyField Tests
# =============================================================================

class TestDSPyField:
    """Tests for DSPyField dataclass."""
    
    def test_input_field(self):
        """Test creating an input field."""
        field = DSPyField(
            name="query",
            description="The user's question",
            is_input=True
        )
        
        assert field.name == "query"
        assert field.description == "The user's question"
        assert field.is_input is True
    
    def test_output_field(self):
        """Test creating an output field."""
        field = DSPyField(
            name="response",
            description="The model's response",
            is_input=False
        )
        
        assert field.is_input is False
    
    def test_to_dspy_field_input(self):
        """Test generating DSPy field definition for input."""
        field = DSPyField(
            name="context",
            description="Background context",
            is_input=True
        )
        
        definition = field.to_dspy_field()
        
        assert "dspy.InputField" in definition
        assert "context" in definition
        assert "Background context" in definition
    
    def test_to_dspy_field_output(self):
        """Test generating DSPy field definition for output."""
        field = DSPyField(
            name="answer",
            description="The answer",
            is_input=False
        )
        
        definition = field.to_dspy_field()
        
        assert "dspy.OutputField" in definition
        assert "answer" in definition
    
    def test_field_with_prefix(self):
        """Test field with custom prefix."""
        field = DSPyField(
            name="query",
            description="The question",
            prefix="Question:",
            is_input=True
        )
        
        definition = field.to_dspy_field()
        
        assert 'prefix="Question:"' in definition


# =============================================================================
# DSPySignature Tests
# =============================================================================

class TestDSPySignature:
    """Tests for DSPySignature dataclass."""
    
    def test_create_signature(self):
        """Test creating a signature."""
        sig = DSPySignature(
            name="QuestionAnswer",
            doc="Answer questions based on context",
            input_fields=[
                DSPyField("context", "Background info", is_input=True),
                DSPyField("question", "The question", is_input=True),
            ],
            output_fields=[
                DSPyField("answer", "The response", is_input=False),
            ]
        )
        
        assert sig.name == "QuestionAnswer"
        assert len(sig.input_fields) == 2
        assert len(sig.output_fields) == 1
    
    def test_to_class_definition(self):
        """Test generating class definition."""
        sig = DSPySignature(
            name="Summarizer",
            doc="Summarize the given text",
            input_fields=[
                DSPyField("text", "Text to summarize", is_input=True),
            ],
            output_fields=[
                DSPyField("summary", "The summary", is_input=False),
            ]
        )
        
        class_def = sig.to_class_definition()
        
        assert "class Summarizer(dspy.Signature):" in class_def
        assert "Summarize the given text" in class_def
        assert "text" in class_def
        assert "summary" in class_def


# =============================================================================
# SignatureBuilder Tests
# =============================================================================

class TestSignatureBuilder:
    """Tests for SignatureBuilder class."""
    
    def test_from_template_basic(self):
        """Test building signature from simple template."""
        template = "Answer the following question: {question}"
        
        sig = SignatureBuilder.from_template(
            template=template,
            template_name="QASignature"
        )
        
        assert sig.name == "QASignature"
        assert len(sig.input_fields) == 1
        assert sig.input_fields[0].name == "question"
        assert len(sig.output_fields) == 1
    
    def test_from_template_multiple_vars(self):
        """Test building signature with multiple variables."""
        template = """
        Context: {context}
        Question: {question}
        Instructions: {instructions}
        
        Please provide a detailed answer.
        """
        
        sig = SignatureBuilder.from_template(
            template=template,
            template_name="ComplexQA"
        )
        
        assert len(sig.input_fields) == 3
        field_names = {f.name for f in sig.input_fields}
        assert "context" in field_names
        assert "question" in field_names
        assert "instructions" in field_names
    
    def test_from_template_with_descriptions(self):
        """Test building signature with custom variable descriptions."""
        template = "Translate {text} from {source_lang} to {target_lang}"
        
        sig = SignatureBuilder.from_template(
            template=template,
            template_name="TranslatorSignature",
            variable_descriptions={
                "text": "The text to translate",
                "source_lang": "Source language code",
                "target_lang": "Target language code"
            }
        )
        
        text_field = next(f for f in sig.input_fields if f.name == "text")
        assert text_field.description == "The text to translate"
    
    def test_from_template_custom_output(self):
        """Test building signature with custom output."""
        template = "Summarize: {document}"
        
        sig = SignatureBuilder.from_template(
            template=template,
            template_name="SummarySignature",
            output_name="summary",
            output_description="A concise summary of the document"
        )
        
        assert sig.output_fields[0].name == "summary"
        assert sig.output_fields[0].description == "A concise summary of the document"
    
    def test_from_template_duplicate_vars(self):
        """Test that duplicate variables are deduplicated."""
        template = "Compare {item1} with {item2}. Especially focus on {item1}."
        
        sig = SignatureBuilder.from_template(
            template=template,
            template_name="CompareSignature"
        )
        
        # Should only have 2 input fields (item1, item2) - no duplicates
        assert len(sig.input_fields) == 2
        field_names = [f.name for f in sig.input_fields]
        assert field_names.count("item1") == 1
    
    def test_from_genie_template(self):
        """Test building signature from PromptGenie result."""
        genie_result = {
            "template": "Summarize the following: {document}",
            "metadata": {
                "intent_tags": ["summarization"],
                "task_type": "summarization"
            },
            "variables": {
                "document": {
                    "description": "The document to summarize",
                    "type": "text"
                }
            }
        }
        
        sig = SignatureBuilder.from_genie_template(genie_result)
        
        assert "Summarization" in sig.name
        assert sig.output_fields[0].name == "summary"
    
    def test_variable_pattern_matches(self):
        """Test the variable pattern regex."""
        pattern = SignatureBuilder.VARIABLE_PATTERN
        
        # Standard {{var}}
        matches = pattern.findall("Hello {{name}}, how are you?")
        assert "name" in matches
        
        # Single brace {var}
        matches = pattern.findall("Hello {name}, how are you?")
        assert "name" in matches
        
        # With spaces
        matches = pattern.findall("Hello {{ name }}, how are you?")
        assert "name" in matches


# =============================================================================
# TrainingExample Tests
# =============================================================================

class TestTrainingExample:
    """Tests for TrainingExample dataclass."""
    
    def test_create_example(self):
        """Test creating a training example."""
        example = TrainingExample(
            inputs={"question": "What is Python?"},
            output="Python is a programming language.",
            quality_score=0.95,
            source_model="gpt-5"
        )
        
        assert example.inputs["question"] == "What is Python?"
        assert example.output == "Python is a programming language."
        assert example.quality_score == 0.95
        assert example.source_model == "gpt-5"
    
    def test_default_values(self):
        """Test default values for training example."""
        example = TrainingExample(
            inputs={"text": "Some text"},
            output="Processed text"
        )
        
        assert example.quality_score == 1.0
        assert example.source_model == "gpt-4o"
