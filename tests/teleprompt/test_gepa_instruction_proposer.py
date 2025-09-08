from dataclasses import dataclass
from typing import Any

import dspy
from dspy.teleprompt.gepa import instruction_proposal
from dspy.utils.dummies import DummyLM


def count_messages_with_image_url_pattern(messages):
    """Helper to count image URLs in messages - borrowed from image adapter tests"""
    pattern = {"type": "image_url", "image_url": {"url": lambda x: isinstance(x, str)}}

    try:

        def check_pattern(obj, pattern):
            if isinstance(pattern, dict):
                if not isinstance(obj, dict):
                    return False
                return all(k in obj and check_pattern(obj[k], v) for k, v in pattern.items())
            if callable(pattern):
                return pattern(obj)
            return obj == pattern

        def count_patterns(obj, pattern):
            count = 0
            if check_pattern(obj, pattern):
                count += 1
            if isinstance(obj, dict):
                count += sum(count_patterns(v, pattern) for v in obj.values())
            if isinstance(obj, (list, tuple)):
                count += sum(count_patterns(v, pattern) for v in obj)
            return count

        return count_patterns(messages, pattern)
    except Exception:
        return 0


@dataclass
class ImagesInHistory:
    has_structured_images: bool
    has_text_serialized_images: bool


def check_images_in_history(history: list[Any]) -> ImagesInHistory:
    def check_text_serialized(item: Any) -> bool:
        if isinstance(item, list):
            return any(check_text_serialized(i) for i in item)
        if isinstance(item, dict):
            return any(check_text_serialized(i) for i in item.values())
        if isinstance(item, str):
            return "CUSTOM-TYPE-START-IDENTIFIER" in item

        return False

    has_structured_images = False

    for call in history:
        if call.get("messages"):
            image_count = count_messages_with_image_url_pattern(call["messages"])
            if image_count > 0:
                has_structured_images = True

                break

    return ImagesInHistory(
        has_structured_images=has_structured_images,
        has_text_serialized_images=any(check_text_serialized(i) for i in history),
    )


def test_reflection_lm_gets_structured_images():
    """
    Verify reflection LM receives structured image messages, not serialized text.
    """
    student = dspy.Predict("image: dspy.Image -> label: str")
    image = dspy.Image.from_url("https://example.com/test.jpg", download=False)
    example = dspy.Example(image=image, label="dog").with_inputs("image")

    reflection_lm = DummyLM(
        [
            {"improved_instruction": "Better instruction"},
            {"improved_instruction": "Enhanced visual analysis instruction"},
            {"improved_instruction": "Focus on key features"},
            {"improved_instruction": "Analyze visual patterns systematically"},
            {"improved_instruction": "Consider distinctive visual elements"},
            {"improved_instruction": "Enhance recognition accuracy"},
            {"improved_instruction": "Improve classification methodology"},
        ]
    )
    lm = DummyLM(
        [
            {"label": "cat"},
            {"label": "dog"},
            {"label": "animal"},
            {"label": "pet"},
            {"label": "feline"},
            {"label": "canine"},
            {"label": "mammal"},
            {"label": "creature"},
            {"label": "species"},
            {"label": "domestic"},
            {"label": "wild"},
            {"label": "carnivore"},
            {"label": "herbivore"},
            {"label": "quadruped"},
            {"label": "vertebrate"},
        ]
    )
    dspy.settings.configure(lm=lm)

    gepa = dspy.GEPA(
        metric=lambda gold, pred, trace=None, pred_name=None, pred_trace=None: 0.3,
        max_metric_calls=2,
        reflection_lm=reflection_lm,
        instruction_proposer=instruction_proposal.MultiModalInstructionProposer(),
    )

    gepa.compile(student, trainset=[example], valset=[example])

    assert len(lm.history) > 0, "LM should have been called"
    assert len(reflection_lm.history) > 0, "Reflection LM should have been called"

    images_in_history = check_images_in_history(reflection_lm.history)

    assert images_in_history.has_structured_images, "Reflection LM should have received structured images"
    assert not images_in_history.has_text_serialized_images, "Reflection LM received serialized images in prompts"


def test_custom_proposer_without_reflection_lm():
    """Test that custom instruction proposers can work without reflection_lm when using updated GEPA core."""

    # External reflection LM managed by the custom proposer
    external_reflection_lm = DummyLM(
        [
            {"improved_instruction": "External LM response"},
            {"improved_instruction": "Enhanced instruction"},
            {"improved_instruction": "Better guidance"},
            {"improved_instruction": "Optimized instruction"},
            {"improved_instruction": "Refined approach"},
        ]
    )

    class ProposerWithExternalLM:
        def __call__(self, candidate, reflective_dataset, components_to_update):
            # This proposer manages its own external reflection LM
            with dspy.context(lm=external_reflection_lm):
                # Use external LM for reflection (optional - could be any custom logic)
                external_reflection_lm([{"role": "user", "content": "Improve this instruction"}])
                return {name: f"Externally-improved: {candidate[name]}" for name in components_to_update}

    student = dspy.Predict("text -> label")
    example = dspy.Example(text="test input", label="test").with_inputs("text")

    # Use a robust dummy LM with enough responses for optimization steps
    lm = DummyLM(
        [
            {"label": "test"},
            {"label": "result"},
            {"label": "output"},
            {"label": "response"},
            {"label": "classification"},
            {"label": "prediction"},
            {"label": "category"},
            {"label": "type"},
            {"label": "class"},
            {"label": "group"},
            {"label": "kind"},
            {"label": "variant"},
            {"label": "form"},
            {"label": "style"},
            {"label": "mode"},
        ]
    )
    dspy.settings.configure(lm=lm)

    # Test the full flexibility: no reflection_lm provided to GEPA at all!
    # The updated GEPA core library now allows this when using custom proposers
    gepa = dspy.GEPA(
        metric=lambda gold, pred, trace=None, pred_name=None, pred_trace=None: 0.7,  # Score to trigger optimization
        max_metric_calls=5,  # More calls to allow proper optimization
        reflection_lm=None,  # No reflection_lm provided - this now works!
        instruction_proposer=ProposerWithExternalLM(),
    )

    result = gepa.compile(student, trainset=[example], valset=[example])

    assert result is not None
    assert len(lm.history) > 0, "Main LM should have been called"
    assert len(external_reflection_lm.history) > 0, "External reflection LM should have been called by custom proposer"


def test_image_serialization_into_strings():
    """
    Test that demonstrates the image serialization problem when calling lm directly with serialized image data.
    """

    class InstructionProposerCallingLMDirectly:
        def __call__(
            self,
            candidate: dict[str, str],
            reflective_dataset: dict[str, list[dict[str, Any]]],
            components_to_update: list[str],
        ) -> dict[str, str]:
            updated_components = {}

            for component_name in components_to_update:
                if component_name not in candidate or component_name not in reflective_dataset:
                    continue

                current_instruction = candidate[component_name]
                component_data = reflective_dataset[component_name]

                feedback_analysis = "Feedback analysis:\n"
                for i, example in enumerate(component_data):
                    feedback_analysis += f"Example {i + 1}:\n"

                    # Non ideal approach: extract and serialize image objects directly
                    inputs = example.get("Inputs", {})
                    for key, value in inputs.items():
                        feedback_analysis += f"  {key}: {value}\n"

                    outputs = example.get("Generated Outputs", {})
                    feedback = example.get("Feedback", "")
                    feedback_analysis += f"  Outputs: {outputs}\n"
                    feedback_analysis += f"  Feedback: {feedback}\n\n"

                context_lm = dspy.settings.lm
                messages = [
                    {"role": "system", "content": "You are an instruction improvement assistant."},
                    {
                        "role": "user",
                        "content": f"Current instruction: {current_instruction}\n\nFeedback: {feedback_analysis}\n\nProvide an improved instruction:",
                    },
                ]

                result = context_lm(messages=messages)
                updated_components[component_name] = result[0]

            return updated_components

    direct_lm_call_proposer = InstructionProposerCallingLMDirectly()

    student = dspy.Predict("image -> label")

    image = dspy.Image.from_url("https://picsum.photos/id/237/200/300", download=False)

    examples = [
        dspy.Example(image=image, label="cat").with_inputs("image"),
        dspy.Example(image=image, label="animal").with_inputs("image"),
    ]

    lm = DummyLM(
        [
            {"label": "cat"},
            {"label": "dog"},
            {"label": "animal"},
            {"label": "pet"},
            {"label": "feline"},
            {"label": "mammal"},
            {"label": "creature"},
            {"label": "species"},
            {"label": "domestic"},
            {"label": "wild"},
            {"label": "carnivore"},
            {"label": "herbivore"},
        ]
    )
    dspy.settings.configure(lm=lm)

    reflection_lm = DummyLM(
        [
            {"improved_instruction": "Be more specific about image analysis"},
            {"improved_instruction": "Focus on visual features when classifying"},
            {"improved_instruction": "Consider contextual clues in the image"},
            {"improved_instruction": "Analyze shape, color, and texture patterns"},
            {"improved_instruction": "Look for distinguishing characteristics"},
        ]
    )

    gepa = dspy.GEPA(
        metric=lambda gold, pred, trace=None, pred_name=None, pred_trace=None: 0.3,
        max_metric_calls=5,
        reflection_lm=reflection_lm,
        instruction_proposer=direct_lm_call_proposer,
    )

    gepa.compile(student, trainset=examples, valset=examples)

    assert len(lm.history) > 0, "LM should have been called"
    assert len(reflection_lm.history) > 0, "Reflection LM should have been called"

    images_in_history = check_images_in_history(reflection_lm.history)

    assert images_in_history.has_text_serialized_images, (
        "Expected to find serialized images (CUSTOM-TYPE-START-IDENTIFIER)"
    )


def test_default_proposer():
    student = dspy.Predict("image -> label")

    image = dspy.Image.from_url("https://picsum.photos/id/237/200/300", download=False)

    examples = [
        dspy.Example(image=image, label="cat").with_inputs("image"),
        dspy.Example(image=image, label="animal").with_inputs("image"),
    ]

    lm = DummyLM(
        [
            {"label": "cat"},
            {"label": "dog"},
            {"label": "animal"},
            {"label": "pet"},
            {"label": "feline"},
            {"label": "mammal"},
            {"label": "creature"},
            {"label": "species"},
            {"label": "domestic"},
            {"label": "wild"},
            {"label": "carnivore"},
            {"label": "herbivore"},
        ]
    )
    dspy.settings.configure(lm=lm)

    reflection_lm = DummyLM(
        [
            {"improved_instruction": "Be more specific about image analysis"},
            {"improved_instruction": "Focus on visual features when classifying"},
            {"improved_instruction": "Consider contextual clues in the image"},
            {"improved_instruction": "Analyze shape, color, and texture patterns"},
            {"improved_instruction": "Look for distinguishing characteristics"},
        ]
    )

    gepa = dspy.GEPA(
        metric=lambda gold, pred, trace=None, pred_name=None, pred_trace=None: 0.3,
        max_metric_calls=5,
        reflection_lm=reflection_lm,
    )

    gepa.compile(student, trainset=examples, valset=examples)

    assert len(lm.history) > 0, "LM should have been called"
    assert len(reflection_lm.history) > 0, "Reflection LM should have been called"

    images_in_history = check_images_in_history(reflection_lm.history)

    assert images_in_history.has_text_serialized_images, (
        "Expected to find serialized images (CUSTOM-TYPE-START-IDENTIFIER)"
    )


def test_verbosity_refined_proposer_formatting():
    """Test that the verbosity refined proposer formats examples correctly."""
    from dspy.teleprompt.gepa.gepa_utils import ReflectiveExample
    from dspy.teleprompt.gepa.instruction_proposal import SingleComponentVerbosityRefinedInstructionProposer

    proposer = SingleComponentVerbosityRefinedInstructionProposer()

    sample_data: list[ReflectiveExample] = [
        {
            "Inputs": {
                "question": "What is the capital of France?",
                "context": "France is a country in Europe."
            },
            "Generated_Outputs": {
                "answer": "London"
            },
            "Feedback": "Incorrect. The capital of France is Paris, not London."
        },
        {
            "Inputs": {
                "question": "What is the capital of Spain?",
                "context": "Spain is located in southwestern Europe."
            },
            "Generated_Outputs": {
                "answer": "Madrid"
            },
            "Feedback": "Correct! Madrid is indeed the capital of Spain."
        }
    ]

    formatted = proposer._format_examples_for_instruction_generation(sample_data)

    # Check for GEPA-compatible structure
    assert "# Example 1" in formatted, "Should have example headers"
    assert "# Example 2" in formatted, "Should have multiple examples"
    assert "## Inputs" in formatted, "Should have input section headers"
    assert "## Generated_Outputs" in formatted, "Should have output section headers"
    assert "## Feedback" in formatted, "Should have feedback section headers"
    assert "### question" in formatted, "Should have nested field headers"
    assert "### context" in formatted, "Should have nested context headers"


def test_verbosity_refined_proposer_with_gepa():
    """Test integration of verbosity refined proposer with GEPA."""
    from dspy.teleprompt.gepa.instruction_proposal import VerbosityRefinedInstructionProposer

    student = dspy.Predict("question, context -> answer")

    examples = [
        dspy.Example(
            question="What is the capital of France?",
            context="France is a country in Europe.",
            answer="Paris"
        ).with_inputs("question", "context"),
        dspy.Example(
            question="What is the capital of Spain?",
            context="Spain is located in southwestern Europe.",
            answer="Madrid"
        ).with_inputs("question", "context")
    ]

    lm = DummyLM([
        {"answer": "London"},  # Incorrect first
        {"answer": "Paris"},   # Correct after optimization
        {"answer": "Barcelona"},  # Incorrect first
        {"answer": "Madrid"},  # Correct after optimization
        {"answer": "Berlin"},
        {"answer": "Rome"},
        {"answer": "Vienna"},
        {"answer": "Amsterdam"},
        {"answer": "Brussels"},
        {"answer": "Prague"}
    ])
    dspy.settings.configure(lm=lm)

    reflection_lm = DummyLM([
        {"reasoning": "Analyzing geography feedback to improve instruction", "improved_instruction": "When answering geography questions about capitals, ensure you correctly identify the capital city of the specified country. For European countries, be especially careful to distinguish between different nations."},
        {"reasoning": "Focusing on European capital accuracy", "improved_instruction": "Focus on providing accurate geographical knowledge about European capitals. When given a question about a country's capital, provide the correct capital city name."},
        {"reasoning": "Ensuring factual geographical accuracy", "improved_instruction": "Answer geography questions with factual accuracy. Pay attention to the specific country mentioned and provide its correct capital city."},
        {"reasoning": "Enhancing geographical knowledge", "improved_instruction": "Provide accurate capital city information for all European countries."},
        {"reasoning": "Improving geographical accuracy", "improved_instruction": "Ensure correct identification of capital cities for European nations."},
        {"reasoning": "Optimizing geographical responses", "improved_instruction": "Focus on precise geographical knowledge about European capitals."},
        {"reasoning": "Refining capital city knowledge", "improved_instruction": "Correctly identify capital cities of European countries."},
        {"reasoning": "Strengthening geographical accuracy", "improved_instruction": "Provide factual information about European capital cities."}
    ])

    # Test with general proposer (simplified interface)
    gepa = dspy.GEPA(
        metric=lambda gold, pred, trace=None, pred_name=None, pred_trace=None: 1.0 if pred.answer.lower() in gold.answer.lower() else 0.0,
        max_metric_calls=4,
        reflection_lm=reflection_lm,
        instruction_proposer=VerbosityRefinedInstructionProposer()
    )

    result = gepa.compile(student, trainset=examples, valset=examples)

    assert result is not None
    assert len(lm.history) > 0, "Main LM should have been called"
    assert len(reflection_lm.history) > 0, "Reflection LM should have been called"


def test_verbosity_refined_proposer_max_length():
    """Test integration of verbosity refined proposer with max_length constraint in GEPA."""
    from dspy.teleprompt.gepa.instruction_proposal import VerbosityRefinedInstructionProposer

    student = dspy.Predict("question, context -> answer")

    examples = [
        dspy.Example(
            question="What is the capital of France?",
            context="France is a country in Europe with a rich history and culture.",
            answer="Paris"
        ).with_inputs("question", "context"),
        dspy.Example(
            question="What is the capital of Spain?",
            context="Spain is located in southwestern Europe on the Iberian Peninsula.",
            answer="Madrid"
        ).with_inputs("question", "context")
    ]

    lm = DummyLM([
        {"answer": "London"},  # Incorrect first
        {"answer": "Paris"},   # Correct after optimization
        {"answer": "Barcelona"},  # Incorrect first
        {"answer": "Madrid"},  # Correct after optimization
        {"answer": "Berlin"},
        {"answer": "Rome"},
        {"answer": "Vienna"},
        {"answer": "Brussels"},
        {"answer": "Prague"}
    ])
    dspy.settings.configure(lm=lm)

    reflection_lm = DummyLM([
        # Standard ChainOfThought responses for GenerateImprovedInstruction
        {"reasoning": "Analyzing feedback with length constraints", "improved_instruction": "Answer geography questions accurately. Focus on capital cities of European countries. Be precise and concise."},
        {"reasoning": "Compressing instruction for brevity", "improved_instruction": "Identify European capitals correctly."},
        {"reasoning": "Further refinement needed", "improved_instruction": "Provide accurate European capital cities."},
        {"reasoning": "Optimizing instruction length", "improved_instruction": "Answer with correct European capitals."},
        {"reasoning": "Final compression pass", "improved_instruction": "Name the correct capital city."},
        {"reasoning": "Ensuring accuracy with brevity", "improved_instruction": "Give accurate capital information."},
        {"reasoning": "Length-aware optimization", "improved_instruction": "Correctly identify capitals."},
        {"reasoning": "Maintaining accuracy while shortening", "improved_instruction": "Provide correct capitals."},
    ])

    # Test with max_length constraint (should trigger refinement when needed)
    gepa = dspy.GEPA(
        metric=lambda gold, pred, trace=None, pred_name=None, pred_trace=None: 1.0 if pred.answer.lower() in gold.answer.lower() else 0.0,
        max_metric_calls=4,
        reflection_lm=reflection_lm,
        instruction_proposer=VerbosityRefinedInstructionProposer(max_length=50)  # Short limit to test refinement
    )

    result = gepa.compile(student, trainset=examples, valset=examples)

    assert result is not None
    assert len(lm.history) > 0, "Main LM should have been called"
    assert len(reflection_lm.history) > 0, "Reflection LM should have been called"
