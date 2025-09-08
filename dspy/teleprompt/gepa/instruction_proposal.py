from typing import Any

from gepa.core.adapter import ProposalFn

import dspy
from dspy.adapters.types.base_type import Type
from dspy.teleprompt.gepa.gepa_utils import ReflectiveExample


class GenerateEnhancedMultimodalInstructionFromFeedback(dspy.Signature):
    """I provided an assistant with instructions to perform a task involving visual content, but the assistant's performance needs improvement based on the examples and feedback below.

    Your task is to write a better instruction for the assistant that addresses the specific issues identified in the feedback, with particular attention to how visual and textual information should be analyzed and integrated.

    ## Analysis Steps:
    1. **Read the inputs carefully** and identify both the visual and textual input formats, understanding how they work together
    2. **Read all the assistant responses and corresponding feedback** to understand what went wrong with visual analysis, text processing, or their integration
    3. **Identify visual analysis patterns** - what visual features, relationships, or details are important for this task
    4. **Identify domain-specific knowledge** about both visual and textual aspects, as this information may not be available to the assistant in the future
    5. **Look for successful visual-textual integration strategies** and include these patterns in the instruction
    6. **Address specific visual analysis issues** mentioned in the feedback

    ## Instruction Requirements:
    - **Clear task definition** explaining how to process both visual and textual inputs
    - **Visual analysis guidance** specific to this task (what to look for, how to describe, what features matter)
    - **Integration strategies** for combining visual observations with textual information
    - **Domain-specific knowledge** about visual concepts, terminology, or relationships
    - **Error prevention guidance** for common visual analysis mistakes shown in the feedback
    - **Precise, actionable language** for both visual and textual processing

    Focus on creating an instruction that helps the assistant properly analyze visual content, integrate it with textual information, and avoid the specific visual analysis mistakes shown in the examples."""

    current_instruction = dspy.InputField(
        desc="The current instruction that was provided to the assistant to perform the multimodal task"
    )
    examples_with_feedback = dspy.InputField(
        desc="Task examples with visual content showing inputs, assistant outputs, and feedback. "
        "Pay special attention to feedback about visual analysis accuracy, visual-textual integration, "
        "and any domain-specific visual knowledge that the assistant missed."
    )

    improved_instruction = dspy.OutputField(
        desc="A better instruction for the assistant that addresses visual analysis issues, provides "
        "clear guidance on how to process and integrate visual and textual information, includes "
        "necessary visual domain knowledge, and prevents the visual analysis mistakes shown in the examples."
    )


class SingleComponentMultiModalProposer(dspy.Module):
    """
    dspy.Module for proposing improved instructions based on feedback.
    """

    def __init__(self):
        super().__init__()
        self.propose_instruction = dspy.Predict(GenerateEnhancedMultimodalInstructionFromFeedback)

    def forward(self, current_instruction: str, reflective_dataset: list[ReflectiveExample]) -> str:
        """
        Generate an improved instruction based on current instruction and feedback examples.

        Args:
            current_instruction: The current instruction that needs improvement
            reflective_dataset: List of examples with inputs, outputs, and feedback
                               May contain dspy.Image objects in inputs

        Returns:
            str: Improved instruction text
        """
        # Format examples with enhanced pattern recognition
        formatted_examples, image_map = self._format_examples_with_pattern_analysis(reflective_dataset)

        # Build kwargs for the prediction call
        predict_kwargs = {
            "current_instruction": current_instruction,
            "examples_with_feedback": formatted_examples,
        }

        # Create a rich multimodal examples_with_feedback that includes both text and images
        predict_kwargs["examples_with_feedback"] = self._create_multimodal_examples(formatted_examples, image_map)

        # Use current dspy LM settings (GEPA will pass reflection_lm via context)
        result = self.propose_instruction(**predict_kwargs)

        return result.improved_instruction

    def _format_examples_with_pattern_analysis(
        self, reflective_dataset: list[ReflectiveExample]
    ) -> tuple[str, dict[int, list[Type]]]:
        """
        Format examples with pattern analysis and feedback categorization.

        Returns:
            tuple: (formatted_text_with_patterns, image_map)
        """
        # First, use the existing proven formatting approach
        formatted_examples, image_map = self._format_examples_for_instruction_generation(reflective_dataset)

        # Enhanced analysis: categorize feedback patterns
        feedback_analysis = self._analyze_feedback_patterns(reflective_dataset)

        # Add pattern analysis to the formatted examples
        if feedback_analysis["summary"]:
            pattern_summary = self._create_pattern_summary(feedback_analysis)
            enhanced_examples = f"{pattern_summary}\n\n{formatted_examples}"
            return enhanced_examples, image_map

        return formatted_examples, image_map

    def _analyze_feedback_patterns(self, reflective_dataset: list[ReflectiveExample]) -> dict[str, Any]:
        """
        Analyze feedback patterns to provide better context for instruction generation.

        Categorizes feedback into:
        - Error patterns: Common mistakes and their types
        - Success patterns: What worked well and should be preserved/emphasized
        - Domain knowledge gaps: Missing information that should be included
        - Task-specific guidance: Specific requirements or edge cases
        """
        analysis = {
            "error_patterns": [],
            "success_patterns": [],
            "domain_knowledge_gaps": [],
            "task_specific_guidance": [],
            "summary": "",
        }

        # Simple pattern recognition - could be enhanced further
        for example in reflective_dataset:
            feedback = example.get("Feedback", "").lower()

            # Identify error patterns
            if any(error_word in feedback for error_word in ["incorrect", "wrong", "error", "failed", "missing"]):
                analysis["error_patterns"].append(feedback)

            # Identify success patterns
            if any(
                success_word in feedback for success_word in ["correct", "good", "accurate", "well", "successfully"]
            ):
                analysis["success_patterns"].append(feedback)

            # Identify domain knowledge needs
            if any(
                knowledge_word in feedback
                for knowledge_word in ["should know", "domain", "specific", "context", "background"]
            ):
                analysis["domain_knowledge_gaps"].append(feedback)

        # Create summary if patterns were found
        if any(analysis[key] for key in ["error_patterns", "success_patterns", "domain_knowledge_gaps"]):
            analysis["summary"] = (
                f"Patterns identified: {len(analysis['error_patterns'])} error(s), {len(analysis['success_patterns'])} success(es), {len(analysis['domain_knowledge_gaps'])} knowledge gap(s)"
            )

        return analysis

    def _create_pattern_summary(self, feedback_analysis: dict[str, Any]) -> str:
        """Create a summary of feedback patterns to help guide instruction generation."""

        summary_parts = ["## Feedback Pattern Analysis\n"]

        if feedback_analysis["error_patterns"]:
            summary_parts.append(f"**Common Issues Found ({len(feedback_analysis['error_patterns'])} examples):**")
            summary_parts.append("Focus on preventing these types of mistakes in the new instruction.\n")

        if feedback_analysis["success_patterns"]:
            summary_parts.append(
                f"**Successful Approaches Found ({len(feedback_analysis['success_patterns'])} examples):**"
            )
            summary_parts.append("Build on these successful strategies in the new instruction.\n")

        if feedback_analysis["domain_knowledge_gaps"]:
            summary_parts.append(
                f"**Domain Knowledge Needs Identified ({len(feedback_analysis['domain_knowledge_gaps'])} examples):**"
            )
            summary_parts.append("Include this specialized knowledge in the new instruction.\n")

        return "\n".join(summary_parts)

    def _format_examples_for_instruction_generation(
        self, reflective_dataset: list[ReflectiveExample]
    ) -> tuple[str, dict[int, list[Type]]]:
        """
        Format examples using GEPA's markdown structure while preserving image objects.

        Returns:
            tuple: (formatted_text, image_map) where image_map maps example_index -> list[images]
        """

        def render_value_with_images(value, level=3, example_images=None):
            if example_images is None:
                example_images = []

            if isinstance(value, Type):
                image_idx = len(example_images) + 1
                example_images.append(value)
                return f"[IMAGE-{image_idx} - see visual content]\n\n"
            elif isinstance(value, dict):
                s = ""
                for k, v in value.items():
                    s += f"{'#' * level} {k}\n"
                    s += render_value_with_images(v, min(level + 1, 6), example_images)
                if not value:
                    s += "\n"
                return s
            elif isinstance(value, (list, tuple)):
                s = ""
                for i, item in enumerate(value):
                    s += f"{'#' * level} Item {i + 1}\n"
                    s += render_value_with_images(item, min(level + 1, 6), example_images)
                if not value:
                    s += "\n"
                return s
            else:
                return f"{str(value).strip()}\n\n"

        def convert_sample_to_markdown_with_images(sample, example_num):
            example_images = []
            s = f"# Example {example_num}\n"

            for key, val in sample.items():
                s += f"## {key}\n"
                s += render_value_with_images(val, level=3, example_images=example_images)

            return s, example_images

        formatted_parts = []
        image_map = {}

        for i, example_data in enumerate(reflective_dataset):
            formatted_example, example_images = convert_sample_to_markdown_with_images(example_data, i + 1)
            formatted_parts.append(formatted_example)

            if example_images:
                image_map[i] = example_images

        formatted_text = "\n\n".join(formatted_parts)

        if image_map:
            total_images = sum(len(imgs) for imgs in image_map.values())
            formatted_text = (
                f"The examples below include visual content ({total_images} images total). "
                "Please analyze both the text and visual elements when suggesting improvements.\n\n" + formatted_text
            )

        return formatted_text, image_map

    def _create_multimodal_examples(self, formatted_text: str, image_map: dict[int, list[Type]]) -> Any:
        """
        Create a multimodal input that contains both text and images for the reflection LM.

        Args:
            formatted_text: The formatted text with image placeholders
            image_map: Dictionary mapping example_index -> list[images] for structured access
        """
        if not image_map:
            return formatted_text

        # Collect all images from all examples
        all_images = []
        for example_images in image_map.values():
            all_images.extend(example_images)

        multimodal_content = [formatted_text]
        multimodal_content.extend(all_images)
        return multimodal_content


class MultiModalInstructionProposer(ProposalFn):
    """GEPA-compatible multimodal instruction proposer.

    This class handles multimodal inputs (like dspy.Image) during GEPA optimization by using
    a single-component proposer for each component that needs to be updated.
    """

    def __init__(self):
        self.single_proposer = SingleComponentMultiModalProposer()

    def __call__(
        self,
        candidate: dict[str, str],
        reflective_dataset: dict[str, list[ReflectiveExample]],
        components_to_update: list[str],
    ) -> dict[str, str]:
        """GEPA-compatible proposal function.

        Args:
            candidate: Current component name -> instruction mapping
            reflective_dataset: Component name -> list of reflective examples
            components_to_update: List of component names to update

        Returns:
            dict: Component name -> new instruction mapping
        """
        updated_components = {}

        for component_name in components_to_update:
            if component_name in candidate and component_name in reflective_dataset:
                current_instruction = candidate[component_name]
                component_reflective_data = reflective_dataset[component_name]

                # Call the single-instruction proposer.
                #
                # In the future, proposals could consider multiple components instructions,
                # instead of just the current instruction, for more holistic instruction proposals.
                new_instruction = self.single_proposer(
                    current_instruction=current_instruction, reflective_dataset=component_reflective_data
                )

                updated_components[component_name] = new_instruction

        return updated_components



class GenerateImprovedInstruction(dspy.Signature):
    """I provided an assistant with instructions to perform a task for me, but the assistant's performance needs improvement based on the examples and feedback below.

    Your task is to write a better instruction for the assistant that addresses the specific issues identified in the feedback.

    ## Analysis Steps:
    1. **Read the inputs carefully** and identify the input format and infer detailed task description about the task I wish to solve with the assistant
    2. **Read all the assistant responses and corresponding feedback** to understand what went wrong and what worked well
    3. **Identify domain-specific knowledge** about the task, as this information may not be available to the assistant in the future
    4. **Look for successful strategies** the assistant may have utilized and include these patterns in the instruction
    5. **Address specific issues** mentioned in the feedback to prevent similar mistakes

    ## Instruction Requirements:
    - **Clear task definition** explaining what the assistant needs to accomplish
    - **Domain-specific knowledge** about concepts, terminology, or relationships relevant to the task
    - **Successful strategy guidance** for approaches that work well for this type of task
    - **Error prevention guidance** for common mistakes shown in the feedback
    - **Precise, actionable language** that helps the assistant perform better

    Focus on creating an instruction that incorporates domain knowledge, successful strategies, and addresses the specific issues shown in the examples."""

    current_instruction = dspy.InputField(
        desc="The current instruction that was provided to the assistant to perform the task"
    )
    examples_with_feedback = dspy.InputField(
        desc="Task execution examples showing inputs, assistant outputs, and feedback on how the assistant's response could be better. "
        "Each example contains the task inputs, what the assistant generated, and detailed feedback about the performance. "
        "Pay special attention to domain-specific knowledge gaps, successful strategies, and error patterns."
    )

    improved_instruction = dspy.OutputField(
        desc="The enhanced instruction that addresses the identified issues, incorporates "
        "domain-specific knowledge, successful strategies, and provides clear actionable guidance. "
        "Write the complete improved instruction directly."
    )


class SingleComponentVerbosityRefinedInstructionProposer(dspy.Module):
    """DSPy-native instruction proposer with verbosity control.
    
    This module provides length-aware instruction generation using DSPy's native patterns.
    When max_length is specified, it uses dspy.Refine to iteratively compress instructions
    that exceed the character limit while preserving essential content.
    
    Features:
    - Standard instruction generation for unconstrained cases
    - Automatic refinement for length-constrained cases using reward-based optimization
    - Preserves GEPA's proven methodology and formatting
    """

    def __init__(self, max_length: int | None = None):
        """Initialize the proposer with single-step generation (matching GEPA core).
        
        Args:
            max_length: Maximum character length for compact instructions. If None, uses standard generation.
        """
        super().__init__()
        self.max_length = max_length
        self.generate_instruction = dspy.ChainOfThought(GenerateImprovedInstruction)

        if max_length is not None:
            def max_length_reward(args, pred: dspy.Prediction) -> float:
                """
                Reward instructions that are at or below max_length get perfect score.

                Instructions should have the max length while maintaining their main key points.
                Compaction should be very thoughtful and not lose any important information.
                """
                character_count = len(pred.improved_instruction)
                if character_count <= max_length:
                    return 1.0  # Perfect score for instructions within limit
                else:
                    # Penalize based on how much it exceeds the limit
                    excess = character_count - max_length
                    penalty = excess / max_length  # Normalize penalty
                    return max(0.0, 1.0 - penalty)

            self.generate_instruction = dspy.Refine(
                self.generate_instruction,
                N=3,
                reward_fn=max_length_reward,
                threshold=1.0,
            )

    def forward(self, current_instruction: str, reflective_dataset: list[ReflectiveExample]) -> str:
        """Generate improved instruction based on current instruction and feedback examples.
        
        Args:
            current_instruction: The current instruction that needs improvement
            reflective_dataset: List of examples with inputs, outputs, and feedback
            
        Returns:
            str: Improved instruction text (compact version if max_length is specified)
        """
        # Format examples using GEPA's proven markdown structure
        formatted_examples = self._format_examples_for_instruction_generation(reflective_dataset)

        result = self.generate_instruction(
            current_instruction=current_instruction,
            examples_with_feedback=formatted_examples
        )
        return result.improved_instruction

    def _format_examples_for_instruction_generation(self, reflective_dataset: list[ReflectiveExample]) -> str:
        """Format examples using GEPA's proven markdown structure.
        
        This method replicates the exact formatting logic from GEPA's default proposer,
        ensuring compatibility and maintaining the proven effectiveness.
        """

        def render_value(value, level=3):
            """Render values with hierarchical markdown structure."""
            if isinstance(value, dict):
                s = ""
                for k, v in value.items():
                    s += f"{'#' * level} {k}\n"
                    s += render_value(v, min(level + 1, 6))
                if not value:
                    s += "\n"
                return s
            elif isinstance(value, (list, tuple)):
                s = ""
                for i, item in enumerate(value):
                    s += f"{'#' * level} Item {i + 1}\n"
                    s += render_value(item, min(level + 1, 6))
                if not value:
                    s += "\n"
                return s
            else:
                return f"{str(value).strip()}\n\n"

        def convert_sample_to_markdown(sample, example_num):
            """Convert a single example to markdown format."""
            s = f"# Example {example_num}\n"
            for key, val in sample.items():
                s += f"## {key}\n"
                s += render_value(val, level=3)
            return s

        formatted_parts = []
        for i, example_data in enumerate(reflective_dataset):
            formatted_example = convert_sample_to_markdown(example_data, i + 1)
            formatted_parts.append(formatted_example)

        return "\n\n".join(formatted_parts)


class VerbosityRefinedInstructionProposer(ProposalFn):
    """GEPA-compatible instruction proposer with verbosity refinement.
    
    This class implements the ProposalFn protocol with optional length constraints.
    Provides DSPy-native instruction generation with automatic refinement for length control.
    
    Features:
    - Optional max_length parameter for verbosity control
    - Maintains single-step methodology when no constraints
    - Uses dspy.Refine for intelligent compression when needed
    """

    def __init__(self, max_length: int | None = None):
        """Initialize the proposer (matching GEPA core's simplicity).
        
        Args:
            max_length: Maximum character length for generated instructions. If None, no length constraint.
        """
        self.proposer = SingleComponentVerbosityRefinedInstructionProposer(max_length=max_length)

    def __call__(
        self,
        candidate: dict[str, str],
        reflective_dataset: dict[str, list[ReflectiveExample]],
        components_to_update: list[str],
    ) -> dict[str, str]:
        """
        Args:
            candidate: Current component name -> instruction mapping
            reflective_dataset: Component name -> list of reflective examples  
            components_to_update: List of component names to update
            
        Returns:
            dict: Component name -> new instruction mapping
        """
        updated_components = {}

        for component_name in components_to_update:
            if component_name in candidate and component_name in reflective_dataset:
                current_instruction = candidate[component_name]
                component_reflective_data = reflective_dataset[component_name]

                # Direct generation (matching GEPA core's single-step approach)
                new_instruction = self.proposer(
                    current_instruction=current_instruction,
                    reflective_dataset=component_reflective_data
                )

                updated_components[component_name] = new_instruction

        return updated_components
