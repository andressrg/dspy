"""
Simple test to demonstrate the core image serialization problem in SIMBA's OfferFeedback
"""

import dspy
from dspy.teleprompt.simba_utils import recursive_mask


def test_recursive_mask_preserves_images():
    """Test that recursive_mask now preserves dspy.Image objects (after fix)"""
    test_image = dspy.Image.from_url("https://example.com/test.jpg", download=False)

    # After our fix, recursive_mask should preserve Image objects
    masked_result = recursive_mask(test_image)

    assert isinstance(masked_result, dspy.Image)
    assert masked_result is test_image  # Should return the same object
    assert masked_result.url == "https://example.com/test.jpg"


def test_trajectory_preserves_images():
    """Test that trajectory data preserves Images in complex structures"""
    test_image = dspy.Image.from_url("https://example.com/test.jpg", download=False)

    # Create trajectory structure like SIMBA uses
    trajectory_data = {
        "program_inputs": {"image": test_image, "prompt": "What is this?"},
        "better_trajectory": [
            {"inputs": {"image": test_image, "prompt": "What is this?"}, "outputs": {"answer": "A test image"}}
        ],
    }

    # After our fix, recursive_mask should preserve Images in complex structures
    masked_data = recursive_mask(trajectory_data)

    assert isinstance(masked_data["program_inputs"]["image"], dspy.Image)
    assert isinstance(masked_data["better_trajectory"][0]["inputs"]["image"], dspy.Image)
    assert masked_data["program_inputs"]["image"].url == "https://example.com/test.jpg"
