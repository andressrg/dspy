"""Test SIMBA's OfferFeedback signature with image handling"""

import dspy
from dspy.teleprompt import SIMBA
from dspy.utils.dummies import DummyLM


def setup_image_classifier():
    """Helper to create image classification program and training data"""

    class ImageClassifier(dspy.Signature):
        """Classify what's in an image"""
        image: dspy.Image = dspy.InputField(desc="Image to classify")
        classification: str = dspy.OutputField(desc="What's in the image")

    trainset = [
        dspy.Example(
            image=dspy.Image.from_url("https://example.com/golden_retriever.jpg", download=False),
            classification="golden retriever",
        ).with_inputs("image"),
        dspy.Example(
            image=dspy.Image.from_url("https://example.com/siamese_cat.jpg", download=False),
            classification="siamese cat",
        ).with_inputs("image"),
    ]

    def metric(example, prediction):
        if not prediction or not hasattr(prediction, "classification"):
            return 0.0
        pred = prediction.classification.lower()
        expected = example.classification.lower()
        return 1.0 if expected in pred or pred in expected else 0.1

    return dspy.Predict(ImageClassifier), trainset, metric


def test_simba_preserves_images_in_advice():
    """Test that SIMBA's OfferFeedback receives actual Images, not placeholders"""
    student, trainset, metric = setup_image_classifier()

    # Responses that will create good/bad trajectories and advice generation
    responses = [
        '{"classification": "golden retriever"}',  # Good response
        '{"classification": "some animal"}',       # Bad response
        '{"classification": "siamese cat"}',       # Good response
        '{"classification": "pet"}',               # Bad response
        # OfferFeedback response for advice generation
        '{"discussion": "Better trajectory provides specific classifications", '
        '"module_advice": {"0": "Identify specific breeds rather than generic terms"}}',
        # Extra responses for robustness
        '{"classification": "classified"}',
        '{"classification": "identified"}',
    ]

    class ImageCapturingLM(DummyLM):
        """LM that captures calls containing images for analysis"""
        def __init__(self, responses):
            super().__init__(responses)
            self.image_calls = []

        def __call__(self, *args, **kwargs):
            if "messages" in kwargs:
                messages_str = str(kwargs["messages"])
                # Check if this call contains our test image URLs
                if any(url in messages_str for url in ["golden_retriever.jpg", "siamese_cat.jpg"]):
                    self.image_calls.append(messages_str)
            return super().__call__(*args, **kwargs)

    lm = ImageCapturingLM(responses)
    dspy.settings.configure(lm=lm)

    # Run SIMBA optimization
    optimizer = SIMBA(metric=metric, bsize=2, num_candidates=2, max_steps=1)
    optimizer.compile(student, trainset=trainset)

    # Verify that at least one call contained actual image URLs (not placeholders)
    assert len(lm.image_calls) > 0, "SIMBA should make calls containing images"

    # Check that images are preserved as URLs, not converted to placeholders
    has_actual_urls = any(
        "golden_retriever.jpg" in call or "siamese_cat.jpg" in call
        for call in lm.image_calls
    )
    has_placeholders = any(
        "<non-serializable" in call or "non-serializable: Image" in call
        for call in lm.image_calls
    )

    assert has_actual_urls, "OfferFeedback should receive actual image URLs"
    assert not has_placeholders, "OfferFeedback should not receive image placeholders"
