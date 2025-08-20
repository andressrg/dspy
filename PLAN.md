# SIMBA Image Handling Fix Plan

## Overview

SIMBA (Stochastic Introspective Mini-Batch Ascent) is a sophisticated DSPy optimizer that implements **"prompt-space reinforcement learning"** - using the LLM to reason about its own performance and generate targeted improvement strategies. Unlike simple few-shot optimizers, SIMBA is self-reflective and adaptive.

### SIMBA's Core Innovation: Introspective Optimization

SIMBA represents an evolution from basic prompt optimization to **self-reflective algorithms** that:
- Analyze their own execution traces
- Identify failure patterns and success indicators  
- Generate specific, actionable improvement rules
- Balance exploration vs. exploitation through intelligent sampling

### Two Strategic Improvement Mechanisms:

1. **Rule Strategy** (`append_a_rule`) - **The Primary Innovation**
   - LLM analyzes successful vs. failed trajectories
   - Generates specific, actionable advice (e.g., "When dealing with math word problems, always identify numerical values first")
   - Creates **meta-cognitive improvements** through self-reflection
   - **Most sophisticated and powerful** optimization strategy

2. **Demo Strategy** (`append_a_demo`) - **Traditional Few-Shot**
   - Simply adds successful execution examples to predictor demos
   - Less sophisticated but reliable fallback mechanism

### Current State:
- ✅ **Demo strategy** works correctly with `dspy.Image` inputs
- ❌ **Rule strategy** has fundamental multimodal limitations preventing image-aware introspection

**This is critical because the rule strategy is SIMBA's primary innovation and most powerful optimization mechanism.**

## How SIMBA Should Work with Images

### SIMBA's Sophisticated Optimization Process:

#### 1. **Mini-Batch Sampling & Multi-Temperature Trajectory Generation**
```python
# SIMBA samples training data and creates multiple LLM variants
for model in prepare_models_for_resampling(program, num_candidates):  # Different temperatures
    for example in mini_batch:
        chosen_program = softmax_sample(top_programs, temperature)  # Intelligent selection
        # Execute with Image inputs - should preserve visual data
        result = program(image=dspy.Image.from_url("golden_retriever.jpg"), prompt="What breed?")
        trace = capture_execution_trace()  # Contains Image objects
```

#### 2. **Trajectory Analysis & Bucketing** 
```python
# Group trajectories by output variability to identify "edge cases"
for trajectory_bucket in sorted_by_variability:
    max_score = best_trajectory["score"]  # e.g., 0.9 for "Golden Retriever"
    min_score = worst_trajectory["score"]  # e.g., 0.2 for "Some dog"
    # Higher variability = more learning opportunity
```

#### 3. **Strategy Selection: Rule vs Demo**

**Demo Strategy** (✅ WORKS): Simple few-shot addition
```python
demo = dspy.Example(image=Image(...), classification="Golden Retriever")
predictor.demos.append(demo)  # Image preserved correctly
```

**Rule Strategy** (❌ BROKEN): **The Core Innovation** - Introspective Analysis
```python
# SHOULD analyze visual differences between successful/failed trajectories
better_trajectory = [{"inputs": {"image": Image(...)}, "outputs": {"classification": "Golden Retriever"}}]
worse_trajectory = [{"inputs": {"image": Image(...)}, "outputs": {"classification": "Some dog"}}]

# OfferFeedback LLM should see actual images to generate advice like:
# "When classifying dog breeds, focus on distinctive coat patterns, ear shape, and body proportions"
# Instead of generic: "Improve your classifications"
```

#### 4. **Self-Reflective Advice Generation** (Currently Broken for Images)
The rule strategy should enable sophisticated visual analysis:

```python
# What OfferFeedback SHOULD receive for image-based introspection:
program_inputs = {"image": dspy.Image(...), "prompt": "What breed?"}
better_trajectory = [{"inputs": {"image": dspy.Image(...)}, "outputs": {...}}] 
worse_trajectory = [{"inputs": {"image": dspy.Image(...)}, "outputs": {...}}]

# Expected advice quality with visual access:
advice = {
    "dog_classifier": "When identifying Golden Retrievers, focus on the distinctive golden coat color, feathered tail, and friendly facial expression visible in the image. The successful trajectory correctly identified these breed-specific visual markers."
}
```

### The Critical Problem:
**SIMBA's most powerful feature (introspective rule generation) is completely broken for multimodal programs** because the OfferFeedback LLM cannot see the visual content it needs to analyze.

## The Problem

### Root Cause: OfferFeedback Signature Design Limitations

The `OfferFeedback` signature in `simba_utils.py` has **string-based input fields** that force serialization of all data:

```python
class OfferFeedback(dspy.Signature):
    program_inputs: str = InputField(...)          # 🚫 Forces serialization
    better_program_trajectory: str = InputField(...)  # 🚫 Forces serialization  
    worse_program_trajectory: str = InputField(...)   # 🚫 Forces serialization
```

### Serialization Chain of Destruction:

1. **SIMBA captures trajectories** with proper `dspy.Image` objects ✅
2. **`append_a_rule()` line 224-225** forces JSON serialization:
   ```python
   kwargs = {k: v if isinstance(v, str) else ujson.dumps(recursive_mask(v), indent=2)
             for k, v in kwargs.items()}
   ```
3. **`recursive_mask()` line 314** converts Images to useless placeholders:
   ```python
   else:
       return f"<non-serializable: {type(o).__name__}>"
   # dspy.Image becomes "<non-serializable: Image>"
   ```
4. **OfferFeedback receives** meaningless text instead of visual data ❌

### Impact:

This breaks SIMBA's core value proposition for multimodal programs:

- **Introspective analysis disabled**: The LLM cannot analyze visual patterns in successful vs. failed trajectories
- **Generic advice only**: 
  - ❌ Current: "Improve your classifications"
  - ✅ Expected: "Focus on the distinctive golden coat pattern and feathered tail visible in successful classifications"
- **No visual meta-cognition**: Cannot develop sophisticated understanding of visual reasoning strategies
- **Optimization regression**: SIMBA performs worse than simpler optimizers for image tasks
- **Wasted potential**: The most advanced DSPy optimizer cannot leverage its key innovation for multimodal programs

**In essence: SIMBA's "prompt-space reinforcement learning" becomes "prompt-space blind guessing" for images.**

## The Solution

### Phase 1: Fix OfferFeedback Signature (Multimodal Support)

**Replace string-based fields with multimodal-aware fields:**

```python
class OfferFeedback(dspy.Signature):
    """
    Multimodal-aware feedback signature that preserves Image objects
    for visual trajectory analysis.
    """
    program_code: str = InputField(desc="The code of the program")
    modules_defn: str = InputField(desc="Module definitions")
    
    # 🔥 KEY CHANGES: Support complex objects including Images
    program_inputs: dict = InputField(desc="Program inputs (may contain Images)")
    oracle_metadata: dict = InputField(desc="Training metadata") 
    
    better_program_trajectory: list[dict] = InputField(
        desc="Successful trajectory with preserved Image objects"
    )
    better_program_outputs: dict = InputField(desc="Successful outputs")
    better_reward_value: float = InputField(desc="Success score")
    
    worse_program_trajectory: list[dict] = InputField(
        desc="Failed trajectory with preserved Image objects"  
    )
    worse_program_outputs: dict = InputField(desc="Failed outputs")
    worse_reward_value: float = InputField(desc="Failure score")
    
    module_names: list[str] = InputField(desc="Module names for advice")
    
    # Outputs remain the same
    discussion: str = OutputField(desc="Analysis of trajectory differences")
    module_advice: dict[str, str] = OutputField(desc="Image-aware advice per module")
```

### Phase 2: Update append_a_rule Function

**Remove forced serialization and preserve Image objects:**

```python
def append_a_rule(bucket, system, **kwargs):
    # ... existing logic ...
    
    # 🔥 KEY CHANGE: Remove forced JSON serialization 
    # OLD (lines 224-225):
    # kwargs = {k: v if isinstance(v, str) else ujson.dumps(recursive_mask(v), indent=2)
    #           for k, v in kwargs.items()}
    
    # NEW: Pass objects directly to multimodal signature
    advice = dspy.Predict(OfferFeedback)(**kwargs).module_advice
    
    # ... rest remains same ...
```

### Phase 3: Update recursive_mask (Image Preservation)

**Preserve Image objects instead of converting to placeholders:**

```python
def recursive_mask(o):
    # ... existing logic for serializable objects ...
    
    # 🔥 KEY CHANGE: Preserve Image objects
    if isinstance(o, Image):
        return o  # Keep Image objects intact for multimodal LMs
    
    # ... existing logic for dicts/lists/tuples ...
    
    # Only convert truly non-serializable objects to placeholders
    else:
        return f"<non-serializable: {type(o).__name__}>"
```

### Phase 4: Backward Compatibility

**Ensure the fix doesn't break non-image use cases:**

1. **String fields still work**: Text-based trajectories serialize normally
2. **Mixed content support**: Both Images and text in same trajectory
3. **Fallback behavior**: If LM doesn't support multimodal, graceful degradation

## Implementation Steps

### Step 1: Create Failing Tests ✅ DONE
- `tests/teleprompt/test_offerfeedback_image_problem.py` - Internal mechanism validation
- `tests/teleprompt/test_simba_offerfeedback_public_api.py` - End-to-end public API validation

### Step 2: Implement OfferFeedback Signature Fix
1. Update signature field types in `simba_utils.py`
2. Test multimodal field handling
3. Verify LM receives proper Image format

### Step 3: Update append_a_rule Function  
1. Remove forced JSON serialization (lines 224-225)
2. Pass kwargs directly to OfferFeedback
3. Handle any adapter compatibility issues

### Step 4: Update recursive_mask Function
1. Add Image object detection and preservation
2. Test mixed content scenarios
3. Ensure backward compatibility

### Step 5: Integration Testing
1. Run existing SIMBA tests to ensure no regressions
2. Test with image-based programs end-to-end
3. Verify advice quality improves for visual tasks

### Step 6: Validate Fix
1. Run failing tests - should now pass
2. Compare advice quality before/after fix
3. Test with various image types and scenarios

## Expected Outcomes

### Before Fix: SIMBA's Core Innovation Disabled
- ❌ **OfferFeedback sees**: `"<non-serializable: Image>"` placeholders
- ❌ **Introspective analysis**: Impossible without visual content
- ❌ **Advice quality**: "Improve your classifications" (generic)
- ❌ **Meta-cognitive learning**: No visual reasoning development
- ❌ **SIMBA effectiveness**: Reduced to basic few-shot optimization

### After Fix: Full Introspective Optimization Restored
- ✅ **OfferFeedback sees**: Actual `dspy.Image` objects with visual content
- ✅ **Introspective analysis**: LLM can analyze visual patterns in trajectories
- ✅ **Advice quality**: "When identifying Golden Retrievers, focus on the distinctive feathered tail and golden coat visible in successful classifications, rather than generic 'dog' labels seen in failed attempts"
- ✅ **Meta-cognitive learning**: Develops sophisticated visual reasoning strategies
- ✅ **SIMBA effectiveness**: Full "prompt-space reinforcement learning" for multimodal programs

**Result: SIMBA becomes the most powerful optimizer for image-based DSPy programs, not the weakest.**

## Risk Assessment

### Low Risk:
- **Backward compatibility**: Non-image programs continue working
- **Incremental changes**: Modify existing functions, don't rewrite architecture
- **Extensive testing**: Both unit and integration tests validate behavior

### Medium Risk:
- **LM adapter compatibility**: Some adapters might need updates for multimodal fields
- **Performance impact**: Preserving Image objects uses more memory than placeholders

### Mitigation:
- **Gradual rollout**: Test with simple cases first
- **Fallback mechanisms**: Graceful degradation if multimodal fails
- **Comprehensive test coverage**: Validate all scenarios

## Success Criteria

1. **Tests pass**: All failing tests demonstrate problem is resolved
2. **Image preservation**: OfferFeedback LM receives actual Image objects  
3. **Advice quality**: Generated advice references visual content specifically
4. **No regressions**: Existing text-based SIMBA optimization still works
5. **End-to-end functionality**: Full image-based program optimization succeeds

---

*This plan addresses the fundamental multimodal limitations in SIMBA's OfferFeedback strategy, enabling effective optimization of image-based DSPy programs.*