# Auto-Quality Reduction System Enhancement

## Overview
Enhanced the auto-quality reduction system in `dependencies/BTQ_Render_Engine/src/rendering/auto_quality.cpp` to better detect performance drops and automatically reduce visual quality to maintain responsiveness.

## Key Enhancements

### 1. Critical Performance Drop Detection
Added `checkCriticalPerformanceDrop()` method that:
- Detects sudden performance drops (frames 3x worse than target)
- Identifies sustained poor performance (60%+ of recent frames poor)
- Performs immediate quality reduction regardless of cooldown periods
- Maintains system responsiveness during critical performance issues

### 2. Enhanced Responsiveness Monitoring
Added `calculateResponsivenessFactor()` method that:
- Calculates a responsiveness score considering recent vs overall performance
- Accounts for frame time variance (jitter) affecting perceived responsiveness
- Includes worst frame time analysis in recent history
- Provides a metric for how responsive the system currently feels

### 3. Improved Quality Adjustment Logic
Enhanced `determineQualityReduction()` method to:
- Incorporate responsiveness factor in quality reduction decisions
- Apply more aggressive quality reduction when responsiveness is poor
- Consider multiple performance metrics simultaneously
- Maintain better balance between quality and performance

### 4. Load-Based Responsiveness Assurance
Added `ensureResponsivenessUnderLoad()` method that:
- Monitors system responsiveness even under heavy load
- Forces immediate quality reduction when responsiveness falls below critical thresholds
- Implements graduated response based on severity (1-3 levels reduction)
- Prevents system lockups during performance-critical moments

### 5. Integration Points
Modified `recordFrameTime()` to:
- Call `checkCriticalPerformanceDrop()` for immediate response to performance issues
- Call `ensureResponsivenessUnderLoad()` to maintain responsiveness under load
- Maintain all existing functionality while adding new features

## Technical Details

### Quality Levels
The system maintains 5 quality levels (0=highest, 4=lowest) with progressive reduction in:
- Rendering resolution scale (1.0 → 0.6)
- Visual effects (shadows, reflections, post-processing)
- Detail levels (textures, lighting, particles)
- Culling and LOD settings
- Anti-aliasing and other quality features

### Performance Metrics
The system monitors:
- Frame time averages and variances
- Performance consistency and stability
- Jank detection and percentage
- GPU/CPU utilization
- Memory and thermal pressure
- Frame pacing irregularity
- Responsiveness factor

### Thresholds and Cooldowns
- Performance threshold: 80% by default
- Adjustment cooldown: 5 seconds by default
- Stability window: 10 seconds before increasing quality
- Adaptive thresholds based on historical performance

## Benefits

1. **Better Responsiveness**: System maintains interactive feel even under load
2. **Automatic Adaptation**: Quality adjusts seamlessly based on performance
3. **Graduated Response**: Appropriate response level based on issue severity
4. **Comprehensive Monitoring**: Multiple metrics considered for accurate assessment
5. **Safety Mechanisms**: Prevents system lockups during performance drops

## Testing
The enhanced system was tested with:
- Stable good performance scenarios
- Gradual performance degradation
- Sudden performance drops
- Recovery phases
- Quality setting validation across all levels

All tests confirmed proper functionality of the new features.