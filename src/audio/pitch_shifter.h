#ifndef PITCH_SHIFTER_H
#define PITCH_SHIFTER_H

#include <vector>
#include <cmath>

/**
 * @brief Implements pitch shifting functionality that scales pitch inversely to volume
 * Big trade = Deep bass effect
 */
class PitchShifter {
public:
    /**
     * Constructor
     * @param base_pitch Base pitch in Hz (default 440Hz for A4 note)
     * @param min_volume Minimum volume threshold for pitch scaling
     * @param max_volume Maximum volume threshold for pitch scaling
     * @param scalar Scalar multiplier for pitch calculation (controls depth of bass effect)
     */
    PitchShifter(double base_pitch = 440.0, double min_volume = 1.0, double max_volume = 10000.0, double scalar = 0.1);

    /**
     * Calculate the shifted pitch based on volume
     * @param volume Trading volume (higher volume = lower pitch)
     * @return Adjusted pitch in Hz
     */
    double calculate_pitch(double volume) const;

    /**
     * Apply pitch shift to audio samples based on volume
     * @param input_samples Input audio samples
     * @param volume Trading volume to determine pitch shift
     * @return Pitch-shifted audio samples
     */
    std::vector<double> shift_pitch(const std::vector<double>& input_samples, double volume) const;

    /**
     * Set the base pitch
     * @param base_pitch Base pitch in Hz
     */
    void set_base_pitch(double base_pitch);

    /**
     * Set the volume range for pitch scaling
     * @param min_volume Minimum volume threshold
     * @param max_volume Maximum volume threshold
     */
    void set_volume_range(double min_volume, double max_volume);

    /**
     * Set the scalar multiplier for pitch calculation
     * @param scalar Scalar value that controls the depth of the bass effect
     */
    void set_scalar(double scalar);

private:
    double base_pitch_;
    double min_volume_;
    double max_volume_;
    double scalar_;

    /**
     * Map volume to pitch multiplier using inverse relationship
     * Higher volume -> Lower pitch (deeper bass)
     */
    double calculate_pitch_multiplier(double volume) const;
};

#endif // PITCH_SHIFTER_H