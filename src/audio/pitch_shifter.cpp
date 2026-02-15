#include "pitch_shifter.h"
#include <algorithm>
#include <stdexcept>

PitchShifter::PitchShifter(double base_pitch, double min_volume, double max_volume)
    : base_pitch_(base_pitch), min_volume_(min_volume), max_volume_(max_volume) {
    if (min_volume_ >= max_volume_) {
        throw std::invalid_argument("Min volume must be less than max volume");
    }
    if (base_pitch <= 0) {
        throw std::invalid_argument("Base pitch must be positive");
    }
}

double PitchShifter::calculate_pitch_multiplier(double volume) const {
    // Clamp volume to the defined range
    volume = std::max(min_volume_, std::min(max_volume_, volume));

    // Use logarithmic scaling to handle the wide range of trade volumes more naturally
    double log_min_volume = std::log10(min_volume_);
    double log_max_volume = std::log10(max_volume_);
    double log_volume = std::log10(std::max(volume, min_volume_));

    // Normalize log volume to [0, 1] range
    double normalized_log_volume = (log_volume - log_min_volume) / (log_max_volume - log_min_volume);

    // Apply inverse relationship: higher volume -> lower pitch (deeper bass)
    // Use a power function to make the effect more pronounced
    double pitch_factor = 1.0 - normalized_log_volume; // Inverse relationship

    // Apply a curve to make the effect more noticeable in the middle range
    // Using a more aggressive curve to emphasize the deep bass effect for large trades
    pitch_factor = std::pow(pitch_factor, 2.0);

    // Ensure the factor is within reasonable bounds (0.05 to 1.0 for more dramatic bass effect)
    pitch_factor = std::max(0.05, pitch_factor);

    return pitch_factor;
}

double PitchShifter::calculate_pitch(double volume) const {
    double multiplier = calculate_pitch_multiplier(volume);
    return base_pitch_ * multiplier;
}

std::vector<double> PitchShifter::shift_pitch(const std::vector<double>& input_samples, double volume) const {
    std::vector<double> output_samples;
    output_samples.reserve(input_samples.size());

    double pitch_multiplier = calculate_pitch_multiplier(volume);
    
    // Calculate how "big" the trade is to determine bass enhancement level
    double normalized_volume = (volume - min_volume_) / (max_volume_ - min_volume_);
    double bass_enhancement_factor = normalized_volume * 0.3; // Up to 30% bass enhancement for largest trades

    // Enhanced pitch shifting with more sophisticated bass effect for large trades
    for (size_t i = 0; i < input_samples.size(); ++i) {
        // Apply pitch shift effect by scaling the sample
        double scaled_sample = input_samples[i] * pitch_multiplier;

        // Add enhanced harmonic content based on the volume for deep bass effect
        if (i > 0 && i < input_samples.size() - 1) {
            // Add more pronounced harmonic distortion for "bass" effect when volume is high
            double harmonic_factor = (1.0 - pitch_multiplier) * 0.2; // More harmonics for lower pitches
            
            // Add sub-harmonic content for deep bass effect on large trades
            double sub_harmonic = 0.0;
            if (normalized_volume > 0.7) { // Only for large trades
                // Create sub-harmonic at half the frequency for deep bass
                sub_harmonic = 0.15 * bass_enhancement_factor * (
                    0.4 * input_samples[i-1] +
                    0.4 * input_samples[i] +
                    0.2 * input_samples[i+1]
                );
            }
            
            scaled_sample += harmonic_factor * (
                0.5 * input_samples[i-1] +
                0.3 * input_samples[i] +
                0.2 * input_samples[i+1]
            ) + sub_harmonic;
        }

        // Apply a simple low-pass filter effect for large trades to emphasize bass
        if (normalized_volume > 0.5) { // For trades above 50% of max volume
            if (i > 0) {
                // Smooth the transition to emphasize lower frequencies
                scaled_sample = scaled_sample * (1.0 - bass_enhancement_factor * 0.5) + 
                               output_samples.back() * (bass_enhancement_factor * 0.5);
            }
        }

        output_samples.push_back(scaled_sample);
    }

    return output_samples;
}

void PitchShifter::set_base_pitch(double base_pitch) {
    if (base_pitch <= 0) {
        throw std::invalid_argument("Base pitch must be positive");
    }
    base_pitch_ = base_pitch;
}

void PitchShifter::set_volume_range(double min_volume, double max_volume) {
    if (min_volume >= max_volume) {
        throw std::invalid_argument("Min volume must be less than max volume");
    }
    min_volume_ = min_volume;
    max_volume_ = max_volume;
}