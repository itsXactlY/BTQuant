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
    
    // Normalize volume to [0, 1] range
    double normalized_volume = (volume - min_volume_) / (max_volume_ - min_volume_);
    
    // Apply inverse relationship: higher volume -> lower pitch (deeper bass)
    // Use a power function to make the effect more pronounced
    double pitch_factor = 1.0 - normalized_volume; // Inverse relationship
    
    // Apply a curve to make the effect more noticeable in the middle range
    pitch_factor = std::pow(pitch_factor, 1.5);
    
    // Ensure the factor is within reasonable bounds (0.1 to 1.0)
    pitch_factor = std::max(0.1, pitch_factor);
    
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
    
    // Simple pitch shifting by scaling amplitude based on pitch multiplier
    // In a real implementation, this would use more sophisticated algorithms like PSOLA or phase vocoder
    for (size_t i = 0; i < input_samples.size(); ++i) {
        // Apply pitch shift effect by scaling the sample
        double scaled_sample = input_samples[i] * pitch_multiplier;
        
        // Add some harmonic content based on the volume
        if (i > 0 && i < input_samples.size() - 1) {
            // Add a slight harmonic distortion for "bass" effect when volume is high
            double harmonic_factor = (1.0 - pitch_multiplier) * 0.1; // More harmonics for lower pitches
            scaled_sample += harmonic_factor * (
                0.5 * input_samples[i-1] + 
                0.3 * input_samples[i] + 
                0.2 * input_samples[i+1]
            );
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