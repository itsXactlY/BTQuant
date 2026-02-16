#pragma once

#include "../data/VolumeDataTypes.h"  // For VolumeAnalysisType and VolumeDataType enums
#include "panel_base.hpp"

namespace BTQuant {

class TimeHistogramPanel : public PanelBase {
 public:
  TimeHistogramPanel(const PanelConfig& config);

  void initialize() override;
  void render_content() override;

  // Volume data type selection for time histogram visualization
  void setVolumeDataType(Data::VolumeDataType vol_type) { volume_data_type_ = vol_type; }
  Data::VolumeDataType getVolumeDataType() const { return volume_data_type_; }

 private:
  // Volume Data Type for Time Histogram Visualization
  Data::VolumeDataType volume_data_type_ = Data::VolumeDataType::BuySellVolume;

  // Auto-scaling and scale locking options
  bool auto_scale_y_axis_ = true;
  bool lock_y_axis_scale_ = false;
  double locked_min_y_ = 0.0;
  double locked_max_y_ = 100.0;
};

}  // namespace BTQuant