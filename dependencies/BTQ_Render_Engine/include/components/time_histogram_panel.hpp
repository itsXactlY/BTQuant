#pragma once

#include "../data/VolumeDataTypes.h"  // For VolumeAnalysisType and VolumeDataType enums
#include "panel_base.hpp"

namespace BTQuant {

class TimeHistogramPanel : public PanelBase {
 public:
  TimeHistogramPanel(const PanelConfig& config);

  void initialize() override;
  void render() override;

  // Volume data type selection for time histogram visualization
  void setVolumeDataType(Data::VolumeDataType vol_type) { volume_data_type_ = vol_type; }
  Data::VolumeDataType getVolumeDataType() const { return volume_data_type_; }

 private:
  // Volume Data Type for Time Histogram Visualization
  Data::VolumeDataType volume_data_type_ = Data::VolumeDataType::BuySellVolume;
};

}  // namespace BTQuant