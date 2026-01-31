#include <gtest/gtest.h>

#include "../include/components/volume_profile_panel.hpp"

// Test the ProfileMode enum values
TEST(ProfileModeEnum, ValuesAreCorrect) {
  EXPECT_EQ(static_cast<int>(BTQuant::ProfileMode::Step), 0);
  EXPECT_EQ(static_cast<int>(BTQuant::ProfileMode::Right), 1);
  EXPECT_EQ(static_cast<int>(BTQuant::ProfileMode::Left), 2);
  EXPECT_EQ(static_cast<int>(BTQuant::ProfileMode::Custom), 3);
}

// Test the ProfileSettings struct default initialization
TEST(ProfileSettingsStruct, DefaultValuesAreCorrect) {
  BTQuant::ProfileSettings settings{};

  EXPECT_DOUBLE_EQ(settings.vaPercent, 70.0);
  EXPECT_EQ(settings.tickStep, 1);
  EXPECT_TRUE(settings.showPOC);
  EXPECT_TRUE(settings.showValueArea);
  EXPECT_EQ(settings.colorScheme, 0);
}

// Test the ProfileSettings struct with custom values
TEST(ProfileSettingsStruct, CustomValuesAssignment) {
  BTQuant::ProfileSettings settings{};

  settings.vaPercent = 80.5;
  settings.tickStep = 5;
  settings.showPOC = false;
  settings.showValueArea = false;
  settings.colorScheme = 2;

  EXPECT_DOUBLE_EQ(settings.vaPercent, 80.5);
  EXPECT_EQ(settings.tickStep, 5);
  EXPECT_FALSE(settings.showPOC);
  EXPECT_FALSE(settings.showValueArea);
  EXPECT_EQ(settings.colorScheme, 2);
}

// Test that ProfileSettings can be copied
TEST(ProfileSettingsStruct, CopyConstructorWorks) {
  BTQuant::ProfileSettings original{};
  original.vaPercent = 75.0;
  original.tickStep = 2;
  original.showPOC = false;
  original.showValueArea = true;
  original.colorScheme = 1;

  BTQuant::ProfileSettings copy = original;

  EXPECT_DOUBLE_EQ(copy.vaPercent, 75.0);
  EXPECT_EQ(copy.tickStep, 2);
  EXPECT_FALSE(copy.showPOC);
  EXPECT_TRUE(copy.showValueArea);
  EXPECT_EQ(copy.colorScheme, 1);
}

// Test that ProfileSettings can be assigned
TEST(ProfileSettingsStruct, AssignmentOperatorWorks) {
  BTQuant::ProfileSettings source{};
  source.vaPercent = 85.0;
  source.tickStep = 3;
  source.showPOC = true;
  source.showValueArea = false;
  source.colorScheme = 3;

  BTQuant::ProfileSettings target{};
  target = source;

  EXPECT_DOUBLE_EQ(target.vaPercent, 85.0);
  EXPECT_EQ(target.tickStep, 3);
  EXPECT_TRUE(target.showPOC);
  EXPECT_FALSE(target.showValueArea);
  EXPECT_EQ(target.colorScheme, 3);
}

// Test integration with VolumeProfilePanel class
TEST(VolumeProfilePanelIntegration, ProfileSettingsUsedInClass) {
  // This test verifies that the VolumeProfilePanel class properly initializes ProfileSettings
  // Since we can't instantiate the full class without dependencies, we test the struct separately
  BTQuant::ProfileSettings settings;

  // Verify defaults are as expected
  EXPECT_DOUBLE_EQ(settings.vaPercent, 70.0);
  EXPECT_EQ(settings.tickStep, 1);
  EXPECT_TRUE(settings.showPOC);
  EXPECT_TRUE(settings.showValueArea);
  EXPECT_EQ(settings.colorScheme, 0);
}