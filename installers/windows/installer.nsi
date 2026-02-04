; PubBTQuant NSIS Installer Script
; This script creates a Windows installer for the PubBTQuant trading terminal

!define APPNAME "PubBTQuant Trading Terminal"
!define COMPANYNAME "PubBTQuant Development Team"
!define DESCRIPTION "Advanced trading terminal with real-time analytics"
!define VERSIONMAJOR 1
!define VERSIONMINOR 0
!define VERSIONPATCH 0
!define INSTALLSIZE 500000 ; Estimated installation size in KB

; Include Modern UI
!include "MUI2.nsh"

; General
Name "${APPNAME}"
OutFile "PubBTQuant-Setup.exe"
InstallDir $PROGRAMFILES64\PubBTQuant
InstallDirRegKey HKCU "Software\${APPNAME}" ""
RequestExecutionLevel admin ; Request administrator privileges for installation

; Interface Settings
!define MUI_ABORTWARNING
!define MUI_ICON "${NSISDIR}\Contrib\Graphics\Icons\modern-install.ico"
!define MUI_UNICON "${NSISDIR}\Contrib\Graphics\Icons\modern-uninstall.ico"

; Pages
!insertmacro MUI_PAGE_WELCOME
!insertmacro MUI_PAGE_LICENSE "license.txt" ; You'll need to create this file
!insertmacro MUI_PAGE_COMPONENTS
!insertmacro MUI_PAGE_DIRECTORY
!insertmacro MUI_PAGE_INSTFILES
!insertmacro MUI_PAGE_FINISH

!insertmacro MUI_UNPAGE_CONFIRM
!insertmacro MUI_UNPAGE_INSTFILES

; Languages
!insertmacro MUI_LANGUAGE "English"

; Registry key to check for directory (so if you install again, it will overwrite the old one automatically)
InstallDirRegKey HKCU "Software\${APPNAME}" ""

; Sections
Section "PubBTQuant Core" SecMain
  SectionIn RO
  
  ; Check for Vulkan support
  Push $0
  Call CheckVulkanSupport
  Pop $0
  StrCmp $0 "OK" vulkan_ok vulkan_missing
  
  vulkan_missing:
    MessageBox MB_OK|MB_ICONEXCLAMATION "Vulkan is not available on this system. Please install the latest graphics drivers."
    Abort "Installation cancelled due to missing Vulkan support."
  
  vulkan_ok:
  
  SetOutPath $INSTDIR
  File /r "bin\*.*"
  File /r "shaders\*.*"
  File "default_layout.json"
  
  ; Store installation folder
  WriteRegStr HKCU "Software\${APPNAME}" "" $INSTDIR
  
  ; Create uninstaller
  WriteUninstaller "$INSTDIR\Uninstall.exe"
  WriteRegStr HKCU "Software\Microsoft\Windows\CurrentVersion\Uninstall\${APPNAME}" "DisplayName" "${APPNAME}"
  WriteRegStr HKCU "Software\Microsoft\Windows\CurrentVersion\Uninstall\${APPNAME}" "UninstallString" "$INSTDIR\Uninstall.exe"
  WriteRegStr HKCU "Software\Microsoft\Windows\CurrentVersion\Uninstall\${APPNAME}" "DisplayIcon" "$INSTDIR\realtime_dashboard.exe"
  WriteRegStr HKCU "Software\Microsoft\Windows\CurrentVersion\Uninstall\${APPNAME}" "Publisher" "${COMPANYNAME}"
  WriteRegStr HKCU "Software\Microsoft\Windows\CurrentVersion\Uninstall\${APPNAME}" "DisplayVersion" "${VERSIONMAJOR}.${VERSIONMINOR}.${VERSIONPATCH}"
  WriteRegDWORD HKCU "Software\Microsoft\Windows\CurrentVersion\Uninstall\${APPNAME}" "VersionMajor" ${VERSIONMAJOR}
  WriteRegDWORD HKCU "Software\Microsoft\Windows\CurrentVersion\Uninstall\${APPNAME}" "VersionMinor" ${VERSIONMINOR}
  WriteRegStr HKCU "Software\Microsoft\Windows\CurrentVersion\Uninstall\${APPNAME}" "URLInfoAbout" "https://pubbtquant.example.com"
  
SectionEnd

Section "Start Menu Shortcuts" SecShortcuts
  CreateDirectory "$SMPROGRAMS\${APPNAME}"
  CreateShortCut "$SMPROGRAMS\${APPNAME}\Uninstall.lnk" "$INSTDIR\Uninstall.exe" "" "$INSTDIR\Uninstall.exe" 0
  CreateShortCut "$SMPROGRAMS\${APPNAME}\PubBTQuant.lnk" "$INSTDIR\realtime_dashboard.exe" "" "$INSTDIR\realtime_dashboard.exe" 0
  CreateShortCut "$SMPROGRAMS\${APPNAME}\Dashboard Advanced.lnk" "$INSTDIR\dashboard_advanced.exe" "" "$INSTDIR\dashboard_advanced.exe" 0
SectionEnd

Section "Desktop Shortcut" SecDesktop
  CreateShortCut "$DESKTOP\PubBTQuant.lnk" "$INSTDIR\realtime_dashboard.exe" "" "$INSTDIR\realtime_dashboard.exe" 0
SectionEnd

; Dependencies section
Section "Dependencies" SecDeps
  ; Check if Visual C++ Redistributables are installed
  Push $0
  Call CheckVCRedist
  Pop $0
  StrCmp $0 "OK" vc_ok vc_missing
  
  vc_missing:
    MessageBox MB_YESNO|MB_ICONQUESTION "Visual C++ Redistributables are not installed. Download and install them now?" IDYES install_vc IDNO skip_vc
  
  install_vc:
    DetailPrint "Downloading Visual C++ Redistributables..."
    NSISdl::download "https://aka.ms/vs/16/release/vc_redist.x64.exe" "$TEMP\vc_redist.x64.exe"
    ExecWait '"$TEMP\vc_redist.x64.exe" /quiet /norestart'
    Delete "$TEMP\vc_redist.x64.exe"
  
  vc_ok:
  skip_vc:
  
  ; Check if Vulkan SDK is installed
  Push $0
  Call CheckVulkanSDK
  Pop $0
  StrCmp $0 "OK" vulkan_sdk_ok install_vulkan_sdk
  
  install_vulkan_sdk:
    MessageBox MB_YESNO|MB_ICONQUESTION "Vulkan SDK is not installed. Download and install it now?" IDYES download_vulkan IDNO skip_vulkan
  
  download_vulkan:
    DetailPrint "Opening Vulkan SDK download page..."
    ExecShell "open" "https://vulkan.lunarg.com/sdk/home#windows"
    MessageBox MB_OK "Please download and install the Vulkan SDK, then restart the installer."
    Abort "Installation paused for Vulkan SDK installation."
  
  vulkan_sdk_ok:
  skip_vulkan:
SectionEnd

; Uninstaller
Section "Uninstall"
  Delete "$INSTDIR\Uninstall.exe"
  Delete "$INSTDIR\realtime_dashboard.exe"
  Delete "$INSTDIR\dashboard_advanced.exe"
  RMDir /r "$INSTDIR\shaders"
  Delete "$INSTDIR\default_layout.json"
  RMDir /r "$INSTDIR\*.*"
  RMDir "$INSTDIR"
  
  Delete "$SMPROGRAMS\${APPNAME}\*.*"
  RMDir "$SMPROGRAMS\${APPNAME}"
  Delete "$DESKTOP\PubBTQuant.lnk"
  
  DeleteRegKey HKCU "Software\Microsoft\Windows\CurrentVersion\Uninstall\${APPNAME}"
  DeleteRegKey HKCU "Software\${APPNAME}"
SectionEnd

; Functions
Function .onInit
  ; Check Windows version (require Windows 10 or later for Vulkan)
  ReadRegStr $0 HKLM "SOFTWARE\Microsoft\Windows NT\CurrentVersion" "ProductName"
  ReadRegStr $1 HKLM "SOFTWARE\Microsoft\Windows NT\CurrentVersion" "CurrentMajorVersionNumber"
  ReadRegStr $2 HKLM "SOFTWARE\Microsoft\Windows NT\CurrentVersion" "CurrentMinorVersionNumber"
  
  ; Convert to numbers
  IntOp $1 $1 * 10
  IntOp $1 $1 + $2
  
  ; Check if Windows 10 (10.0) or later
  IntCmp $1 100 0 win_ok win_old
  
  win_old:
    MessageBox MB_OK|MB_ICONSTOP "This application requires Windows 10 or later."
    Abort "Unsupported Windows version"
  
  win_ok:
FunctionEnd

Function CheckVulkanSupport
  ; Simple check for Vulkan support
  ; In a real implementation, this would check for vulkan-1.dll and validate GPU support
  Push "OK"
FunctionEnd

Function CheckVCRedist
  ; Check if Visual C++ Redistributables are installed
  ReadRegStr $0 HKLM "SOFTWARE\Microsoft\VisualStudio\14.0\VC\Runtimes\x64" "Installed"
  StrCmp $0 "1" vc_installed vc_not_installed
  
  vc_installed:
    Push "OK"
    Return
  
  vc_not_installed:
    Push "MISSING"
FunctionEnd

Function CheckVulkanSDK
  ; Check if Vulkan SDK is installed
  ReadRegStr $0 HKLM "SOFTWARE\Khronos\Vulkan\Drivers" ""
  StrCmp $0 "" sdk_not_installed sdk_installed
  
  sdk_installed:
    Push "OK"
    Return
  
  sdk_not_installed:
    Push "MISSING"
FunctionEnd

; Default section selections
SectionGroup /e "Installation Options" SecOptions
  SectionIn "SecMain" "SecDeps" "SecShortcuts"
SectionGroupEnd

; Set section properties
!insertmacro MUI_FUNCTION_DESCRIPTION_BEGIN
!macro MUI_DESCRIPTION_TEXT ${SecMain} "Main application files"
!macro MUI_DESCRIPTION_TEXT ${SecDeps} "Required dependencies (Vulkan, Visual C++ Redistributables)"
!macro MUI_DESCRIPTION_TEXT ${SecShortcuts} "Start menu shortcuts"
!macro MUI_DESCRIPTION_TEXT ${SecDesktop} "Desktop shortcut"
!insertmacro MUI_FUNCTION_DESCRIPTION_END