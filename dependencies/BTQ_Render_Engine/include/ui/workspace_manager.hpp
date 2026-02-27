#pragma once

#include <string>

namespace BTQuant {

class QuantWorkspaceComponent;

class WorkspaceManager {
public:
    explicit WorkspaceManager(QuantWorkspaceComponent* workspace);

    /**
     * @brief Export the entire workspace (layouts, settings, symbols) to a file
     * @param filepath Path to the file where the workspace will be saved
     * @return True if export was successful, false otherwise
     */
    bool export_workspace(const std::string& filepath);

    /**
     * @brief Import a workspace from a file
     * @param filepath Path to the file containing the workspace data
     * @return True if import was successful, false otherwise
     */
    bool import_workspace(const std::string& filepath);

private:
    QuantWorkspaceComponent* workspace_;
};

}  // namespace BTQuant