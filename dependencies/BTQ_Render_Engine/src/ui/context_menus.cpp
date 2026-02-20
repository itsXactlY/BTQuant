#include "ui/context_menus.hpp"
namespace BTQuant {
ContextMenuManager::ContextMenuManager(PanelManager* manager) : panel_manager_(manager) {}
void ContextMenuManager::show_context_menu(PanelBase*) {}
void ContextMenuManager::render_generic_context_menu(PanelBase*, const char*) {}
}  // namespace BTQuant
