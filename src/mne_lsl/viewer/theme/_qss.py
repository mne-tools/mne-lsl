"""Thin, palette-role-based style sheets and the Qt-ADS icon slots.

Every color goes through the QSS ``palette(<role>)`` function instead of a hardcoded
hex value, so one sheet serves both modes and survives a theme flip.
"""

from __future__ import annotations

# Thin, targeted control skin over Fusion, for the few things a palette cannot express:
# radius, padding, an integrated combo arrow, flat tabs and header styling. Metrics: 4
# px radius, 1 px borders, padding on a small grid. Checkbox / radio indicators are left
# to Fusion, as styling them through QSS needs bundled images and would blank the native
# check.
_QSS = """
/* --- Toolbars --------------------------------------------------------- */
QToolBar { border: none; background: transparent; padding: 3px; spacing: 3px; }
QToolBar::separator {
    width: 1px; background: palette(mid); margin: 5px 6px;
}

/* --- Tool buttons: flat; hover/pressed/checked fills from palette ------ */
QToolButton {
    border: none; border-radius: 4px; padding: 4px;
    background: transparent; color: palette(button-text);
}
QToolButton:hover { background-color: palette(alternate-base); }
QToolButton:pressed { background-color: palette(mid); }
QToolButton:checked {
    background-color: palette(highlight); color: palette(highlighted-text);
}
QToolButton:disabled { color: palette(placeholder-text); }
QToolButton::menu-indicator { image: none; }

/* --- Push buttons: raised face; accent-filled default ----------------- */
QPushButton {
    border: 1px solid palette(mid); border-radius: 4px; padding: 5px 14px;
    background-color: palette(button); color: palette(button-text);
}
QPushButton:hover { background-color: palette(midlight); }
QPushButton:pressed { background-color: palette(mid); }
QPushButton:checked {
    background-color: palette(highlight); color: palette(highlighted-text);
    border-color: palette(highlight);
}
QPushButton:default {
    background-color: palette(highlight); color: palette(highlighted-text);
    border-color: palette(highlight);
}
QPushButton:default:hover {
    background-color: palette(link); border-color: palette(link);
}
QPushButton:disabled {
    color: palette(placeholder-text); background-color: palette(window);
    border-color: palette(alternate-base);
}

/* --- Text fields, spin boxes, combo box: filled, subtle border -------- */
QLineEdit, QAbstractSpinBox, QComboBox {
    border: 1px solid palette(mid); border-radius: 4px; padding: 4px 8px;
    background-color: palette(base); color: palette(text);
    selection-background-color: palette(highlight);
    selection-color: palette(highlighted-text);
}
QLineEdit:focus, QAbstractSpinBox:focus, QComboBox:focus, QComboBox:on {
    border-color: palette(highlight);
}
QLineEdit:disabled, QAbstractSpinBox:disabled, QComboBox:disabled {
    color: palette(placeholder-text); background-color: palette(window);
}

/* --- Combo box: integrated drop-down (no dated separated button) ------ */
QComboBox::drop-down {
    subcontrol-origin: padding; subcontrol-position: center right;
    width: 20px; border: none; background: transparent;
}
QComboBox QAbstractItemView {
    border: 1px solid palette(mid); border-radius: 4px;
    background-color: palette(base); padding: 3px; outline: none;
    selection-background-color: palette(highlight);
    selection-color: palette(highlighted-text);
}
QComboBox QAbstractItemView::item {
    border-radius: 3px; padding: 4px 8px; min-height: 20px;
}

/* --- Tabs: flat, muted inactive, accented selected (any orientation) -- */
QTabWidget::pane {
    border: 1px solid palette(mid); border-radius: 4px; top: -1px;
}
QTabBar { qproperty-drawBase: 0; background: transparent; }
QTabBar::tab {
    background: transparent; border: none; border-radius: 4px;
    padding: 7px 16px; margin: 1px; color: palette(mid);
}
QTabBar::tab:hover { color: palette(text); background: palette(alternate-base); }
QTabBar::tab:selected { color: palette(highlight); background: palette(base); }

/* --- Item views + tables: clean grid, styled header ------------------- */
QListView, QTreeView, QTableView {
    border: 1px solid palette(mid); border-radius: 4px;
    background-color: palette(base);
    alternate-background-color: palette(alternate-base);
    gridline-color: palette(alternate-base);
    selection-background-color: palette(highlight);
    selection-color: palette(highlighted-text);
}
QListView::item, QTreeView::item, QTableView::item { padding: 3px 6px; }
QHeaderView { background-color: transparent; }
QHeaderView::section {
    background-color: palette(alternate-base); color: palette(text);
    padding: 5px 8px; border: none; border-bottom: 1px solid palette(mid);
}
QHeaderView::section:!last { border-right: 1px solid palette(base); }
QTableCornerButton::section {
    background-color: palette(alternate-base); border: none;
    border-bottom: 1px solid palette(mid);
}

/* --- Scrollbars: thin, rounded handle, no arrows ---------------------- */
QScrollBar:vertical { background: transparent; width: 12px; margin: 0; }
QScrollBar:horizontal { background: transparent; height: 12px; margin: 0; }
QScrollBar::handle { background: palette(mid); border-radius: 4px; }
QScrollBar::handle:vertical { min-height: 28px; margin: 2px; }
QScrollBar::handle:horizontal { min-width: 28px; margin: 2px; }
QScrollBar::handle:hover { background: palette(text); }
QScrollBar::add-line, QScrollBar::sub-line { width: 0; height: 0; }
QScrollBar::add-page, QScrollBar::sub-page { background: transparent; }

/* --- Group boxes + tooltips: soft, no heavy bevel --------------------- */
QCheckBox, QRadioButton { spacing: 6px; }
QGroupBox {
    border: 1px solid palette(alternate-base); border-radius: 6px;
    margin-top: 8px; padding: 10px 8px 6px 8px;
}
QGroupBox::title {
    subcontrol-origin: margin; subcontrol-position: top left;
    left: 8px; padding: 0 4px; color: palette(mid);
}
QToolTip {
    color: palette(text); background-color: palette(base);
    border: 1px solid palette(mid); padding: 3px 6px;
}
"""

# Thin, palette-role-based skin for the Qt-ADS chrome (tabs, title bars, splitters,
# container). It replaces Qt-ADS's own bundled style sheet so the dock area reads like
# the rest of the application; it is re-set on every theme change.
_ADS_QSS = """
ads--CDockContainerWidget, ads--CDockAreaWidget { background: palette(window); }
ads--CDockWidget { background: palette(window); border: none; }

/* Splitter handles between split panes (the clean 50/50 divide). */
ads--CDockContainerWidget ads--CDockSplitter::handle { background: palette(window); }
ads--CDockContainerWidget ads--CDockSplitter::handle:hover {
    background: palette(highlight);
}

/* Dock-area title bar: no bottom border of its own. Qt-ADS already draws one
   clean, full-width divider between the title-bar strip and the content (it
   tracks the palette), whereas a border here is clipped by the tab widgets and
   only shows in the empty space right of the tabs — the stray partial line that
   read as broken. The focus accent lives on the active tab below instead. */
ads--CDockAreaTitleBar { background: palette(window); }

/* Tabs: flat, muted inactive, accented active (mirrors the app QTabBar). Every
   tab reserves a transparent 2px underline so activating one never shifts the
   strip; the active tab fills with base + accent label, and the focused area
   paints that underline in the highlight role — a full-tab-width accent bounded
   to the tab, never a partial line hanging off a tab edge. */
ads--CDockWidgetTab {
    background: transparent; border: none; padding: 4px 10px;
    border-bottom: 2px solid transparent;
}
ads--CDockWidgetTab QLabel { color: palette(mid); }
ads--CDockWidgetTab[activeTab="true"] {
    background: palette(base);
    border-top-left-radius: 4px; border-top-right-radius: 4px;
}
ads--CDockWidgetTab[activeTab="true"] QLabel { color: palette(highlight); }
ads--CDockAreaWidget[focused="true"] ads--CDockWidgetTab[activeTab="true"] {
    border-bottom: 2px solid palette(highlight);
}

/* Per-tab close button (Qt-ADS names it #tabCloseButton): reset the app-wide
   raised QPushButton chrome — border + 5px/14px padding wraps the 16px glyph in
   a giant bordered box — to a small, borderless IDE-style "×" that sits neatly at
   the tab's right edge, with a subtle palette-role hover. */
ads--CDockWidgetTab #tabCloseButton {
    background: transparent; border: none; border-radius: 4px;
    padding: 1px; margin-left: 4px; qproperty-iconSize: 14px 14px;
}
ads--CDockWidgetTab #tabCloseButton:hover { background: palette(alternate-base); }
ads--CDockWidgetTab #tabCloseButton:pressed { background: palette(mid); }

/* Title-bar buttons (close / tabs menu / undock). */
ads--CTitleBarButton { background: transparent; border: none; border-radius: 4px; }
ads--CTitleBarButton:hover { background: palette(alternate-base); }
ads--CTitleBarButton:pressed { background: palette(mid); }
"""

# Qt-ADS title-bar glyphs, keyed by the name of the 'ads.eIcon' slot they replace, so
# this module stays free of the docking dependency (the consumer resolves the slot).
_ADS_ICONS: dict[str, str] = {
    "TabCloseIcon": "mdi6.close",
    "DockAreaCloseIcon": "mdi6.close",
    "DockAreaMenuIcon": "mdi6.chevron-down",
    "DockAreaUndockIcon": "mdi6.dock-window",
}
