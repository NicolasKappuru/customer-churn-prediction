"""Customer data panel — three switchable tabs (Q / W / E).

Layout: 2 columns, compact rows.  Content taller than the panel is scrollable
(UP / DOWN arrows; caller manages scroll_y and passes it in).
"""

from __future__ import annotations
import pygame
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from customer_data import CustomerDataModel

# ── palette ───────────────────────────────────────────────────────────────────
_BG        = ( 26,  26,  34)
_BORDER    = ( 52,  52,  68)
_TAB_ON    = ( 46,  46,  60)
_TAB_OFF   = ( 28,  28,  38)
_LINE      = ( 44,  44,  58)
_LABEL     = (175, 175, 200)
_SCROLL_BG = ( 36,  36,  48)
_SCROLL_TH = ( 80,  80, 105)   # scrollbar thumb

ROW_H      = 48    # px per feature row: ~18 label + ~24 value + 6 gap
TAB_H      = 30
SCROLL_W   = 8

# ── font cache ────────────────────────────────────────────────────────────────
_F: dict = {}


def _font(size: int, bold: bool = False) -> pygame.font.Font:
    k = (size, bold)
    if k not in _F:
        _F[k] = pygame.font.SysFont("Courier New", size, bold=bold)
    return _F[k]


# ── public ────────────────────────────────────────────────────────────────────

def draw_customer_panel(
    surface: pygame.Surface,
    rect: pygame.Rect,
    model: "CustomerDataModel",
    active_tab: int,
    scroll_y: int = 0,
) -> int:
    """Draw panel; returns max_scroll (0 when content fits)."""
    pygame.draw.rect(surface, _BG, rect)
    pygame.draw.rect(surface, _BORDER, rect, 1)

    _draw_tabs(surface, rect, active_tab)

    content_rect = pygame.Rect(
        rect.x, rect.y + TAB_H,
        rect.width - SCROLL_W, rect.height - TAB_H,
    )

    if   active_tab == 0:
        rows = model.member_rows()
    elif active_tab == 1:
        rows = model.transaction_rows()
    else:
        rows = model.log_rows()

    max_scroll = _draw_rows(surface, content_rect, rows, model, scroll_y)

    _draw_scrollbar(surface,
                    pygame.Rect(rect.right - SCROLL_W, rect.y + TAB_H,
                                SCROLL_W, rect.height - TAB_H),
                    scroll_y, max_scroll)
    return max_scroll


# ── tabs ──────────────────────────────────────────────────────────────────────

def _draw_tabs(surface, rect, active):
    labels = ["[Q] PROFILE", "[W] TRANSACTIONS", "[E] LOGS"]
    tw     = rect.width // 3
    for i, lbl in enumerate(labels):
        tab = pygame.Rect(rect.x + i * tw, rect.y, tw, TAB_H)
        pygame.draw.rect(surface, _TAB_ON if i == active else _TAB_OFF, tab)
        pygame.draw.rect(surface, _BORDER, tab, 1)
        col = (198, 198, 218) if i == active else (95, 95, 120)
        txt = _font(11, bold=True).render(lbl, True, col)
        surface.blit(txt, (tab.x + (tw - txt.get_width()) // 2,
                           tab.y + (TAB_H - txt.get_height()) // 2))


# ── row renderer ──────────────────────────────────────────────────────────────

def _draw_rows(surface, rect, rows, model, scroll_y):
    """Draw two-column feature rows; return max scroll offset."""
    n        = len(rows)
    half     = (n + 1) // 2            # rows in left column
    col_w    = rect.width // 2

    total_h  = half * ROW_H
    max_scroll = max(0, total_h - rect.height)
    scroll_y   = max(0, min(scroll_y, max_scroll))

    # Draw onto a virtual surface then clip-blit
    vsurf = pygame.Surface((rect.width, total_h + 2), pygame.SRCALPHA)
    vsurf.fill((0, 0, 0, 0))

    lf  = _font(17)
    vf  = _font(20, bold=True)

    for idx, (key, val) in enumerate(rows):
        col  = idx // half
        row  = idx  % half
        x    = col * col_w + 10
        y    = row * ROW_H + 3

        lbl = lf.render(key, True, _LABEL)
        vsurf.blit(lbl, (x, y))

        vcol = model.field_color(key)
        vs   = vf.render(val, True, vcol)
        vsurf.blit(vs, (x, y + 18))

        # divider line at bottom of row
        if row < half - 1 or col == 0:
            pygame.draw.line(vsurf, _LINE,
                             (x, y + ROW_H - 2),
                             (x + col_w - 14, y + ROW_H - 2), 1)

    # Blit the visible slice
    surface.blit(vsurf, (rect.x, rect.y),
                 pygame.Rect(0, scroll_y, rect.width, rect.height))

    return max_scroll


# ── scrollbar ─────────────────────────────────────────────────────────────────

def _draw_scrollbar(surface, rect, scroll_y, max_scroll):
    pygame.draw.rect(surface, _SCROLL_BG, rect)
    if max_scroll <= 0:
        return
    total_h   = rect.height + max_scroll
    thumb_h   = max(20, rect.height * rect.height // total_h)
    thumb_y   = rect.y + int(scroll_y / max_scroll * (rect.height - thumb_h))
    pygame.draw.rect(surface, _SCROLL_TH,
                     pygame.Rect(rect.x + 1, thumb_y, rect.width - 2, thumb_h),
                     border_radius=3)
