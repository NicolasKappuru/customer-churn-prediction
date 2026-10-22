"""Agent strategies panel — right bottom panel."""

from __future__ import annotations
import pygame

# ── palette ───────────────────────────────────────────────────────────────────
_BG          = ( 28,  28,  36)
_BORDER      = ( 55,  55,  70)
_HEADER_BG   = ( 36,  36,  50)
_ROW_EVEN    = ( 32,  32,  44)
_ROW_ODD     = ( 36,  36,  50)
_BADGE_DEF   = ( 50,  50,  68)
_TEXT_NAME   = (200, 200, 220)

# Per-action selection theme: (row_bg, border, badge_bg, text)
_SEL_THEMES = [
    (( 52,  44,  80), (110,  80, 180), ( 88,  65, 140), (220, 200, 255)),  # purple
    (( 55,  50,  18), (200, 175,  50), (120, 100,  22), (255, 235, 120)),  # yellow
    (( 18,  44,  62), ( 60, 150, 210), ( 22,  85, 140), (140, 210, 255)),  # light blue
]
_TEXT_COST   = (195, 185, 135)
_LABEL       = (120, 120, 145)

# ── action definitions ────────────────────────────────────────────────────────
_ACTIONS = [
    ("1", "SUBSCRIPTION RENEWAL",  "Cost: 0%"),
    ("2", "ANNUAL PLAN + RE-ENG",  "Cost: 10%"),
    ("3", "5% DISCOUNT",           "Cost: 15%"),
]

# ── font cache ────────────────────────────────────────────────────────────────
_F: dict = {}


def _font(size: int, bold: bool = False) -> pygame.font.Font:
    key = (size, bold)
    if key not in _F:
        _F[key] = pygame.font.SysFont("Courier New", size, bold=bold)
    return _F[key]


# ── public ────────────────────────────────────────────────────────────────────

def draw_agent_panel(surface: pygame.Surface, rect: pygame.Rect,
                     selected_action: int | None) -> None:
    pygame.draw.rect(surface, _BG, rect)
    pygame.draw.rect(surface, _BORDER, rect, 1)

    # header bar
    hdr = pygame.Rect(rect.x, rect.y, rect.width, 32)
    pygame.draw.rect(surface, _HEADER_BG, hdr)
    pygame.draw.rect(surface, _BORDER, hdr, 1)

    h_lbl  = _font(12, bold=True).render("STRATEGIES", True, (175, 175, 200))
    h_cost = _font(12, bold=True).render("COST", True, _LABEL)
    surface.blit(h_lbl,  (rect.x + 18, rect.y + 8))
    surface.blit(h_cost, (rect.right - h_cost.get_width() - 18, rect.y + 8))

    # rows
    row_h = (rect.height - 32) // 3
    for i, (num, label, cost) in enumerate(_ACTIONS):
        ry  = rect.y + 32 + i * row_h
        row = pygame.Rect(rect.x, ry, rect.width, row_h)

        sel  = (selected_action == i)
        if sel:
            row_bg, border_col, badge_col, text_col = _SEL_THEMES[i]
        else:
            row_bg, border_col, badge_col, text_col = (
                _ROW_EVEN if i % 2 == 0 else _ROW_ODD, None, _BADGE_DEF, _TEXT_NAME)
        pygame.draw.rect(surface, row_bg, row)
        if sel:
            pygame.draw.rect(surface, border_col, row, 2)

        # number badge
        badge = pygame.Rect(row.x + 14, ry + row_h // 2 - 20, 38, 38)
        pygame.draw.rect(surface, badge_col, badge)
        num_s = _font(18, bold=True).render(num, True, (220, 220, 240))
        surface.blit(num_s, (badge.x + badge.width  // 2 - num_s.get_width()  // 2,
                              badge.y + badge.height // 2 - num_s.get_height() // 2))

        # label
        l_s   = _font(20, bold=True).render(label, True, text_col)
        surface.blit(l_s, (row.x + 64, ry + row_h // 2 - l_s.get_height() // 2))

        # cost
        c_s = _font(16).render(cost, True, _TEXT_COST)
        surface.blit(c_s, (row.right - c_s.get_width() - 18,
                            ry + row_h // 2 - c_s.get_height() // 2))

        # divider
        if i < 2:
            pygame.draw.line(surface, _BORDER,
                             (row.x, row.bottom), (row.right, row.bottom), 1)
