"""HUD overlay elements drawn on top of everything else.

- Churn probability box (top-left of customer panel)
- Reward box (top-left of agent panel)
- Accept / reject circle overlay on the customer sprite
"""

from __future__ import annotations
import pygame

# ── font cache ────────────────────────────────────────────────────────────────
_F: dict = {}


def _font(name: str, size: int, bold: bool = False) -> pygame.font.Font:
    key = (name, size, bold)
    if key not in _F:
        _F[key] = pygame.font.SysFont(name, size, bold=bold)
    return _F[key]


# ── churn colour ramp (green → yellow → red) ─────────────────────────────────

def _churn_color(p: float) -> tuple:
    r = int(min(255, p * 2 * 255))
    g = int(min(255, (1.0 - p) * 2 * 255))
    return (r, g, 40)


def draw_churn_box(surface: pygame.Surface, x: int, y: int,
                   probability: float) -> None:
    """Draw the P(CHURN) indicator box."""
    w, h = 162, 78
    box = pygame.Surface((w, h), pygame.SRCALPHA)
    box.fill((14, 14, 22, 210))
    pygame.draw.rect(box, (65, 65, 80), (0, 0, w, h), 1)

    # label
    lbl = _font("Courier New", 12).render("P(CHURN)", True, (170, 170, 190))
    box.blit(lbl, (8, 6))

    # percentage
    color = _churn_color(probability)
    pct   = _font("Courier New", 28, bold=True).render(f"{probability:.0%}", True, color)
    box.blit(pct, (8, 20))

    # bar
    bx, by, bw, bh = 8, 58, w - 16, 11
    pygame.draw.rect(box, (45, 45, 58), (bx, by, bw, bh))
    filled = int(bw * probability)
    for i in range(filled):
        seg = _churn_color(i / max(bw, 1))
        pygame.draw.rect(box, seg, (bx + i, by, 1, bh))

    surface.blit(box, (x, y))


def draw_reward_box(surface: pygame.Surface, x: int, y: int,
                    reward: float) -> None:
    """Draw the REWARD indicator box."""
    w, h = 162, 78
    box = pygame.Surface((w, h), pygame.SRCALPHA)
    box.fill((14, 14, 22, 210))
    pygame.draw.rect(box, (65, 65, 80), (0, 0, w, h), 1)

    lbl = _font("Courier New", 12).render("REWARD", True, (170, 170, 190))
    box.blit(lbl, (8, 6))

    if reward > 0:
        color = ( 70, 210, 100)
        sign  = "+"
    elif reward < 0:
        color = (210,  70,  70)
        sign  = ""
    else:
        color = (160, 160, 160)
        sign  = ""

    val = _font("Courier New", 28, bold=True).render(f"{sign}{reward:.2f}", True, color)
    box.blit(val, (8, 20))

    surface.blit(box, (x, y))


