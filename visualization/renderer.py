"""Main renderer — composes all layers into the 1280×720 window."""

from __future__ import annotations
import math
import pygame

from assets.sprites        import load_customer_sprites, load_robot_sprite, load_acceptance_sprites
from assets.backgrounds    import generate_background_customer, generate_background_agent
from panels.customer_panel import draw_customer_panel
from panels.agent_panel    import draw_agent_panel
from hud                   import draw_churn_box, draw_reward_box
from customer_data         import CustomerDataModel

# ── layout ────────────────────────────────────────────────────────────────────
WIN_W    = 1280
WIN_H    = 720
TITLE_H  = 36
TOP_H    = 210
BOTTOM_H = WIN_H - TITLE_H - TOP_H   # 474
HALF_W   = WIN_W // 2

_BG     = ( 16,  16,  24)
_TITLE  = ( 11,  11,  18)
_DIV_V  = ( 52,  52,  70)
_DIV_H  = ( 46,  46,  62)
_BORDER = ( 58,  58,  76)

_SCROLL_STEP = 18

_F: dict = {}


def _font(size, bold=False):
    k = (size, bold)
    if k not in _F:
        _F[k] = pygame.font.SysFont("Courier New", size, bold=bold)
    return _F[k]


class KKBoxRenderer:
    def __init__(self) -> None:
        pygame.init()
        self.screen = pygame.display.set_mode((WIN_W, WIN_H))
        pygame.display.set_caption("KKBOX — RETENTION")
        self.clock  = pygame.time.Clock()

        self.customer_sprites  = load_customer_sprites((150, 200))
        self.robot_sprite      = load_robot_sprite((150, 200))
        self.spr_accept, self.spr_reject = load_acceptance_sprites((72, 72))
        self.bg_customer      = generate_background_customer((HALF_W, TOP_H))
        self.bg_agent         = generate_background_agent((HALF_W, TOP_H))

        self.active_tab    = 0
        self._sprite_idx   = 0          # index into customer_sprites list (0–23)
        self._tick         = 0.0
        self._scroll       = {0: 0, 1: 0, 2: 0}
        self._max_scroll   = {0: 0, 1: 0, 2: 0}

    # ── public ────────────────────────────────────────────────────────────────

    def set_sprite(self, idx: int) -> None:
        """Pick which customer sprite to show (0–23)."""
        self._sprite_idx = idx % len(self.customer_sprites)

    def render(self, model: CustomerDataModel) -> None:
        self._tick += 0.05
        self.screen.fill(_BG)
        self._draw_title()
        self._draw_top(model)
        self._draw_bottom(model)
        pygame.display.flip()

    def handle_event(self, event: pygame.event.Event) -> bool:
        if event.type == pygame.QUIT:
            return False
        if event.type == pygame.KEYDOWN:
            if   event.key == pygame.K_q:      self.active_tab = 0
            elif event.key == pygame.K_w:      self.active_tab = 1
            elif event.key == pygame.K_e:      self.active_tab = 2
            elif event.key == pygame.K_UP:
                self._scroll[self.active_tab] = max(
                    0, self._scroll[self.active_tab] - _SCROLL_STEP)
            elif event.key == pygame.K_DOWN:
                self._scroll[self.active_tab] = min(
                    self._max_scroll[self.active_tab],
                    self._scroll[self.active_tab] + _SCROLL_STEP)
            elif event.key == pygame.K_ESCAPE: return False
        return True

    def close(self) -> None:
        pygame.quit()

    # ── private ───────────────────────────────────────────────────────────────

    def _draw_title(self) -> None:
        pygame.draw.rect(self.screen, _TITLE, (0, 0, WIN_W, TITLE_H))
        pygame.draw.line(self.screen, _BORDER, (0, TITLE_H - 1), (WIN_W, TITLE_H - 1), 1)
        t = _font(16, bold=True).render("KKBOX — RETENTION", True, (215, 215, 235))
        self.screen.blit(t, (18, (TITLE_H - t.get_height()) // 2))

    def _draw_top(self, model: CustomerDataModel) -> None:
        y     = TITLE_H
        div_y = y + TOP_H

        self.screen.blit(self.bg_customer, (0,      y))
        self.screen.blit(self.bg_agent,    (HALF_W, y))
        pygame.draw.line(self.screen, _DIV_V, (HALF_W, y), (HALF_W, div_y), 2)

        # Purely upward bob — sprites stand on the divider line
        bob_c = int(abs(math.sin(self._tick)) * 7)
        bob_r = int(abs(math.cos(self._tick)) * 7)

        sp   = self.customer_sprites[self._sprite_idx]
        sp_x = HALF_W // 2 - sp.get_width() // 2
        sp_y = div_y - sp.get_height() - bob_c
        self.screen.blit(sp, (sp_x, sp_y))

        rb   = self.robot_sprite
        rb_x = HALF_W + HALF_W // 2 - rb.get_width() // 2
        rb_y = div_y - rb.get_height() - bob_r
        self.screen.blit(rb, (rb_x, rb_y))

        draw_churn_box( self.screen,          10, y + 8, model.churn_probability)
        draw_reward_box(self.screen, HALF_W + 10, y + 8, model.reward)

        if model.action_accepted is not None:
            icon = self.spr_accept if model.action_accepted else self.spr_reject
            icon_x = sp_x + sp.get_width()  - icon.get_width()  // 2
            icon_y = sp_y + 10
            self.screen.blit(icon, (icon_x, icon_y))

        # Floor frame drawn in front of sprites — covers feet so they look grounded
        FRAME_H = 14
        pygame.draw.rect(self.screen, _DIV_H,  (0, div_y - FRAME_H, WIN_W, FRAME_H))
        pygame.draw.line(self.screen, _BORDER, (0, div_y - FRAME_H), (WIN_W, div_y - FRAME_H), 1)
        pygame.draw.line(self.screen, _BORDER, (0, div_y),           (WIN_W, div_y),            1)

    def _draw_bottom(self, model: CustomerDataModel) -> None:
        y     = TITLE_H + TOP_H
        left  = pygame.Rect(0,      y, HALF_W, BOTTOM_H)
        right = pygame.Rect(HALF_W, y, HALF_W, BOTTOM_H)

        max_s = draw_customer_panel(
            self.screen, left, model,
            self.active_tab,
            self._scroll[self.active_tab],
        )
        self._max_scroll[self.active_tab] = max_s
        draw_agent_panel(self.screen, right, model.action_selected)
        pygame.draw.line(self.screen, _DIV_V, (HALF_W, y), (HALF_W, y + BOTTOM_H), 2)
