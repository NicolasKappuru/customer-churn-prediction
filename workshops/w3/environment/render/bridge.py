"""Bridge between KKBoxRetentionEnv and the pygame visualization."""

from __future__ import annotations
import random
import sys
from pathlib import Path

_VIZ_PATH = Path(__file__).parents[4] / "visualization"
if str(_VIZ_PATH) not in sys.path:
    sys.path.insert(0, str(_VIZ_PATH))

import pygame
from renderer import KKBoxRenderer
from customer_data import CustomerDataModel


class KKBoxRenderBridge:
    """Keeps the pygame window alive across episodes and feeds it env state."""

    def __init__(self, render_delay: float = 2.0) -> None:
        self._renderer = KKBoxRenderer()
        self._model: CustomerDataModel | None = None
        self._alive = True
        self._render_delay_ms = int(render_delay * 1000)

    # ── called by env ─────────────────────────────────────────────────────────

    def on_reset(self, features: dict, churn_prob: float) -> None:
        """Show a fresh customer after env.reset()."""
        self._model = CustomerDataModel(
            features=features,
            churn_probability=churn_prob,
        )
        self._renderer.set_sprite(random.randint(0, 23))
        self._renderer._scroll     = {0: 0, 1: 0, 2: 0}
        self._renderer._max_scroll = {0: 0, 1: 0, 2: 0}
        self._draw()

    def on_step(
        self,
        new_features: dict,
        new_churn: float,
        action: int,
        reward: float,
        accepted: bool,
    ) -> None:
        """Update the display after env.step()."""
        if self._model is None:
            return
        self._model.apply_action_result(
            new_features=new_features,
            new_churn=new_churn,
            reward=reward,
            action=action,
            accepted=accepted,
        )
        self._draw()
        self._hold()

    def tick(self) -> bool:
        """Pump events and redraw. Returns False when user closes/ESC."""
        for event in pygame.event.get():
            if not self._renderer.handle_event(event):
                self._alive = False
        if self._alive and self._model is not None:
            self._renderer.render(self._model)
        return self._alive

    def close(self) -> None:
        self._renderer.close()

    # ── internal ──────────────────────────────────────────────────────────────

    def _draw(self) -> None:
        if self._model is not None:
            self._renderer.render(self._model)
        self.tick()

    def _hold(self) -> None:
        """Hold the current frame for render_delay seconds, pumping events."""
        if self._render_delay_ms <= 0:
            return
        deadline = pygame.time.get_ticks() + self._render_delay_ms
        while self._alive and pygame.time.get_ticks() < deadline:
            for event in pygame.event.get():
                if not self._renderer.handle_event(event):
                    self._alive = False
                    return
            if self._model is not None:
                self._renderer.render(self._model)
            self._renderer.clock.tick(60)
