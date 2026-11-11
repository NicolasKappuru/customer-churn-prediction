"""Load customer and robot sprite PNGs.

Customer sprites: assets/sprites/customers/customer_01.png … customer_24.png
Robot sprite:     assets/sprites/robot/robot.png
"""

from pathlib import Path
import pygame

_HERE        = Path(__file__).parent
_CUST_DIR    = _HERE / "sprites" / "customers"
_ROBOT_DIR   = _HERE / "sprites" / "robot"
_ACCEPT_DIR  = _HERE / "sprites" / "acceptance"

N_CUSTOMERS = 24


def load_customer_sprites(size: tuple[int, int] = (150, 200)) -> list[pygame.Surface]:
    """Return list[0..23] of scaled customer Surfaces."""
    sprites = []
    for i in range(1, N_CUSTOMERS + 1):
        path = _CUST_DIR / f"customer_{i:02d}.png"
        img  = pygame.image.load(str(path)).convert_alpha()
        sprites.append(pygame.transform.scale(img, size))
    return sprites


def load_robot_sprite(size: tuple[int, int] = (150, 200)) -> pygame.Surface:
    """Return scaled robot Surface."""
    path = _ROBOT_DIR / "robot.png"
    img  = pygame.image.load(str(path)).convert_alpha()
    return pygame.transform.scale(img, size)


def load_acceptance_sprites(size: tuple[int, int] = (72, 72)) -> tuple[pygame.Surface, pygame.Surface]:
    """Return (accept_surface, reject_surface)."""
    accept = pygame.image.load(str(_ACCEPT_DIR / "accept.png")).convert_alpha()
    reject = pygame.image.load(str(_ACCEPT_DIR / "reject.png")).convert_alpha()
    return pygame.transform.scale(accept, size), pygame.transform.scale(reject, size)
