"""Load landscape background PNGs for customer and agent panels."""

from pathlib import Path
import pygame

_HERE        = Path(__file__).parent
_LAND_DIR    = _HERE / "sprites" / "landscapes"


def generate_background_customer(size=(640, 210)) -> pygame.Surface:
    img = pygame.image.load(str(_LAND_DIR / "landscape_customer.png")).convert()
    return pygame.transform.scale(img, size)


def generate_background_agent(size=(640, 210)) -> pygame.Surface:
    img = pygame.image.load(str(_LAND_DIR / "landscape_robot.png")).convert()
    return pygame.transform.scale(img, size)
