from enum import Enum
from pydantic import BaseModel

class ImageSize(BaseModel):
    width: int
    height: int


class Variants(Enum):
    GRAYSCALE = "grayscale" # grayscale images
    CONTRAST = "contrast" # increased contrast
    BRIGHT = "bright" # increased brightness
    COMBINED = "combined" # combination of all variants
    ORIGINAL = "original" # no modification