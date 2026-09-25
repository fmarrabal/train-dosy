from typing import Literal
import numpy as np
from pydantic import BaseModel, ConfigDict, Field, model_validator

class FitRequest(BaseModel):
    """Signed phased observations. b is s/m² and all diffusion values are m²/s."""
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)
    Y: list[list[float]]
    b: list[float]
    ppm: list[float]
    sigma: float = Field(gt=0)
    method: Literal["RAI-S", "DOME-S", "MF-AUTO"] = "RAI-S"
    mask: list[bool] | None = None
    diffusion_bounds: tuple[float, float] = (0.1e-9, 15e-9)
    bins: int = Field(default=256, ge=256, le=2048)
    max_components: int = Field(default=4, ge=1, le=4)

    @model_validator(mode="after")
    def validate_arrays(self):
        n,p=len(self.b),len(self.ppm)
        if not 12 <= n <= 256 or not 1 <= p <= 8192 or n*p > 524288:
            raise ValueError("Require 12..256 rows, 1..8192 frequencies and at most 524288 observations")
        if len(self.Y)!=n or any(len(row)!=p for row in self.Y):
            raise ValueError("Y must have shape (len(b), len(ppm))")
        y=np.asarray(self.Y); b=np.asarray(self.b); ppm=np.asarray(self.ppm)
        lo,hi=self.diffusion_bounds
        if not all(np.isfinite(a).all() for a in (y,b,ppm)) or not np.isfinite([lo,hi]).all():
            raise ValueError("All values must be finite")
        if np.any(b<0) or len(np.unique(b))!=n or len(np.unique(ppm))!=p or not 0<lo<hi:
            raise ValueError("Distinct b >= 0 and ppm; 0 < diffusion lower bound < upper bound")
        if self.mask is not None and (len(self.mask)!=p or not any(self.mask)):
            raise ValueError("mask must match ppm and select at least one frequency")
        if self.method=="MF-AUTO" and self.max_components!=4:
            raise ValueError("Frozen MF automatic selector uses a fixed search limit of 4")
        return self
