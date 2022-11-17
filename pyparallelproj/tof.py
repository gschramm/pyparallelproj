from pydantic import BaseModel, Field


class TOFParameters(BaseModel):
    """
    time of flight (TOF) parameters
    """
    num_tofbins: int = Field(None,
                             description='Number of time of flight bins',
                             ge=0)
    tofbin_width: float = Field(
        None, description='width of the TOF bin in spatial units', ge=0)
    sigma_tof: float = Field(
        None,
        description=
        'standard deviation of Gaussian TOF kernel in spatial units',
        ge=0)
    num_sigmas: float = Field(
        3,
        description='number of sigmas after which TOF kernel is truncated',
        ge=0)
    tofcenter_offset: float = Field(
        0,
        description=
        'offset of center of central TOF bin from LOR center in spatial units')