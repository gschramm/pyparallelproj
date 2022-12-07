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


ge_discovery_mi_tof_parameters = TOFParameters(
    num_tofbins=29,
    tofbin_width=13 * 0.01302 * 299.792 /
    2,  # 13 TOF "small" TOF bins of 0.01302[ns] * (speed of light / 2) [mm/ns]
    sigma_tof=(299.792 / 2) *
    (0.385 / 2.355),  # (speed_of_light [mm/ns] / 2) * TOF FWHM [ns] / 2.355
    num_sigmas=3,
    tofcenter_offset=0)