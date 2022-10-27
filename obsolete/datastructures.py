import attrs


@attrs.frozen(kw_only=True)
class TimeOfFlightDataParameters:
    """parameters related to time of flight data
    
    Parameters
    ----------
    num_tofbins: int
        number of total tof bins
    tofbin_width: float
        width of the tof bin in spatial units (mm)
    """
    num_tofbins: int = attrs.field(
        validator=[attrs.validators.instance_of(int),
                   attrs.validators.gt(0)])
    tofbin_width: float = attrs.field(validator=[
        attrs.validators.instance_of(float),
        attrs.validators.gt(0)
    ])


@attrs.frozen(kw_only=True)
class TimeOfFlightModelParameters:
    """parameters related to time of flight model
    
    Parameters
    ----------
    sigma_tof: float
        standard deviation of the Gaussian kernel used for tof projections
        in spatial units (mm)
    num_sigmas: float
        number of sigmas after which Gaussian tof kernel is truncated, default 3
    tofcenter_offset: float
        offset of the central tof bin in spatial units (mm)
    """
    sigma_tof: float = attrs.field(validator=[
        attrs.validators.instance_of(float),
        attrs.validators.gt(0)
    ])
    num_sigmas: float = attrs.field(default=3.,
                                    validator=[
                                        attrs.validators.instance_of(float),
                                        attrs.validators.gt(0)
                                    ])
    tofcenter_offset: float = attrs.field(validator=[
        attrs.validators.instance_of(float),
        attrs.validators.gt(0)
    ])