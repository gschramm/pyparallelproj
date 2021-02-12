#ifndef JOSEPH3D_FWD_TOF_SINO
#define JOSEPH3D_FWD_TOF_SINO

void joseph3d_fwd_tof_sino(const float *xstart, 
                           const float *xend, 
                           const float *img,
                           const float *img_origin, 
                           const float *voxsize, 
                           float *p,
                           long long nlors, 
                           const int *img_dim,
                           float tofbin_width,
                           const float *sigma_tof,
                           const float *tofcenter_offset,
                           float n_sigmas,
                           short n_tofbins);

#endif
