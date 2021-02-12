#ifndef JOSEPH3D_FWD_TOF_LM_CUDA
#define JOSEPH3D_FWD_TOF_LM_CUDA

extern "C" void joseph3d_fwd_tof_lm_cuda(const float *h_xstart, 
                                         const float *h_xend, 
                                         const float *h_img,
                                         const float *h_img_origin, 
                                         const float *h_voxsize, 
                                         float *h_p,
                                         long long nlors, 
                                         const int *h_img_dim,
                                         float tofbin_width,
                                         const float *h_sigma_tof,
                                         const float *h_tofcenter_offset,
                                         float n_sigmas,
                                         const short *h_tof_bin,
                                         int threadsperblock,
                                         int num_devices);

#endif
