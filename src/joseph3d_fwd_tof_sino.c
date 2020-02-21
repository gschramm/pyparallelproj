/**
 * @file joseph3d_fwd_tof_sino.c
 */

#include<stdio.h>
#include<stdlib.h>
#include<math.h>
#include<omp.h>

#include "tof_utils.h"

/** @brief 3D sinogram tof joseph forward projector
 *
 *  @param xstart array of shape [3*nlors] with the coordinates of the start points of the LORs.
 *                The start coordinates of the n-th LOR are at xstart[n*3 + i] with i = 0,1,2 
 *  @param xend   array of shape [3*nlors] with the coordinates of the end   points of the LORs.
 *                The start coordinates of the n-th LOR are at xstart[n*3 + i] with i = 0,1,2 
 *  @param img    array of shape [n0*n1*n2] containing the 3D image to be projected.
 *                The pixel [i,j,k] ist stored at [n1*n2+i + n2*k + j].
 *  @param img_origin  array [x0_0,x0_1,x0_2] of coordinates of the center of the [0,0,0] voxel
 *  @param voxsize     array [vs0, vs1, vs2] of the voxel sizes
 *  @param p           array of length nlors*n_tofbins (output) used to store the projections
 *  @param nlors       number of geomtrical LORs
 *  @param img_dim     array with dimensions of image [n0,n1,n2]
 *  @param n_tofbins        number of TOF bins
 *  @param tofbin_width     width of the TOF bins in spatial units (units of xstart and xend)
 *  @param sigma_tof        array of length nlors with the TOF resolution (sigma) for each LOR in
 *                          spatial units (units of xstart and xend) 
 *  @param tofcenter_offset array of length nlors with the offset of the central TOF bin from the 
 *                          midpoint of each LOR in spatial units (units of xstart and xend) 
 *  @param n_sigmas         number of sigmas to consider for calculation of TOF kernel
 */
void joseph3d_fwd_tof_sino(float *xstart, 
                           float *xend, 
                           float *img,
                           float *img_origin, 
                           float *voxsize, 
                           float *p,
                           long long nlors, 
                           unsigned int *img_dim,
		                       int n_tofbins,
		                       float tofbin_width,
		                       float *sigma_tof,
		                       float *tofcenter_offset,
		                       unsigned int n_sigmas)
{
  long long i;

  unsigned int n0 = img_dim[0];
  unsigned int n1 = img_dim[1];
  unsigned int n2 = img_dim[2];

  int n_half = n_tofbins/2;

  # pragma omp parallel for schedule(static)
  for(i = 0; i < nlors; i++)
  {
    float d0, d1, d2, d0_sq, d1_sq, d2_sq; 
    float lsq, cos0_sq, cos1_sq, cos2_sq;
    unsigned short direction; 
    unsigned int i0, i1, i2;
    int i0_floor, i1_floor, i2_floor;
    int i0_ceil, i1_ceil, i2_ceil;
    float x_pr0, x_pr1, x_pr2;
    float tmp_0, tmp_1, tmp_2;
   
    float u0, u1, u2, d_norm;
    float x_m0, x_m1, x_m2;    
    float x_v0, x_v1, x_v2;    

    int it, it1, it2;
    float dtof, tw;

    // correction factor for cos(theta) and voxsize
    float cf;
    float toAdd;

    float sig_tof   = sigma_tof[i];
    float tc_offset = tofcenter_offset[i];

    float xstart0 = xstart[i*3 + 0];
    float xstart1 = xstart[i*3 + 1];
    float xstart2 = xstart[i*3 + 2];

    float xend0 = xend[i*3 + 0];
    float xend1 = xend[i*3 + 1];
    float xend2 = xend[i*3 + 2];

    float voxsize0 = voxsize[0];
    float voxsize1 = voxsize[1];
    float voxsize2 = voxsize[2];

    float img_origin0 = img_origin[0];
    float img_origin1 = img_origin[1];
    float img_origin2 = img_origin[2];

    // test whether the ray between the two detectors is most parallel
    // with the 0, 1, or 2 axis
    d0 = xend0 - xstart0;
    d1 = xend1 - xstart1;
    d2 = xend2 - xstart2;

    d0_sq = d0*d0;
    d1_sq = d1*d1;
    d2_sq = d2*d2;

    lsq = d0_sq + d1_sq + d2_sq;

    cos0_sq = d0_sq / lsq;
    cos1_sq = d1_sq / lsq;
    cos2_sq = d2_sq / lsq;

    direction = 0;
    if ((cos1_sq >= cos0_sq) && (cos1_sq >= cos2_sq))
    {
      direction = 1;
    }
    else
    {
      if ((cos2_sq >= cos0_sq) && (cos2_sq >= cos1_sq))
      {
        direction = 2;
      }
    }
 
    //---------------------------------------------------------
    //--- calculate TOF related quantities
    
    // unit vector (u0,u1,u2) that points from xstart to end
    d_norm = sqrt(lsq);
    u0 = d0 / d_norm; 
    u1 = d1 / d_norm; 
    u2 = d2 / d_norm; 

    // calculate mid point of LOR
    x_m0 = 0.5*(xstart0 + xend0);
    x_m1 = 0.5*(xstart1 + xend1);
    x_m2 = 0.5*(xstart2 + xend2);

    //---------------------------------------------------------

    if (direction == 0)
    {
      cf = voxsize0 / sqrt(cos0_sq);

      // case where ray is most parallel to the 0 axis
      // we step through the volume along the 0 direction
      for(i0 = 0; i0 < n0; i0++)
      {
        // get the indices where the ray intersects the image plane
        x_pr1 = xstart1 + (img_origin0 + i0*voxsize0 - xstart0)*d1 / d0;
        x_pr2 = xstart2 + (img_origin1 + i0*voxsize1 - xstart0)*d2 / d0;
  
        i1_floor = (int)floor((x_pr1 - img_origin1)/voxsize1);
        i1_ceil  = i1_floor + 1;
  
        i2_floor = (int)floor((x_pr2 - img_origin2)/voxsize2);
        i2_ceil  = i2_floor + 1; 
 
        // calculate the distances to the floor normalized to [0,1]
        // for the bilinear interpolation
        tmp_1 = (x_pr1 - (i1_floor*voxsize1 + img_origin1)) / voxsize1;
        tmp_2 = (x_pr2 - (i2_floor*voxsize2 + img_origin2)) / voxsize2;

	      toAdd = 0;

        if ((i1_floor >= 0) && (i1_floor < n1) && (i2_floor >= 0) && (i2_floor < n2))
        {
          toAdd += img[n1*n2*i0 + n2*i1_floor + i2_floor] * (1 - tmp_1) * (1 - tmp_2);
        }
        if ((i1_ceil >= 0) && (i1_ceil < n1) && (i2_floor >= 0) && (i2_floor < n2))
        {
          toAdd += img[n1*n2*i0 + n2*i1_ceil + i2_floor] * tmp_1 * (1 - tmp_2);
        }
        if ((i1_floor >= 0) && (i1_floor < n1) && (i2_ceil >= 0) && (i2_ceil < n2))
        {
          toAdd += img[n1*n2*i0 + n2*i1_floor + i2_ceil] * (1 - tmp_1) * tmp_2;
        }
        if ((i1_ceil >= 0) && (i1_ceil < n1) && (i2_ceil >= 0) && (i2_ceil < n2))
        {
          toAdd += img[n1*n2*i0 + n2*i1_ceil + i2_ceil] * tmp_1 * tmp_2;
        }

        //--------- TOF related quantities
        // calculate the voxel center needed for TOF weights
        x_v0 = img_origin0 + i0*voxsize0;
        x_v1 = x_pr1;
        x_v2 = x_pr2;

	      it1 = -n_half;
	      it2 =  n_half;

        // get the relevant tof bins (the TOF bins where the TOF weight is not close to 0)
        relevant_tof_bins(x_m0, x_m1, x_m2, x_v0, x_v1, x_v2, u0, u1, u2, 
			                    tofbin_width, tc_offset, sig_tof, n_sigmas, n_half,
		                      &it1, &it2);

        if(toAdd != 0){
          for(it = it1; it <= it2; it++){
            // calculate distance of voxel to tof bin center
            dtof = sqrtf(powf((x_m0 + (it*tofbin_width + tc_offset)*u0 - x_v0), 2) + 
                         powf((x_m1 + (it*tofbin_width + tc_offset)*u1 - x_v1), 2) + 
                         powf((x_m2 + (it*tofbin_width + tc_offset)*u2 - x_v2), 2));

            //calculate the TOF weight
            tw = 0.5*(erff((dtof + 0.5*tofbin_width)/(sqrtf(2)*sig_tof)) - 
                      erff((dtof - 0.5*tofbin_width)/(sqrtf(2)*sig_tof)));

            p[i*n_tofbins + it + n_half] += (tw * cf * toAdd);
          }
        }
      }
    }

    //--------------------------------------------------------------------------------- 
    if (direction == 1)
    {
      cf = voxsize1 / sqrt(cos1_sq);

      // case where ray is most parallel to the 1 axis
      // we step through the volume along the 1 direction
      for (i1 = 0; i1 < n1; i1++)
      {
        // get the indices where the ray intersects the image plane
        x_pr0 = xstart0 + (img_origin1 + i1*voxsize1 - xstart1)*d0 / d1;
        x_pr2 = xstart2 + (img_origin1 + i1*voxsize1 - xstart1)*d2 / d1;
  
        i0_floor = (int)floor((x_pr0 - img_origin0)/voxsize0);
        i0_ceil  = i0_floor + 1; 
  
        i2_floor = (int)floor((x_pr2 - img_origin2)/voxsize2);
        i2_ceil  = i2_floor + 1;
  
        // calculate the distances to the floor normalized to [0,1]
        // for the bilinear interpolation
        tmp_0 = (x_pr0 - (i0_floor*voxsize0 + img_origin0)) / voxsize0;
        tmp_2 = (x_pr2 - (i2_floor*voxsize2 + img_origin2)) / voxsize2;
 
        toAdd = 0;

        if ((i0_floor >= 0) && (i0_floor < n0) && (i2_floor >= 0) && (i2_floor < n2))
        {
          toAdd += img[n1*n2*i0_floor + n2*i1 + i2_floor] * (1 - tmp_0) * (1 - tmp_2);
        }
        if ((i0_ceil >= 0) && (i0_ceil < n0) && (i2_floor >= 0) && (i2_floor < n2))
        {
          toAdd += img[n1*n2*i0_ceil + n2*i1 + i2_floor] * tmp_0 * (1 - tmp_2);
        }
        if ((i0_floor >= 0) && (i0_floor < n0) && (i2_ceil >= 0) && (i2_ceil < n2))
        {
          toAdd += img[n1*n2*i0_floor + n2*i1 + i2_ceil] * (1 - tmp_0) * tmp_2;
        }
        if ((i0_ceil >= 0) && (i0_ceil < n0) && (i2_ceil >= 0) && (i2_ceil < n2))
        {
          toAdd += img[n1*n2*i0_ceil + n2*i1 + i2_ceil] * tmp_0 * tmp_2;
        }

        //--------- TOF related quantities
        // calculate the voxel center needed for TOF weights
        x_v0 = x_pr0;
        x_v1 = img_origin1 + i1*voxsize1;
        x_v2 = x_pr2;

	      it1 = -n_half;
	      it2 =  n_half;

        // get the relevant tof bins (the TOF bins where the TOF weight is not close to 0)
        relevant_tof_bins(x_m0, x_m1, x_m2, x_v0, x_v1, x_v2, u0, u1, u2, 
			                    tofbin_width, tc_offset, sig_tof, n_sigmas, n_half,
		                      &it1, &it2);

        if(toAdd != 0){
          for(it = it1; it <= it2; it++){
            // calculate distance of voxel to tof bin center
            dtof = sqrtf(powf((x_m0 + (it*tofbin_width + tc_offset)*u0 - x_v0), 2) + 
                         powf((x_m1 + (it*tofbin_width + tc_offset)*u1 - x_v1), 2) + 
                         powf((x_m2 + (it*tofbin_width + tc_offset)*u2 - x_v2), 2));

            //calculate the TOF weight
            tw = 0.5*(erff((dtof + 0.5*tofbin_width)/(sqrtf(2)*sig_tof)) - 
                      erff((dtof - 0.5*tofbin_width)/(sqrtf(2)*sig_tof)));


            p[i*n_tofbins + it + n_half] += (tw * cf * toAdd);
	        }
	      }
      }
    }

    //--------------------------------------------------------------------------------- 
    if (direction == 2)
    {
      cf = voxsize[direction] / sqrt(cos2_sq);

      // case where ray is most parallel to the 2 axis
      // we step through the volume along the 2 direction

      for(i2 = 0; i2 < n2; i2++)
      {
        // get the indices where the ray intersects the image plane
        x_pr0 = xstart0 + (img_origin2 + i2*voxsize2 - xstart2)*d0 / d2;
        x_pr1 = xstart1 + (img_origin2 + i2*voxsize2 - xstart2)*d1 / d2;
  
        i0_floor = (int)floor((x_pr0 - img_origin0)/voxsize0);
        i0_ceil  = i0_floor + 1;
  
        i1_floor = (int)floor((x_pr1 - img_origin1)/voxsize1);
        i1_ceil  = i1_floor + 1; 
  
        // calculate the distances to the floor normalized to [0,1]
        // for the bilinear interpolation
        tmp_0 = (x_pr0 - (i0_floor*voxsize0 + img_origin0)) / voxsize0;
        tmp_1 = (x_pr1 - (i1_floor*voxsize1 + img_origin1)) / voxsize1;

        toAdd = 0;

        if ((i0_floor >= 0) && (i0_floor < n0) && (i1_floor >= 0) && (i1_floor < n1))
        {
          toAdd += img[n1*n2*i0_floor + n2*i1_floor + i2] * (1 - tmp_0) * (1 - tmp_1);
        }
        if ((i0_ceil >= 0) && (i0_ceil < n0) && (i1_floor >= 0) && (i1_floor < n1))
        {
          toAdd += img[n1*n2*i0_ceil + n2*i1_floor + i2] * tmp_0 * (1 - tmp_1);
        }
        if ((i0_floor >= 0) && (i0_floor < n0) && (i1_ceil >= 0) & (i1_ceil < n1))
        {
          toAdd += img[n1*n2*i0_floor + n2*i1_ceil + i2] * (1 - tmp_0) * tmp_1;
        }
        if ((i0_ceil >= 0) && (i0_ceil < n0) && (i1_ceil >= 0) && (i1_ceil < n1))
        {
          toAdd += img[n1*n2*i0_ceil + n2*i1_ceil + i2] * tmp_0 * tmp_1;
        }

        //--------- TOF related quantities
        // calculate the voxel center needed for TOF weights
        x_v0 = x_pr0;
        x_v1 = x_pr1;
        x_v2 = img_origin2 + i2*voxsize2;

	      it1 = -n_half;
	      it2 =  n_half;

        // get the relevant tof bins (the TOF bins where the TOF weight is not close to 0)
        relevant_tof_bins(x_m0, x_m1, x_m2, x_v0, x_v1, x_v2, u0, u1, u2, 
			                    tofbin_width, tc_offset, sig_tof, n_sigmas, n_half,
		                      &it1, &it2);

        if(toAdd != 0){
          for(it = it1; it <= it2; it++){
            // calculate distance of voxel to tof bin center
            dtof = sqrtf(powf((x_m0 + (it*tofbin_width + tc_offset)*u0 - x_v0), 2) + 
                         powf((x_m1 + (it*tofbin_width + tc_offset)*u1 - x_v1), 2) + 
                         powf((x_m2 + (it*tofbin_width + tc_offset)*u2 - x_v2), 2));

            //calculate the TOF weight
            tw = 0.5*(erff((dtof + 0.5*tofbin_width)/(sqrtf(2)*sig_tof)) - 
                      erff((dtof - 0.5*tofbin_width)/(sqrtf(2)*sig_tof)));

            p[i*n_tofbins + it + n_half] += (tw * cf * toAdd);
	        }
	      }
      }
    }
  }
}