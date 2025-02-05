//========================================================================================
// Athena++ astrophysical MHD code
// Copyright(C) 2014 James M. Stone <jmstone@princeton.edu> and other code contributors
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
//! \file turbfield.cpp
//  \brief implementation of functions in Turbulent B Field class

// C headers

// C++ headers
#include <algorithm>
#include <cmath>
#include <complex>
#include <iostream>
#include <random>     // mt19937, normal_distribution, uniform_real_distribution
#include <sstream>    // sstream
#include <stdexcept>  // runtime_error
#include <string>     // c_str()

// Athena++ headers
#include "../athena.hpp"
#include "../athena_arrays.hpp"
#include "../coordinates/coordinates.hpp"
#include "../globals.hpp"
#include "../hydro/hydro.hpp"
#include "../field/field.hpp"
#include "../mesh/mesh.hpp"
#include "../utils/utils.hpp"
#include "athena_fft.hpp"
#include "turbfield.hpp"

//----------------------------------------------------------------------------------------
//! \fn RandFieldDriver::RandFieldDriver(Mesh *pm, ParameterInput *pin)
//! \brief RandFieldDriver constructor
//!
//! \note
//! Random field is only generated before the main loop - added onto the B field 
//!     after the problem generator is run.

RandFieldDriver::RandFieldDriver(Mesh *pm, ParameterInput *pin) :
    FFTDriver(pm, pin),
    rseed(pin->GetOrAddInteger("randBfield", "rseed", -1)), // seed for RNG
    // cut-off wavenumbers, low and high:
    nlow(pin->GetOrAddInteger("randBfield", "nlow", 0)),
    nhigh(pin->GetOrAddInteger("randBfield", "nhigh", pm->mesh_size.nx1/2)),
    // power spectrum exponential law
    expo(pin->GetOrAddReal("randBfield", "expo", 2)), // power-law exponent
    dB(pin->GetReal("randBfield", "dB")), // turbulent field amplitude
    deltaB{ {nmb, pm->my_blocks(0)->ncells3,
               pm->my_blocks(0)->ncells2, pm->my_blocks(0)->ncells1},
         {nmb, pm->my_blocks(0)->ncells3,
               pm->my_blocks(0)->ncells2, pm->my_blocks(0)->ncells1},
         {nmb, pm->my_blocks(0)->ncells3,
               pm->my_blocks(0)->ncells2, pm->my_blocks(0)->ncells1} }//,
    // deltaA{ {nmb, pm->my_blocks(0)->ncells3,
    //            pm->my_blocks(0)->ncells2, pm->my_blocks(0)->ncells1},
    //      {nmb, pm->my_blocks(0)->ncells3,
    //            pm->my_blocks(0)->ncells2, pm->my_blocks(0)->ncells1},
    //      {nmb, pm->my_blocks(0)->ncells3,
    //            pm->my_blocks(0)->ncells2, pm->my_blocks(0)->ncells1} }
     {


#ifndef FFT
    std::stringstream msg;
    msg << "### FATAL ERROR in RandFieldDriver::RandFieldDriver" << std::endl
        << "non zero Random Field flag is set without FFT!" << std::endl;
    ATHENA_ERROR(msg);
    return;
#endif
  

  InitializeFFTBlock(true);
  // note, pmy_fb won't be defined until InitializeFFTBlock is called:
  dvol = pmy_fb->dx1*pmy_fb->dx2*pmy_fb->dx3;
  QuickCreatePlan();

  fv_ = new std::complex<Real>*[3];
  fv_sh_ = new std::complex<Real>*[3];
  fv_co_ = new std::complex<Real>*[3];
  for (int nv=0; nv<3; nv++) {
    fv_[nv] = new std::complex<Real>[pmy_fb->cnt_];
    fv_sh_[nv] = new std::complex<Real>[pmy_fb->cnt_];
    fv_co_[nv] = new std::complex<Real>[pmy_fb->cnt_];
  }


  // initialize MT19937 random number generator
  if (rseed < 0) {
    std::random_device device;
    rseed = static_cast<std::int64_t>(device());
  } else {
    // If rseed is specified with a non-negative value,
    // PS is generated with a global random number sequence.
    // This would make perturbation identical irrespective of number of MPI ranks,
    // but the cost of the PowerSpectrum() function call is huge.
    global_ps_ = true;
  }
  rng_generator.seed(rseed);
}

// destructor
RandFieldDriver::~RandFieldDriver() {
  for (int nv=0; nv<3; nv++) {
    delete [] fv_[nv];
  }
  delete [] fv_;
}

//----------------------------------------------------------------------------------------
//! \fn void RandFieldDriver::Driving()
//! \brief Generate and Perturb the velocity field

void RandFieldDriver::Driving() {
  Mesh *pm = pmy_mesh_;

  Generate();
  Perturb();
  
  return;
}

//----------------------------------------------------------------------------------------
//! \fn void RandFieldDriver::Generate()
//! \brief Generate random components of vector potential.

void RandFieldDriver::Generate() {
  Mesh *pm = pmy_mesh_;
  FFTBlock *pfb = pmy_fb;
  AthenaFFTPlan *plan = pfb->bplan_;

  int nbs = nslist_[Globals::my_rank];
  int nbe = nbs + nblist_[Globals::my_rank] - 1;

  if (!initialized_) {
    for (int nv=0; nv<3; nv++) {
      std::complex<Real> *fv = fv_[nv];
      PowerSpectrum(fv);
    }
    Project(fv_);
  } 

  for (int nv=0; nv<3; nv++) {
    AthenaArray<Real> &dB = deltaB[nv], dB_mb;
    for (int kidx=0; kidx<pfb->cnt_; kidx++) pfb->in_[kidx] = fv_[nv][kidx];
    pfb->Execute(plan);
    for (int nb=0; nb<pm->nblocal; ++nb) {
      MeshBlock *pmb = pm->my_blocks(nb);
      dB_mb.InitWithShallowSlice(dB, 4, nb, 1);
      pfb->RetrieveResult(dB_mb, 0, NGHOST, pmb->loc, pmb->block_size);
    }
  }
}


//----------------------------------------------------------------------------------------
//! \fn void RandFieldDriver::PowerSpectrum(std::complex<Real> *amp)
//! \brief Generate Power spectrum in Fourier space with power-law

void RandFieldDriver::PowerSpectrum(std::complex<Real> *amp) {
  Real pcoeff;
  FFTBlock *pfb = pmy_fb;
  AthenaFFTIndex *idx = pfb->b_in_;
  int kNx1 = pfb->kNx[0], kNx2 = pfb->kNx[1], kNx3 = pfb->kNx[2];
  int knx1 = pfb->knx[0], knx2 = pfb->knx[1], knx3 = pfb->knx[2];
  int kdisp1 = pfb->kdisp[0], kdisp2 = pfb->kdisp[1], kdisp3 = pfb->kdisp[2];

  std::normal_distribution<Real> ndist(0.0,1.0); // standard normal distribution
  std::uniform_real_distribution<Real> udist(0.0,1.0); // uniform in [0,1)

  // set random amplitudes with gaussian deviation
  // loop over entire Mesh
  if (global_ps_) {
    for (int gk=0; gk<kNx3; gk++) {
      for (int gj=0; gj<kNx2; gj++) {
        for (int gi=0; gi<kNx1; gi++) {
          std::int64_t nx = GetKcomp(gi, 0, kNx1);
          std::int64_t ny = GetKcomp(gj, 0, kNx2);
          std::int64_t nz = GetKcomp(gk, 0, kNx3);
          Real nmag = std::sqrt(nx*nx+ny*ny+nz*nz);
          int k = gk - kdisp3;
          int j = gj - kdisp2;
          int i = gi - kdisp1;
          // Draw random number only in the cutoff range.
          // This ensures that the perturbed velocity field is independent of
          // the number of cells, which is useful property for resolution study.
          // Note that this only applies for global_ps_ = true (i.e., rseed >= 0).
          if ((nmag > nlow) && (nmag < nhigh)) {
            if ((k >= 0) && (k < knx3) &&
                (j >= 0) && (j < knx2) &&
                (i >= 0) && (i < knx1)) {
              std::int64_t kidx = pfb->GetIndex(i,j,k,idx);
              Real A = ndist(rng_generator);
              Real ph = udist(rng_generator)*TWO_PI;
              amp[kidx] = A*std::complex<Real>(std::cos(ph), std::sin(ph));
            } else { // if it is not in FFTBlock, just burn unused random numbers
              Real A = ndist(rng_generator);
              Real ph = udist(rng_generator)*TWO_PI;
            }
          }
        }
      }
    }
  }

  // set power spectrum: only power-law

  // find the reference 2PI/L along the longest axis
  Real dkx = pfb->dkx[0];
  if (knx2>1) dkx = dkx < pfb->dkx[1] ? dkx : pfb->dkx[1];
  if (knx3>2) dkx = dkx < pfb->dkx[2] ? dkx : pfb->dkx[2];

  for (int k=0; k<knx3; k++) {
    for (int j=0; j<knx2; j++) {
      for (int i=0; i<knx1; i++) {
        std::int64_t nx = GetKcomp(i,pfb->kdisp[0],pfb->kNx[0]);
        std::int64_t ny = GetKcomp(j,pfb->kdisp[1],pfb->kNx[1]);
        std::int64_t nz = GetKcomp(k,pfb->kdisp[2],pfb->kNx[2]);
        Real nmag = std::sqrt(nx*nx+ny*ny+nz*nz);
        Real kx = nx*pfb->dkx[0];
        Real ky = ny*pfb->dkx[1];
        Real kz = nz*pfb->dkx[2];
        Real kmag = std::sqrt(kx*kx+ky*ky+kz*kz);

        std::int64_t gidx = pfb->GetGlobalIndex(i,j,k);

        if (gidx == 0) {
          pcoeff = 0.0;
        } else {
          if ((kmag/dkx > nlow) && (kmag/dkx < nhigh)) {
            pcoeff = 1.0/std::pow(kmag,(expo+2.0)/2.0);
          } else {
            pcoeff = 0.0;
          }
        }
        std::int64_t kidx=pfb->GetIndex(i,j,k,idx);

        if (global_ps_) {
          amp[kidx] *= pcoeff;
        } else {
          Real A = ndist(rng_generator);
          Real ph = udist(rng_generator)*TWO_PI;
          amp[kidx] = pcoeff*A*std::complex<Real>(std::cos(ph), std::sin(ph));
        }
      }
    }
  }
  return;
}

//----------------------------------------------------------------------------------------
//! \fn void RandFieldeDriver::Project(std::complex<Real> **fv, Real f_shear)
//! \brief calculate velocity field with a given ratio of shear to comp.

void RandFieldDriver::Project(std::complex<Real> **fv) {
  FFTBlock *pfb = pmy_fb;
  Project(fv, fv_sh_, fv_co_);
  for (int nv=0; nv<3; nv++) {
    for (int kidx=0; kidx<pfb->cnt_; kidx++) {
      fv[nv][kidx] = fv_sh_[nv][kidx];
    }
  }
}

//----------------------------------------------------------------------------------------
//! \fn void RandFieldDriver::Project(std::complex<Real> **fv,
//!                                    std::complex<Real> **fv_sh,
//!                                    std::complex<Real> **fv_co)
//! \brief calculates shear and compressible components
void RandFieldDriver::Project(std::complex<Real> **fv, std::complex<Real> **fv_sh,
                               std::complex<Real> **fv_co) {
  FFTBlock *pfb = pmy_fb;
  AthenaFFTIndex *idx = pfb->b_in_;
  int knx1 = pfb->knx[0], knx2 = pfb->knx[1], knx3 = pfb->knx[2];

  for (int k=0; k<knx3; k++) {
    for (int j=0; j<knx2; j++) {
      for (int i=0; i<knx1; i++) {
        // Get khat
        std::int64_t nx = GetKcomp(i, pfb->kdisp[0], pfb->kNx[0]);
        std::int64_t ny = GetKcomp(j, pfb->kdisp[1], pfb->kNx[1]);
        std::int64_t nz = GetKcomp(k, pfb->kdisp[2], pfb->kNx[2]);
        Real kx = nx*pfb->dkx[0];
        Real ky = ny*pfb->dkx[1];
        Real kz = nz*pfb->dkx[2];
        Real kmag = std::sqrt(kx*kx+ky*ky+kz*kz);

        std::int64_t kidx = pfb->GetIndex(i, j, k, idx);
        std::int64_t gidx = pfb->GetGlobalIndex(i,j,k);
        if (gidx == 0.0) {
          fv_co[0][kidx] = std::complex<Real>(0,0);
          fv_co[1][kidx] = std::complex<Real>(0,0);
          fv_co[2][kidx] = std::complex<Real>(0,0);

          fv_sh[0][kidx] = std::complex<Real>(0,0);
          fv_sh[1][kidx] = std::complex<Real>(0,0);
          fv_sh[2][kidx] = std::complex<Real>(0,0);
        } else {
          kx /= kmag;
          ky /= kmag;
          kz /= kmag;
          // Form (khat.f)
          std::complex<Real> kdotf = kx*fv[0][kidx] + ky*fv[1][kidx] + kz*fv[2][kidx];

          fv_co[0][kidx] = kdotf * kx;
          fv_co[1][kidx] = kdotf * ky;
          fv_co[2][kidx] = kdotf * kz;

          fv_sh[0][kidx] = fv[0][kidx] - fv_co[0][kidx];
          fv_sh[1][kidx] = fv[1][kidx] - fv_co[1][kidx];
          fv_sh[2][kidx] = fv[2][kidx] - fv_co[2][kidx];
        }
      }
    }
  }
  return;
}


//----------------------------------------------------------------------------------------
//! \fn void RandFieldDriver::Perturb()
//! \brief Add magnetic perturbation to the field variables

void RandFieldDriver::Perturb() {
  Mesh *pm = pmy_mesh_;
  int nbs = nslist_[Globals::my_rank];
  int nbe = nbs+nblist_[Globals::my_rank]-1;

  int il = pm->my_blocks(0)->is-1, iu = pm->my_blocks(0)->ie+1;
  int jl = pm->my_blocks(0)->js-1, ju = pm->my_blocks(0)->je+1;
  int kl = pm->my_blocks(0)->ks-1, ku = pm->my_blocks(0)->ke+1;

  Real aa, b, c, s, de, b1, b2, b3, B1, B2, B3;
  Real m[3] = {0};
  // AthenaArray<Real> &dA1 = deltaA[0], &dA2 = deltaA[1], &dA3 = deltaA[2];
  AthenaArray<Real> &dB1 = deltaB[0], &dB2 = deltaB[1], &dB3 = deltaB[2];
  // Real dA1dx2, dA1dx3, dA2dx1, dA2dx3, dA3dx1, dA3dx2;
  // Real dx1,dx2,dx3;
  // Calculate B field from vector potential perturbations A
  // for (int nb=0; nb<pm->nblocal; ++nb) {
  //   MeshBlock *pmb = pm->my_blocks(nb);
  //   for (int k=kl; k<=ku; k++) {
  //     for (int j=jl; j<=ju; j++) {
  //       for (int i=il; i<=iu; i++) {
  //         dx1 = pmb->pcoord->GetEdge1Length(k,j,i);
  //         dx2 = pmb->pcoord->GetEdge2Length(k,j,i);
  //         dx3 = pmb->pcoord->GetEdge3Length(k,j,i);
  //         dA1dx2 = (dA1(nb,k,j+1,i) - dA1(nb,k,j,i))/dx2;
  //         dA1dx3 = (dA1(nb,k+1,j,i) - dA1(nb,k,j,i))/dx3;
  //         dA2dx1 = (dA2(nb,k,j,i+1) - dA2(nb,k,j,i))/dx1;
  //         dA2dx3 = (dA2(nb,k+1,j,i) - dA2(nb,k,j,i))/dx3;
  //         dA3dx1 = (dA3(nb,k,j,i+1) - dA3(nb,k,j,i))/dx1;
  //         dA3dx2 = (dA3(nb,k,j+1,i) - dA3(nb,k,j,i))/dx2;
  //         dB1(nb,k,j,i) = dA2dx3 - dA3dx2; 
  //         dB2(nb,k,j,i) = dA3dx1 - dA1dx3; 
  //         dB3(nb,k,j,i) = dA1dx2 - dA2dx1; 
  //       }
  //     }
  //   }
  // }
  for (int nb=0; nb<pm->nblocal; ++nb) {
    MeshBlock *pmb = pm->my_blocks(nb);
    for (int k=kl; k<=ku; k++) {
      for (int j=jl; j<=ju; j++) {
        for (int i=il; i<=iu; i++) {
          m[0] += dB1(nb,k,j,i);
          m[1] += dB2(nb,k,j,i);
          m[2] += dB3(nb,k,j,i);
        }
      }
    }
  }

#ifdef MPI_PARALLEL
  int mpierr;
  // Sum the perturbations over all processors
  mpierr = MPI_Allreduce(MPI_IN_PLACE, m, 3, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
  if (mpierr) {
    std::stringstream msg;
    msg << "[normalize]: MPI_Allreduce error = " << mpierr << std::endl;
    ATHENA_ERROR(msg);
  }
#endif // MPI_PARALLEL

  for (int nb=0; nb<nmb; nb++) {
    for (int k=kl; k<=ku; k++) {
      for (int j=jl; j<=ju; j++) {
        for (int i=il; i<=iu; i++) {
          dB1(nb,k,j,i) -= m[0];
          dB2(nb,k,j,i) -= m[1];
          dB3(nb,k,j,i) -= m[2];
        }
      }
    }
  }

  // Calculate unscaled energy of perturbations
  m[0] = 0.0;
  m[1] = 0.0; 
  for (int nb=0; nb<pm->nblocal; ++nb) {
    MeshBlock *pmb = pm->my_blocks(nb);
    for (int k=kl; k<=ku; k++) {
      for (int j=jl; j<=ju; j++) {
        for (int i=il; i<=iu; i++) {
          b1 = dB1(nb,k,j,i);
          b2 = dB2(nb,k,j,i);
          b3 = dB3(nb,k,j,i);
          B1 = pmb->pfield->bcc(IB1,k,j,i);
          B2 = pmb->pfield->bcc(IB2,k,j,i);
          B3 = pmb->pfield->bcc(IB3,k,j,i);
          m[0] += SQR(b1) + SQR(b2) + SQR(b3);
          m[1] += B1*b1 + B2*b2 + B3*b3;
        }
      }
    }
  }

#ifdef MPI_PARALLEL
  // Sum the perturbations over all processors
  mpierr = MPI_Allreduce(MPI_IN_PLACE, m, 2, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
  if (mpierr) {
    std::stringstream msg;
    msg << "[normalize]: MPI_Allreduce error = "
        << mpierr << std::endl;
    ATHENA_ERROR(msg);
  }
#endif // MPI_PARALLEL


  // decaying turbulence (all in one shot)
  de = dB;
  
  aa = 0.5*m[0];
  aa = std::max(aa,static_cast<Real>(1.0e-20));
  b = m[1];
  c = -de/dvol;
  if (b >= 0.0)
    s = (-2.0*c)/(b + std::sqrt(b*b - 4.0*aa*c));
  else
    s = (-b + std::sqrt(b*b - 4.0*aa*c))/(2.0*aa);

  if (std::isnan(s)) std::cout << "[perturb]: s is NaN!" << std::endl;

  // Apply momentum pertubations
  for (int nb=0; nb<pm->nblocal; ++nb) {
    MeshBlock *pmb = pm->my_blocks(nb);
    for (int k=kl; k<=ku; k++) {
      for (int j=jl; j<=ju; j++) {
        for (int i=il; i<=iu; i++) {
          
          B1 = pmb->pfield->bcc(IB1,k,j,i);
          B2 = pmb->pfield->bcc(IB2,k,j,i);
          B3 = pmb->pfield->bcc(IB3,k,j,i); 
          b1 = dB1(nb,k,j,i);
          b2 = dB2(nb,k,j,i);
          b3 = dB3(nb,k,j,i);
          
          if (NON_BAROTROPIC_EOS) {
            // pmb->phydro->u(IEN,k,j,i) -= 0.5*(SQR(B1) + SQR(B2) + SQR(B3));
            pmb->phydro->u(IEN,k,j,i) += s*(b1*B1 + b2*B2 + b3*B3)+0.5*s*s*(SQR(b1) + SQR(b2) + SQR(b3));
          }
          pmb->pfield->bcc(IB1,k,j,i) += s*b1;
          pmb->pfield->bcc(IB2,k,j,i) += s*b2;
          pmb->pfield->bcc(IB3,k,j,i) += s*b3;
        }
      }
    }
    for (int k=kl; k<=ku; ++k) {
      for (int j=jl; j<=ju; ++j) {
        for (int i=il; i<=iu+1; ++i) {
          pmb->pfield->b.x1f(k,j,i) +=0.5*s*(dB1(nb,k,j,i) + dB1(nb,k,j,i+1));
        }
      }
    }
    for (int k=kl; k<=ku; ++k) {
      for (int j=jl; j<=ju+1; ++j) {
        for (int i=il; i<=iu; ++i) {
          pmb->pfield->b.x2f(k,j,i) +=0.5*s*(dB2(nb,k,j,i) + dB2(nb,k,j+1,i)) ;
        }
      }
    }
    for (int k=kl; k<=ku+1; ++k) {
      for (int j=jl; j<=ju; ++j) {
        for (int i=il; i<=iu; ++i) {
          pmb->pfield->b.x3f(k,j,i) += 0.5*s*(dB3(nb,k,j,i) + dB3(nb,k+1,j,i));
        }
      }
    }
    
    pmb->pfield->CalculateCellCenteredField(pmb->pfield->b,pmb->pfield->bcc,pmb->pcoord,pmb->is,pmb->ie,pmb->js,pmb->je,pmb->ks,pmb->ke);
    // for (int k=kl; k<=ku; k++) {
    //   for (int j=jl; j<=ju; j++) {
    //     for (int i=il; i<=iu; i++) {
    //       B1 = pmb->pfield->bcc(IB1,k,j,i);
    //       B2 = pmb->pfield->bcc(IB2,k,j,i);
    //       B3 = pmb->pfield->bcc(IB3,k,j,i);
    //       if (NON_BAROTROPIC_EOS) {
    //         pmb->phydro->u(IEN,k,j,i) += 0.5*(SQR(B1) + SQR(B2) + SQR(B3));
    //       }
    //     }
    //   }
    // }
  }
  
  
  Real divB1, divB2, divB3;
  m[0] = 0.0;
  for (int nb=0; nb<pm->nblocal; ++nb) {
    MeshBlock *pmb = pm->my_blocks(nb);
    for (int k=kl; k<=ku; k++) {
      for (int j=jl; j<=ju; j++) {
        for (int i=il; i<=iu; i++) {
          divB1 = pmb->pfield->bcc(IB1,k,j,i+1) - pmb->pfield->bcc(IB1,k,j,i);
          divB2 = pmb->pfield->bcc(IB2,k,j+1,i) - pmb->pfield->bcc(IB2,k,j,i);
          divB3 = pmb->pfield->bcc(IB3,k+1,j,i) - pmb->pfield->bcc(IB3,k,j,i);
          m[0] += fabs(divB1+divB2+divB3);
        }
      }
    }
  }
#ifdef MPI_PARALLEL
  // Sum the perturbations over all processors
  mpierr = MPI_Allreduce(MPI_IN_PLACE, m, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
  if (mpierr) {
    std::stringstream msg;
    msg << "[normalize]: MPI_Allreduce error = " << mpierr << std::endl;
    ATHENA_ERROR(msg);
  }
#endif // MPI_PARALLEL
  std::cout << "[RandField]: sum(|divB|)  = " <<  m[0] << std::endl;

  return;
}


//----------------------------------------------------------------------------------------
//! \fn void RandFieldDriver::GetKcomp(int idx, int disp, int Nx)
//! \brief Get k index, which runs from 0, 1, ... Nx/2-1, -Nx/2, -Nx/2+1, ..., -1.

std::int64_t RandFieldDriver::GetKcomp(int idx, int disp, int Nx) {
  return ((idx+disp) - static_cast<std::int64_t>(2*(idx+disp)/Nx)*Nx);
}
