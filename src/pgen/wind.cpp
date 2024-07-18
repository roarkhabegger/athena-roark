//========================================================================================
// Athena++ astrophysical MHD code
// Copyright(C) 2014 James M. Stone <jmstone@princeton.edu> and other code contributors
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
//! \file wind.cpp
//! \brief Problem generator for building spherical wind from multiple shocks
//!
//! Problem generator for shock tube (1-D Riemann) problems. Initializes plane-parallel
//! shock along x1 (in 1D, 2D, 3D), along x2 (in 2D, 3D), and along x3 (in 3D).
//========================================================================================

// C headers

// C++ headers
#include <cmath>      // sqrt()
#include <cstdio>     // fopen(), freopen(), fprintf(), fclose()
#include <iostream>   // endl
#include <sstream>    // stringstream
#include <stdexcept>  // runtime_error
#include <string>

// Athena++ headers
#include "../athena.hpp"
#include "../athena_arrays.hpp"
#include "../coordinates/coordinates.hpp"
#include "../eos/eos.hpp"
#include "../field/field.hpp"
#include "../hydro/hydro.hpp"
#include "../mesh/mesh.hpp"
#include "../parameter_input.hpp"
#include "../scalars/scalars.hpp"

Real P0,rho0;

Real r0, tau;

Real E1,M1;

Real E2,t2,M2;

void ProjectAndDriveInnerX1(MeshBlock *pmb, Coordinates *pco, AthenaArray<Real> &prim,
                            FaceField &b, Real time, Real dt,
                            int il, int iu, int jl, int ju, int kl, int ku, int ngh);

void ShockSource(MeshBlock *pmb, const Real time, const Real dt,
               const AthenaArray<Real> &prim, const AthenaArray<Real> &prim_scalar,
               const AthenaArray<Real> &bcc, AthenaArray<Real> &cons,
               AthenaArray<Real> &cons_scalar);

//========================================================================================
//! \fn void Mesh::InitUserMeshData(ParameterInput *pin)
//  \brief Function to initialize problem-specific data in mesh class.  Can also be used
//  to initialize variables which are global to (and therefore can be passed to) other
//  functions in this file.  Called in Mesh constructor.
//========================================================================================

void Mesh::InitUserMeshData(ParameterInput *pin) {

  // parse ambient, unshocked medium parameters
  rho0 = pin->GetReal("problem","d0");
  P0 = pin->GetReal("problem","p0");

  r0 = pin->GetReal("problem","r0");
  tau = pin->GetReal("problem","tau");

  E1 = pin->GetReal("problem","E1");
  M1 = pin->GetReal("problem","M1");
  E2 = pin->GetReal("problem","E2");
  M2 = pin->GetReal("problem","M2");
  t2 = pin->GetReal("problem","t2");

  EnrollUserExplicitSourceFunction(ShockSource);
  if (pin->GetString("mesh","ix1_bc") == "user") EnrollUserBoundaryFunction(BoundaryFace::inner_x1, ProjectAndDriveInnerX1);
  return;
}

//========================================================================================
//! \fn void MeshBlock::ProblemGenerator(ParameterInput *pin)
//! \brief Problem Generator for the shock tube tests
//========================================================================================

void MeshBlock::ProblemGenerator(ParameterInput *pin) {
  
  Real gm1 = peos->GetGamma() - 1;
  Real Vol = 4*M_PI/3*pow(r0,3);
  // Initialize hydro variable
  for(int k=ks; k<=ke; ++k) {
    for (int j=js; j<=je; ++j) {
      for (int i=is; i<=ie; ++i) {
        phydro->u(IDN, k, j, i) = rho0;
        phydro->u(IM1, k, j, i) = 0.0;
        phydro->u(IM2, k, j, i) = 0.0;
        phydro->u(IM3, k, j, i) = 0.0;
        Real x = pcoord->x1v(i);
        if (NON_BAROTROPIC_EOS) {
            phydro->u(IEN, k, j, i) = P0/gm1;
        }
        Real r = pcoord->x1v(i);
        if ((r <= r0)){
          phydro->u(IDN,k,j,i) +=  M1/Vol ;
          phydro->u(IEN,k,j,i) += (E1/Vol);
        }
      }
    }
  }

  // Add horizontal magnetic field lines, to show streaming and diffusion
  // along magnetic field ines
  if (MAGNETIC_FIELDS_ENABLED) {
    for (int k=ks; k<=ke; ++k) {
      for (int j=js; j<=je; ++j) {
        for (int i=is; i<=ie+1; ++i) {
          pfield->b.x1f(k,j,i) = 0.0;
        }
      }
    }
    if (block_size.nx2 > 1) {
      for (int k=ks; k<=ke; ++k) {
        for (int j=js; j<=je+1; ++j) {
          for (int i=is; i<=ie; ++i) {
            pfield->b.x2f(k,j,i) = 0.0;
          }
        }
      }
    }
    if (block_size.nx3 > 1) {
      for (int k=ks; k<=ke+1; ++k) {
        for (int j=js; j<=je; ++j) {
          for (int i=is; i<=ie; ++i) {
            pfield->b.x3f(k,j,i) = 0.0;
          }
        }
      }
    }

    // set cell centerd magnetic field
    // Add magnetic energy density to the total energy
    pfield->CalculateCellCenteredField(pfield->b,pfield->bcc,pcoord,is,ie,js,je,ks,ke);

    for(int k=ks; k<=ke; ++k) {
      for(int j=js; j<=je; ++j) {
        for(int i=is; i<=ie; ++i) {
          phydro->u(IEN,k,j,i) +=
            0.5*(SQR((pfield->bcc(IB1,k,j,i)))
               + SQR((pfield->bcc(IB2,k,j,i)))
               + SQR((pfield->bcc(IB3,k,j,i))));
        }
      }
    }
  }
  


  return;
}

void ShockSource(MeshBlock *pmb, const Real time, const Real dt,
               const AthenaArray<Real> &prim, const AthenaArray<Real> &prim_scalar,
               const AthenaArray<Real> &bcc, AthenaArray<Real> &cons,
               AthenaArray<Real> &cons_scalar){

  Mesh *pm = pmb->pmy_mesh;
  Real Vol = 4*M_PI/3*pow(r0,3);
  Real gm1 = pmb->peos->GetGamma() - 1;
  if ((time <= t2) && (time+dt > t2) ) {
    for (int k=pmb->ks; k<=pmb->ke; ++k) {
      for (int j=pmb->js; j<=pmb->je; ++j) {
  #pragma omp simd
        for (int i=pmb->is; i<=pmb->ie; ++i) {
          Real r = pmb->pcoord->x1v(i);
          if (r < r0){
            cons(IDN,k,j,i) +=  M2/Vol ;
            cons(IEN,k,j,i) +=  E2/Vol;
          }
        }
      }
    } 
  }

  return;
}

//----------------------------------------------------------------------------------------
//! \fn void ProjectAndDriveInnerX1()
//  \brief Sets boundary condition on left x1 boundary to be double shocked state w2

void ProjectAndDriveInnerX1(MeshBlock *pmb, Coordinates *pco, AthenaArray<Real> &prim, FaceField &b,
                Real time, Real dt,
                int il, int iu, int jl, int ju, int kl, int ku, int ngh) {
  // set primitive variables in inlet ghost zones
  Real Vol = 4*M_PI/3*pow(r0,3);
  Real gm1 = pmb->peos->GetGamma() - 1;
  for (int k=kl; k<=ku; ++k) {
    for (int j=jl; j<=ju; ++j) {
      for (int i=1; i<=ngh; ++i) {
        // if (time <= tau){
        //   prim(IDN,k,j,il-i) = M1/Vol;
        //   prim(IVX,k,j,il-i) = 0;
        //   prim(IVY,k,j,il-i) = 0;
        //   prim(IVZ,k,j,il-i) = 0;
        //   prim(IPR,k,j,il-i) = (E1/Vol)*gm1;
        // } else if ( (time > t2) && (time <= (t2+tau))){
        //   prim(IDN,k,j,il-i) = M2/Vol;
        //   prim(IVX,k,j,il-i) = 0;
        //   prim(IVY,k,j,il-i) = 0;
        //   prim(IVZ,k,j,il-i) = 0;
        //   prim(IPR,k,j,il-i) = (E2/Vol)*gm1;
        // } else {
          for (int n=0; n<NHYDRO; ++n) {
            Real q1 = prim(n,k,j,il);
            Real q2 = prim(n,k,j,il+1);
            Real x1 = pco->x1v(il);
            Real x2 = pco->x1v(il+1);
            Real slope = (q2-q1)/(x2-x1);
            Real x = pco->x1v(il-i);
            prim(n,k,j,il-i) = q1+slope*(x-x1);
            // prim(n,k,j,il-i) = prim(n,k,j,il) ;
          }
        // }
        if (prim(IVX,k,j,il-i) < 0.0) {
          prim(IVX,k,j,il-i) = 0.0;
        }
        pmb->peos->ApplyPrimitiveFloors(prim, k, j, il-i);
      }
    }
  }

  // set magnetic field in inlet ghost zones
  if (MAGNETIC_FIELDS_ENABLED) {
    for (int k=kl; k<=ku; ++k) {
      for (int j=jl; j<=ju; ++j) {
#pragma omp simd
        for (int i=1; i<=ngh; ++i) {
          b.x1f(k,j,il-i) = 0.0;
        }
      }
    }

    for (int k=kl; k<=ku; ++k) {
      for (int j=jl; j<=ju+1; ++j) {
#pragma omp simd
        for (int i=1; i<=ngh; ++i) {
          b.x2f(k,j,il-i) = 0.0;
        }
      }
    }

    for (int k=kl; k<=ku+1; ++k) {
      for (int j=jl; j<=ju; ++j) {
#pragma omp simd
        for (int i=1; i<=ngh; ++i) {
          b.x3f(k,j,il-i) = 0.0;
        }
      }
    }
  }
}
