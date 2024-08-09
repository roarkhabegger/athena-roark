//======================================================================================
/* Athena++ astrophysical MHD code
 * Copyright (C) 2014 James M. Stone  <jmstone@princeton.edu>
 *
 * This program is free software: you can redistribute and/or modify it under the terms
 * of the GNU General Public License (GPL) as published by the Free Software Foundation,
 * either version 3 of the License, or (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful, but WITHOUT ANY
 * WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A
 * PARTICULAR PURPOSE.  See the GNU General Public License for more details.
 *
 * You should have received a copy of GNU GPL in the file LICENSE included in the code
 * distribution.  If not see <http://www.gnu.org/licenses/>.
 *====================================================================================*/

// C++ headers
#include <algorithm>  // min
#include <cmath>      // sqrt
#include <fstream>
#include <iostream>   // endl
#include <sstream>    // stringstream
#include <stdexcept>  // runtime_error
#include <string>     // c_str()
#include <random>     // distributions
#include <cfloat>      // FLT_MAX
#include <vector> 
#include <chrono>

// Athena++ headers
#include "../athena.hpp"
#include "../athena_arrays.hpp"
#include "../bvals/bvals.hpp"
#include "../coordinates/coordinates.hpp"
#include "../cr/cr.hpp"
#include "../cr/integrators/cr_integrators.hpp"
#include "../eos/eos.hpp"
#include "../field/field.hpp"
#include "../globals.hpp"
#include "../hydro/hydro.hpp"
#include "../hydro/srcterms/hydro_srcterms.hpp"
#include "../mesh/mesh.hpp"
#include "../parameter_input.hpp"
 
const Real v_scale = 97779222.16807891 ; //Scale constants in CGS
const Real l_scale = 3.0856775814913673e18;
const Real e_scale = 1.5991564026460636e-08;
const Real k_B = 1.380649e-16;

Real r0;
Real L0;
Real tf;
Real ti;
Real inv_crLosstime;
int dim;


Real sigmaParl, sigmaPerp; //CR diffusion 
                           //decouple parallel (to local B field) and perpendicular diffusion coefficients

void Diffusion(MeshBlock *pmb, AthenaArray<Real> &u_cr,
                AthenaArray<Real> &prim, AthenaArray<Real> &bcc);

void CRSource(MeshBlock *pmb, const Real time, const Real dt,
                const AthenaArray<Real> &prim, FaceField &b, 
                AthenaArray<Real> &u_cr);

// void NTFInnerCRX1(MeshBlock *pmb, Coordinates *pco, CosmicRay *pcr,
//                 const AthenaArray<Real> &w, FaceField &b,
//                 AthenaArray<Real> &u_cr, Real time, Real dt, int is, int ie,
//                 int js, int je, int ks, int ke, int ngh);

// void NTFInnerX1(MeshBlock *pmb, Coordinates *pco, AthenaArray<Real> &prim, FaceField &b,
//                 Real time, Real dt,
//                 int il, int iu, int jl, int ju, int kl, int ku, int ngh);

void MeshBlock::InitUserMeshBlockData(ParameterInput *pin) {
  if (CR_ENABLED) {
    pcr->EnrollOpacityFunction(Diffusion);
    inv_crLosstime = pin->GetOrAddReal("problem","inv_crLosstime",0.0);
    pcr->EnrollUserCRSource(CRSource);
  }
  return;
}

void Mesh::InitUserMeshData(ParameterInput *pin) {

  if(CR_ENABLED){
    //Load CR Variables
    Real vmax = pin->GetReal("cr","vmax") ;
    Real kappaPerp = pin->GetOrAddReal("cr","kappaPerp",3e28)/(v_scale*l_scale) ;
    Real kappaParl = pin->GetOrAddReal("cr","kappaParl",3e28)/(v_scale*l_scale) ;
    sigmaPerp = vmax / (3*kappaPerp);
    sigmaParl = vmax / (3*kappaParl);
    r0 = pin->GetReal("problem","r0") ;
    ti = pin->GetOrAddReal("problem","ti",0.0) ;
    tf = pin->GetReal("problem","tf");
    L0 = pin->GetReal("problem","L0");

    if (pin->GetInteger("mesh", "nx2") == 1) {
      dim = 1;
    } else if (pin->GetInteger("mesh", "nx3") == 1) {
      dim = 2;
    } else {
      dim = 3;
    }
    
    //vcr_inj = pin->GetReal("problem","vcr_inj");

    // Real T0 = pin->GetReal("problem", "T0"); // now T is in K
    // Real n0 = pin->GetReal("problem", "n0"); // midplane particles per ccm
    // Real gm1 = pin->GetReal("hydro","gamma") -1;
    // Real P0 = (n0*k_B*T0)/e_scale;
  }
  // if (mesh_bcs[BoundaryFace::inner_x1] == BoundaryFlag::user) {
  //     EnrollUserBoundaryFunction(BoundaryFace::inner_x1, NTFInnerX1);
  //     EnrollUserCRBoundaryFunction(BoundaryFace::inner_x1, NTFInnerCRX1);
  // }  
  return;
}

//======================================================================================
//! \fn void MeshBlock::ProblemGenerator(ParameterInput *pin)
//  \brief cosmic ray modified turbulence with radiative cooling
//======================================================================================

void MeshBlock::ProblemGenerator(ParameterInput *pin) {
  Mesh *pm = pmy_mesh; 
  Real myGamma = pin->GetReal("hydro","gamma");

  // Load variables
  Real T0 = pin->GetReal("problem", "T0"); // now T is in K
  Real n0 = pin->GetReal("problem", "n0"); // midplane particles per ccm
  Real gm1 = myGamma -1;
  Real P0 = (n0*k_B*T0)/e_scale;
  Real inv_beta = pin->GetReal("problem", "inv_beta"); 
  Real inv_betaCR = pin->GetReal("problem", "inv_betaCR"); 

  Real b0 = sqrt(2*inv_beta*P0);
  // Real maxErr = FLT_MIN;

  // Initialize hydro variable
  for(int k=ks; k<=ke; ++k) {
    for (int j=js; j<=je; ++j) {
      for (int i=is; i<=ie; ++i) {

        phydro->u(IDN, k, j, i) = n0;
        phydro->u(IM1, k, j, i) = 0.0;
        phydro->u(IM2, k, j, i) = 0.0;
        phydro->u(IM3, k, j, i) = 0.0;
        //energy
        if (NON_BAROTROPIC_EOS) {
            phydro->u(IEN, k, j, i) = P0/gm1;
        }
        if (CR_ENABLED) {
            pcr->u_cr(CRE,k,j,i) = 3*inv_betaCR*P0;
            pcr->u_cr(CRF1,k,j,i) = 0.0;
            pcr->u_cr(CRF2,k,j,i) = 0.0;
            pcr->u_cr(CRF3,k,j,i) = 0.0;
        }
      }
    }
  }
  // std::cout << " Max HSE Err = " << maxErr << std::endl;
  //Need to set opactiy sigma in the ghost zones
  if (CR_ENABLED) {
  // Default values are 1/3
    int nz1 = block_size.nx1 + 2*(NGHOST);
    int nz2 = block_size.nx2;
    if (nz2 > 1) nz2 += 2*(NGHOST);
    int nz3 = block_size.nx3;
    if (nz3 > 1) nz3 += 2*(NGHOST);
    for(int k=0; k<nz3; ++k) {
      for(int j=0; j<nz2; ++j) {
        for(int i=0; i<nz1; ++i) {
          pcr->sigma_diff(0,k,j,i) = sigmaParl;
          pcr->sigma_diff(1,k,j,i) = sigmaPerp;
          pcr->sigma_diff(2,k,j,i) = sigmaPerp;
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
          pfield->b.x1f(k,j,i) = b0;
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

//----------------------------------------------------------------------------------------
//! \fn void NTFInnerCRX1()
//  \brief Sets boundary condition on left X boundary (iib) for jet problem

// void NTFInnerCRX1(MeshBlock *pmb, Coordinates *pco, CosmicRay *pcr,
//     const AthenaArray<Real> &w, FaceField &b,
//     AthenaArray<Real> &u_cr, Real time, Real dt, int is, int ie,
//     int js, int je, int ks, int ke, int ngh){
//  for (int k=ks; k<=ke; ++k) {
//     for (int j=js; j<=je; ++j) {
// #pragma omp simd
//       for (int i=1; i<=ngh; ++i) {
//         //Real rad = std::sqrt(SQR(pco->x2v(j)) + SQR(pco->x3v(k)));

//         //if ((rad <= r0) && (time >= ti) && (time <= tf)) { //Use if not 1d
//         if ((time >= ti) && (time <= tf)) {
//           Real ucr = L0 * dt / (pco->GetEdge1Length(k,j,i) * M_PI * SQR(r0));

//           u_cr(CRE,k,j,is-i) = ucr;
//           u_cr(CRF1,k,j,is-i) = 4.0/3.0 * ucr * vcr_inj;
//           u_cr(CRF2,k,j,is-i) = u_cr(CRF2,k,j,is);
//           u_cr(CRF3,k,j,is-i) = u_cr(CRF3,k,j,is);
//         } else {
//           u_cr(CRE,k,j,is-i) = u_cr(CRE,k,j,is);
//           u_cr(CRF1,k,j,is-i) = u_cr(CRF1,k,j,is);
//           u_cr(CRF2,k,j,is-i) = u_cr(CRF2,k,j,is);
//           u_cr(CRF3,k,j,is-i) = u_cr(CRF3,k,j,is);
//         }
        
//       }
//     }
//   }
//   return;
// }

//----------------------------------------------------------------------------------------
//! \fn void NTFInnerX1()
//  \brief Sets boundary condition on left X boundary (iib) for jet problem

// void NTFInnerX1(MeshBlock *pmb, Coordinates *pco, AthenaArray<Real> &prim, FaceField &b,
//                 Real time, Real dt,
//                 int il, int iu, int jl, int ju, int kl, int ku, int ngh) {
//   // set primitive variables in inlet ghost zones
//   for (int k=kl; k<=ku; ++k) {
//     for (int j=jl; j<=ju; ++j) {
// #pragma omp simd
//       for (int i=1; i<=ngh; ++i) {
//         prim(IDN,k,j,il-i) = prim(IDN,k,j,il);
//         prim(IVX,k,j,il-i) = prim(IVX,k,j,il);
//         prim(IVY,k,j,il-i) = prim(IVY,k,j,il);
//         prim(IVZ,k,j,il-i) = prim(IVZ,k,j,il);
//         prim(IPR,k,j,il-i) = prim(IPR,k,j,il);
//       }
//     }
//   }

//   // set magnetic field in inlet ghost zones
//   if (MAGNETIC_FIELDS_ENABLED) {
//     for (int k=kl; k<=ku; ++k) {
//       for (int j=jl; j<=ju; ++j) {
// #pragma omp simd
//         for (int i=1; i<=ngh; ++i) {
//           b.x1f(k,j,il-i) = b.x1f(k,j,il);
//         }
//       }
//     }

//     for (int k=kl; k<=ku; ++k) {
//       for (int j=jl; j<=ju+1; ++j) {
// #pragma omp simd
//         for (int i=1; i<=ngh; ++i) {
//           b.x2f(k,j,il-i) = b.x2f(k,j,il);
//         }
//       }
//     }

//     for (int k=kl; k<=ku+1; ++k) {
//       for (int j=jl; j<=ju; ++j) {
// #pragma omp simd
//         for (int i=1; i<=ngh; ++i) {
//           b.x3f(k,j,il-i) = b.x3f(k,j,il);
//         }
//       }
//     }
//   }
// }


void Diffusion(MeshBlock *pmb, AthenaArray<Real> &u_cr,
               AthenaArray<Real> &prim, AthenaArray<Real> &bcc) {
  // set the default opacity to be a large value in the default hydro case
  CosmicRay *pcr=pmb->pcr;
  int kl=pmb->ks, ku=pmb->ke;
  int jl=pmb->js, ju=pmb->je;
  int il=pmb->is-1, iu=pmb->ie+1;
  if (pmb->block_size.nx2 > 1) {
    jl -= 1;
    ju += 1;
  }
  if (pmb->block_size.nx3 > 1) {
    kl -= 1;
    ku += 1;
  }

  for(int k=kl; k<=ku; ++k) {
    for(int j=jl; j<=ju; ++j) {
#pragma omp simd
      for(int i=il; i<=iu; ++i) {
        pcr->sigma_diff(0,k,j,i) = sigmaParl; //sigma_diff is defined with x0 coordinate parallel to local B field
        pcr->sigma_diff(1,k,j,i) = sigmaPerp;
        pcr->sigma_diff(2,k,j,i) = sigmaPerp;
      }
    }
  }

  Real invlim=1.0/pcr->vmax;

  // The information stored in the array
  // b_angle is
  // b_angle[0]=sin_theta_b
  // b_angle[1]=cos_theta_b
  // b_angle[2]=sin_phi_b
  // b_angle[3]=cos_phi_b




  if (MAGNETIC_FIELDS_ENABLED) {
    //First, calculate B_dot_grad_Pc
    for(int k=kl; k<=ku; ++k) {
      for(int j=jl; j<=ju; ++j) {
        // x component
        pmb->pcoord->CenterWidth1(k,j,il-1,iu+1,pcr->cwidth);
        for(int i=il; i<=iu; ++i) {
          Real distance = 0.5*(pcr->cwidth(i-1) + pcr->cwidth(i+1))
                          + pcr->cwidth(i);
          Real dprdx=(u_cr(CRE,k,j,i+1) - u_cr(CRE,k,j,i-1))/3.0;
          dprdx /= distance;
          pcr->b_grad_pc(k,j,i) = bcc(IB1,k,j,i) * dprdx;
        }
        // y component
        pmb->pcoord->CenterWidth2(k,j-1,il,iu,pcr->cwidth1);
        pmb->pcoord->CenterWidth2(k,j,il,iu,pcr->cwidth);
        pmb->pcoord->CenterWidth2(k,j+1,il,iu,pcr->cwidth2);
        for(int i=il; i<=iu; ++i) {
          Real distance = 0.5*(pcr->cwidth1(i) + pcr->cwidth2(i))
                         + pcr->cwidth(i);
          Real dprdy=(u_cr(CRE,k,j+1,i) - u_cr(CRE,k,j-1,i))/3.0;
          dprdy /= distance;
          pcr->b_grad_pc(k,j,i) += bcc(IB2,k,j,i) * dprdy;
        }
        // z component
        pmb->pcoord->CenterWidth3(k-1,j,il,iu,pcr->cwidth1);
        pmb->pcoord->CenterWidth3(k,j,il,iu,pcr->cwidth);
        pmb->pcoord->CenterWidth3(k+1,j,il,iu,pcr->cwidth2);

        for(int i=il; i<=iu; ++i) {
          Real distance = 0.5*(pcr->cwidth1(i) + pcr->cwidth2(i))
                          + pcr->cwidth(i);
          Real dprdz=(u_cr(CRE,k+1,j,i) - u_cr(CRE,k-1,j,i))/3.0;
          dprdz /= distance;
          pcr->b_grad_pc(k,j,i) += bcc(IB3,k,j,i) * dprdz;

          // now only get the sign
          //  if (pcr->b_grad_pc(k,j,i) > TINY_NUMBER) pcr->b_grad_pc(k,j,i) = 1.0;
          //  else if (-pcr->b_grad_pc(k,j,i) > TINY_NUMBER) pcr->b_grad_pc(k,j,i)
          //    = -1.0;
          //  else pcr->b_grad_pc(k,j,i) = 0.0;
        }

      // now calculate the streaming velocity
      // streaming velocity is calculated with respect to the current coordinate
      //  system
      // diffusion coefficient is calculated with respect to B direction
        for(int i=il; i<=iu; ++i) {
          Real pb= bcc(IB1,k,j,i)*bcc(IB1,k,j,i)
                  +bcc(IB2,k,j,i)*bcc(IB2,k,j,i)
                  +bcc(IB3,k,j,i)*bcc(IB3,k,j,i);
          Real inv_sqrt_rho = 1.0/std::sqrt(prim(IDN,k,j,i));
          Real va1 = bcc(IB1,k,j,i)*inv_sqrt_rho;
          Real va2 = bcc(IB2,k,j,i)*inv_sqrt_rho;
          Real va3 = bcc(IB3,k,j,i)*inv_sqrt_rho;

          Real va = std::sqrt(pb/prim(IDN,k,j,i));

          Real dpc_sign = 0.0;
          if (pcr->b_grad_pc(k,j,i) > TINY_NUMBER) dpc_sign = 1.0;
          else if (-pcr->b_grad_pc(k,j,i) > TINY_NUMBER) dpc_sign = -1.0;

          pcr->v_adv(0,k,j,i) = -va1 * dpc_sign;
          pcr->v_adv(1,k,j,i) = -va2 * dpc_sign;
          pcr->v_adv(2,k,j,i) = -va3 * dpc_sign;

          // now the diffusion coefficient

          if (va < TINY_NUMBER) {
            pcr->sigma_adv(0,k,j,i) = pcr->max_opacity;
          } else {
            pcr->sigma_adv(0,k,j,i) = std::abs(pcr->b_grad_pc(k,j,i))
                          /(std::sqrt(pb)* va * (1.0 + 1.0/3.0)
                                    * invlim * u_cr(CRE,k,j,i));
          }

          pcr->sigma_adv(1,k,j,i) = pcr->max_opacity;
          pcr->sigma_adv(2,k,j,i) = pcr->max_opacity;

          // Now calculate the angles of B
          Real bxby = std::sqrt(bcc(IB1,k,j,i)*bcc(IB1,k,j,i) +
                           bcc(IB2,k,j,i)*bcc(IB2,k,j,i));
          Real btot = std::sqrt(pb);
          if (btot > TINY_NUMBER) {
            pcr->b_angle(0,k,j,i) = bxby/btot;
            pcr->b_angle(1,k,j,i) = bcc(IB3,k,j,i)/btot;
          } else {
            pcr->b_angle(0,k,j,i) = 1.0;
            pcr->b_angle(1,k,j,i) = 0.0;
          }
          if (bxby > TINY_NUMBER) {
            pcr->b_angle(2,k,j,i) = bcc(IB2,k,j,i)/bxby;
            pcr->b_angle(3,k,j,i) = bcc(IB1,k,j,i)/bxby;
          } else {
            pcr->b_angle(2,k,j,i) = 0.0;
            pcr->b_angle(3,k,j,i) = 1.0;
          }
        }
      }
    }
  } else {
  for(int k=kl; k<=ku; ++k) {
    for(int j=jl; j<=ju; ++j) {
  // x component
      pmb->pcoord->CenterWidth1(k,j,il-1,iu+1,pcr->cwidth);
      for(int i=il; i<=iu; ++i) {
         Real distance = 0.5*(pcr->cwidth(i-1) + pcr->cwidth(i+1))
                        + pcr->cwidth(i);
         Real grad_pr=(u_cr(CRE,k,j,i+1) - u_cr(CRE,k,j,i-1))/3.0;
         grad_pr /= distance;
         Real va = 0.0;
         if (va < TINY_NUMBER) {
           pcr->sigma_adv(0,k,j,i) = pcr->max_opacity;
           pcr->v_adv(0,k,j,i) = 0.0;
         } else {
           Real sigma2 = std::abs(grad_pr)/(va * (1.0 + 1.0/3.0)
                             * invlim * u_cr(CRE,k,j,i));
           if (std::abs(grad_pr) < TINY_NUMBER) {
             pcr->sigma_adv(0,k,j,i) = 0.0;
             pcr->v_adv(0,k,j,i) = 0.0;
           } else {
             pcr->sigma_adv(0,k,j,i) = sigma2;
             pcr->v_adv(0,k,j,i) = -va * grad_pr/std::abs(grad_pr);
           }
        }
        pcr->sigma_adv(1,k,j,i) = pcr->max_opacity;
        pcr->sigma_adv(2,k,j,i) = pcr->max_opacity;

        pcr->v_adv(1,k,j,i) = 0.0;
        pcr->v_adv(2,k,j,i) = 0.0;
      }
    }
  }
  }
}

void CRSource(MeshBlock *pmb, const Real time, const Real dt,
                const AthenaArray<Real> &prim, FaceField &b, 
                AthenaArray<Real> &u_cr){ 
  if ((COORDINATE_SYSTEM == "cylindrical") && (dim == 3)) {
    for (int k=pmb->ks; k<=pmb->ke; ++k) {
      for (int j=pmb->js; j<=pmb->je; ++j) {
    #pragma omp simd
        for (int i=pmb->is; i<=pmb->ie; ++i) {
          //CRLoss Term
          u_cr(CRE,k,j,i) -= inv_crLosstime * dt * u_cr(CRE,k,j,i);

          if ((pmb->pcoord->x3f(k) == 0) && (time >= ti) && (time <= tf) && (pmb->pcoord->x1v(i) < r0)) {
            //CRSource Term
            Real rad_percent = std::min(1.0, std::max(0.0, (SQR(r0) - SQR(pmb->pcoord->x1f(i))) / pmb->pcoord->dx1f(i))); //Returns percentage of cylindrical cell within r0.
            u_cr(CRE,k,j,i) += L0 * dt * rad_percent / (pmb->pcoord->GetEdge1Length(k,j,i) * M_PI * SQR(r0));
          }
        }
      }
    }
  } else if ((COORDINATE_SYSTEM == "cartesian") && (dim == 1)) {
    for (int k=pmb->ks; k<=pmb->ke; ++k) {
      for (int j=pmb->js; j<=pmb->je; ++j) {
    #pragma omp simd
        for (int i=pmb->is; i<=pmb->ie; ++i) {
          //CRLoss Term
          u_cr(CRE,k,j,i) -= inv_crLosstime * dt * u_cr(CRE,k,j,i);

          if ((pmb->pcoord->x1f(i) == 0) && (time >= ti) && (time <= tf)) {
            //CRSource Term
            u_cr(CRE,k,j,i) += L0 * dt / pmb->pcoord->GetCellVolume(k,j,i);
          }
        }
      }
    }
  } else if ((COORDINATE_SYSTEM == "cartesian") && (dim == 3)) { //NOT FINISHED
    for (int k=pmb->ks; k<=pmb->ke; ++k) {
      for (int j=pmb->js; j<=pmb->je; ++j) {
    #pragma omp simd
        for (int i=pmb->is; i<=pmb->ie; ++i) {
          //CRLoss Term
          u_cr(CRE,k,j,i) -= inv_crLosstime * dt * u_cr(CRE,k,j,i);

          Real rad = std::sqrt(SQR(pmb->pcoord->x2v(j)) + SQR(pmb->pcoord->x3v(k)));
          Real rad_percent = 0.0;
          if ((pmb->pcoord->x1f(i) == 0) && (rad <= r0) && (time >= ti) && (time <= tf)) {
            //CRSource Term
            u_cr(CRE,k,j,i) += L0 * dt / (pmb->pcoord->GetEdge1Length(k,j,i) * 4 * SQR(r0)); //4 if Cartesian, M_PI if Cylindrical
          }
        }
      }
    }
  }
  return;
}