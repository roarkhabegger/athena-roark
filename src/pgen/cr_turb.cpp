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
#include "../scalars/scalars.hpp"
#include "../parameter_input.hpp"

//======================================================================================
//! \fn void MeshBlock::ProblemGenerator(ParameterInput *pin)
//  \brief cosmic ray modified turbulence with radiative cooling
//======================================================================================


Real sigmaParlHot, sigmaPerpHot, sigmaParlCold, sigmaPerpCold; //CR diffusion 
                           //decouple parallel (to local B field) and perpendicular diffusion coefficients
Real T_decouple;
Real crLoss; //CR Loss term. E.g. Hadronic losses, proportional to local CR energy
             // can be useful to give a decay to CR energy during turbulent driving

int cooling_flag;

void CRSource(MeshBlock *pmb, const Real time, const Real dt,
                const AthenaArray<Real> &prim, FaceField &b, 
              AthenaArray<Real> &u_cr);

void mySource(MeshBlock *pmb, const Real time, const Real dt,
               const AthenaArray<Real> &prim, const AthenaArray<Real> &prim_scalar,
               const AthenaArray<Real> &bcc, AthenaArray<Real> &cons,
               AthenaArray<Real> &cons_scalar);          

void Diffusion(MeshBlock *pmb, AthenaArray<Real> &u_cr,
        AthenaArray<Real> &prim, AthenaArray<Real> &bcc);

void MeshBlock::InitUserMeshBlockData(ParameterInput *pin) {
  if (CR_ENABLED) {
    pcr->EnrollOpacityFunction(Diffusion);
    bool lossFlag = (pin->GetOrAddReal("problem","crLoss",0.0) > 0.0);
    if (lossFlag) {
        pcr->EnrollUserCRSource(CRSource);
    }
  }
}

Real clumping(MeshBlock *pmb, int iout)
{
  int is=pmb->is, ie=pmb->ie, js=pmb->js, je=pmb->je, ks=pmb->ks, ke=pmb->ke;
  Real d = 0;
  for(int k=ks; k<=ke; k++) {
    for(int j=js; j<=je; j++) {
      for(int i=is; i<=ie; i++) {
        if (iout == 0) d += (pmb->phydro->u(IDN,k,j,i));
        if (iout == 1) d += SQR(pmb->phydro->u(IDN,k,j,i));
      }
    }
  }
  return d;
}

void Mesh::InitUserMeshData(ParameterInput *pin) {
  if(CR_ENABLED){
    //Load CR Variables
    Real vmax = pin->GetReal("cr","vmax") ;
    Real kappaPerpHot = pin->GetReal("cr","kappaPerpHot");
    Real kappaParlHot = pin->GetReal("cr","kappaParlHot");
    Real kappaPerpCold = pin->GetReal("cr","kappaPerpCold");
    Real kappaParlCold = pin->GetReal("cr","kappaParlCold"); // 0.9943153210629e5 for 3e28
    T_decouple = pin->GetReal("cr","T_decouple"); // Base is 8.25439974158e1 for 10^4 K
    sigmaPerpHot = vmax/(3*kappaPerpHot);
    sigmaParlHot = vmax/(3*kappaParlHot);
    sigmaPerpCold = vmax/(3*kappaPerpCold);
    sigmaParlCold = vmax/(3*kappaParlCold);
    crLoss = pin->GetOrAddReal("problem","crLoss",0.0);
  }
  cooling_flag = pin->GetInteger("problem","cooling");
  if (cooling_flag != 0) {
    EnrollUserExplicitSourceFunction(mySource);
  }
  AllocateUserHistoryOutput(2);
  EnrollUserHistoryOutput(0, clumping, "d");
  EnrollUserHistoryOutput(1, clumping, "dSqr");

  // turb_flag is initialzed in the Mesh constructor to 0 by default;
  // turb_flag = 1 for decaying turbulence
  // turb_flag = 2 for driven turbulence
  turb_flag = pin->GetInteger("problem","turb_flag");
  if (turb_flag != 0) {
#ifndef FFT
    std::stringstream msg;
    msg << "### FATAL ERROR in TurbulenceDriver::TurbulenceDriver" << std::endl
        << "non zero Turbulence flag is set without FFT!" << std::endl;
    throw std::runtime_error(msg.str().c_str());
    return;
#endif
  }
  return;
}

// sub-cycling method for computing the cooling time step
void mySource(MeshBlock *pmb, const Real time, const Real dt,
               const AthenaArray<Real> &prim, const AthenaArray<Real> &prim_scalar,
               const AthenaArray<Real> &bcc, AthenaArray<Real> &cons,
               AthenaArray<Real> &cons_scalar){
    
  // Assuming units of v = 10^5 cm/s, l = 1 pc, n = 1/cm^3, m = 1 m_p
  // therefore conversions are:
  //         multiply by 5.420598489365e-28 for cgs erg/s
  //         multiply by 5.420598489365e-28 for cgs erg cm^3 /s
  //         multiply by 1.21147513e+02 for K
  const Real Heat    =  3.68962948e+01 ;
  const Real T_floor =  1.65087995e-01 ; 
  // Inoue
  const Real Lamb1_I   =  3.65000000e+05 ; // Note these are Lambda/Gamma Terms
  const Real Lamb2_I   =  3.95000000e-01 ;
  const Real T1a_I     =  9.77320931e+02 ;
  const Real T1b_I     =  1.23815996e+01 ;
  const Real T2_I      =  7.59404778e-01 ;
  // Koyama
  const Real Lamb1_K   =  1.00000000e+07 ;
  const Real Lamb2_K   =  1.54093843e+02 ; // 154.093843316
  const Real T1a_K     =  9.47605092e+02 ;
  const Real T1b_K     =  8.25439976e+00 ;
  const Real T2_K      =  7.59404778e-01 ;
  Real pfloor = pmb->peos->GetPressureFloor();
  Real dfloor = pmb->peos->GetDensityFloor();
  Real      g = pmb->peos->GetGamma();
  const Real k_b = 1.381e-16;
  Real E, E_ergs, T, T0, dEDdt, dEdt;
  AthenaArray<Real> &u = cons;

  Real massRate = 7*1; 
  // the 1 is for a MW like 1 solar mass per yr SFR, in a box with about 1e-6 volume of the MW

  for (int k=pmb->ks; k<=pmb->ke; ++k) {
    for (int j=pmb->js; j<=pmb->je; ++j) {
#pragma omp simd
      for (int i=pmb->is; i<=pmb->ie; ++i) {
        Real d = prim(IDN,k,j,i);
        Real p = prim(IPR,k,j,i);
        if ((d> dfloor) && (p> pfloor) ) {
            Real T = p/d;
            Real Lamb = 0.0;
            if (cooling_flag == 1) {
              Lamb = Lamb1_I*exp(-1*T1a_I/(T + T1b_I)) + Lamb2_I*exp(-1*T2_I/T);
            } else if (cooling_flag == 2) {
              Lamb = Lamb1_K*exp(-1*T1a_K/(T + T1b_K)) + Lamb2_K*sqrt(T)*exp(-1*T2_K/T);
            }
            Real dEdt = ((T > T_floor) ) ? d*Heat*( d*Lamb - 1) : 0.0;
            cons(IEN,k,j,i) -= dEdt*dt;
            if ( T <= T_floor) {
                Real cellVol = pmb->pcoord->GetCellVolume(k, j, i);
                cons_scalar(0,k,j,i) += dt * massRate/cellVol;
            }
        }
      }
    }
  }
  return;
}

void CRSource(MeshBlock *pmb, const Real time, const Real dt,
                const AthenaArray<Real> &prim, FaceField &b, 
                AthenaArray<Real> &u_cr){ 
  for (int k=pmb->ks; k<=pmb->ke; ++k) {
    for (int j=pmb->js; j<=pmb->je; ++j) {
  #pragma omp simd
      for (int i=pmb->is; i<=pmb->ie; ++i) {
        //CRLoss Term
        u_cr(CRE,k,j,i) -= crLoss*dt*u_cr(CRE,k,j,i);
      }
    }
  }
  return;
}

void MeshBlock::ProblemGenerator(ParameterInput *pin) {
  // read in the mean velocity, diffusion coefficient
  const Real nH = pin->GetReal("problem", "nH"); //density
  const Real iso_cs = pin->GetReal("hydro", "iso_sound_speed");
  const Real pres = nH*SQR(iso_cs);
  const Real gm1  = peos->GetGamma() - 1.0;
  
  const Real invbeta = pin->GetOrAddReal("problem","invbeta",0.0);
  const Real b0 = sqrt(2*invbeta*pres); //mean field strength
  const Real angle = (PI/180.0)*pin->GetOrAddReal("problem","angle",0.0);

  const Real invbetaCR = pin->GetOrAddReal("problem","invbetaCR",0.0);
  const Real crp = pres*invbetaCR;

  Real gamma = peos->GetGamma();
    
  // Initialize hydro variable
  for(int k=ks; k<=ke; ++k) {
    for (int j=js; j<=je; ++j) {
      for (int i=is; i<=ie; ++i) {
        Real x1 = pcoord->x1v(i);
        Real x2 = pcoord->x2v(j);
        Real x3 = pcoord->x3v(k);

        phydro->u(IDN, k, j, i) = nH;
        phydro->u(IM1, k, j, i) = 0.0;
        phydro->u(IM2, k, j, i) = 0.0;
        phydro->u(IM3, k, j, i) = 0.0;
        //energy
        if (NON_BAROTROPIC_EOS) {
            phydro->u(IEN, k, j, i) = pres/gm1;
        }
        if (NSCALARS == 1) {
            pscalars->s(0,k,j,i) = 0.0;
        }

        if (CR_ENABLED) {
            pcr->u_cr(CRE,k,j,i) = 3*crp;
            pcr->u_cr(CRF1,k,j,i) = 0.0;
            pcr->u_cr(CRF2,k,j,i) = 0.0;
            pcr->u_cr(CRF3,k,j,i) = 0.0;
        }
      }
    }
  }
  //Need to set opactiy sigma in the ghost zones
  if (CR_ENABLED) {
  // Default values are 1/3
    for(int k=ks; k<=ke; ++k) {
      for (int j=js; j<=je; ++j) {
        for (int i=is; i<=ie; ++i) {
            Real T = pres/nH;
            Real fracCold = 0.0;
            Real width = 0.1 * T_decouple;  //8.25439974158e-1;
            fracCold = 0.5*(1 - std::tanh( (T - T_decouple)/width));
            Real parl = std::exp((std::log(sigmaParlCold) - std::log(sigmaParlHot)) * fracCold + std::log(sigmaParlHot));
            Real perp = std::exp((std::log(sigmaPerpCold) - std::log(sigmaPerpHot)) * fracCold + std::log(sigmaPerpHot));

            pcr->sigma_diff(0,k,j,i) = parl; 
            //sigma_diff is defined with x0 coordinate parallel to local B field
            pcr->sigma_diff(1,k,j,i) = perp;
            pcr->sigma_diff(2,k,j,i) = perp;
            // pcr->sigma_adv(0,k,j,i) = pcr->max_opacity;
            // pcr->sigma_adv(1,k,j,i) = pcr->max_opacity;
            // pcr->sigma_adv(2,k,j,i) = pcr->max_opacity;
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
          pfield->b.x1f(k,j,i) = b0* std::cos(angle);
        }
      }
    }
    if (block_size.nx2 > 1) {
      for (int k=ks; k<=ke; ++k) {
        for (int j=js; j<=je+1; ++j) {
          for (int i=is; i<=ie; ++i) {
            pfield->b.x2f(k,j,i) = b0* std::sin(angle);
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
  Real invlim = 1.0/pcr->vmax;
  Real pfloor = pmb->peos->GetPressureFloor();
  Real dfloor = pmb->peos->GetDensityFloor();
  for(int k=kl; k<=ku; ++k) {
    for(int j=jl; j<=ju; ++j) {
#pragma omp simd
      for(int i=il; i<=iu; ++i) {
        Real d = prim(IDN,k,j,i);
        Real p = prim(IPR,k,j,i);
        Real T = p/d;
        Real width =   0.1 * T_decouple;
        Real fracCold = 0.0;
        if ((d> dfloor) && (p> pfloor) ) {
             fracCold = 0.5*(1 - std::tanh( (T - T_decouple)/width));
        }
        Real parl = std::exp((std::log(sigmaParlCold) - std::log(sigmaParlHot)) * fracCold + std::log(sigmaParlHot));
        Real perp = std::exp((std::log(sigmaPerpCold) - std::log(sigmaPerpHot)) * fracCold + std::log(sigmaPerpHot));
        pcr->sigma_diff(0,k,j,i) = parl; 
        //sigma_diff is defined with x0 coordinate parallel to local B field
        pcr->sigma_diff(1,k,j,i) = perp;
        pcr->sigma_diff(2,k,j,i) = perp;
      }
    }
  }
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
          if (pcr->stream_flag > 0) va = 0;

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

