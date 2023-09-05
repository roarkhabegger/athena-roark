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


//======================================================================================
//! \fn void MeshBlock::ProblemGenerator(ParameterInput *pin)
//  \brief cosmic ray modified turbulence with radiative cooling
//======================================================================================


Real sigmaParl, sigmaPerp; //CR diffusion 
                           //decouple parallel (to local B field) and perpendicular diffusion coefficients
Real crLoss; //CR Loss term. E.g. Hadronic losses, proportional to local CR energy
             // can be useful to give a decay to CR energy during turbulent driving

Real nGrav, h, g0, pres0, dens0, alpha, beta;

int cooling_flag;
int HSE_CR_Forcing;
int HSE_Gamma;
int uniformInj;

std::vector<double> X1Inj = {};
std::vector<double> X2Inj = {};
std::vector<double> X3Inj = {};
int NInjs = 0;
int TotalInjs = 0;
// double lastInjT = 0.0;
double SNRate = 0.0;
double injH = 100;
double Esn_cr = 0.0;
double Esn_th = 0.0;
double injL = 0.0;
double StopT = -1.0;

double phasesX[5];
double phasesY[5];
double amplitudes[5];

// All in cgs units
const Real k_B = 1.380649e-16;
const Real M_sun = 1.98840987e+33;
const Real G = 6.6743e-08;
const Real c = 2.99792458e+10;
const Real l_scale = 3.08568e+18;
const Real t_scale = 3.15576e+13;
const Real n_scale = 1; 
const Real m_scale = 1.67262192e-24;
const Real v_scale = l_scale/t_scale;
const Real rho_scale = m_scale*n_scale;
const Real e_scale = rho_scale*v_scale*v_scale;
const Real T_scale = e_scale/(n_scale*k_B);
const Real B_scale = 4*PI*sqrt(e_scale);

void CRSource(MeshBlock *pmb, const Real time, const Real dt,
                const AthenaArray<Real> &prim, FaceField &b, 
              AthenaArray<Real> &u_cr);

void mySource(MeshBlock *pmb, const Real time, const Real dt,
               const AthenaArray<Real> &prim, const AthenaArray<Real> &prim_scalar,
               const AthenaArray<Real> &bcc, AthenaArray<Real> &cons,
               AthenaArray<Real> &cons_scalar);
              

void Diffusion(MeshBlock *pmb, AthenaArray<Real> &u_cr,
        AthenaArray<Real> &prim, AthenaArray<Real> &bcc);

//Floors for Diode boundary conds
Real dfloor, pfloor; // Floor values for density and rpessure

//x2 boundaries with vacuum
void DiodeInnerX2(MeshBlock *pmb, Coordinates *pco, AthenaArray<Real> &prim,
                            FaceField &b, Real time, Real dt,
                            int il, int iu, int jl, int ju, int kl, int ku, int ngh);
void DiodeOuterX2(MeshBlock *pmb, Coordinates *pco, AthenaArray<Real> &prim,
                            FaceField &b, Real time, Real dt,
                            int il, int iu, int jl, int ju, int kl, int ku, int ngh);

//CR boundary conditions

// vacuum at x2 bounds
void DiodeCROuterX2(MeshBlock *pmb, Coordinates *pco, CosmicRay *pcr,
    const AthenaArray<Real> &w, FaceField &b,
    AthenaArray<Real> &u_cr, Real time, Real dt, int is, int ie,
    int js, int je, int ks, int ke, int ngh);

void DiodeCRInnerX2(MeshBlock *pmb, Coordinates *pco, CosmicRay *pcr,
    const AthenaArray<Real> &w, FaceField &b,
    AthenaArray<Real> &u_cr, Real time, Real dt, int is, int ie,
    int js, int je, int ks, int ke, int ngh);

void MeshBlock::InitUserMeshBlockData(ParameterInput *pin) {
  if (CR_ENABLED) {
    pcr->EnrollUserCRSource(CRSource);
    pcr->EnrollOpacityFunction(Diffusion);
  }
}

void Mesh::InitUserMeshData(ParameterInput *pin) {
  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  if (rank == 0){
    std::cout << "Temp Scale = " << T_scale << std::endl;
    std::cout << "v Scale    = " << v_scale << std::endl;
    std::cout << "e Scale    = " << e_scale << std::endl;
    std::cout << "B Scale    = " << B_scale << std::endl;
  }


  if(CR_ENABLED){
    //Load CR Variables
    Real vmax = pin->GetReal("cr","vmax") ;
    Real kappaPerp = pin->GetOrAddReal("problem","kappaPerp",3e28)/(v_scale*l_scale) ;
    Real kappaParl = pin->GetOrAddReal("problem","kappaParl",3e28)/(v_scale*l_scale) ;
    sigmaPerp = vmax/(3*kappaPerp);
    sigmaParl = vmax/(3*kappaParl);
    crLoss = pin->GetOrAddReal("problem","crLoss",0.0);
    if (rank == 0){
      std::cout << "Vmax = " << vmax / (c / (v_scale)) << " c" << std::endl;
      std::cout << "sigmaParl = " << sigmaParl << std::endl;
      std::cout << "sigmaPerp = " << sigmaPerp << std::endl;
    }
  }
  cooling_flag = pin->GetInteger("problem","cooling");

  SNRate = pin->GetReal("problem","SNRate");
  injH = pin->GetOrAddReal("problem","InjH",100); 
  StopT = pin->GetReal("problem","StopT");
  injL = pin->GetReal("problem","InjL");
  Esn_th = pin->GetOrAddReal("problem","Esn_th",1) * 1.0e51/(e_scale*pow(l_scale,3));
  Esn_cr = pin->GetOrAddReal("problem","Esn_cr",0.1) * 1.0e51/(e_scale*pow(l_scale,3));

  if (rank == 0){
    std::cout << "Ecr   = " << Esn_cr << std::endl;
    std::cout << "Eth   = " << Esn_th << std::endl;
  }
  HSE_CR_Forcing = pin->GetOrAddInteger("problem","HSE_CR",0);
  HSE_Gamma= pin->GetOrAddInteger("problem","HSE_G",0);
  uniformInj = pin->GetOrAddInteger("problem","uniformInj",0);

  EnrollUserExplicitSourceFunction(mySource);

  if (uniformInj==1) {
    std::uniform_real_distribution<double> distPhase(-1*M_PI,M_PI);
    std::uniform_real_distribution<double> distAmp(-1,1);

    unsigned seed1 = std::chrono::system_clock::now().time_since_epoch().count();
    std::default_random_engine gen(seed1);
    if (rank == 0){
      for (int m =1; m<=5; m++){
        phasesX[m-1] = distPhase(gen);
        phasesY[m-1] = distPhase(gen);
        amplitudes[m-1] = distAmp(gen);
      }

    }
    MPI_Bcast(&phasesX[0],5,MPI_DOUBLE,0,MPI_COMM_WORLD);
    MPI_Bcast(&phasesY[0],5,MPI_DOUBLE,0,MPI_COMM_WORLD);
    MPI_Bcast(&amplitudes[0],5,MPI_DOUBLE,0,MPI_COMM_WORLD);

  }

  
  return;
}


void MeshBlock::ProblemGenerator(ParameterInput *pin) {
  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  Mesh *pm = pmy_mesh; 
  Real myGamma = pin->GetReal("hydro","gamma");

  // Load variables
  
  beta = pin->GetReal("problem","beta");
  alpha = pin->GetReal("problem","alpha");
  Real angle = pin->GetReal("problem","angle");

  Real T0 = pin->GetReal("problem", "T0"); // now T is in K
  Real n0 = pin->GetReal("problem", "n0"); // midplane particles per ccm

  Real SigmaStar = pin->GetReal("problem","SigmaStar")*(M_sun/SQR(l_scale)); // in Msun / pc^2
  Real HStar =pin->GetReal("problem","HStar") ;

  Real gm1 = peos->GetGamma() - 1;
  
  dens0 = n0;

  pres0 = k_B*T0 *n0 / e_scale;

  g0 = 2*PI*G * SigmaStar / (v_scale / t_scale);

  h = pres0/(g0*dens0)*(1 + alpha + beta);

  nGrav = HStar/h;

  if (rank == 0){
    std::cout << " Dens = " << dens0 << std::endl;
    std::cout << " Pres = " << pres0 << std::endl;
    std::cout << " Sig  = " << SigmaStar << std::endl;
    std::cout << " Grav = " << g0 << std::endl;
    std::cout << " h = " << h << std::endl;
    std::cout << " n = " << nGrav << std::endl;
  }
  // Real maxErr = FLT_MIN;

  // Initialize hydro variable
  for(int k=ks; k<=ke; ++k) {
    for (int j=js; j<=je; ++j) {
      for (int i=is; i<=ie; ++i) {
        Real x2 = pcoord->x2v(j);
        Real density = dens0*pow(cosh(x2/(nGrav*h)),-1.0*nGrav);
        Real pressure = pres0*pow(cosh(x2/(nGrav*h)),-1.0*nGrav);

        Real gravity = -1*g0*tanh(x2/(nGrav*h));
        Real dz = pcoord->dx2v(j);
        Real dPdz =  (pres0*pow(cosh((x2 + dz)/(nGrav*h)),-1.0*nGrav) - pressure) / dz;
        // maxErr = fmax(fabs( (dPdz*(1+alpha+beta) - density * gravity )/(pres0 ) ),maxErr);

        phydro->u(IDN, k, j, i) = density;
        phydro->u(IM1, k, j, i) = 0.0;
        phydro->u(IM2, k, j, i) = 0.0;
        phydro->u(IM3, k, j, i) = 0.0;
        //energy
        if (NON_BAROTROPIC_EOS) {
            phydro->u(IEN, k, j, i) = pressure/gm1;
        }
        if (uniformInj==1) {
          Real x1 = pcoord->x1v(i);
          Real x3 = pcoord->x3v(k);
          Real pert = 0.0;
          Real x1scale = (pm->mesh_size.x1max - pm->mesh_size.x1min)/2.0;
          Real x3scale = (pm->mesh_size.x3max - pm->mesh_size.x3min)/2.0;
          for (int m =1; m<=5; m++){
            pert += amplitudes[m-1]*sin(M_PI*m*x1/x1scale+phasesX[m-1])*sin(M_PI*m*x3/x3scale+phasesY[m-1]);
          }
          pert *= 1e-2;
          phydro->u(IDN,k,j,i) += density*pert;
        }

        if (CR_ENABLED) {
            Real crp = beta*pressure;
            Real  fcz = pow(cosh(x2/(nGrav*h)),-1.0*nGrav)*tanh(x2/(nGrav*h));
            fcz *= beta*pres0/(h*sigmaPerp) ;
            pcr->u_cr(CRE,k,j,i) = 3*crp;
            pcr->u_cr(CRF1,k,j,i) = 0.0;
            pcr->u_cr(CRF2,k,j,i) = fcz;
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
          Real x2 = pcoord->x2v(j);
          Real pressure = pres0*pow(cosh(x2/(nGrav*h)),-1.0*nGrav);
          Real b0 = sqrt(2*alpha*pressure);
          pfield->b.x1f(k,j,i) = b0* std::cos(angle);
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
            Real x2 = pcoord->x2v(j);
            Real pressure = pres0*pow(cosh(x2/(nGrav*h)),-1.0*nGrav);
            Real b0 = sqrt(2*alpha*pressure);
            pfield->b.x3f(k,j,i) = b0* std::sin(angle);
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
void Mesh::UserWorkAfterLoop(ParameterInput *pin) {
  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  if (rank==0) {
    std::cout << "Total Number of Injections = " << TotalInjs << std::endl;
  }
}


//----------------------------------------------------------------------------------------
void Mesh::UserWorkInLoop(void)
{
  if (uniformInj != 1){
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    X1Inj.clear();
    X2Inj.clear();
    X3Inj.clear();
    NInjs = 0;
    if ((dt < FLT_MAX) && (time < StopT) && (time > 0.0)) {
    if (rank == 0) {
      Real x1d = (mesh_size.x1max - mesh_size.x1min)/float(mesh_size.nx1);
      Real x2d = (mesh_size.x2max - mesh_size.x2min)/float(mesh_size.nx2);;
      Real x3d = (mesh_size.x3max - mesh_size.x3min)/float(mesh_size.nx3);;
      //std::cout << mesh_size.x1min << "," << mesh_size.x1max << std::endl;

      // std::exponential_distribution<double> distDt(SNRate);
      std::poisson_distribution<int> distN(SNRate*dt);

      std::uniform_real_distribution<double> distx1(mesh_size.x1min+injL/2,mesh_size.x1max - x1d-injL/2);
      std::uniform_real_distribution<double> distx2(-1*injH,injH-x2d);
      std::uniform_real_distribution<double> distx3(mesh_size.x3min+injL/2,mesh_size.x3max - x3d-injL/2);

      unsigned seed1 = std::chrono::system_clock::now().time_since_epoch().count();
      std::default_random_engine gen(seed1);

      NInjs = distN(gen);
      for (int n = 1; n <= NInjs; n++){
        X1Inj.insert(X1Inj.end(), round((distx1(gen)-mesh_size.x1min)/x1d)*x1d + mesh_size.x1min + 0.5*x1d);
        X2Inj.insert(X2Inj.end(), round((distx2(gen)-mesh_size.x2min)/x2d)*x2d + mesh_size.x2min + 0.5*x2d);
        X3Inj.insert(X3Inj.end(), round((distx3(gen)-mesh_size.x3min)/x3d)*x3d + mesh_size.x3min + 0.5*x3d);
        std::cout << "X1: " << X1Inj[n-1] << std::endl;
        std::cout << "X2: " << X2Inj[n-1] << std::endl;
        std::cout << "X3: " << X3Inj[n-1] << std::endl;
      }

    } 
    
    //MPI_Bcast(&lastInjT,1,MPI_DOUBLE,0,MPI_COMM_WORLD);
    MPI_Bcast(&NInjs,1,MPI_INT,0,MPI_COMM_WORLD);

    if ((NInjs > 0) && (rank != 0)){
      X1Inj.insert(X1Inj.end(),NInjs,FLT_MAX);
      X2Inj.insert(X2Inj.end(),NInjs,FLT_MAX);
      X3Inj.insert(X3Inj.end(),NInjs,FLT_MAX);
    }

    MPI_Bcast(&X1Inj[0],NInjs,MPI_DOUBLE,0,MPI_COMM_WORLD);
    MPI_Bcast(&X2Inj[0],NInjs,MPI_DOUBLE,0,MPI_COMM_WORLD);
    MPI_Bcast(&X3Inj[0],NInjs,MPI_DOUBLE,0,MPI_COMM_WORLD);
    TotalInjs += NInjs;
    }
  }
}



void mySource(MeshBlock *pmb, const Real time, const Real dt,
               const AthenaArray<Real> &prim, const AthenaArray<Real> &prim_scalar,
               const AthenaArray<Real> &bcc, AthenaArray<Real> &cons,
               AthenaArray<Real> &cons_scalar){

  Mesh *pm = pmb->pmy_mesh;
    
  const Real Heat    =  2e-26 / (e_scale/t_scale) ;
  const Real T_floor =  30 / T_scale ; 
  // Inoue
  Real lamb_scale = (e_scale * n_scale * n_scale) /t_scale;
  const Real Lamb1_I   =  7.3e-21 / lamb_scale; 
  const Real Lamb2_I   =  7.9e-27 / lamb_scale; 
  const Real T1a_I     =  118400.0 /T_scale;
  const Real T1b_I     =  1500.0 /T_scale;
  const Real T2_I      =  92.0 / T_scale ;
  // Koyama
  const Real Lamb1_K   =  2e-19 / lamb_scale;
  const Real Lamb2_K   =  28e-26 / lamb_scale * (T_scale / 1);
  const Real T1a_K     =  114800.0 / T_scale;
  const Real T1b_K     =  1000.0 / T_scale;
  const Real T2_K      =  92.0 /T_scale;
  //CIE
  const Real T0_C  =  3.0e2 / T_scale ;
  const Real T1_C  =  2.0e3 / T_scale;
  const Real T2_C  =  8.0e3 / T_scale ;
  const Real T3_C  =  1.0e5 / T_scale ;
  const Real T4_C  =  4.0e7 / T_scale;

  const Real a1_C  =  2.0;
  const Real a2_C  =  1.5;
  const Real a3_C  =  2.867;
  const Real a4_C  = -0.65;
  const Real a5_C  =  0.5;

  const Real A1_C  =  (2.2380e-32 / lamb_scale) * pow(T_scale,a1_C) ;
  const Real A2_C  =  (1.0012e-30 / lamb_scale) * pow(T_scale,a2_C) ;
  const Real A3_C  =  (4.6240e-36 / lamb_scale) * pow(T_scale,a3_C) ;
  const Real A4_C  =  (1.7800e-18 / lamb_scale) * pow(T_scale,a4_C) ;
  const Real A5_C  =  (3.2217e-27 / lamb_scale) * pow(T_scale,a5_C) ;


  const Real T_switch = 1.40413e4 / T_scale;

  for (int k=pmb->ks; k<=pmb->ke; ++k) {
    for (int j=pmb->js; j<=pmb->je; ++j) {
#pragma omp simd
      for (int i=pmb->is; i<=pmb->ie; ++i) {
        Real d = prim(IDN,k,j,i);
        Real p = prim(IPR,k,j,i);

        // GRAVITY
        Real x2 = pmb->pcoord->x2v(j);
        Real gravity = -1*g0*tanh(x2/(nGrav*h));
        Real src = dt*d*gravity;

        cons(IM2,k,j,i) += src;
        if (NON_BAROTROPIC_EOS) cons(IEN,k,j,i) += src*prim(IVY,k,j,i);

        //COOLING
        if ((d> dfloor) && (p> pfloor) && (cooling_flag != 0) ) {
          Real T = p/d;
          Real Lamb = 0.0;
          Real Gam = Heat;
          if (HSE_Gamma == 1) Gam *=  pow(cosh(x2/(nGrav*h)),-1.0*nGrav);

          if (cooling_flag == 1) {
            Lamb = Lamb1_I*exp(-1*T1a_I/(T + T1b_I)) + Lamb2_I*exp(-1*T2_I/T);
          } else if (cooling_flag == 2) {
            Lamb = Lamb1_K*exp(-1*T1a_K/(T + T1b_K)) + Lamb2_K*sqrt(T)*exp(-1*T2_K/T);
          } else if (cooling_flag == 3) {
            if ((T >= T0_C) && (T < T1_C)) { 
              Lamb = A1_C * pow(T,a1_C);
            } else if ((T >= T1_C) && (T< T2_C)){
              Lamb = A2_C * pow(T,a2_C);
            } else if ((T>= T2_C) && (T < T3_C)){
              Lamb = A3_C * pow(T,a3_C);
            } else if ((T>= T3_C) && (T < T4_C)){
              Lamb = A4_C * pow(T,a4_C);
            } else if ((T >= T4_C) ){
              Lamb = A5_C * pow(T,a5_C);
            }
          } else if (cooling_flag == 4) {
            if (T< T_switch){
              Lamb = Lamb1_I*exp(-1*T1a_I/(T + T1b_I)) + Lamb2_I*exp(-1*T2_I/T);
            } else if ((T>= T_switch) && (T < T3_C)){
              Lamb = A3_C * pow(T,a3_C);
            } else if ((T>= T3_C) && (T < T4_C)){
              Lamb = A4_C * pow(T,a4_C);
            } else if ((T >= T4_C) ){
              Lamb = A5_C * pow(T,a5_C);
            }
          }
          Real dEdt = ((T > T_floor) ) ? (d*d*Lamb - d*Gam) : (-1*d*Gam);
          cons(IEN,k,j,i) -= dEdt*dt;
        }

        //INJECTION
        if (uniformInj != 1){
          Real x1 = pmb->pcoord->x1v(i);
          Real x2 = pmb->pcoord->x2v(j);
          Real x3 = pmb->pcoord->x3v(k);
          Real dx1 = pmb->pcoord->dx1v(i+1);
          Real dx2 = pmb->pcoord->dx2v(j+1);
          Real dx3 = pmb->pcoord->dx3v(k+1);
          Real cellVol = pmb->pcoord->GetCellVolume(k,j,i);
          for (int m = 0 ; m < NInjs; ++m) {
            Real x10   = X1Inj[m];
            Real x20   = X2Inj[m];
            Real x30   = X3Inj[m];
            Real ax = (x1 - x10 - dx1/2)/injL;
            Real bx = (x1 - x10 + dx1/2)/injL;
            Real ay = (x2 - x20 - dx2/2)/injL;
            Real by = (x2 - x20 + dx2/2)/injL;
            Real az = (x3 - x30 - dx3/2)/injL;
            Real bz = (x3 - x30 + dx3/2)/injL;
            Real frac = (fmin(0.5,bx) - fmax(-0.5,ax))*(fmin(0.5,by) - fmax(-0.5,ay))*(fmin(0.5,bz) - fmax(-0.5,az));

            if ( ( bx > -0.5) && ( ax < 0.5) && ( by > -0.5) && ( ay < 0.5) && ( bz > -0.5) && ( az < 0.5)) {
              cons(IEN,k,j,i) += Esn_th/cellVol*frac;
            }
          }
        }
        if (uniformInj == 1){
          Real x2v = abs(pmb->pcoord->x2v(j));
          Real Vol = 2*injH*(pm->mesh_size.x1max - pm->mesh_size.x1min)*(pm->mesh_size.x3max - pm->mesh_size.x3min);
          if (x2v < injH) {
            cons(IEN,k,j,i) += Esn_th/Vol*SNRate*dt;
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
  Mesh *pm = pmb->pmy_mesh;
  if ((HSE_CR_Forcing == 1) || (Esn_cr > 0.0)) {
  for (int k=pmb->ks; k<=pmb->ke; ++k) {
    for (int j=pmb->js; j<=pmb->je; ++j) {
  #pragma omp simd
      for (int i=pmb->is; i<=pmb->ie; ++i) {
        //HSE forcing
        if (HSE_CR_Forcing == 1) {
          Real xi = pmb->pcoord->x2v(j)/(nGrav*h);
          Real arg = (1/(nGrav*SQR(cosh(xi))) - SQR(tanh(xi)))*pow(cosh(xi),-1.0*nGrav);
          Real coeff = (beta* pres0)/ (sigmaPerp*SQR(h));
          u_cr(CRE,k,j,i) += arg*coeff*dt;
        }

        //INJECTION
        if (uniformInj != 1){
          Real x1 = pmb->pcoord->x1v(i);
          Real x2 = pmb->pcoord->x2v(j);
          Real x3 = pmb->pcoord->x3v(k);
          Real dx1 = pmb->pcoord->dx1v(i+1);
          Real dx2 = pmb->pcoord->dx2v(j+1);
          Real dx3 = pmb->pcoord->dx3v(k+1);
          Real cellVol = pmb->pcoord->GetCellVolume(k,j,i);
          for (int m = 0 ; m < NInjs; ++m) {
            Real x10   = X1Inj.at(m);
            Real x20   = X2Inj.at(m);
            Real x30   = X3Inj.at(m);
            Real ax = (x1 - x10 - dx1/2)/injL;
            Real bx = (x1 - x10 + dx1/2)/injL;
            Real ay = (x2 - x20 - dx2/2)/injL;
            Real by = (x2 - x20 + dx2/2)/injL;
            Real az = (x3 - x30 - dx3/2)/injL;
            Real bz = (x3 - x30 + dx3/2)/injL;
            Real frac = (fmin(0.5,bx) - fmax(-0.5,ax))*(fmin(0.5,by) - fmax(-0.5,ay))*(fmin(0.5,bz) - fmax(-0.5,az));

            if ( ( bx > -0.5) && ( ax < 0.5) && ( by > -0.5) && ( ay < 0.5) && ( bz > -0.5) && ( az < 0.5)) {
              u_cr(CRE,k,j,i) += Esn_cr/cellVol*frac;
            }
          }
        }
        if (uniformInj == 1){
          Real x2v = abs(pmb->pcoord->x2v(j));
          Real Vol = 2*injH*(pm->mesh_size.x1max - pm->mesh_size.x1min)*(pm->mesh_size.x3max - pm->mesh_size.x3min);
          if (x2v < injH) {
            u_cr(CRE,k,j,i) += Esn_cr/Vol*SNRate*dt;
          }
        }

        // HADRONIC LOSSES
        if (crLoss > 0.0) {
          u_cr(CRE,k,j,i) -= crLoss*dt*u_cr(CRE,k,j,i);
        }
        
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

