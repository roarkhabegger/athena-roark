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
#include "../parameter_input.hpp"
#include "../fft/turbulence.hpp"


//======================================================================================
//! \fn void MeshBlock::ProblemGenerator(ParameterInput *pin)
//  \brief cosmic ray modified turbulence in the intracluster medium
//======================================================================================
const double k_B = 1.380648999999999994e-16;               
const double c = 2.9979245800e+10;                  
const double l_scale = 3.085677581491367313408e+18; // pc
const double t_scale = 3.15576000000e+13; // Myr
const double m_scale = 1.672621923689999956e-24; //m_p
const double n_scale = 1.0e0; // cm^-3
const double v_scale = l_scale/t_scale;
const double rho_scale = m_scale*n_scale;
const double e_scale = rho_scale*v_scale*v_scale;
const double T_scale = m_scale*v_scale*v_scale/k_B;
const double B_scale = 4*PI*sqrt(e_scale);
const double lamb_scale = e_scale/(t_scale*n_scale*n_scale);

// double totalVolume;

double sigmaParl, sigmaPerp; //CR diffusion 
                           //decouple parallel (to local B field) and perpendicular diffusion coefficients
Real crLoss; //CR Loss term. E.g. Hadronic losses, proportional to local CR energy
             // can be useful to give a decay to CR energy during turbulent driving
Real n0, T0;
int cooling_flag;
int heating_flag;
// Real turb_dedt;
// 
static Real f_i, T_f_i, dT_f_i;


// int nHigh, nLow, Ntriples;
// Real Lrand;

// Cooling Function parameters: 
// Tupper   [8.e+03 1.e+05 4.e+07 1.e+08] K
// Tlower   [2.e+03 8.e+03 1.e+05 4.e+07] K
// Lks/LN   [9.8273346163844594e-09 4.5387130709310564e-14 1.7471689589656751e+04 3.1622776601683795e-05]
// alphaks  [ 1.5    2.867 -0.65   0.5  ]
// TEF bounds for Tmax= 100000000.0  K
// Yks      [1.1275108240104612 1.1203155181743534 1.1184058966906867 0.7350889359326482 0. ]
double Tupps[4] = {8.0e+3, 1.0e+5, 4.0e+7, 1.0e+8};
double Tlows[4] = {2.0e+3, 8.0e+3, 1.0e+5, 4.0e+7};
double Lks[4] = {1.0012e-30, 4.6240e-36, 1.7800e-18, 3.2217e-27};
double aks[4] = {1.5, 2.867, -0.65, 0.5};
double Yks[5] = {1.1275108240104612, 1.1203155181743534, 1.1184058966906867, 0.7350889359326482, 0.0};

// double Tupps[3] = {1.0e+5, 4.0e+7, 1.0e+8};
// double Tlows[3] = {8.0e+3, 1.0e+5, 4.0e+7};
// double Lks[3] = {4.6240e-36, 1.7800e-18, 3.2217e-27};
// double aks[3] = {2.867, -0.65, 0.5};
// double Yks[4] = {1.1203155181743534, 1.1184058966906867, 0.7350889359326482, 0.0};


double Tmax = Tupps[2];
double LN =  Lks[2] * std::pow(Tmax,aks[2]);

double TEF(double T);
double invTEF(double T);

double TEF(double T){
  double val = 0.0;
  int j = 0;
  double Temp = T*T_scale;
  
  //Figure out which T bin we are in
  while(Temp >= Tupps[j]) j++;

  //Calculate Y function value at Temparture T*T_scale
  val = (1/(1-aks[j]))*(LN/Lks[j])*std::pow(Tlows[j],-1*aks[j])*(Tlows[j]/Tmax);
  val *= (1-std::pow(Tlows[j]/(Temp),aks[j]-1));
  val += Yks[j];
  return val;
}

double invTEF(double y) {
  double val = 0.0;
  int j = 0;
  //Figure out which T bin we are in
  if (y> Yks[0]) {
    val = Tlows[0];
  } else {
    while(y<= Yks[j]) j++;
    j -= 1;
  }
  // Calculate invY
  val = (Lks[j]/LN)*std::pow(Tlows[j],aks[j])*(Tmax/Tlows[j]);
  val *= (y-Yks[j])*(1-aks[j]);
  val = 1-val;
  val = (Tlows[j]/T_scale)*std::pow(val,1.0/(1-aks[j]));
  return val;
}   

void CRSource(MeshBlock *pmb, const Real time, const Real dt,
                const AthenaArray<Real> &prim, FaceField &b, 
              AthenaArray<Real> &u_cr);

void mySource(MeshBlock *pmb, const Real time, const Real dt,
               const AthenaArray<Real> &prim, const AthenaArray<Real> &prim_scalar,
               const AthenaArray<Real> &bcc, AthenaArray<Real> &cons,
               AthenaArray<Real> &cons_scalar);
              

void Opacity(MeshBlock *pmb, AthenaArray<Real> &u_cr,
        AthenaArray<Real> &prim, AthenaArray<Real> &bcc);

void Streaming(MeshBlock *pmb, AthenaArray<Real> &u_cr,
             AthenaArray<Real> &prim, AthenaArray<Real> &bcc,
             AthenaArray<Real> &grad_pc, int k, int j, int is, int ie);

Real TotalHeating(MeshBlock *pmb, int iout);


Real TotalHeating(MeshBlock *pmb, int iout){
  // Real heat=0;
  int is=pmb->is, ie=pmb->ie, js=pmb->js, je=pmb->je, ks=pmb->ks, ke=pmb->ke;
  Real pfloor = pmb->peos->GetPressureFloor();
  Real dfloor = pmb->peos->GetDensityFloor();
  Real Tfloor = Tlows[0]/T_scale;
  double gm1 = pmb->peos->GetGamma()-1.0;

  double totdE = 0.0;
  double totV = 0.0;
  
  AthenaArray<Real> &cons = pmb->phydro->u;
  AthenaArray<Real> &bcc = pmb->pfield->bcc;
  Real dt = pmb->pmy_mesh->dt;
  for (int k=pmb->ks; k<=pmb->ke; ++k) {
    for (int j=pmb->js; j<=pmb->je; ++j) {
#pragma omp simd
      for (int i=pmb->is; i<=pmb->ie; ++i) {
        double d = cons(IDN,k,j,i);
        double p = gm1*(cons(IEN,k,j,i) - 0.5*(SQR(cons(IM1,k,j,i))+SQR(cons(IM2,k,j,i))+SQR(cons(IM3,k,j,i)))/d - 0.5*(SQR(bcc(IB1,k,j,i))+SQR(bcc(IB2,k,j,i))+SQR(bcc(IB3,k,j,i))));

        if ((d> dfloor) && (p> pfloor) ) {
          double T = p/d;
          if ((T > Tfloor)){
            double time0 = t_scale*gm1*d*n_scale*LN/(k_B*Tmax);
            double newT = invTEF(TEF(T) + dt*time0);
            double dE = d*(newT-T)/gm1;
            totdE += -1*dE * pmb->pcoord->GetCellVolume(k,j,i);
            totV += pmb->pcoord->GetCellVolume(k,j,i);
          }
        }
      }
    }
  }
  Real heat = totdE / totV;
  if (iout == 0) return totdE;
  if (iout == 1) return totV;
  
  return heat;
}


void MeshBlock::InitUserMeshBlockData(ParameterInput *pin) {
  AllocateUserOutputVariables(6);
  if (CR_ENABLED) {
    pcr->EnrollOpacityFunction(Opacity);
    pcr->EnrollStreamingFunction(Streaming);
    bool lossFlag = (pin->GetOrAddReal("problem","crLoss",0.0) > 0.0);
    if (lossFlag) {
        pcr->EnrollUserCRSource(CRSource);
    }
  }
}
void MeshBlock::UserWorkBeforeOutput(ParameterInput *pin)
{
  Real IQX = IPR;
  Real IQY = IPR;
  Real IQZ = IPR;
  Real tol = 10.0;

  for(int k=ks; k<=ke; k++) {
    for(int j=js; j<=je; j++) {
      for(int i=is; i<=ie; i++) {
        
        Real shx = 0.0;
        Real vsx = 0.0;
        // Real vx0 = phydro->w(IVX,k,j,i);
        Real apx = std::abs(phydro->w(IQX,k,j,i+1) - phydro->w(IQX,k,j,i));
        Real amx = std::abs(phydro->w(IQX,k,j,i) - phydro->w(IQX,k,j,i-1));
        if (apx+amx > tol){
          shx = SQR((apx - amx)/(apx+amx));
          if (user_out_var(0,k,j,i - 1) > 0.5){
            Real comp = phydro->w(IDN,k,j,i) / phydro->w(IDN,k,j,i-1);
            vsx = (comp * phydro->w(IVX,k,j,i) -  phydro->w(IVX,k,j,i-1)) / (comp - 1);
          } else {
            Real comp = phydro->w(IDN,k,j,i) / phydro->w(IDN,k,j,i+1);
            vsx = (comp * phydro->w(IVX,k,j,i) -  phydro->w(IVX,k,j,i+1)) / (comp - 1);
          }
        } 
        Real shy = 0.0;
        Real vsy = 0.0;
        Real apy = std::abs(phydro->w(IQY,k,j+1,i) - phydro->w(IQY,k,j,i));
        Real amy = std::abs(phydro->w(IQY,k,j,i) - phydro->w(IQY,k,j-1,i));
        if (apy+amy > tol){
          shy = SQR((apy - amy)/(apy+amy));
          if (user_out_var(0,k,j-1,i) > 0.5){
            Real comp = phydro->w(IDN,k,j,i) / phydro->w(IDN,k,j-1,i);
            vsy = (comp * phydro->w(IVX,k,j,i) -  phydro->w(IVX,k,j-1,i)) / (comp - 1);
          } else {
            Real comp = phydro->w(IDN,k,j,i) / phydro->w(IDN,k,j+1,i);
            vsy = (comp * phydro->w(IVX,k,j,i) -  phydro->w(IVX,k,j+1,i)) / (comp - 1);
          }
        } 
        Real shz = 0.0;
        Real vsz = 0.0;
        Real apz = std::abs(phydro->w(IQZ,k+1,j,i) - phydro->w(IQZ,k,j,i));
        Real amz = std::abs(phydro->w(IQZ,k,j,i) - phydro->w(IQZ,k-1,j,i));
        if (apy+amz > tol){
          shz = SQR((apz - amz)/(apz+amz));
          if (user_out_var(0,k-1,j,i) > 0.5){
            Real comp = phydro->w(IDN,k,j,i) / phydro->w(IDN,k-1,j,i);
            vsz = (comp * phydro->w(IVX,k,j,i) -  phydro->w(IVX,k-1,j,i)) / (comp - 1);
          } else {
            Real comp = phydro->w(IDN,k,j,i) / phydro->w(IDN,k+1,j,i);
            vsz = (comp * phydro->w(IVX,k,j,i) -  phydro->w(IVX,k+1,j,i)) / (comp - 1);
          }
        } 

        user_out_var(0,k,j,i) = shx;
        user_out_var(1,k,j,i) = shy;
        user_out_var(2,k,j,i) = shz;
        user_out_var(3,k,j,i) = vsx;
        user_out_var(4,k,j,i) = vsy;
        user_out_var(5,k,j,i) = vsz;

        



        // Real pmag = 0.5*(SQR(pfield->bcc(IB1,k,j,i))
        //                 +SQR(pfield->bcc(IB2,k,j,i))
        //                 +SQR(pfield->bcc(IB3,k,j,i)));
        // user_out_var(0,k,j,i) = phydro->w(IPR,k,j,i)/phydro->w(IDN,k,j,i);
        // user_out_var(1,k,j,i) = phydro->w(IPR,k,j,i)/pmag;
      }
    }
  }
}

void Mesh::InitUserMeshData(ParameterInput *pin) {
  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank); //Just for print statements

  // Real dx =  (pin->GetReal("mesh","x1max") -  pin->GetReal("mesh","x1min") );
  // Real dy =  (pin->GetReal("mesh","x2max") -  pin->GetReal("mesh","x2min") );
  // Real dz =  (pin->GetReal("mesh","x3max") -  pin->GetReal("mesh","x3min") );
  // totalVolume = dx*dy*dz;

  // turb_dedt = pin->GetOrAddReal("turbulence","dedt",0.0);
  
  if(CR_ENABLED){
    //Load CR Variables
    Real vmax = pin->GetReal("cr","vmax") ;
    Real kappaPerp = pin->GetOrAddReal("cr","kappaPerp",3e28)/(v_scale*l_scale) ;
    Real kappaParl = pin->GetOrAddReal("cr","kappaParl",3e28)/(v_scale*l_scale) ;
    sigmaPerp = vmax/(3*kappaPerp);
    sigmaParl = vmax/(3*kappaParl);
    f_i = pin->GetOrAddReal("cr","f_i",1);
    T_f_i = pin->GetOrAddReal("cr","T_f_i",10000)/T_scale;
    dT_f_i = pin->GetOrAddReal("cr","dT_f_i",1000)/T_scale;
    crLoss = pin->GetOrAddReal("problem","crLoss",0.0);

    if (rank == 0){
      std::cout << "Vmax = " << vmax / (c / (v_scale)) << " c" << std::endl;
      std::cout << "sigmaParl = " << sigmaParl << std::endl;
      std::cout << "sigmaPerp = " << sigmaPerp << std::endl;
      std::cout << "Tceil = " << Tmax / T_scale << std::endl;
    }
  }
  cooling_flag = pin->GetInteger("problem","cooling");
  // heating_flag = pin->GetOrAddInteger("problem","heating",1);
  // Real gm1 = pin->GetReal("hydro","gamma")-1.0;
  n0 = pin->GetReal("problem", "n0")/n_scale; //density
  T0 = pin->GetReal("problem", "T0")/T_scale;
  
  
  if (cooling_flag != 0) {
    // EnrollUserTimeStepFunction(CoolingTimeStep);
    
    EnrollUserExplicitSourceFunction(mySource);
    // constant heating rate is n^2 Lambda at n0, T0 in computational units
  }

  AllocateUserHistoryOutput(2);
  EnrollUserHistoryOutput(0, TotalHeating, "totdE_heat");
  EnrollUserHistoryOutput(1, TotalHeating, "totV_heat");

  

  // EnrollUserHistoryOutput(7, div_correlation, "corr_pc_div");

  // generate parameters for random field
  // AllocateRealUserMeshDataField(1);
  // int Nx1 =pin->GetInteger("mesh","nx1");
  // int Nx2 =pin->GetInteger("mesh","nx2");
  // int Nx3 =pin->GetInteger("mesh","nx3");
  // nHigh= pin->GetOrAddInteger("problem","randB_nhigh",1);
  // nLow = pin->GetOrAddInteger("problem","randB_nlow",1);
  // int my_count = 0;
  
  // for (int l=0; l<=nHigh; l++){
  //   for(int m=0; m<=nHigh; m++) {
  //     for(int n=0; n<=nHigh; n++) {
  //       Real nmag = std::sqrt(SQR(n)+SQR(m)+SQR(l));
  //       if ((nmag <= nHigh) && (nmag > nLow)) {
  //         my_count +=1;
  //       }
  //     }
  //   }
  // }

  // Ntriples = my_count;
  // Real x1min = pin->GetReal("mesh","x1min");
  // Real x1max = pin->GetReal("mesh","x1max");
  // Lrand = pin->GetOrAddReal("problem","randB_Lrand",x1max-x1min);
  // ruser_mesh_data[0].NewAthenaArray(9,Ntriples);
  // if (rank==0){

  //   std::int64_t rseed = pin->GetOrAddInteger("problem","randB_rseed",-1);
  //   std::mt19937_64 rng_generator;
  //   if (rseed < 0) {
  //     std::random_device device;
  //     rseed = static_cast<std::int64_t>(device());
  //   } 
  //   rng_generator.seed(rseed);

  //   std::uniform_real_distribution<Real> udist(0.0,1.0);
  //   // std::normal_distribution<Real> gaussian(0.0,1.0);
  //   int my_count = 0;
  //   Real s1 = 0.0;
  //   Real s2 = 0.0;
  //   Real s3 = 0.0;
  //   Real scaleFactor = 0.0;
  //   for (int l=0; l<=nHigh; l++){
  //     for(int m=0; m<=nHigh; m++) {
  //       for(int n=0; n<=nHigh; n++) {
  //         // initialization
  //         Real nmag = std::sqrt(SQR(n)+SQR(m)+SQR(l));
  //         if ((nmag <= nHigh) && (nmag > nLow)) {
  //           ruser_mesh_data[0](0,my_count)=2*M_PI/Lrand*n;
  //           ruser_mesh_data[0](1,my_count)=2*M_PI/Lrand*m;
  //           ruser_mesh_data[0](2,my_count)=2*M_PI/Lrand*l;
  //           ruser_mesh_data[0](3,my_count)=udist(rng_generator);
  //           ruser_mesh_data[0](4,my_count)=udist(rng_generator);
  //           ruser_mesh_data[0](5,my_count)=udist(rng_generator);
  //           ruser_mesh_data[0](6,my_count)=2*M_PI*udist(rng_generator);
  //           ruser_mesh_data[0](7,my_count)=2*M_PI*udist(rng_generator);
  //           ruser_mesh_data[0](8,my_count)=2*M_PI*udist(rng_generator);
  //           s1 += SQR(ruser_mesh_data[0](3,my_count));
  //           s2 += SQR(ruser_mesh_data[0](4,my_count));
  //           s3 += SQR(ruser_mesh_data[0](5,my_count));
  //           scaleFactor += SQR(2*M_PI/Lrand)*SQR(nmag);
  //           my_count += 1;
  //         }
  //       }
  //     }
  //   }
  //   for (int q=0; q < Ntriples; ++q){
  //     ruser_mesh_data[0](3,q) *= 1/std::sqrt(s1*scaleFactor);
  //     ruser_mesh_data[0](4,q) *= 1/std::sqrt(s2*scaleFactor);
  //     ruser_mesh_data[0](5,q) *= 1/std::sqrt(s3*scaleFactor);
  //   }
  //   std::cout << "N_expected=" << Ntriples << " vs Ncount" << my_count << std::endl;
  // }
  // MPI_Bcast(ruser_mesh_data[0].data(), 9*Ntriples,MPI_ATHENA_REAL,0,MPI_COMM_WORLD);

  return;
}

void mySource(MeshBlock *pmb, const Real time, const Real dt,
               const AthenaArray<Real> &prim, const AthenaArray<Real> &prim_scalar,
               const AthenaArray<Real> &bcc, AthenaArray<Real> &cons,
               AthenaArray<Real> &cons_scalar){
    

  Real pfloor = pmb->peos->GetPressureFloor();
  Real dfloor = pmb->peos->GetDensityFloor();
  Real Tfloor = Tlows[0]/T_scale;
  Real Tceil = Tmax / T_scale;
  double gm1 = pmb->peos->GetGamma()-1.0;

  Real t0 = t_scale*gm1*n0*n_scale*LN/(k_B*Tmax);
  Real T2 = invTEF(TEF(T0) + dt*t0);
  // Real const_heating = -n0*(T2-T0)/gm1;

  double totdE = 0.0;
  double totV = 0.0;
  
  for (int k=pmb->ks; k<=pmb->ke; ++k) {
    for (int j=pmb->js; j<=pmb->je; ++j) {
#pragma omp simd
      for (int i=pmb->is; i<=pmb->ie; ++i) {
        double d = cons(IDN,k,j,i);
        double p = gm1*(cons(IEN,k,j,i) - 0.5*(SQR(cons(IM1,k,j,i))+SQR(cons(IM2,k,j,i))+SQR(cons(IM3,k,j,i)))/d - 0.5*(SQR(bcc(IB1,k,j,i))+SQR(bcc(IB2,k,j,i))+SQR(bcc(IB3,k,j,i))));

        if ((d> dfloor) && (p> pfloor) ) {
          double T = p/d;
          if ((T > Tfloor) && (T < Tceil)){
            double time0 = t_scale*gm1*d*n_scale*LN/(k_B*Tmax);
            double newT = invTEF(TEF(T) + dt*time0);
            double dE = d*(newT-T)/gm1;
            totdE += -1*dE * pmb->pcoord->GetCellVolume(k,j,i);
            totV += d * pmb->pcoord->GetCellVolume(k,j,i);
            cons(IEN,k,j,i) += dE;
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
        //CRLoss Term Note that crLoss is zeta_cr total loss rate cm^3/s in computational units
        u_cr(CRE,k,j,i) -= crLoss*dt*u_cr(CRE,k,j,i)*prim(IDN,k,j,i);
      }
    }
  }
  return;
}

void MeshBlock::ProblemGenerator(ParameterInput *pin) {
  // read in the mean velocity, diffusion coefficient
  // const Real n0 = pin->GetReal("problem", "n0")/n_scale; //density
  // const Real T0 = pin->GetReal("problem", "T0")/T_scale;
  const Real pres = n0*T0*(n_scale*k_B*T_scale/e_scale);
  const Real gm1  = peos->GetGamma() - 1.0;
  // std::cout << Ntriples << std::endl;
  const Real invbeta = pin->GetOrAddReal("problem","invbeta",0.0);
  Real dBoverB= pin->GetOrAddReal("problem","delta_B",0.0);
  
  const Real bx_0 = sqrt(2*invbeta*pres/(1 + SQR(dBoverB))); //mean field strength
  const Real dB = dBoverB*bx_0;
  // const Real b_amp = dBrat*bx_0;
  const Real invbetaCR = pin->GetOrAddReal("problem","invbetaCR",0.0);
  const Real crp = pres*invbetaCR;

  const Real Einj = (1e51/(e_scale*l_scale*l_scale*l_scale))*pin->GetOrAddReal("problem","injE1e51",1.0);
  const Real Rinj = pin->GetOrAddReal("problem","injRpc",1.0);
  Real volinj = (4*M_PI/3) * Rinj*Rinj * Rinj; 

  Real gamma = peos->GetGamma();
    
  // Initialize hydro variable
  for(int k=ks; k<=ke; ++k) {
    for (int j=js; j<=je; ++j) {
      for (int i=is; i<=ie; ++i) {
        Real x1 = pcoord->x1v(i);
        Real x2 = pcoord->x2v(j);
        Real x3 = pcoord->x3v(k);
        Real r = std::sqrt(SQR(x1) +SQR(x2) + SQR(x3) );

        phydro->u(IDN, k, j, i) = n0;
        phydro->u(IM1, k, j, i) = 0.0;
        phydro->u(IM2, k, j, i) = 0.0;
        phydro->u(IM3, k, j, i) = 0.0;
        //energy
        if (NON_BAROTROPIC_EOS) {
          phydro->u(IEN, k, j, i) = pres/gm1;
          if (r <= Rinj){
            phydro->u(IEN, k, j, i) += Einj/volinj;
          }
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

  
  if (MAGNETIC_FIELDS_ENABLED) {
    // std::cout << " Amp,phase = " << pmy_mesh->ruser_mesh_data[0](0,0) << "," << pmy_mesh->ruser_mesh_data[0](0,4) << std::endl;
    for (int k=ks; k<=ke; ++k) {
      for (int j=js; j<=je; ++j) {
        for (int i=is; i<=ie+1; ++i) {
          Real x1 = pcoord->x1f(i);
          Real x2 = pcoord->x2v(j);
          Real x3 = pcoord->x3v(k);
          pfield->b.x1f(k,j,i) = bx_0;
          // for (int q=0; q < Ntriples; ++q){
            
          //   Real k1 = pmy_mesh->ruser_mesh_data[0](0,q);
          //   Real k2 = pmy_mesh->ruser_mesh_data[0](1,q);
          //   Real k3 = pmy_mesh->ruser_mesh_data[0](2,q);
          //   Real A1 = dB*pmy_mesh->ruser_mesh_data[0](3,q);
          //   Real A2 = dB*pmy_mesh->ruser_mesh_data[0](4,q);
          //   Real A3 = dB*pmy_mesh->ruser_mesh_data[0](5,q);
          //   Real phi1 = pmy_mesh->ruser_mesh_data[0](6,q);
          //   Real phi2 = pmy_mesh->ruser_mesh_data[0](7,q);
          //   Real phi3 = pmy_mesh->ruser_mesh_data[0](8,q);
          //   // if ((i==10)&&(j==10)&&(k==10)){
          //   //   std::cout << "(" << k1 <<"," << k2 << "," << k3 << ") A1=" << A1 << std::endl;
          //   // }

          //   pfield->b.x1f(k,j,i) += k2*A3*cos(k1*x1+k2*x2+k3*x3+phi3) - k3*A2*cos(k1*x1+k2*x2+k3*x3+phi2);
          // }
          
        }
      }
    }
    if (block_size.nx2 > 1) {
      for (int k=ks; k<=ke; ++k) {
        for (int j=js; j<=je+1; ++j) {
          for (int i=is; i<=ie; ++i) {
            Real x1 = pcoord->x1v(i);
            Real x2 = pcoord->x2f(j);
            Real x3 = pcoord->x3v(k);
            pfield->b.x2f(k,j,i) = 0.0;
            // for (int q=0; q < Ntriples; ++q){
            //   Real k1 = pmy_mesh->ruser_mesh_data[0](0,q);
            //   Real k2 = pmy_mesh->ruser_mesh_data[0](1,q);
            //   Real k3 = pmy_mesh->ruser_mesh_data[0](2,q);
            //   Real A1 = dB*pmy_mesh->ruser_mesh_data[0](3,q);
            //   Real A2 = dB*pmy_mesh->ruser_mesh_data[0](4,q);
            //   Real A3 = dB*pmy_mesh->ruser_mesh_data[0](5,q);
            //   Real phi1 = pmy_mesh->ruser_mesh_data[0](6,q);
            //   Real phi2 = pmy_mesh->ruser_mesh_data[0](7,q);
            //   Real phi3 = pmy_mesh->ruser_mesh_data[0](8,q);
            //   pfield->b.x2f(k,j,i) += k3*A1*cos(k1*x1+k2*x2+k3*x3+phi1) - k1*A3*cos(k1*x1+k2*x2+k3*x3+phi3);
            // }
          }
        }
      }
    }
    if (block_size.nx3 > 1) {
      for (int k=ks; k<=ke+1; ++k) {
        for (int j=js; j<=je; ++j) {
          for (int i=is; i<=ie; ++i) {
            Real x1 = pcoord->x1v(i);
            Real x2 = pcoord->x2v(j);
            Real x3 = pcoord->x3f(k);
            pfield->b.x3f(k,j,i) = 0.0;
            // for (int q=0; q < Ntriples; ++q){
            //   Real k1 = pmy_mesh->ruser_mesh_data[0](0,q);
            //   Real k2 = pmy_mesh->ruser_mesh_data[0](1,q);
            //   Real k3 = pmy_mesh->ruser_mesh_data[0](2,q);
            //   Real A1 = dB*pmy_mesh->ruser_mesh_data[0](3,q);
            //   Real A2 = dB*pmy_mesh->ruser_mesh_data[0](4,q);
            //   Real A3 = dB*pmy_mesh->ruser_mesh_data[0](5,q);
            //   Real phi1 = pmy_mesh->ruser_mesh_data[0](6,q);
            //   Real phi2 = pmy_mesh->ruser_mesh_data[0](7,q);
            //   Real phi3 = pmy_mesh->ruser_mesh_data[0](8,q);
            //   pfield->b.x3f(k,j,i) += k1*A2*cos(k1*x1+k2*x2+k3*x3+phi2) - k2*A1*cos(k1*x1+k2*x2+k3*x3+phi1);
            // }
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


void Opacity(MeshBlock *pmb, AthenaArray<Real> &u_cr,
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
        pcr->sigma_diff(0,k,j,i) = sigmaParl;
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
          Real Temp = prim(IPR,k,j,i)/prim(IDN,k,j,i);
          Real switch_func = 0.5*(1+std::tanh( (Temp - T_f_i)/dT_f_i));
          Real my_fi = (1-f_i)*switch_func + f_i;
          // Real my_fi = std::pow(10,(1-std::log10(f_i))*switch_func + std::log10(f_i));
          Real inv_sqrt_rho = 1.0/std::sqrt(prim(IDN,k,j,i) * my_fi);
          Real va1 = bcc(IB1,k,j,i)*inv_sqrt_rho;
          Real va2 = bcc(IB2,k,j,i)*inv_sqrt_rho;
          Real va3 = bcc(IB3,k,j,i)*inv_sqrt_rho;


          Real va = std::sqrt(SQR(va1) + SQR(va2) + SQR(va3));

          Real dpc_sign = 0.0;
          if (pcr->b_grad_pc(k,j,i) > TINY_NUMBER) dpc_sign = 1.0;
          else if (-pcr->b_grad_pc(k,j,i) > TINY_NUMBER) dpc_sign = -1.0;
          if (pcr->stream_flag > 0) {
            pcr->v_adv(0,k,j,i) = -va1 * dpc_sign;
            pcr->v_adv(1,k,j,i) = -va2 * dpc_sign;
            pcr->v_adv(2,k,j,i) = -va3 * dpc_sign;
          } else {
            pcr->v_adv(0,k,j,i) = 0.0;
            pcr->v_adv(1,k,j,i) = 0.0;
            pcr->v_adv(2,k,j,i) = 0.0;
          }

          // now the diffusion coefficient
          if (pcr->stream_flag > 0) {
            if (va < TINY_NUMBER) {
              pcr->sigma_adv(0,k,j,i) = pcr->max_opacity;
            } else {
              pcr->sigma_adv(0,k,j,i) = std::abs(pcr->b_grad_pc(k,j,i))
                            /(std::sqrt(pb)* va * (1.0 + 1.0/3.0)
                                      * invlim * u_cr(CRE,k,j,i));
            }

            pcr->sigma_adv(1,k,j,i) = pcr->max_opacity;
            pcr->sigma_adv(2,k,j,i) = pcr->max_opacity;
          } else {
            pcr->sigma_adv(0,k,j,i)  = pcr->max_opacity;
            pcr->sigma_adv(1,k,j,i)  = pcr->max_opacity;
            pcr->sigma_adv(2,k,j,i)  = pcr->max_opacity;
          }
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
  }
}

void Streaming(MeshBlock *pmb, AthenaArray<Real> &u_cr,
             AthenaArray<Real> &prim, AthenaArray<Real> &bcc,
             AthenaArray<Real> &grad_pc, int k, int j, int is, int ie) {
  CosmicRay *pcr=pmb->pcr;
  Real invlim = 1.0/pcr->vmax;

  for(int i=is; i<=ie; ++i) {
    Real Temp = prim(IPR,k,j,i)/prim(IDN,k,j,i);
    Real switch_func = 0.5*(1+std::tanh( (Temp - T_f_i)/dT_f_i));
    Real my_fi = (1-f_i)*switch_func + f_i;
    // Real my_fi = std::pow(10,(1-std::log10(f_i))*switch_func + std::log10(f_i));
    Real inv_sqrt_rho = 1.0/std::sqrt(prim(IDN,k,j,i) * my_fi);
    Real bsq = bcc(IB1,k,j,i)*bcc(IB1,k,j,i)
              +bcc(IB2,k,j,i)*bcc(IB2,k,j,i)
              +bcc(IB3,k,j,i)*bcc(IB3,k,j,i);

    Real b_grad_pc = bcc(IB1,k,j,i) * grad_pc(0,k,j,i)
                   + bcc(IB2,k,j,i) * grad_pc(1,k,j,i)
                   + bcc(IB3,k,j,i) * grad_pc(2,k,j,i);

    Real va1 = bcc(IB1,k,j,i) * inv_sqrt_rho;
    Real va2 = bcc(IB2,k,j,i) * inv_sqrt_rho;
    Real va3 = bcc(IB3,k,j,i) * inv_sqrt_rho;

    Real va = std::sqrt(bsq) * inv_sqrt_rho;
    Real dpc_sign = 0.0;

    if (b_grad_pc > TINY_NUMBER) dpc_sign = 1.0;
    else if (-b_grad_pc > TINY_NUMBER) dpc_sign = -1.0;

    if (pcr->stream_flag > 0) {
      pcr->v_adv(0,k,j,i) = -va1 * dpc_sign;
      pcr->v_adv(1,k,j,i) = -va2 * dpc_sign;
      pcr->v_adv(2,k,j,i) = -va3 * dpc_sign;
      if (va > TINY_NUMBER) {
        pcr->sigma_adv(0,k,j,i) = std::abs(b_grad_pc)/(std::sqrt(bsq) * va *
                               (4.0/3.0) * invlim * u_cr(CRE,k,j,i));
        pcr->sigma_adv(1,k,j,i) = pcr->max_opacity;
        pcr->sigma_adv(2,k,j,i) = pcr->max_opacity;
      }
    } else {
      pcr->v_adv(0,k,j,i) = 0.0;
      pcr->v_adv(1,k,j,i) = 0.0;
      pcr->v_adv(2,k,j,i) = 0.0;
      pcr->sigma_adv(0,k,j,i)  = pcr->max_opacity;
      pcr->sigma_adv(1,k,j,i)  = pcr->max_opacity;
      pcr->sigma_adv(2,k,j,i)  = pcr->max_opacity;
    }
  }
}
