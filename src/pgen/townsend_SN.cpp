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
#include "../eos/eos.hpp"
#include "../field/field.hpp"
#include "../globals.hpp"
#include "../hydro/hydro.hpp"
#include "../hydro/srcterms/hydro_srcterms.hpp"
#include "../mesh/mesh.hpp"
#include "../parameter_input.hpp"
#include "../inputs/hdf5_reader.hpp"


//======================================================================================
//! \fn void MeshBlock::ProblemGenerator(ParameterInput *pin)
//  \brief Single Supernova injection with Switch between Explicit Cooling and Exact Cooling
//======================================================================================

//Initial paramters
Real pres0, dens0, invbeta, angle;

//Cooling Parameters
Real HeatingRate;
Real cool_CFL;
int Tbins;
AthenaArray<Real> Lks, aks, Tlows, Tupps, Yks, Tmax_arr, LN_arr;
Real Y(Real T);
Real invY(Real T);


//Injection information
double Esn_th = 0.0;
double Esn_mom = 0.0;
double Msn = 0.0;
double injL = 0.0;
Real max_dt = FLT_MAX;



// All in cgs units
const Real k_B = 1.380649e-16;
const Real M_sun = 1.98840987e+33;
const Real parsec = 3.08568e+18;  
const Real G = 6.6743e-08;
const Real c = 2.99792458e+10;
const Real l_scale = 1*parsec;
const Real t_scale = 3.15576e+13;
const Real n_scale = 1; 
const Real m_scale = 1.67262192e-24;
const Real v_scale = l_scale/t_scale;
const Real rho_scale = m_scale*n_scale;
const Real e_scale = rho_scale*v_scale*v_scale;
const Real T_scale = e_scale/(n_scale*k_B);
const Real B_scale = 4*PI*sqrt(e_scale);


void mySource(MeshBlock *pmb, const Real time, const Real dt,
               const AthenaArray<Real> &prim, const AthenaArray<Real> &prim_scalar,
               const AthenaArray<Real> &bcc, AthenaArray<Real> &cons,
               AthenaArray<Real> &cons_scalar);

Real MyTimeStep(MeshBlock *pmb);

//Floors for Diode boundary conds
Real dfloor, pfloor; // Floor values for density and rpessure


Real Y(Real T) {
  //uniform bins in log T
  int z = std::floor((std::log10(T*T_scale) -2)/ 0.01);
  if (z < 0) {
    z = 0;
  }
  if (z >= Tbins) {
    z = Tbins - 1;
  }
  //Now compute Y(T)
  Real a1 = 1.0/(1.0 - aks(z));
  Real a2 = (LN_arr(0) / Lks(z));
  Real a3 = std::pow( Tlows(z) , -aks(z));
  Real a4 = Tlows(z) / Tmax_arr(0);
  Real a5 = 1.0 - std::pow( (Tlows(z) / (T*T_scale)) , (aks(z) - 1.0) );
  return Yks(z) + a1*a2*a3*a4*a5;
}
Real alpha(Real T) {
  //uniform bins in log T
  int z = std::floor((std::log10(T*T_scale) -2)/ 0.01);
  if (z < 0) {
    z = 0;
  }
  if (z >= Tbins) {
    z = Tbins - 1;
  }
  
  return aks(z);
}

Real lambda(Real T) {
  //uniform bins in log T
  int z = std::floor((std::log10(T*T_scale) -2)/ 0.01);
  if (z < 0) {
    z = 0;
  }
  if (z >= Tbins) {
    z = Tbins - 1;
  }
  
  return (Lks(z)* std::pow(T*T_scale, aks(z)))/(e_scale  /(n_scale*n_scale*t_scale)) ;
}


Real Yinv(Real y) {
  //find bracketing indicies of y in Yks array with binary tree
  if (y <= Yks(Tbins)) {
    // std::cout << "y = " << y << " Yks(Tbins) = " << Yks(Tbins) << std::endl;
    return Tupps(Tbins-1)/ T_scale; // return in code units
  } 
  if (y >= Yks(0)) {
    // std::cout << "y = " << y << " Yks(0) = " << Yks(0) << std::endl;
    return Tlows(0)/T_scale; // return in code units
  }
  int low = 0;
  int high = Tbins;
  int mid;
  while (high - low > 1.01) {
    mid = std::floor((high + low) / 2);
    if (Yks(mid) < y) {
      high = mid;
    } else {
      low = mid;
    }
  }
  // std::cout << "low = " << low << " high = " << high << std::endl;
  if (high != low + 1) {
    throw std::runtime_error("### FATAL ERROR in realistic_grav_SN.cpp: Binary search failed in Yinv!");
  }
  Real a1 = (Lks(low)/LN_arr(0));
  Real a2 = std::pow( Tlows(low) , aks(low));
  Real a3 = Tmax_arr(0)/(Tlows(low));
  Real a4 = (y - Yks(low));
  Real q = 1.0 - (1.0 - aks(low))*a1*a2*a3*a4;
  Real T = Tlows(low) * std::pow(q, 1.0/(1.0 - aks(low)));
  return T / T_scale; // return in code units
}

Real Heating(Real z){
  return HeatingRate ;//* std::exp(-1* (potential(z) - potential(0)) * dens0/(pres0 * (1 + invbeta)));
}

void Mesh::InitUserMeshData(ParameterInput *pin) {
  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  if (rank == 0){
    std::cout << "Temp Scale = " << T_scale << std::endl;
    std::cout << "v Scale    = " << v_scale << std::endl;
    std::cout << "e Scale    = " << e_scale << std::endl;
    std::cout << "B Scale    = " << B_scale << std::endl;
    std::cout << "multilevel = " << multilevel << std::endl;
  }

  //Load in parameters
  if (MAGNETIC_FIELDS_ENABLED){
    invbeta = pin->GetReal("problem","invbeta");
    angle = pin->GetOrAddReal("problem","B_angle",0.0) * PI/180.0; // in radians
  }
  
  Real T0 = pin->GetReal("problem", "T0"); // now T is in K
  Real n0 = pin->GetReal("problem", "n0"); // midplane particles per cm^3

  Real gm1 = pin->GetReal("hydro","gamma") - 1;
  
  dens0 = n0 / n_scale; 

  pres0 = k_B*T0 *n0 / e_scale;

  pfloor = pin->GetReal("hydro","pfloor");
  dfloor = pin->GetReal("hydro","dfloor");

  max_dt = pin->GetOrAddReal("problem","max_dt",FLT_MAX);

  Real dx = (pin->GetReal("mesh","x1max") - pin->GetReal("mesh","x1min"))/(pin->GetInteger("mesh","nx1"));
  injL = pin->GetReal("problem","InjL") * dx;
  Esn_th = pin->GetOrAddReal("problem","Esn_th",1) * 1.0e51/(e_scale*pow(l_scale,3));
  Esn_mom = pin->GetOrAddReal("problem","Esn_mom",0.0) * 1.0e51/(e_scale*pow(l_scale,3));
  Msn = pin->GetOrAddReal("problem","Msn",1.0) * M_sun/(rho_scale*pow(l_scale,3));
  EnrollUserTimeStepFunction(MyTimeStep);

  
  Tbins = 600;
  int start_file[2] = {0, 20};
  int count_fileY[2] = {Tbins + 1, 1};
  int count_fileT[2] = {Tbins, 1};
  int start_mem[1] = {0};
  int count_memY[1] = {Tbins + 1};
  int count_memT[1] = {Tbins};
  int start_fileLN[1] = {20};
  int count_scalar[1] = {1};
  

  Yks.NewAthenaArray(Tbins + 1);
  Tlows.NewAthenaArray(Tbins);
  Tupps.NewAthenaArray(Tbins);
  Lks.NewAthenaArray(Tbins);
  aks.NewAthenaArray(Tbins);
  LN_arr.NewAthenaArray(1);
  Tmax_arr.NewAthenaArray(1);
  HDF5ReadRealArray("cooling.hdf5", "Yks", 2, start_file, count_fileY, 1, start_mem, count_memY, Yks);
  HDF5ReadRealArray("cooling.hdf5", "Lks", 2, start_file, count_fileT, 1, start_mem, count_memT, Lks);
  HDF5ReadRealArray("cooling.hdf5", "aks", 2, start_file, count_fileT, 1, start_mem, count_memT, aks);
  HDF5ReadRealArray("cooling.hdf5", "Tlows", 1, start_mem, count_memT, 1, start_mem, count_memT, Tlows);
  HDF5ReadRealArray("cooling.hdf5", "Tupps", 1, start_mem, count_memT, 1, start_mem, count_memT, Tupps);
  HDF5ReadRealArray("cooling.hdf5", "Tmax", 1, start_mem, count_scalar, 1, start_mem, count_scalar, Tmax_arr);
  HDF5ReadRealArray("cooling.hdf5", "LN", 1, start_fileLN, count_scalar, 1, start_mem, count_scalar, LN_arr);
  
  HeatingRate = dens0 * lambda(T0/T_scale) ;//pin->GetOrAddReal("problem","HeatingRate",2e-26)/(e_scale/t_scale);
  cool_CFL = pin->GetOrAddReal("problem","cool_CFL",0.5);
  EnrollUserExplicitSourceFunction(mySource);

  if (rank == 0) {
    std::cout << "Loaded Cooling Function from cooling.hdf5: " << std::endl;
    std::cout << "Tmax = " << Tmax_arr(0) << std::endl;
    std::cout << "LN   = " << LN_arr(0) << std::endl;
    Real Tmax = Tmax_arr(0) / T_scale;
    Real LN = LN_arr(0) / (e_scale/t_scale * SQR(n_scale)); 
    Real dt = 1e-3;
    Real Temp0 = pres0/dens0;
    Real tcool = pow(dens0*gm1 * LN / (Tmax),-1);
    Real Tnp1 = Yinv( Y(Temp0) + dt / tcool );
    Real cool1 = 1/(gm1) * (Temp0 - Tnp1) / dt;
    Real heat = Heating(0) ;  
    Real cool2 = dens0*lambda(Temp0);
    std::cout << "Cooling rate from Yinv = " << cool1 << std::endl;
    std::cout << "Cooling rate at d0, T0 = " << cool2 << std::endl;
    std::cout << "Heating rate at d0, T0 = " << heat << std::endl;
    std::cout << "net Time = " << (Temp0/(gm1))/(heat - cool2) << " t_scale " << std::endl;
    std::cout << "First 5 Yks, Lks, aks: " << std::endl;
    for (int i=0; i<5; i++) {
      std::cout << Yks(i) << " " << Lks(i) << " " << aks(i) << std::endl;
    }
    std::cout << "Last 5 Yks, Lks, aks: " << std::endl;
    for (int i=595; i<600; i++) {
      std::cout << Yks(i+1) << " " << Lks(i) << " " << aks(i) << std::endl;
    }
    std::cout << "Tupps and Tlows: " << std::endl;
    for (int i=0; i<5; i++) {
      std::cout << Tlows(i) << " " << Tupps(i) << std::endl;
    }
    std::cout << "..." << std::endl;
    for (int i=595; i<600; i++) {
      std::cout << Tlows(i) << " " << Tupps(i) << std::endl;
    } 
    AthenaArray<Real> errT;
    errT.NewAthenaArray(2*Tbins);
    Real MaxErr = 1e-9;

    for (int i=0; i<Tbins; i++) {
      errT(2*i) = Yinv( Y(Tlows(i)/T_scale) ) - Tlows(i) / T_scale; 
      Real T = 0.5*(Tlows(i) + Tupps(i))  ;
      errT(2*i+1) = Yinv( Y(T/T_scale) ) - T/T_scale ;
      if ( std::abs(errT(2*i)) > MaxErr ) {
        std::cout << "i = " << i << " T = " << Tlows(i) << " Yinv(Y(T)) - T = " << errT(2*i) << std::endl;
      }
      if ( std::abs(errT(2*i+1)) > MaxErr ) {
        std::cout << "i = " << i << " T = " << T << " Yinv(Y(T)) - T = " << errT(2*i+1) << std::endl;   
      }
    }
    // std::cout << "Max error in Y and Yinv = " << MaxErr << " at T = " << MaxT << std::endl;
    errT.DeleteAthenaArray();
    // Test values
    // std::cout << "57395 K = " <<  57395/T_scale << std::endl;
    // std::cout << "Y(57395) = " << Y( 57395/T_scale) << std::endl;
    // std::cout << "Yinv(Y(57395)) = " << Yinv(Y( 57395/T_scale)) << std::endl;
  }
  // throw std::runtime_error("### FATAL ERROR break point to check cooling function");
  
  return;
}


void MeshBlock::ProblemGenerator(ParameterInput *pin) {
  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  Mesh *pm = pmy_mesh; 
  Real myGamma = pin->GetReal("hydro","gamma");

  Real gm1 = myGamma - 1;

  Real x10 = 0.5*( pm->mesh_size.x1max + pm->mesh_size.x1min);
  Real x20 = 0.5*( pm->mesh_size.x2max + pm->mesh_size.x2min);
  Real x30 = 0.5*( pm->mesh_size.x3max + pm->mesh_size.x3min);
  // Initialize hydro variable
  for(int k=ks; k<=ke; ++k) {
    for (int j=js; j<=je; ++j) {
      for (int i=is; i<=ie; ++i) {
        Real x1 = pcoord->x1v(i);
        Real x2 = pcoord->x2v(j) ;
        Real x3 = pcoord->x3v(k);
        Real dx1 = pcoord->dx1v(i);
        Real dx2 = pcoord->dx2v(j);
        Real dx3 = pcoord->dx3v(k);
        Real cellVol = pcoord->GetCellVolume(k,j,i);

        Real T0 = pres0/dens0;
        Real dens = dens0 ;
        Real pres = dens * T0 ;


        phydro->u(IDN, k, j, i) = dens;
        phydro->u(IM1, k, j, i) = 0.0;
        phydro->u(IM2, k, j, i) = 0.0;
        phydro->u(IM3, k, j, i) = 0.0;
        //energy
        if (NON_BAROTROPIC_EOS) {
            phydro->u(IEN, k, j, i) = pres/gm1;
        }

        //Apply Injection
        Real dist = std::sqrt(SQR(x1 - (x10+0.5*dx1)) +  SQR(x2 - (x20+0.5*dx2)) +  SQR(x3 - (x30+0.5*dx3)) );
        Real SN_Vol = 4*M_PI/3*std::pow(injL,3);
        
        if (dist <= injL) {
          phydro->u(IEN,k,j,i) += Esn_th/SN_Vol;
          phydro->u(IDN,k,j,i) += Msn/SN_Vol;
          Real mom0 = std::sqrt(2*(Esn_mom/SN_Vol) *(phydro->u(IDN,k,j,i)));

          if ((dist > 0) ){
            phydro->u(IM1,k,j,i) += mom0 * (x1)/dist;
            phydro->u(IM2,k,j,i) += mom0 * (x2)/dist;
            phydro->u(IM3,k,j,i) += mom0 * (x3)/dist;
            phydro->u(IEN,k,j,i) += 0.5*SQR(mom0)/(phydro->u(IDN,k,j,i));
          } 
        }
      }
    }
  }


  if (MAGNETIC_FIELDS_ENABLED) {
    for (int k=ks; k<=ke; ++k) {
      for (int j=js; j<=je; ++j) {
        for (int i=is; i<=ie+1; ++i) {

          Real T0 = pres0/dens0;
          Real dens = dens0 ;
          Real pres = dens * T0 ;
          Real b0 = sqrt(2*pres*invbeta);
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
            Real T0 = pres0/dens0;
            Real dens = dens0 ;
            Real pres = dens * T0 ;
            Real b0 = sqrt(2*pres*invbeta);
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

}


//----------------------------------------------------------------------------------------
void Mesh::UserWorkInLoop(void)
{
 
}



void mySource(MeshBlock *pmb, const Real time, const Real dt,
               const AthenaArray<Real> &prim, const AthenaArray<Real> &prim_scalar,
               const AthenaArray<Real> &bcc, AthenaArray<Real> &cons,
               AthenaArray<Real> &cons_scalar){

  Mesh *pm = pmb->pmy_mesh;
  Real gm1 = pmb->peos->GetGamma() - 1;
  

  // Build Townsend Cooling Functions
  Real Tmax = Tmax_arr(0) / T_scale;
  Real LN = LN_arr(0) / (e_scale/(t_scale * SQR(n_scale))); 
  Real Tfloor = Tlows(0) / T_scale; // in code units
  Real Tceil = Tmax_arr(0) / T_scale; // in code units
  
  

  for (int k=pmb->ks; k<=pmb->ke; ++k) {
    for (int j=pmb->js; j<=pmb->je; ++j) {
#pragma omp simd
      for (int i=pmb->is; i<=pmb->ie; ++i) {
        Real d = prim(IDN,k,j,i);
        Real p = prim(IPR,k,j,i);
        Real x1 = pmb->pcoord->x1v(i);
        Real x2 = pmb->pcoord->x2v(j);
        Real x3 = pmb->pcoord->x3v(k);
        Real dx1 = pmb->pcoord->dx1v(i);
        Real dx2 = pmb->pcoord->dx2v(j);
        Real dx3 = pmb->pcoord->dx3v(k);
        Real cellVol = pmb->pcoord->GetCellVolume(k,j,i);
        
        
        //COOLING and HEATING
        if ((d> dfloor) && (p> pfloor) ) {
        // if (d> dfloor)  {
          Real T = p/d;
          if (T <= Tfloor) { 
            // Apply floor heating
            cons(IEN,k,j,i) += Heating(x2) * d * dt;//(Tfloor - T)*d/(gm1);
          } else if (T >= Tceil) {
            // Apply ceiling cooling
            cons(IEN,k,j,i) -= d*d*dt*lambda(T);
            // cons(IEN,k,j,i) += (T - Tceil)*d/(gm1);
          } else {
            //Find cooling and heating rates
            Real heat_rate = Heating(x2);
            Real cool_rate = d*lambda(T);
            Real tnet =  cons(IEN,k,j,i) / (heat_rate - cool_rate);
            Real net = 0.0;
            if (cool_CFL * std::fabs(tnet) > dt ) {
              net = (heat_rate - cool_rate)*dt*d;
            } else {
              Real heat = Heating(x2) * d * dt;
              Real cool = 0.0;
              Real tcool = pow(d*gm1 * LN / (Tmax),-1);
              Real Tnp1 = Yinv( Y(T) + dt / tcool );
              cool = d/(gm1) * (T - Tnp1);
              net = heat - cool;
              Real newT = T + net * gm1 / d;
              if (newT < Tfloor) {
                net = (Tfloor - T)* d / gm1;
              } else if (newT > Tceil) {
                // net = -1*d*d*dt*lambda(T);
                net = (Tceil - T)* d / gm1;
              } else {
                // do nothing
              }
            }
            cons(IEN,k,j,i) += net;
          }
        }
      }
    }
  }
  return;
}

Real MyTimeStep(MeshBlock *pmb)
{
  // Real min_dt=1e-4;

  // for (int k=pmb->ks; k<=pmb->ke; ++k) {
  //   for (int j=pmb->js; j<=pmb->je; ++j) {
  //     for (int i=pmb->is; i<=pmb->ie; ++i) {
  //       Real dt;
  //       dt = ... // calculate your own time step here
  //       min_dt = std::min(min_dt, dt);
  //     }
  //   }
  // }
  // if (Ninjs >1) {
  //   return 1e-6;
  // } else {
  return max_dt;
  // }
}
