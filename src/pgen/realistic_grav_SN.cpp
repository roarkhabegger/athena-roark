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
//  \brief Stratified medium with supernova injections and realistic gravitational profile
//======================================================================================

//Gravitational parameters
Real A, B, C, D, E;

//Initial paramters
Real pres0, dens0, invbeta, angle;

//Cooling Parameters
Real HeatingRate;
int Tbins;
AthenaArray<Real> Lks, aks, Tlows, Tupps, Yks, Tmax_arr, LN_arr;
Real Y(Real T);
Real invY(Real T);


//Injection information
std::vector<double> X1Inj = {};
std::vector<double> X2Inj = {};
std::vector<double> X3Inj = {};
unsigned seed_inj;
std::default_random_engine gen;
int NInjs = 0;
int TotalInjs = 0;
double SNRate = 0.0;
double injH = 100;
double Esn_th = 0.0;
double injL = 0.0;



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


void mySource(MeshBlock *pmb, const Real time, const Real dt,
               const AthenaArray<Real> &prim, const AthenaArray<Real> &prim_scalar,
               const AthenaArray<Real> &bcc, AthenaArray<Real> &cons,
               AthenaArray<Real> &cons_scalar);



//Floors for Diode boundary conds
Real dfloor, pfloor; // Floor values for density and rpessure

//x2 boundaries with vacuum
void DiodeInnerX2(MeshBlock *pmb, Coordinates *pco, AthenaArray<Real> &prim,
                            FaceField &b, Real time, Real dt,
                            int il, int iu, int jl, int ju, int kl, int ku, int ngh);
void DiodeOuterX2(MeshBlock *pmb, Coordinates *pco, AthenaArray<Real> &prim,
                            FaceField &b, Real time, Real dt,
                            int il, int iu, int jl, int ju, int kl, int ku, int ngh);

Real (*gravity)(Real z);
// Return signed gravitational acceleration at height z

Real (*density)(Real z);
// Return density at height z
// Calculated by dens0* Exp[ ( \int dz g(z) ) / ( pres0/dens0 * (1 + invbeta) ) ]


// SILCC FUNCTIONS
Real gravity_SILCC(Real z) {
  //In SILCC Case, Assume A is sigma_star in Msun/pc^2 and B is z_star in code units (pc)
  Real g0 = 2*PI*G*(A*(M_sun/pow(l_scale,2))) /(v_scale*v_scale/l_scale); // in code units
  Real zd = B ; // in code units
  return -1* g0 * tanh(z/(2*zd));
}
Real density_SILCC(Real z) {
  //In SILCC Case, Assume A is sigma_star in Msun/pc^2 and B is z_star in code units (pc)
  Real g0 = 2*PI*G*(A*(M_sun/pow(l_scale,2))) /(v_scale*v_scale/l_scale); // in code units
  Real zd = B ; // in code units
  return dens0* exp( -2* g0*zd/(pres0*(1+invbeta)/dens0) * std::log( cosh(z/(2*zd)) ) );
}

//TIGRESS FUNCTIONS
Real gravity_TIGRESS(Real z) {
  return gravity_SILCC(z);
}
Real density_TIGRESS(Real z) {
  return density_SILCC(z);
}


//GalPot Functions
Real gravity_GALPOT(Real z) {
  return gravity_SILCC(z);
}
Real density_GALPOT(Real z) {
  return density_SILCC(z);
}


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

Real Yinv(Real y) {
  //find bracketing indicies of y in Yks array with binary tree
  if (y >= Yks(Tbins)) {
    return Tlows(0) / T_scale; // return in code units
  } 
  if (y <= Yks(0)) {
    return Tupps(Tbins-1)/T_scale; // return in code units
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


void Mesh::InitUserMeshData(ParameterInput *pin) {
  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  seed_inj = pin->GetOrAddInteger("problem","seed_inj",0);
  if (rank == 0){
    std::cout << "Temp Scale = " << T_scale << std::endl;
    std::cout << "v Scale    = " << v_scale << std::endl;
    std::cout << "e Scale    = " << e_scale << std::endl;
    std::cout << "B Scale    = " << B_scale << std::endl;
    std::cout << "multilevel = " << multilevel << std::endl;
    unsigned seed1 = std::chrono::system_clock::now().time_since_epoch().count();
    if (seed_inj != 0 ) seed1 = seed_inj;
    gen.seed(seed1);
  }

  //Load in parameters
  if (MAGNETIC_FIELDS_ENABLED){
    invbeta = pin->GetReal("problem","invbeta");
    angle = pin->GetOrAddReal("problem","B_angle",0.0) * PI/180.0; // in radians
  }
  
  Real T0 = pin->GetReal("problem", "T0"); // now T is in K
  Real n0 = pin->GetReal("problem", "n0"); // midplane particles per cm^3

  Real gm1 = pin->GetReal("hydro","gamma") - 1;
  
  dens0 = n0;

  pres0 = k_B*T0 *n0 / e_scale;

  int Grav_flag = pin->GetOrAddInteger("problem","Grav_flag",0);
  if (Grav_flag == 0) {
    gravity = gravity_SILCC;
    density = density_SILCC;
  } else if (Grav_flag == 1) {
    gravity = gravity_TIGRESS;
    density = density_TIGRESS;
  } else if (Grav_flag == 2) {
    gravity = gravity_GALPOT;
    density = density_GALPOT;
  } else {
    throw std::runtime_error("### FATAL ERROR in realistic_grav_SN.cpp: Invalid Grav_flag");
  }

  A = pin->GetOrAddReal("problem","A",30.0);
  B = pin->GetOrAddReal("problem","B",100.0);
  C = pin->GetOrAddReal("problem","C",0.0);
  D = pin->GetOrAddReal("problem","D",0.0);
  E = pin->GetOrAddReal("problem","E",0.0);  

  pfloor = pin->GetReal("hydro","pfloor");
  dfloor = pin->GetReal("hydro","dfloor");


  SNRate = pin->GetReal("problem","SNRate");
  injH = pin->GetOrAddReal("problem","InjH",100); 
  injL = pin->GetReal("problem","InjL");

  
  Esn_th = pin->GetOrAddReal("problem","Esn_th",1) * 1.0e51/(e_scale*pow(l_scale,3));

  HeatingRate = pin->GetOrAddReal("problem","HeatingRate",2e-26)/(e_scale/t_scale);
  
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
  
  
  EnrollUserExplicitSourceFunction(mySource);

  if (rank == 0) {
    std::cout << "Loaded Cooling Function from cooling.hdf5: " << std::endl;
    std::cout << "Tmax = " << Tmax_arr(0) << std::endl;
    std::cout << "LN   = " << LN_arr(0) << std::endl;
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

  }
  // throw std::runtime_error("### FATAL ERROR break point to check cooling function");
  
  
  if (mesh_bcs[BoundaryFace::inner_x2] == BoundaryFlag::user) {
    EnrollUserBoundaryFunction(BoundaryFace::inner_x2, DiodeInnerX2);
  }

  if (mesh_bcs[BoundaryFace::outer_x2] == BoundaryFlag::user) {
    EnrollUserBoundaryFunction(BoundaryFace::outer_x2, DiodeOuterX2);
  } 

  return;
}


void MeshBlock::ProblemGenerator(ParameterInput *pin) {
  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  if (rank==0) {
    std::ofstream myfile;
    myfile.open("injections.csv",std::ios::out | std::ios::app);
    myfile << "Cell,X1,X2,X3\n";
    myfile.close();
  }
  Mesh *pm = pmy_mesh; 
  Real myGamma = pin->GetReal("hydro","gamma");


  Real gm1 = myGamma - 1;


  // Initialize hydro variable
  for(int k=ks; k<=ke; ++k) {
    for (int j=js; j<=je; ++j) {
      for (int i=is; i<=ie; ++i) {
        Real x2 = pcoord->x2v(j);
        Real dens = density(x2);
        Real pres = density(x2)/dens0 * pres0;
        // Real grav = gravity(x2);

        phydro->u(IDN, k, j, i) = dens;
        phydro->u(IM1, k, j, i) = 0.0;
        phydro->u(IM2, k, j, i) = 0.0;
        phydro->u(IM3, k, j, i) = 0.0;
        //energy
        if (NON_BAROTROPIC_EOS) {
            phydro->u(IEN, k, j, i) = pres/gm1;
        }
      }
    }
  }
  // std::cout << " Max HSE Err = " << maxErr << std::endl;
  
  // Add horizontal magnetic field lines, to show streaming and diffusion
  // along magnetic field ines
  if (MAGNETIC_FIELDS_ENABLED) {
    for (int k=ks; k<=ke; ++k) {
      for (int j=js; j<=je; ++j) {
        for (int i=is; i<=ie+1; ++i) {
          Real x2 = pcoord->x2v(j);
          Real press =density(x2)/dens0 * pres0;
          Real b0 = sqrt(2*press*invbeta);
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
            Real press = density(x2)/dens0 * pres0;
            Real b0 = sqrt(2*press*invbeta);
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
  
  int size, rank;
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  X1Inj.clear();
  X2Inj.clear();
  X3Inj.clear();
  NInjs = 0;
  if ((dt < FLT_MAX) && (time > 0.0)) {
    if (rank == 0) {
        std::ofstream myfile;
        myfile.open("injections.csv",std::ios::out | std::ios::app);
        std::poisson_distribution<int> distN(SNRate*dt);
        NInjs = distN(gen);
        Real x1d = (mesh_size.x1max - mesh_size.x1min)/float(mesh_size.nx1);
        Real x2d = (mesh_size.x2max - mesh_size.x2min)/float(mesh_size.nx2);
        Real x3d = (mesh_size.x3max - mesh_size.x3min)/float(mesh_size.nx3);
        std::uniform_real_distribution<double> distx1(mesh_size.x1min+injL/2,mesh_size.x1max - x1d-injL/2);
        std::uniform_real_distribution<double> distx2(-1*injH,injH-x2d);
        std::uniform_real_distribution<double> distx3(mesh_size.x3min+injL/2,mesh_size.x3max - x3d-injL/2);
        for (int n = 1; n <= NInjs; n++){
          X1Inj.insert(X1Inj.end(), round((distx1(gen)-mesh_size.x1min)/x1d)*x1d + mesh_size.x1min + 0.5*x1d);
          X2Inj.insert(X2Inj.end(), round((distx2(gen)-mesh_size.x2min)/x2d)*x2d + mesh_size.x2min + 0.5*x2d);
          X3Inj.insert(X3Inj.end(), round((distx3(gen)-mesh_size.x3min)/x3d)*x3d + mesh_size.x3min + 0.5*x3d);
          myfile <<  0 << ","<< X1Inj[n-1] << "," <<  X2Inj[n-1] << "," <<  X3Inj[n-1] << "\n";
        }
        myfile.close();
    }
    
  } 
  
  MPI_Bcast(&NInjs,1,MPI_INT,0,MPI_COMM_WORLD);

  if ((NInjs > 0) && (rank != 0)){
    X1Inj.insert(X1Inj.end(),NInjs,FLT_MAX);
    X2Inj.insert(X2Inj.end(),NInjs,FLT_MAX);
    X3Inj.insert(X3Inj.end(),NInjs,FLT_MAX);
  }

  MPI_Bcast(X1Inj.data(),NInjs,MPI_DOUBLE,0,MPI_COMM_WORLD);
  MPI_Bcast(X2Inj.data(),NInjs,MPI_DOUBLE,0,MPI_COMM_WORLD);
  MPI_Bcast(X3Inj.data(),NInjs,MPI_DOUBLE,0,MPI_COMM_WORLD);
  TotalInjs += NInjs;
  
}



void mySource(MeshBlock *pmb, const Real time, const Real dt,
               const AthenaArray<Real> &prim, const AthenaArray<Real> &prim_scalar,
               const AthenaArray<Real> &bcc, AthenaArray<Real> &cons,
               AthenaArray<Real> &cons_scalar){

  Mesh *pm = pmb->pmy_mesh;
  Real gm1 = pmb->peos->GetGamma() - 1;
  

  // Build Townsend Cooling Functions
  Real Tmax = Tmax_arr(0) / T_scale;
  Real LN = LN_arr(0) / (e_scale/t_scale * SQR(n_scale)); 
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

        // GRAVITY
        
        Real grav = gravity(x2);
        Real src = dt*d*grav;

        cons(IM2,k,j,i) += src;
        if (NON_BAROTROPIC_EOS) cons(IEN,k,j,i) += src*prim(IVY,k,j,i);

        //COOLING and HEATING
        if ((d> dfloor) && (p> pfloor) ) {
          Real T = p/d;
          if (T <= Tfloor) { 
            // Apply floor heating
            cons(IEN,k,j,i) += (Tfloor - T)*d/(gm1);
          } else if (T >= Tceil) {
            // Apply ceiling cooling
            cons(IEN,k,j,i) += (T - Tceil)*d/(gm1);
          } else {
            //Find cooling and heating rates
            Real heat = HeatingRate* d * dt;

            Real cool = 0.0;
            Real tcool = pow(d*gm1 * LN / (Tmax),-1);
            Real error = Yinv(Y(T)) - T; 
            if ( std::abs(error) > 1e-13 ) {
              std::cout << "### WARNING in realistic_grav_SN.cpp: Inconsistent Y and Yinv! Error = " << error << std::endl;
              throw std::runtime_error("### FATAL ERROR in realistic_grav_SN.cpp: Inconsistent Y and Yinv!");
            }
            Real Tnp1 = Yinv( Y(T) + dt / tcool );
            cool = d/(gm1) * (T - Tnp1);
            Real net = heat - cool;
            Real newT = T + net * gm1 / d;
            if (newT < Tfloor) {
              net = (Tfloor - T)* d / gm1;
            } else if (newT > Tceil) {
              net = (Tceil - T)* d / gm1;
            } else {
              // do nothing
            }
            cons(IEN,k,j,i) += net;
          }
        }

        //INJECTION
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
    }
  }
  return;
}

//----------------------------------------------------------------------------------------
//! \fn void DiodeInnerX2(MeshBlock *pmb, Coordinates *pco,
//!                             AthenaArray<Real> &prim, FaceField &b, Real time, Real dt,
//!                             int il, int iu, int jl, int ju, int kl, int ku, int ngh)
//! \brief set vacuum outside the simulation. 

void DiodeInnerX2(MeshBlock *pmb, Coordinates *pco, AthenaArray<Real> &prim,
                            FaceField &b, Real time, Real dt,
                            int il, int iu, int jl, int ju, int kl, int ku, int ngh) {
  for (int n=0; n<(NHYDRO); ++n) {
    for (int k=kl; k<=ku; ++k) {
      for (int j=1; j<=ngh; ++j) {
        if (n==(IPR)) {
#pragma omp simd
          for (int i=il; i<=iu; ++i) {
            prim(IPR,k,jl-j,i) = pfloor;
          }
        } else if (n==IDN) {
#pragma omp simd
          for (int i=il; i<=iu; ++i) {
            prim(n,k,jl-j,i) = dfloor;
          }
        } else {
#pragma omp simd
          for (int i=il; i<=iu; ++i) {
            prim(n,k,jl-j,i) = 0.0;
          }
        }
      }
    }
  }

  // zero face-centered magnetic fields 
  if (MAGNETIC_FIELDS_ENABLED) {
    for (int k=kl; k<=ku; ++k) {
      for (int j=1; j<=ngh; ++j) {
#pragma omp simd
        for (int i=il; i<=iu+1; ++i) {
          b.x1f(k,(jl-j),i) =  0.0;
        }
      }
    }

    for (int k=kl; k<=ku; ++k) {
      for (int j=1; j<=ngh; ++j) {
#pragma omp simd
        for (int i=il; i<=iu; ++i) {
          b.x2f(k,(jl-j),i) = 0.0;  
        }
      }
    }

    for (int k=kl; k<=ku+1; ++k) {
      for (int j=1; j<=ngh; ++j) {
#pragma omp simd
        for (int i=il; i<=iu; ++i) {
          b.x3f(k,(jl-j),i) =  0.0;
        }
      }
    }
  }

  return;
}
//----------------------------------------------------------------------------------------
//! \fn void DiodeOuterX2(MeshBlock *pmb, Coordinates *pco,
//!                             AthenaArray<Real> &prim, FaceField &b, Real time, Real dt,
//!                             int il, int iu, int jl, int ju, int kl, int ku, int ngh)
//! \brief  Vacuum conditions outside boundary

void DiodeOuterX2(MeshBlock *pmb, Coordinates *pco, AthenaArray<Real> &prim,
                            FaceField &b, Real time, Real dt,
                            int il, int iu, int jl, int ju, int kl, int ku, int ngh) {
  for (int n=0; n<(NHYDRO); ++n) {
    for (int k=kl; k<=ku; ++k) {
      for (int j=1; j<=ngh; ++j) {
        if (n==(IPR)) {
#pragma omp simd
          for (int i=il; i<=iu; ++i) {
            prim(IPR,k,ju+j,i) = pfloor;
          }
        } else if (n==(IDN))  {
#pragma omp simd
          for (int i=il; i<=iu; ++i) {
            prim(IDN,k,ju+j,i) = dfloor;
          }
        } else {
#pragma omp simd
          for (int i=il; i<=iu; ++i) {
            prim(n,k,ju+j,i) = 0.0;
          }
        }
      }
    }
  }

  // zero face-centered magnetic fields
  if (MAGNETIC_FIELDS_ENABLED) {
    for (int k=kl; k<=ku; ++k) {
      for (int j=1; j<=ngh; ++j) {
#pragma omp simd
        for (int i=il; i<=iu+1; ++i) {
          b.x1f(k,(ju+j  ),i) =  0.0;
        }
      }
    }

    for (int k=kl; k<=ku; ++k) {
      for (int j=1; j<=ngh; ++j) {
#pragma omp simd
        for (int i=il; i<=iu; ++i) {
          b.x2f(k,(ju+j+1),i) = 0.0;  
        }
      }
    }

    for (int k=kl; k<=ku+1; ++k) {
      for (int j=1; j<=ngh; ++j) {
#pragma omp simd
        for (int i=il; i<=iu; ++i) {
          b.x3f(k,(ju+j  ),i) =  0.0;
        }
      }
    }
  }

  return;
}