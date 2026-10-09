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
#include <limits>     // numeric_limits
#include <sstream>    // stringstream
#include <stdexcept>  // runtime_error
#include <string>     // c_str()
#include <cfloat>      // FLT_MAX

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
#include "../scalars/scalars.hpp"


//======================================================================================
//! \fn void MeshBlock::ProblemGenerator(ParameterInput *pin)
//  \brief Stratified medium with supernova injections and realistic gravitational profile
//======================================================================================

//Gravitational parameters
Real A, B, C, D, E, R, rh, rho_0, z_peak, rho_tigress;

//Initial paramters
Real pres0, dens0, invbeta, angle;

//Cooling Parameters
Real HeatingRate, HeatingMin;
int Tbins;
AthenaArray<Real> Lks, aks, Tlows, Tupps, Yks, Tmax_arr, LN_arr;
Real Y(Real T);
Real invY(Real T);


//Injection information
int NInjections = 0;
double Esn_th;
double Esn_mom;
double Msn;
double injL;
Real max_dt;
Real eps_floor;
Real sh_tol;
Real IsShock(Real rho1, Real v1, Real p1, Real rho2, Real v2, Real p2);


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
const Real lam_scale = e_scale/(n_scale*n_scale*t_scale);

const Real footprints[204] = {
     1,     7,    19,    27,    57,    81,   123,   147,   171,   179,   203,   251,  
   305,   341,   365,   389,   461,   587,   619,   739,   799,   847,   895,   925, 
   949,  1021,  1213,  1237,  1357,  1503,  1551,  1575,  1647,  1743,  1791,  1839,  
  1863,  1935,  2007,  2103,  2205,  2325,  2373,  2517,  2801,  2945,  3071,  3119,
  3191,  3239,  3287,  3407,  3575,  3743,  3791,  3887,  4067,  4139,  4385,  4457, 
  4553,  4625,  4697,  4729,  4801,  4945,  5041,  5137,  5185,  5257,  5449,  5695, 
  5743,  5887,  6031,  6187,  6235,  6403,  6451,  6619,  6667,  6763,  6859,  6931,
  6979,  7075,  7123,  7441,  7497,  7689,  7809,  7881,  8217,  8289,  8385,  8601,
  8709,  8733,  8829,  8925,  9045,  9315,  9435,  9459,  9771,  9795, 10059, 10131,
 10251, 10395, 10827, 11067, 11363, 11459, 11513, 11633, 11753, 11981, 12053, 12149,
 12533, 13205, 13301, 13397, 13517, 13613, 13805, 13949, 14147, 14243, 14363, 14531,
 14771, 14939, 15275, 15515, 15791, 15895, 15967, 16135, 16279, 16375, 16519, 16831,
 16879, 17071, 17269, 17365, 17461, 17845, 18325, 18853, 18949, 19093, 19309, 19381,      
 21631, 21975, 22119, 22335, 22743, 22983, 23583, 23703, 23847, 24111, 24207, 24573, 
 24837, 24885, 25413, 25533, 25677, 26145, 27201, 27369, 27609, 27825, 28017, 28113, 
 28353, 28425, 28887, 28991, 29279, 29423, 29711, 29855, 30047, 30095, 30215, 30647, 
 31103, 31439, 31799, 31919, 32231, 32423, 32531, 32675, 32795, 32987, 33059, 33371};

void mySource(MeshBlock *pmb, const Real time, const Real dt,
               const AthenaArray<Real> &prim, const AthenaArray<Real> &prim_scalar,
               const AthenaArray<Real> &bcc, AthenaArray<Real> &cons,
               AthenaArray<Real> &cons_scalar);

Real MyTimeStep(MeshBlock *pmb);

//Floors for Diode boundary conds
Real dfloor, pfloor; // Floor values for density and rpessure

//x3 boundaries with vacuum
void DiodeInnerX3(MeshBlock *pmb, Coordinates *pco, AthenaArray<Real> &prim,
                            FaceField &b, Real time, Real dt,
                            int il, int iu, int jl, int ju, int kl, int ku, int ngh);
void DiodeOuterX3(MeshBlock *pmb, Coordinates *pco, AthenaArray<Real> &prim,
                            FaceField &b, Real time, Real dt,
                            int il, int iu, int jl, int ju, int kl, int ku, int ngh);

Real (*gravity)(Real z);
// Return signed gravitational acceleration at height z

Real (*potential)(Real z);
// Return gravitational potential at height z

Real grav_none(Real z) {

  return 0.0;
}
Real pot_none(Real z) {
  return 0.0;
}

// SILCC FUNCTIONS
Real gravity_SILCC(Real z) {
  //In SILCC Case, Assume A is sigma_star in Msun/pc^2 and B is z_star in code units (pc)
  Real g0 = 2*PI*G*(A*(M_sun/pow(parsec,2))) /(l_scale / pow(t_scale,2)); // in code units
  Real zd = B  * parsec; // in code units
  return -1* g0 * tanh(z * l_scale/(2*zd));
}
Real potential_SILCC(Real z) {
  //In SILCC Case, Assume A is sigma_star in Msun/pc^2 and B is z_star in code units (pc)
  Real g0 = 2*PI*G*(A*(M_sun/pow(parsec,2))) /(l_scale / pow(t_scale,2)); // in code units
  Real zd = B  * parsec; // in code units
  // std::cout << "My Estimate = "<<  g0 << " cm/s^2"<< std::endl;
  return  g0*(2*zd/l_scale) * std::log( cosh(z*l_scale/(2*zd)) );
}

//TIGRESS FUNCTIONS
Real gravity_TIGRESS(Real z) {
  Real g0 = 2*PI*G;
  //Not sure about if scaling stuff is correct
  Real zstar = B * parsec ;
  Real full_rho = rho_tigress * (M_sun/pow(parsec,3));
  return -1 * g0 * (((2*z*l_scale*full_rho)/(1+pow((z*l_scale/R*l_scale),2))) + ((A*(M_sun/pow(parsec,2))*z*l_scale)/(zstar*(pow((1+pow((z*l_scale/zstar),2)), 0.5))))) / (l_scale / pow(t_scale,2));
  //return gravity_SILCC(z);
}
Real potential_TIGRESS(Real z) {
  Real g0 = 2*PI*G*A*(M_sun/pow(parsec,2))/(l_scale / pow(t_scale,2));
  Real zstar = B * parsec;
  Real R = E * 1000 * parsec;
  Real full_rho = rho_tigress * (M_sun/pow(parsec,3));
  Real rhodm = 2*PI*G* rho_tigress ;
  return g0*(zstar/l_scale)*(pow((1+pow((z*l_scale/zstar),2)),0.5) - 1) + rhodm * (pow(R*l_scale,2)) * std::log(1 + pow((z*l_scale/R*l_scale),2)) / (l_scale / pow(t_scale,2));
  //return potential_SILCC(z);
}


//GalPot Functions
Real gravity_GALPOT(Real z) {
  Real z_peaks = z_peak * parsec * 1000 ;
  //std::cout << "z_peaks =" << z_peak << std::endl ;
  Real gravpin = -1*A*std::tanh((z_peaks/l_scale)/B) - C*std::tanh((z_peaks/l_scale)/D) ;
  //std::cout << "gravpin =" << gravpin << std::endl;
  Real g0 ; 
  if (std::abs(z) <= -1*z_peaks/l_scale) g0 = -1*A*std::tanh(z/B) - C*std::tanh(z/D) ;
  else if (z < z_peaks/l_scale) g0 = gravpin*std::exp(E* pow((std::log(std::abs(z)/(-1*z_peaks/l_scale))), 2)) ;
  else g0 = -1*gravpin*std::exp(E* pow((std::log(std::abs(z)/(-1*z_peaks/l_scale))), 2)) ;
  return g0* 2*PI*G*(M_sun/pow(parsec,2))/(l_scale / pow(t_scale,2));
  //return gravity_SILCC(z);
}
Real potential_GALPOT(Real z) {
  Real rh = rh ;
  Real rho_0_cgs = rho_0 * (M_sun/pow(parsec,3));
  Real z_peaks = z_peak * parsec * 1000 ;
  Real gravpin = -1*A*std::tanh((z_peaks/l_scale)/B) - C*std::tanh((z_peaks/l_scale)/D);
  Real int_const_1 = A*B*std::log(std::cosh((-1*z_peaks/l_scale)/B)) + C*D*std::log(std::cosh((-1*z_peaks/l_scale)/D)) ; 
  Real int_const_2 = ((gravpin*std::exp((1/4)/(-1*E))*(-1*z_peaks/l_scale)*std::sqrt(PI)*std::erf((-1+2*(-1*E)*std::log(-1*z_peaks/(-1*z_peaks)))/(2*std::sqrt(-1*E)))) / (2*std::sqrt(-1*E))) ;
  Real pot ;
  if (std::abs(z) <= -1*z_peaks/l_scale) {
    pot = A*B*std::log(std::cosh(z/B)) + C*D*std::log(std::cosh(z/D));
    //std::cout << "inner potential =" << pot << std::endl;
  }
  else if (z < z_peaks/l_scale) {
    pot = int_const_1 + ((gravpin*std::exp((1/4)/(-1*E))*(-1*z_peaks/l_scale)*std::sqrt(PI)*std::erf((-1+2*(-1*E)*std::log(-1*z/(-1*z_peaks/l_scale)))/(2*std::sqrt(-1*E)))) / (2*std::sqrt(-1*E))) - int_const_2 ;
    //std::cout << "left arm potential =" << pot << std::endl;
  }
  else {
    pot = int_const_1 + ((gravpin*std::exp((1/4)/(-1*E))*(-1*z_peaks/l_scale)*std::sqrt(PI)*std::erf((-1+2*(-1*E)*std::log(z/(-1*z_peaks/l_scale)))/(2*std::sqrt(-1*E)))) / (2*std::sqrt(-1*E))) - int_const_2 ;
    //std::cout << "right arm potential =" << pot << std::endl;
  }
  if (std::isnan(pot)) {
    throw std::runtime_error("### FATAL ERROR in realistic_grav_SN.cpp: potential is nan!");
  }
  return (pot*(2*PI*G)*(M_sun/pow(parsec,2))/(l_scale / pow(t_scale,2)))  ;
  //return potential_SILCC(z);
}
// To get erf not erfi when integrating grav to get pot, must assume E<0, z_peak>0 (specifically when solving in mathematica)


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

Real IsShock(Real rho1, Real v1, Real p1, Real rho2, Real v2, Real p2) {
  if (std::abs(rho2 - rho1) / std::max(rho1,rho2) < 1e-10) {
    return 0.0;
  }
  Real vs = (rho2 * v2 - rho1*v1) / (rho2 - rho1);
  Real mom1 = rho1 * SQR(v1 - vs) + p1;
  Real mom2 = rho2 * SQR(v2 - vs) + p2;
  return std::abs(mom2 - mom1) / std::max(mom1,mom2);
}


Real Heating(Real z){
  Real pos = HeatingRate * std::exp(-1* (potential(z) - potential(0)) * dens0/(pres0 * (1 + invbeta))) ;
  return std::max(pos, HeatingMin);
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

  int Grav_flag = pin->GetOrAddInteger("problem","Grav_flag",0);
  if (Grav_flag == 0) {
    gravity = gravity_SILCC;
    potential = potential_SILCC;
  } else if (Grav_flag == 1) {
    gravity = gravity_TIGRESS;
    potential = potential_TIGRESS;
    rho_tigress = pin->GetReal("problem", "rho_tigress");
  } else if (Grav_flag == 2) {
    gravity = gravity_GALPOT;
    potential = potential_GALPOT;
    z_peak = pin->GetReal("problem", "z_peak");
    rh = pin->GetReal("problem","rh");
    rho_0 = pin->GetReal("problem","rho_0");
  } else {
    gravity = grav_none;
    potential = pot_none;
  }

  A = pin->GetOrAddReal("problem","A",30.0); // surface density in Msun/pc^2
  B = pin->GetOrAddReal("problem","B",100.0); // scale height in pc
  C = pin->GetOrAddReal("problem","C",1.0); 
  D = pin->GetOrAddReal("problem","D",1.0);
  E = pin->GetOrAddReal("problem","E",1.0);  
  R = pin->GetReal("problem","R");


  pfloor = pin->GetReal("hydro","pfloor");
  dfloor = pin->GetReal("hydro","dfloor");

  max_dt = pin->GetOrAddReal("problem","max_dt",FLT_MAX);

  Real dx = (pin->GetReal("mesh","x1max") - pin->GetReal("mesh","x1min"))/(pin->GetInteger("mesh","nx1"));
  Real guessInjL = pin->GetReal("problem","InjL");
  Real finalInjL = std::pow(3.0 * footprints[0] / (4.0 * PI), 1.0 / 3.0);
  for (int Ncells : footprints) {
    Real trueInjL = std::pow(3.0 * Ncells / (4.0 * PI), 1.0 / 3.0);
    if (trueInjL < guessInjL && trueInjL > finalInjL) {
      finalInjL = trueInjL;
    }
  }
  if (rank == 0) {
    std::cout << "Initial guess for injection length (in cells): " << guessInjL << std::endl;
    std::cout << "Final injection length (in cells): " << finalInjL << std::endl;
  }
  injL = finalInjL * dx;
  


  Esn_th = pin->GetOrAddReal("problem","Esn_th",0.8) * 1.0e51/(e_scale*pow(l_scale,3));
  Esn_mom = pin->GetOrAddReal("problem","Esn_mom",0.2) * 1.0e51/(e_scale*pow(l_scale,3));
  Msn = pin->GetOrAddReal("problem","Msn",1.0) * M_sun/(rho_scale*pow(l_scale,3));
  eps_floor = pin->GetOrAddReal("problem","eps_floor",FLT_MIN);
  sh_tol = pin->GetOrAddReal("problem","sh_tol",0.1);
  const Real SN_Vol = 4.0 * PI / 3.0 * std::pow(injL, 3);
  if (rank == 0) {
    std::cout << "Injection radius = " << injL << " code units ("
              << injL * l_scale / parsec << " pc)" << std::endl;
    std::cout << "Injection volume = " << SN_Vol << " code-volume units"
              << std::endl;
    std::cout << "Injected mass = " << Msn << " code-mass units ("
              << Msn * rho_scale * std::pow(l_scale, 3) << " g)" << std::endl;
    std::cout << "Injected thermal energy = " << Esn_th
              << " code-energy units ("
              << Esn_th * e_scale * std::pow(l_scale, 3) << " erg)"
              << std::endl;
    std::cout << "Injected kinetic energy = " << Esn_mom
              << " code-energy units ("
              << Esn_mom * e_scale * std::pow(l_scale, 3) << " erg)"
              << std::endl;

    
  }

  

  EnrollUserTimeStepFunction(MyTimeStep);

  std::string injection_table =
      pin->GetOrAddString("problem", "InjectionTable", "injection_table.txt");
  std::ifstream injection_file(injection_table);
  if (!injection_file.is_open()) {
    throw std::runtime_error("### FATAL ERROR in realGrav_SN_table.cpp: Could not open " + injection_table);
  }
  std::string line;
  double injection_time, injection_x1, injection_x2, injection_x3;
  while (std::getline(injection_file, line)) {
    if (line.empty() || line[0] == '#') {
      continue;
    }
    std::stringstream row(line);
    if (!(row >> injection_time >> injection_x1 >> injection_x2 >> injection_x3)) {
      throw std::runtime_error("### FATAL ERROR in realGrav_SN_table.cpp: Invalid row in " + injection_table);
    }
    ++NInjections;
  }
  AllocateRealUserMeshDataField(4);
  for (int n = 0; n < 4; ++n) {
    ruser_mesh_data[n].NewAthenaArray(NInjections);
  }
  injection_file.clear();
  injection_file.seekg(0);
  int injection_number = 0;
  while (std::getline(injection_file, line)) {
    if (line.empty() || line[0] == '#') {
      continue;
    }
    std::stringstream row(line);
    if (!(row >> injection_time >> injection_x1 >> injection_x2 >> injection_x3)) {
      throw std::runtime_error("### FATAL ERROR in realGrav_SN_table.cpp: Invalid row in " + injection_table);
    }
    ruser_mesh_data[0](injection_number) = injection_time;
    ruser_mesh_data[1](injection_number) = injection_x1 * (parsec / l_scale);
    ruser_mesh_data[2](injection_number) = injection_x2 * (parsec / l_scale);
    ruser_mesh_data[3](injection_number) = injection_x3 * (parsec / l_scale);
    ++injection_number;
  }
  injection_file.close();

  
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
  Tmax_arr(0) = HDF5ReadRealScalar("cooling.hdf5", "Tmax");
  HDF5ReadRealArray("cooling.hdf5", "LN", 1, start_fileLN, count_scalar, 1, start_mem, count_scalar, LN_arr);
  Real MinFactor = pin->GetOrAddReal("problem","HeatingMinFactor",1e-2);
  HeatingRate = dens0 * lambda(T0/T_scale) ;//pin->GetOrAddReal("problem","HeatingRate",2e-26)/(e_scale/t_scale);
  HeatingMin = dens0*MinFactor * lambda(T0/T_scale) ;
  // cool_CFL = pin->GetOrAddReal("problem","cool_CFL",0.5);
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
  
  
  if (mesh_bcs[BoundaryFace::inner_x3] == BoundaryFlag::user) {
    EnrollUserBoundaryFunction(BoundaryFace::inner_x3, DiodeInnerX3);
  }

  if (mesh_bcs[BoundaryFace::outer_x3] == BoundaryFlag::user) {
    EnrollUserBoundaryFunction(BoundaryFace::outer_x3, DiodeOuterX3);
  } 

  //Check if Temp_inj will be higher than T_ceil

  if (rank == 0) {
    Real SN_vol = 4.0 * PI / 3.0 * std::pow(injL, 3);
    Real rho_tot = dens0 + Msn / SN_vol;
    Real e_inj = Esn_th / SN_vol;
    Real T_inj = e_inj / (gm1 * rho_tot );
    Real Tceil = Tmax_arr(0) / T_scale;
    if (T_inj > Tceil) {
      std::cout << "Warning: T_inj (" << T_inj << ") exceeds T_ceil (" << Tceil << ")" << " by a factor " << T_inj / Tceil << std::endl;
    }
  }

  return;
}

void MeshBlock::InitUserMeshBlockData(ParameterInput *pin)
{
  AllocateUserOutputVariables(2);
  return;
}

// can just change it to the user out var being potential and gravity, respectively, and the init stuff above will take care of which one

void MeshBlock::UserWorkBeforeOutput(ParameterInput *pin)
{
  for(int k=ks; k<=ke; k++) {
    for(int j=js; j<=je; j++) {
      for(int i=is; i<=ie; i++) {
        Real z = pcoord->x3v(k);
        user_out_var(0,k,j,i) = potential(z);
        user_out_var(1,k,j,i) = gravity(z);
      }
    }
  }
}


void MeshBlock::ProblemGenerator(ParameterInput *pin) {
  Mesh *pm = pmy_mesh; 
  Real myGamma = pin->GetReal("hydro","gamma");
  
  Real qshear = pin->GetReal("orbital_advection","qshear");
  Real Omega0 = pin->GetReal("orbital_advection","Omega0");

  Real gm1 = myGamma - 1;

  
  // Initialize hydro variable
  for(int k=ks; k<=ke; ++k) {
    for (int j=js; j<=je; ++j) {
      for (int i=is; i<=ie; ++i) {
        Real x3 = pcoord->x3v(k);
        Real x1 = pcoord->x1v(i);

        Real T0 = pres0/dens0;
        Real dens = dens0 * std::exp( -1* (potential(x3) - potential(0)) * dens0/(pres0 * (1 + invbeta)));
        Real pres = dens * T0 ;
        // Real Tz = T0 + (potential(x2) - potential(0))/((1+invbeta)* (alpha(T0)-1));
        // Real dens = HeatingRate / lambda(Tz);  
        // Real pres = dens * T0 ;
        // Real grav = gravity(x2);

        phydro->u(IDN, k, j, i) = dens;
        phydro->u(IM1, k, j, i) = 0.0;
        phydro->u(IM2, k, j, i) = -1*qshear*Omega0*x1*dens;
        phydro->u(IM3, k, j, i) = 0.0;
        //energy
        if (NON_BAROTROPIC_EOS) {
          phydro->u(IEN, k, j, i) = pres/gm1
            + 0.5 * (SQR(phydro->u(IM1, k, j, i))+ SQR(phydro->u(IM2, k, j, i))+ SQR(phydro->u(IM3, k, j, i)))/ phydro->u(IDN, k, j, i);
        }
        if (NSCALARS > 0) {
          pscalars->s(0,k,j,i) = 0.0;
          pscalars->s(1,k,j,i) = 0.0;
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
          Real x3 = pcoord->x3v(k);
          // Real T0 = pres0/dens0;
          // Real Tz = T0 + (potential(x2) - potential(0))/((1+invbeta)* (alpha(T0))-1);
          // Real dens = HeatingRate / lambda(Tz);  
          // Real pres = dens * Tz ;
          Real T0 = pres0/dens0;
          Real dens = dens0 * std::exp( -1* (potential(x3) - potential(0)) * dens0/(pres0 * (1 + invbeta)));
          Real pres = dens * T0 ;
          Real b0 = sqrt(2*pres*invbeta);
          pfield->b.x1f(k,j,i) = b0* std::cos(angle);
        }
      }
    }
    if (block_size.nx3 > 1) {
      for (int k=ks; k<=ke; ++k) {
        for (int j=js; j<=je+1; ++j) {
          for (int i=is; i<=ie; ++i) {
            pfield->b.x3f(k,j,i) = 0.0;
          }
        }
      }
    }
    if (block_size.nx2 > 1) {
      for (int k=ks; k<=ke+1; ++k) {
        for (int j=js; j<=je; ++j) {
          for (int i=is; i<=ie; ++i) {
            Real x3 = pcoord->x3v(k);
            Real T0 = pres0/dens0;
            Real dens = dens0 * std::exp( -1* (potential(x3) - potential(0)) * dens0/(pres0 * (1 + invbeta)));
            Real pres = dens * T0 ;
            // Real T0 = pres0/dens0;
            // Real Tz = T0 + (potential(x2) - potential(0))/((1+invbeta)* (alpha(T0))-1);
            // Real dens = HeatingRate / lambda(Tz);  
            // Real pres = dens * Tz ;
            Real b0 = sqrt(2*pres*invbeta);
            pfield->b.x2f(k,j,i) = b0* std::sin(angle);
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
}


//----------------------------------------------------------------------------------------
void Mesh::UserWorkInLoop(void)
{
}

void MeshBlock::UserWorkInLoop()
{
}



void mySource(MeshBlock *pmb, const Real time, const Real dt,
               const AthenaArray<Real> &prim, const AthenaArray<Real> &prim_scalar,
               const AthenaArray<Real> &bcc, AthenaArray<Real> &cons,
               AthenaArray<Real> &cons_scalar){

  Mesh *pm = pmb->pmy_mesh;
  Real gm1 = pmb->peos->GetGamma() - 1;
  Real source_start = 0.0;
  Real source_end = 0.0;
  const Real time_tolerance = 1.0e-12 * std::max(1.0, std::abs(pm->dt));
  const bool rk2_final_stage =
      std::abs(time - (pm->time + pm->dt)) <= time_tolerance &&
      std::abs(dt - 0.5 * pm->dt) <= time_tolerance;
  const bool rk3_final_stage =
      std::abs(time - (pm->time + 0.5 * pm->dt)) <= time_tolerance &&
      std::abs(dt - (2.0 / 3.0) * pm->dt) <= time_tolerance;
  if ((rk2_final_stage || rk3_final_stage) && dt > 0.0) {
    // Apply impulsive events only after the final RK stage averaging.
    source_start = pm->time;
    source_end = pm->time + pm->dt;
  }
  int first_injection = 0;
  int last_injection = 0;
  if (NInjections > 0 && source_end > source_start) {
    const Real *injection_times = pm->ruser_mesh_data[0].data();
    first_injection = static_cast<int>(
        std::lower_bound(injection_times, injection_times + NInjections,
                         source_start) - injection_times);
    last_injection = static_cast<int>(
        std::lower_bound(injection_times, injection_times + NInjections,
                         source_end) - injection_times);
  }

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

        // Estimate heat dissipation
        if (NSCALARS > 0) {
            // Check both neighboring cells for a shock in each direction.
            Real x_sh = std::max(
                IsShock(prim(IDN,k,j,i), prim(IVX,k,j,i), prim(IPR,k,j,i),
                  prim(IDN,k,j,i-1), prim(IVX,k,j,i-1), prim(IPR,k,j,i-1)),
                IsShock(prim(IDN,k,j,i), prim(IVX,k,j,i), prim(IPR,k,j,i),
                  prim(IDN,k,j,i+1), prim(IVX,k,j,i+1), prim(IPR,k,j,i+1)));
            Real y_sh = std::max(
                IsShock(prim(IDN,k,j,i), prim(IVY,k,j,i), prim(IPR,k,j,i),
                  prim(IDN,k,j-1,i), prim(IVY,k,j-1,i), prim(IPR,k,j-1,i)),
                IsShock(prim(IDN,k,j,i), prim(IVY,k,j,i), prim(IPR,k,j,i),
                  prim(IDN,k,j+1,i), prim(IVY,k,j+1,i), prim(IPR,k,j+1,i)));
            Real z_sh = std::max(
                IsShock(prim(IDN,k,j,i), prim(IVZ,k,j,i), prim(IPR,k,j,i),
                  prim(IDN,k-1,j,i), prim(IVZ,k-1,j,i), prim(IPR,k-1,j,i)),
                IsShock(prim(IDN,k,j,i), prim(IVZ,k,j,i), prim(IPR,k,j,i),
                  prim(IDN,k+1,j,i), prim(IVZ,k+1,j,i), prim(IPR,k+1,j,i)));
          Real is_sh = std::sqrt(SQR(x_sh) + SQR(y_sh) + SQR(z_sh));

          // Calculate dissipated energy
          Real P_ad = p * pow(cons(IDN,k,j,i)/d,gm1+1);
          Real ME = 0.5*(bcc(IB1,k,j,i)*bcc(IB1,k,j,i) + bcc(IB2,k,j,i)*bcc(IB2,k,j,i) + bcc(IB3,k,j,i)*bcc(IB3,k,j,i));
          Real KE = 0.5 * (cons(IM1,k,j,i)*cons(IM1,k,j,i) + cons(IM2,k,j,i)*cons(IM2,k,j,i) + cons(IM3,k,j,i)*cons(IM3,k,j,i)) / cons(IDN,k,j,i);
          Real Eth = cons(IEN,k,j,i) - KE - ME;
          Real P_true = gm1*(Eth);
          Real dS = std::max(P_true - P_ad, 0.0);
          if ((dS > eps_floor) && (is_sh >= sh_tol)) {
            cons_scalar(0,k,j,i) += dS/gm1;
          }
        }


        // GRAVITY
        
        Real grav = gravity(x3);
        Real src = dt*d*grav;

        cons(IM3,k,j,i) += src;
        if (NON_BAROTROPIC_EOS) cons(IEN,k,j,i) += src*prim(IVZ,k,j,i);
        

        //COOLING and HEATING
        if ((d> dfloor) && (p> pfloor) ) {
        // if (d> dfloor)  {
          Real T = p/d;
          if (T <= Tfloor) { 
            // Apply floor heating
            cons(IEN,k,j,i) += (Tfloor - T)*d/(gm1);
          } else if (T >= Tceil) {
            // Apply ceiling cooling
            // cons(IEN,k,j,i) -= d*d*dt*lambda(T);
            cons(IEN,k,j,i) += (Tceil - T)*d/(gm1);
          } else {
            // get basic temp prediction from operator split
            Real T_heat = Heating(x3) * d * dt + T;
            Real tcool = pow(d*gm1 * LN / (Tmax),-1);
            Real T_cool = Yinv( Y(T) + dt / tcool );

            Real T_new = T + (T_heat-T) + (T_cool-T);
            Real sign = (Heating(x3) - d*lambda(T)) > 0 ? 1.0 : -1.0;

            // check for near critical temperatures
            int bin1 = std::floor((std::log10(T_heat*T_scale) -2)/ 0.01);
            int bin2 = std::floor((std::log10(T_cool*T_scale) -2)/ 0.01);
            for (int b =bin2; b <= bin1; b++) {
              if ((b >= 0) && (b < Tbins)) {
                Real Tcrit = std::pow(Heating(x3) /(Lks(b) * d / lam_scale ), 1.0/(aks(b))) * T_scale;
                if ((Tcrit>=Tlows(b)) && (Tcrit<=Tupps(b)) && aks(b) > 1.0) {
                  Real crit_sign = (Tcrit/T_scale - T) > 0 ? 1.0 : -1.0;
                  if (sign == crit_sign) {
                    
                    T_new = Tcrit/T_scale;  
                  }
                }
              }
            }
            if (T_new < Tfloor) {
              T_new = Tfloor;
            } else if (T_new > Tceil) {
              T_new = Tceil;
            } 
            cons(IEN,k,j,i) += (T_new - T)*d/(gm1);
            if (NSCALARS > 0) {
              cons_scalar(1,k,j,i) += (T_new - T)*d/(gm1);
            }
          }
        }

        // Apply events that fall within this explicit source interval.
        if (first_injection < last_injection) {
          const Real SN_Vol = 4*M_PI/3*std::pow(injL,3);
          for (int m = first_injection; m < last_injection; ++m) {
            Real x10 = pm->ruser_mesh_data[1](m);
            Real x20 = pm->ruser_mesh_data[2](m);
            Real x30 = pm->ruser_mesh_data[3](m);
            Real dist = std::sqrt(SQR(x1-x10) + SQR(x2-x20) + SQR(x3-x30));

            if (dist <= injL) {
              
              const Real dmass = Msn / SN_Vol;
              const Real dthermal = Esn_th / SN_Vol;
              cons(IDN,k,j,i) += dmass;
              cons(IM1,k,j,i) += dmass * prim(IVX,k,j,i);
              cons(IM2,k,j,i) += dmass * prim(IVY,k,j,i);
              cons(IM3,k,j,i) += dmass * prim(IVZ,k,j,i);
              cons(IEN,k,j,i) += dthermal + 0.5 * dmass * (SQR(prim(IVX,k,j,i)) + SQR(prim(IVY,k,j,i)) + SQR(prim(IVZ,k,j,i)));

              if ((dist > 0.0) && (Esn_mom > 0.0))  {
                const Real locDens = prim(IDN,k,j,i) + dmass;
                const Real dv_max = std::sqrt(
                    10.0 * Esn_mom / (3.0 * locDens * SN_Vol));
                const Real dv = dv_max * dist / injL;
                cons(IM1,k,j,i) += dv*locDens * (x1-x10)/dist;
                cons(IM2,k,j,i) += dv*locDens * (x2-x20)/dist;
                cons(IM3,k,j,i) += dv*locDens * (x3-x30)/dist;
                const Real oldKE = 0.5 * (SQR(prim(IVX,k,j,i)) + SQR(prim(IVY,k,j,i))
                         + SQR(prim(IVZ,k,j,i))) *  (prim(IDN,k,j,i) + dmass);

                const Real newKE = 0.5 * (SQR( prim(IVX,k,j,i) + dv * (x1-x10)/dist)
                    + SQR( prim(IVY,k,j,i) + dv* (x2-x20)/dist)
                    + SQR( prim(IVZ,k,j,i) + dv* (x3-x30)/dist))
                    * (prim(IDN,k,j,i) + dmass);
                cons(IEN,k,j,i) += newKE - oldKE;
              }
            }
          }
        }

      }
    }
  }
  return;
}

Real MyTimeStep(MeshBlock *pmb)
{
  Mesh *pm = pmb->pmy_mesh;
  Real inj_dt = FLT_MAX;
  Real t_next = pm->time + pm->dt;
  const Real *injection_times = pm->ruser_mesh_data[0].data();
  const int next_injection = static_cast<int>(
      NInjections > 0 ? std::lower_bound(injection_times, injection_times + NInjections,
                             t_next) - injection_times : 0);
  const Real injection_volume = 4.0 * PI / 3.0 * std::pow(injL, 3);
  if (NInjections > 0 && pm->dt > 0.0 && next_injection < NInjections &&
      injection_times[next_injection] <= t_next + 2*pm->dt) {
    const Real gamma = pmb->peos->GetGamma();
    const Real injection_density = Msn / injection_volume;
    const Real injection_pressure = (gamma - 1.0) * Esn_th / injection_volume;
    const Real sound_speed = std::sqrt(gamma * injection_pressure
                                       / injection_density);
    const Real ejecta_speed = std::sqrt(2.0 * Esn_mom / Msn);
    const Real signal_speed = sound_speed + ejecta_speed;
    const Real dx = pmb->pcoord->dx1v(pmb->is);
    inj_dt = pm->cfl_number * dx / signal_speed;
  }
  return std::min(max_dt, inj_dt);
}

//----------------------------------------------------------------------------------------
//! \fn void DiodeInnerX3(MeshBlock *pmb, Coordinates *pco,
//!                             AthenaArray<Real> &prim, FaceField &b, Real time, Real dt,
//!                             int il, int iu, int jl, int ju, int kl, int ku, int ngh)
//! \brief set vacuum outside the simulation. 

void DiodeInnerX3(MeshBlock *pmb, Coordinates *pco, AthenaArray<Real> &prim,
                            FaceField &b, Real time, Real dt,
                            int il, int iu, int jl, int ju, int kl, int ku, int ngh) {
  for (int n=0; n<(NHYDRO); ++n) {
    for (int k=1; k<=ngh; ++k) {
      for (int j=jl; j<=ju; ++j) { 
        if (n==(IPR)) {
#pragma omp simd
          for (int i=il; i<=iu; ++i) {
            prim(IPR,kl-k,j,i) = pfloor;
          }
        } else if (n==IDN) {
#pragma omp simd
          for (int i=il; i<=iu; ++i) {
            prim(n,kl-k,j,i) = dfloor;
          }
        } else {
#pragma omp simd
          for (int i=il; i<=iu; ++i) {
            prim(n,kl-k,j,i) = 0.0;
          }
        }
      }
    }
  }

  // zero face-centered magnetic fields 
  if (MAGNETIC_FIELDS_ENABLED) {
    for (int k=1; k<=ngh; ++k) {
      for (int j=jl; j<=ju; ++j) {   
#pragma omp simd
        for (int i=il; i<=iu+1; ++i) {
          b.x1f((kl-k),j,i) =  0.0;
        }
      }
    }

    for (int k=1; k<=ngh; ++k) {
      for (int j=jl; j<=ju+1; ++j) {   
#pragma omp simd
        for (int i=il; i<=iu; ++i) {
          b.x2f((kl-k),j,i) =  0.0;
        }
      }
    }

    for (int k=1; k<=ngh; ++k) {
      for (int j=jl; j<=ju; ++j) {   
#pragma omp simd
        for (int i=il; i<=iu; ++i) {
          b.x3f((kl-k),j,i) = 0.0;
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

void DiodeOuterX3(MeshBlock *pmb, Coordinates *pco, AthenaArray<Real> &prim,
                            FaceField &b, Real time, Real dt,
                            int il, int iu, int jl, int ju, int kl, int ku, int ngh) {
  for (int n=0; n<(NHYDRO); ++n) {
    for (int k=1; k<=ngh; ++k) {
      for (int j=jl; j<=ju; ++j) {   
        if (n==(IPR)) {
#pragma omp simd
          for (int i=il; i<=iu; ++i) {
            prim(IPR,ku+k,j,i) = pfloor;
          }
        } else if (n==(IDN))  {
#pragma omp simd
          for (int i=il; i<=iu; ++i) {
            prim(IDN,ku+k,j,i) = dfloor;
          }
        } else {
#pragma omp simd
          for (int i=il; i<=iu; ++i) {
            prim(n,ku+k,j,i) = 0.0;
          }
        }
      }
    }
  }

  // zero face-centered magnetic fields
  if (MAGNETIC_FIELDS_ENABLED) {
    for (int k=1; k<=ngh; ++k) {
      for (int j=jl; j<=ju; ++j) {
#pragma omp simd
        for (int i=il; i<=iu+1; ++i) {
          b.x1f((ku+k),j,i) =  0.0;
        }
      }
    }

    for (int k=1; k<=ngh; ++k) {
      for (int j=jl; j<=ju+1; ++j) {  
#pragma omp simd
        for (int i=il; i<=iu; ++i) {
          b.x2f((ku+k),j,i) =  0.0;
        }
      }
    }
    for (int k=1; k<=ngh; ++k) {
      for (int j=jl; j<=ju; ++j) {  
#pragma omp simd
        for (int i=il; i<=iu; ++i) {
          b.x3f((ku+k+1),j,i) = 0.0;
        }
      }
    }
  }

  return;
}