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
#include <iostream>   // endl
#include <fstream>
#include <sstream>    // stringstream
#include <stdexcept>  // runtime_error
#include <string>     // c_str()
#include <cmath>      // sqrt
#include <algorithm>  // min

// Athena++ headers
#include "../athena.hpp"
#include "../athena_arrays.hpp"
#include "../mesh.hpp"
#include "../parameter_input.hpp"
#include "../hydro/hydro.hpp"
#include "../hydro/eos/eos.hpp"
#include "../bvals/bvals.hpp"
#include "../hydro/srcterms/srcterms.hpp"
#include "../field/field.hpp"
#include "../coordinates/coordinates.hpp"
#include "../radiation/radiation.hpp"
#include "../radiation/integrators/rad_integrators.hpp"


//======================================================================================
/*! \file beam.cpp
 *  \brief Beam test for the radiative transfer module
 *
 *====================================================================================*/


void Mesh::InitUserMeshProperties(ParameterInput *pin)
{

  return;
}

//======================================================================================
//! \fn void Mesh::TerminateUserMeshProperties(void)
//  \brief Clean up the Mesh properties
//======================================================================================
void Mesh::TerminateUserMeshProperties(void)
{
  // nothing to do
  return;
}



//======================================================================================
//! \fn void MeshBlock::ProblemGenerator(ParameterInput *pin)
//  \brief beam test
//======================================================================================
void MeshBlock::ProblemGenerator(ParameterInput *pin)
{

  Real tgas, er;
  int flag=1;
  if(flag==1){
    tgas=1.0;
    er =1.0;
  
  }
  
  Real gamma = phydro->peos->GetGamma();
  
  // Initialize hydro variable
  for(int k=ks; k<=ke; ++k) {
    for (int j=js; j<=je; ++j) {
      for (int i=is; i<=ie; ++i) {
        phydro->u(IDN,k,j,i) = 1.0;
        phydro->u(IM1,k,j,i) = 3.0;
        phydro->u(IM2,k,j,i) = 0.0;
        phydro->u(IM3,k,j,i) = 0.0;
        if (NON_BAROTROPIC_EOS){

          phydro->u(IEN,k,j,i) = tgas/(gamma-1.0);
          phydro->u(IEN,k,j,i) += 0.5*SQR(phydro->u(IM1,k,j,i))/phydro->u(IDN,k,j,i);
          phydro->u(IEN,k,j,i) += 0.5*SQR(phydro->u(IM2,k,j,i))/phydro->u(IDN,k,j,i);
          phydro->u(IEN,k,j,i) += 0.5*SQR(phydro->u(IM3,k,j,i))/phydro->u(IDN,k,j,i);
        }
      }
    }
  }
  
  //Now initialize opacity and specific intensity
  if(RADIATION_ENABLED){
    int nfreq = prad->nfreq;
    int nang = prad->nang;
    AthenaArray<Real> ir_cm;
    ir_cm.NewAthenaArray(prad->n_fre_ang);

    Real *ir_lab;
    
    for(int k=ks; k<=ke; ++k) {
      for (int j=js; j<=je; ++j) {
        for (int i=is; i<=ie; ++i) {
         

          Real vx = phydro->u(IM1,k,j,i)/phydro->u(IDN,k,j,i);
          Real vy = phydro->u(IM2,k,j,i)/phydro->u(IDN,k,j,i);
          Real vz = phydro->u(IM3,k,j,i)/phydro->u(IDN,k,j,i);
          Real *mux = &(prad->mu(0,k,j,i,0));
          Real *muy = &(prad->mu(1,k,j,i,0));
          Real *muz = &(prad->mu(2,k,j,i,0));
          
          ir_lab = &(prad->ir(k,j,i,0));
        
//          prad->pradintegrator->ComToLab(vx,vy,vz,mux,muy,muz,ir_cm,ir_lab);
          for(int n=0; n<prad->n_fre_ang; n++){
             ir_lab[n] = er;
          }
          
          for (int ifr=0; ifr < nfreq; ++ifr){
            prad->sigma_s(k,j,i,ifr) = 0.0;
            prad->sigma_a(k,j,i,ifr) = 100.0;
            prad->sigma_ae(k,j,i,ifr) = 100.0;
            
          }
        }
      }
    }
    
    ir_cm.DeleteAthenaArray();
    
  }// End Rad
  
  return;
}

//======================================================================================
//! \fn void MeshBlock::UserWorkInLoop(void)
//  \brief User-defined work function for every time step
//======================================================================================

void MeshBlock::UserWorkInLoop(void)
{
  // nothing to do
  return;
}

void Radiation::LoadInternalVariable()
{
  int n1z = pmy_block->block_size.nx1 + 2*(NGHOST);
  int n2z = pmy_block->block_size.nx2;
  int n3z = pmy_block->block_size.nx3;
  if(n2z > 1) n2z += (2*(NGHOST));
  if(n3z > 1) n3z += (2*(NGHOST));
  
  for(int n=0; n<NRADFOV; ++n)
    for(int k=0; k<n3z; ++k)
      for(int j=0; j<n2z; ++j)
        for(int i=0; i<n1z; ++i){
          rad_ifov(n,k,j,i) = 0.0;
        }
 
  for(int n=0; n<NRADFOV; ++n)
  for(int k=0; k<n3z; ++k)
    for(int j=0; j<n2z; ++j)
      for(int i=0; i<n1z; ++i)
        for(int ifr=0; ifr<nfreq; ++ifr){
          rad_ifov(n,k,j,i) += wfreq(ifr) * ir(k,j,i,ifr*nang+n);
        }
  
  
  
  return;
}




