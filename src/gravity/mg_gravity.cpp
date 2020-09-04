//========================================================================================
// Athena++ astrophysical MHD code
// Copyright(C) 2014 James M. Stone <jmstone@princeton.edu> and other code contributors
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
//! \file mg_gravity.cpp
//  \brief create multigrid solver for gravity

// C headers

// C++ headers
#include <algorithm>
#include <iostream>
#include <sstream>    // sstream
#include <stdexcept>  // runtime_error
#include <string>     // c_str()

// Athena++ headers
#include "../athena.hpp"
#include "../athena_arrays.hpp"
#include "../coordinates/coordinates.hpp"
#include "../globals.hpp"
#include "../hydro/hydro.hpp"
#include "../mesh/mesh.hpp"
#include "../multigrid/multigrid.hpp"
#include "../parameter_input.hpp"
#include "gravity.hpp"
#include "mg_gravity.hpp"

#ifdef MPI_PARALLEL
#include <mpi.h>
#endif

class MeshBlock;

//----------------------------------------------------------------------------------------
//! \fn MGGravityDriver::MGGravityDriver(Mesh *pm, ParameterInput *pin)
//  \brief MGGravityDriver constructor

MGGravityDriver::MGGravityDriver(Mesh *pm, ParameterInput *pin)
    : MultigridDriver(pm, pm->MGGravityBoundaryFunction_,
                      pm->MGGravitySourceMaskFunction_, 1) {
  four_pi_G_ = pmy_mesh_->four_pi_G_;
  eps_ = pin->GetOrAddReal("gravity", "threshold", -1.0);
  niter_ = pin->GetOrAddInteger("gravity", "niteration", -1);
  ffas_ = pin->GetOrAddBoolean("gravity", "fas", ffas_);
  std::string m = pin->GetOrAddString("gravity", "mgmode", "none");
  std::transform(m.begin(), m.end(), m.begin(), ::tolower);
  if (m == "fmg") {
    mode_ = 0;
  } else if (m == "mgi") {
    mode_ = 1; // Iterative
  } else {
    std::stringstream msg;
    msg << "### FATAL ERROR in MGGravityDriver::MGGravityDriver" << std::endl
        << "The \"mgmode\" parameter in the <gravity> block is invalid." << std::endl
        << "FMG: Full Multigrid + Multigrid iteration (default)" << std::endl
        << "MGI: Multigrid Iteration" << std::endl;
    ATHENA_ERROR(msg);
  }
  if (eps_ < 0.0 && niter_ < 0) {
    std::stringstream msg;
    msg << "### FATAL ERROR in MGGravityDriver::MGGravityDriver" << std::endl
        << "Either \"threshold\" or \"niteration\" parameter must be set "
        << "in the <gravity> block." << std::endl
        << "When both parameters are specified, \"niteration\" is ignored." << std::endl
        << "Set \"threshold = 0.0\" for automatic convergence control." << std::endl;
    ATHENA_ERROR(msg);
  }
  if (four_pi_G_ == 0.0) {
    std::stringstream msg;
    msg << "### FATAL ERROR in MGGravityDriver::MGGravityDriver" << std::endl
        << "Gravitational constant must be set in the Mesh::InitUserMeshData "
        << "using the SetGravitationalConstant or SetFourPiG function." << std::endl;
    ATHENA_ERROR(msg);
  }

  mg_bcs_[inner_x1]=GetMGBoundaryFlag(pin->GetOrAddString("gravity", "ix1_bc", "none"));
  mg_bcs_[outer_x1]=GetMGBoundaryFlag(pin->GetOrAddString("gravity", "ox1_bc", "none"));
  mg_bcs_[inner_x2]=GetMGBoundaryFlag(pin->GetOrAddString("gravity", "ix2_bc", "none"));
  mg_bcs_[outer_x2]=GetMGBoundaryFlag(pin->GetOrAddString("gravity", "ox2_bc", "none"));
  mg_bcs_[inner_x3]=GetMGBoundaryFlag(pin->GetOrAddString("gravity", "ix3_bc", "none"));
  mg_bcs_[outer_x3]=GetMGBoundaryFlag(pin->GetOrAddString("gravity", "ox3_bc", "none"));
  SetBoundaryFunctions();
  AllocateMultipoleCoefficients();

  // Allocate the root multigrid
  mgroot_ = new MGGravity(this, nullptr);
}


//----------------------------------------------------------------------------------------
//! \fn void MGGravityDriver::SetBoundaryFunctions()
//  \brief Set Multigrid boundary functions from boundary flags.

void MGGravityDriver::SetBoundaryFunctions() {
  fsubtract_average_ = true;
  mp_order_ = 0;
  fskip_dipole_ = false;
  switch(mg_bcs_[BoundaryFace::inner_x1]) {
    case MGBoundaryFlag::user:
      if (MGBoundaryFunction_[BoundaryFace::inner_x1] == nullptr) {
        std::stringstream msg;
        msg << "### FATAL ERROR in MGGravityDriver::SetBoundaryFunctions" << std::endl
            << "A user-defined boundary condition is specified for " << std::endl
            << "inner_x1 but no function is enrolled." << std::endl;
        ATHENA_ERROR(msg);
      }
      fsubtract_average_ = false;
      break;
    case MGBoundaryFlag::periodic:
      MGBoundaryFunction_[BoundaryFace::inner_x1] = MGPeriodicInnerX1;
      break;
    case MGBoundaryFlag::zerograd:
      MGBoundaryFunction_[BoundaryFace::inner_x1] = MGZeroGradientInnerX1;
      break;
    case MGBoundaryFlag::zerofixed:
      MGBoundaryFunction_[BoundaryFace::inner_x1] = MGZeroFixedInnerX1;
      break;
    case MGBoundaryFlag::multipole4:
      MGBoundaryFunction_[BoundaryFace::inner_x1] = MGMultipole4InnerX1;
      fsubtract_average_ = false;
      mp_order_ = std::max(mp_order_, 2);
      break;
    case MGBoundaryFlag::multipole16:
      MGBoundaryFunction_[BoundaryFace::inner_x1] = MGMultipole16InnerX1;
      fsubtract_average_ = false;
      mp_order_ = 4;
      break;
    default:
      std::stringstream msg;
      msg << "### FATAL ERROR in MGGravityDriver::SetBoundaryFunctions" << std::endl
          << "Invalid or no boundary type is specified." << std::endl;
      ATHENA_ERROR(msg);
      break;
  }
  switch(mg_bcs_[BoundaryFace::outer_x1]) {
    case MGBoundaryFlag::user:
      if (MGBoundaryFunction_[BoundaryFace::outer_x1] == nullptr) {
        std::stringstream msg;
        msg << "### FATAL ERROR in MGGravityDriver::SetBoundaryFunctions" << std::endl
            << "A user-defined boundary condition is specified for " << std::endl
            << "outer_x1 but no function is enrolled." << std::endl;
        ATHENA_ERROR(msg);
      }
      fsubtract_average_ = false;
      break;
    case MGBoundaryFlag::periodic:
      MGBoundaryFunction_[BoundaryFace::outer_x1] = MGPeriodicOuterX1;
      break;
    case MGBoundaryFlag::zerograd:
      MGBoundaryFunction_[BoundaryFace::outer_x1] = MGZeroGradientOuterX1;
      break;
    case MGBoundaryFlag::zerofixed:
      MGBoundaryFunction_[BoundaryFace::outer_x1] = MGZeroFixedOuterX1;
      break;
    case MGBoundaryFlag::multipole4:
      MGBoundaryFunction_[BoundaryFace::outer_x1] = MGMultipole4OuterX1;
      fsubtract_average_ = false;
      mp_order_ = std::max(mp_order_, 2);
      break;
    case MGBoundaryFlag::multipole16:
      MGBoundaryFunction_[BoundaryFace::outer_x1] = MGMultipole16OuterX1;
      fsubtract_average_ = false;
      mp_order_ = 4;
      break;
    default:
      std::stringstream msg;
      msg << "### FATAL ERROR in MGGravityDriver::SetBoundaryFunctions" << std::endl
          << "Invalid or no boundary type is specified." << std::endl;
      ATHENA_ERROR(msg);
      break;
  }
  switch(mg_bcs_[BoundaryFace::inner_x2]) {
    case MGBoundaryFlag::user:
      if (MGBoundaryFunction_[BoundaryFace::inner_x2] == nullptr) {
        std::stringstream msg;
        msg << "### FATAL ERROR in MGGravityDriver::SetBoundaryFunctions" << std::endl
            << "A user-defined boundary condition is specified for " << std::endl
            << "inner_x2 but no function is enrolled." << std::endl;
        ATHENA_ERROR(msg);
      }
      fsubtract_average_ = false;
      break;
    case MGBoundaryFlag::periodic:
      MGBoundaryFunction_[BoundaryFace::inner_x2] = MGPeriodicInnerX2;
      break;
    case MGBoundaryFlag::zerograd:
      MGBoundaryFunction_[BoundaryFace::inner_x2] = MGZeroGradientInnerX2;
      break;
    case MGBoundaryFlag::zerofixed:
      MGBoundaryFunction_[BoundaryFace::inner_x2] = MGZeroFixedInnerX2;
      break;
    case MGBoundaryFlag::multipole4:
      MGBoundaryFunction_[BoundaryFace::inner_x2] = MGMultipole4InnerX2;
      fsubtract_average_ = false;
      mp_order_ = std::max(mp_order_, 2);
      break;
    case MGBoundaryFlag::multipole16:
      MGBoundaryFunction_[BoundaryFace::inner_x2] = MGMultipole16InnerX2;
      fsubtract_average_ = false;
      mp_order_ = 4;
      break;
    default:
      std::stringstream msg;
      msg << "### FATAL ERROR in MGGravityDriver::SetBoundaryFunctions" << std::endl
          << "Invalid or no boundary type is specified." << std::endl;
      ATHENA_ERROR(msg);
      break;
  }
  switch(mg_bcs_[BoundaryFace::outer_x2]) {
    case MGBoundaryFlag::user:
      if (MGBoundaryFunction_[BoundaryFace::outer_x2] == nullptr) {
        std::stringstream msg;
        msg << "### FATAL ERROR in MGGravityDriver::SetBoundaryFunctions" << std::endl
            << "A user-defined boundary condition is specified for " << std::endl
            << "outer_x2 but no function is enrolled." << std::endl;
        ATHENA_ERROR(msg);
      }
      fsubtract_average_ = false;
      break;
    case MGBoundaryFlag::periodic:
      MGBoundaryFunction_[BoundaryFace::outer_x2] = MGPeriodicOuterX2;
      break;
    case MGBoundaryFlag::zerograd:
      MGBoundaryFunction_[BoundaryFace::outer_x2] = MGZeroGradientOuterX2;
      break;
    case MGBoundaryFlag::zerofixed:
      MGBoundaryFunction_[BoundaryFace::outer_x2] = MGZeroFixedOuterX2;
      break;
    case MGBoundaryFlag::multipole4:
      MGBoundaryFunction_[BoundaryFace::outer_x2] = MGMultipole4OuterX2;
      fsubtract_average_ = false;
      mp_order_ = std::max(mp_order_, 2);
      break;
    case MGBoundaryFlag::multipole16:
      MGBoundaryFunction_[BoundaryFace::outer_x2] = MGMultipole16OuterX2;
      fsubtract_average_ = false;
      mp_order_ = 4;
      break;
    default:
      std::stringstream msg;
      msg << "### FATAL ERROR in MGGravityDriver::SetBoundaryFunctions" << std::endl
          << "Invalid or no boundary type is specified." << std::endl;
      ATHENA_ERROR(msg);
      break;
  }
  switch(mg_bcs_[BoundaryFace::inner_x3]) {
    case MGBoundaryFlag::user:
      if (MGBoundaryFunction_[BoundaryFace::inner_x3] == nullptr) {
        std::stringstream msg;
        msg << "### FATAL ERROR in MGGravityDriver::SetBoundaryFunctions" << std::endl
            << "A user-defined boundary condition is specified for " << std::endl
            << "inner_x3 but no function is enrolled." << std::endl;
        ATHENA_ERROR(msg);
      }
      fsubtract_average_ = false;
      break;
    case MGBoundaryFlag::periodic:
      MGBoundaryFunction_[BoundaryFace::inner_x3] = MGPeriodicInnerX3;
      break;
    case MGBoundaryFlag::zerograd:
      MGBoundaryFunction_[BoundaryFace::inner_x3] = MGZeroGradientInnerX3;
      break;
    case MGBoundaryFlag::zerofixed:
      MGBoundaryFunction_[BoundaryFace::inner_x3] = MGZeroFixedInnerX3;
      break;
    case MGBoundaryFlag::multipole4:
      MGBoundaryFunction_[BoundaryFace::inner_x3] = MGMultipole4InnerX3;
      fsubtract_average_ = false;
      mp_order_ = std::max(mp_order_, 2);
      break;
    case MGBoundaryFlag::multipole16:
      MGBoundaryFunction_[BoundaryFace::inner_x3] = MGMultipole16InnerX3;
      fsubtract_average_ = false;
      mp_order_ = 4;
      break;
    default:
      std::stringstream msg;
      msg << "### FATAL ERROR in MGGravityDriver::SetBoundaryFunctions" << std::endl
          << "Invalid or no boundary type is specified." << std::endl;
      ATHENA_ERROR(msg);
      break;
  }
  switch(mg_bcs_[BoundaryFace::outer_x3]) {
    case MGBoundaryFlag::user:
      if (MGBoundaryFunction_[BoundaryFace::outer_x3] == nullptr) {
        std::stringstream msg;
        msg << "### FATAL ERROR in MGGravityDriver::SetBoundaryFunctions" << std::endl
            << "A user-defined boundary condition is specified for " << std::endl
            << "outer_x3 but no function is enrolled." << std::endl;
        ATHENA_ERROR(msg);
      }
      fsubtract_average_ = false;
      break;
    case MGBoundaryFlag::periodic:
      MGBoundaryFunction_[BoundaryFace::outer_x3] = MGPeriodicOuterX3;
      break;
    case MGBoundaryFlag::zerograd:
      MGBoundaryFunction_[BoundaryFace::outer_x3] = MGZeroGradientOuterX3;
      break;
    case MGBoundaryFlag::zerofixed:
      MGBoundaryFunction_[BoundaryFace::outer_x3] = MGZeroFixedOuterX3;
      break;
    case MGBoundaryFlag::multipole4:
      MGBoundaryFunction_[BoundaryFace::outer_x3] = MGMultipole4OuterX3;
      fsubtract_average_ = false;
      mp_order_ = std::max(mp_order_, 2);
      break;
    case MGBoundaryFlag::multipole16:
      MGBoundaryFunction_[BoundaryFace::outer_x3] = MGMultipole16OuterX3;
      fsubtract_average_ = false;
      mp_order_ = 4;
      break;
    default:
      std::stringstream msg;
      msg << "### FATAL ERROR in MGGravityDriver::SetBoundaryFunctions" << std::endl
          << "Invalid or no boundary type is specified." << std::endl;
      ATHENA_ERROR(msg);
      break;
  }
}


//----------------------------------------------------------------------------------------
//! \fn MGGravityDriver::~MGGravityDriver()
//  \brief MGGravityDriver destructor

MGGravityDriver::~MGGravityDriver() {
  delete mgroot_;
}


//----------------------------------------------------------------------------------------
//! \fn MGGravity::MGGravity(MultigridDriver *pmd, MeshBlock *pmb)
//  \brief MGGravity constructor

MGGravity::MGGravity(MultigridDriver *pmd, MeshBlock *pmb) : Multigrid(pmd, pmb, 1, 1) {
  btype = BoundaryQuantity::mggrav;
  btypef = BoundaryQuantity::mggrav_f;
  defscale_ = rdx_*rdx_;
  if (pmy_block_ != nullptr)
    pmgbval = new MGGravityBoundaryValues(this, pmy_block_->pbval->block_bcs);
  else
    pmgbval = new MGGravityBoundaryValues(this, pmy_driver_->pmy_mesh_->mesh_bcs);
}


//----------------------------------------------------------------------------------------
//! \fn MGGravity::~MGGravity()
//  \brief MGGravity deconstructor

MGGravity::~MGGravity() {
  delete pmgbval;
}


//----------------------------------------------------------------------------------------
//! \fn void MGGravityDriver::Solve(int stage)
//  \brief load the data and solve

void MGGravityDriver::Solve(int stage) {
  // Construct the Multigrid array
  vmg_.clear();
  for (int i=0; i<pmy_mesh_->nblocal; ++i)
    vmg_.push_back(pmy_mesh_->my_blocks(i)->pmg);

  // load the source
  for (Multigrid* pmg : vmg_) {
    // assume all the data are located on the same node
    pmg->LoadSource(pmg->pmy_block_->phydro->u, IDN, NGHOST, four_pi_G_);
    if (mode_ ==1) // iterative mode - load initial guess
      pmg->LoadFinestData(pmg->pmy_block_->pgrav->phi, 0, NGHOST);
  }

  SetupMultigrid();
  Real mean_rho = 0.0;
  if (fsubtract_average_)
    mean_rho = last_ave_/four_pi_G_;

  if (mode_ == 0) {
    SolveFMGCycle();
  } else {
    if (eps_ >= 0.0)
      SolveIterative();
    else
      SolveIterativeFixedTimes();
  }

  // Return the result
  for (Multigrid* pmg : vmg_) {
    Gravity *pgrav = pmg->pmy_block_->pgrav;
    pmg->RetrieveResult(pgrav->phi, 0, NGHOST);
    pgrav->grav_mean_rho = mean_rho;
    if(pgrav->output_defect)
      pmg->RetrieveDefect(pgrav->def, 0, NGHOST);
  }
  return;
}


//----------------------------------------------------------------------------------------
//! \fn  void MGGravity::Smooth(AthenaArray<Real> &u, const AthenaArray<Real> &src,
//            int rlev, int il, int iu, int jl, int ju, int kl, int ku, int color)
//  \brief Implementation of the Red-Black Gauss-Seidel Smoother
//         rlev = relative level from the finest level of this Multigrid block

void MGGravity::Smooth(AthenaArray<Real> &u, const AthenaArray<Real> &src, int rlev,
                       int il, int iu, int jl, int ju, int kl, int ku, int color) {
  int c = color;
  Real dx;
  if (rlev <= 0) dx = rdx_*static_cast<Real>(1<<(-rlev));
  else           dx = rdx_/static_cast<Real>(1<<rlev);
  Real dx2 = SQR(dx);
  Real isix = omega_/6.0;
  for (int k=kl; k<=ku; k++) {
    for (int j=jl; j<=ju; j++) {
      for (int i=il+c; i<=iu; i+=2)
        u(0,k,j,i) -= ((6.0*u(0,k,j,i) - u(0,k+1,j,i) - u(0,k,j+1,i) - u(0,k,j,i+1)
                      - u(0,k-1,j,i) - u(0,k,j-1,i) - u(0,k,j,i-1))
                       + src(0,k,j,i)*dx2)*isix;
      c ^= 1;  // bitwise XOR assignment
    }
    c ^= 1;
  }

// Jacobi
/*  const Real isix = 1.0/7.0;
  static AthenaArray<Real> temp;
  if (!temp.IsAllocated())
    temp.NewAthenaArray(1,18,18,18);
  for (int k=kl; k<=ku; k++) {
    for (int j=jl; j<=ju; j++) {
      for (int i=il; i<=iu; i++)
        temp(0,k,j,i) = u(0,k,j,i) - (((6.0*u(0,k,j,i) - u(0,k+1,j,i) - u(0,k,j+1,i) - u(0,k,j,i+1)
                      - u(0,k-1,j,i) - u(0,k,j-1,i) - u(0,k,j,i-1)) + src(0,k,j,i)*dx2)*isix);
    }
  }
  for (int k=kl; k<=ku; k++) {
    for (int j=jl; j<=ju; j++) {
      for (int i=il; i<=iu; i++)
      u(0,k,j,i) = temp(0,k,j,i);
    }
  }*/
  return;
}


//----------------------------------------------------------------------------------------
//! \fn  void MGGravity::CalculateDefect(AthenaArray<Real> &def,
//                       const AthenaArray<Real> &u, const AthenaArray<Real> &src,
//                       int rlev, int il, int iu, int jl, int ju, int kl, int ku)
//  \brief Implementation of the Defect calculation
//         rlev = relative level from the finest level of this Multigrid block

void MGGravity::CalculateDefect(AthenaArray<Real> &def, const AthenaArray<Real> &u,
                                const AthenaArray<Real> &src, int rlev,
                                int il, int iu, int jl, int ju, int kl, int ku) {
  Real dx;
  if (rlev <= 0) dx = rdx_*static_cast<Real>(1<<(-rlev));
  else           dx = rdx_/static_cast<Real>(1<<rlev);
  Real idx2 = 1.0/SQR(dx);
  for (int k=kl; k<=ku; k++) {
    for (int j=jl; j<=ju; j++) {
      for (int i=il; i<=iu; i++)
        def(0,k,j,i) = (6.0*u(0,k,j,i) - u(0,k+1,j,i) - u(0,k,j+1,i) - u(0,k,j,i+1)
                       - u(0,k-1,j,i) - u(0,k,j-1,i) - u(0,k,j,i-1))*idx2
                       + src(0,k,j,i);
    }
  }

  return;
}


//----------------------------------------------------------------------------------------
//! \fn  void MGGravity::CalculateFASRHS(AthenaArray<Real> &src,
//   const AthenaArray<Real> &u, int rlev, int il, int iu, int jl, int ju, int kl, int ku)
//  \brief Implementation of the RHS calculation for FAS
//         rlev = relative level from the finest level of this Multigrid block

void MGGravity::CalculateFASRHS(AthenaArray<Real> &src, const AthenaArray<Real> &u,
                         int rlev, int il, int iu, int jl, int ju, int kl, int ku) {
  Real dx;
  if (rlev <= 0) dx = rdx_*static_cast<Real>(1<<(-rlev));
  else           dx = rdx_/static_cast<Real>(1<<rlev);
  Real idx2 = 1.0/SQR(dx);
  for (int k=kl; k<=ku; k++) {
    for (int j=jl; j<=ju; j++) {
      for (int i=il; i<=iu; i++)
        src(0,k,j,i) -= (6.0*u(0,k,j,i) - u(0,k+1,j,i) - u(0,k,j+1,i) - u(0,k,j,i+1)
                        - u(0,k-1,j,i) - u(0,k,j-1,i) - u(0,k,j,i-1))*idx2;
    }
  }

  return;
}


//----------------------------------------------------------------------------------------
//! \fn void MGGravityDriver::ProlongateOctetBoundariesFluxCons(AthenaArray<Real> &dst)
//  \brief prolongate octet boundaries using the flux conservation formula

void MGGravityDriver::ProlongateOctetBoundariesFluxCons(AthenaArray<Real> &dst) {
  constexpr Real ot = 1.0/3.0;
  const int ngh = mgroot_->ngh_;
  const AthenaArray<Real> &u = dst;
  const int ci = ngh, cj = ngh, ck = ngh, l = ngh, r = ngh + 1;

  // x1face
  for (int ox1=-1; ox1<=1; ox1+=2) {
    if (ncoarse_[1][1][ox1+1]) {
      int i, fi, fig;
      if (ox1 > 0) i = ngh + 1, fi = ngh + 1, fig = ngh + 2;
      else         i = ngh - 1, fi = ngh,     fig = ngh - 1;
      Real ccval = cbuf_(0, ck, cj, i);
      Real gx2m = ccval - cbuf_(0, ck, cj-1, i);
      Real gx2p = cbuf_(0, ck, cj+1, i) - ccval;
      Real gx2c = 0.125*(SIGN(gx2m) + SIGN(gx2p))*std::min(std::abs(gx2m),
                                                           std::abs(gx2p));
      Real gx3m = ccval - cbuf_(0, ck-1, cj, i);
      Real gx3p = cbuf_(0, ck+1, cj, i) - ccval;
      Real gx3c = 0.125*(SIGN(gx3m) + SIGN(gx3p))*std::min(std::abs(gx3m),
                                                           std::abs(gx3p));
      dst(0, l, l, fig) = ot*(2.0*(ccval - gx2c - gx3c) + u(0, l, l, fi));
      dst(0, l, r, fig) = ot*(2.0*(ccval + gx2c - gx3c) + u(0, l, r, fi));
      dst(0, r, l, fig) = ot*(2.0*(ccval - gx2c + gx3c) + u(0, r, l, fi));
      dst(0, r, r, fig) = ot*(2.0*(ccval + gx2c + gx3c) + u(0, r, r, fi));
    }
  }

  // x2face
  for (int ox2=-1; ox2<=1; ox2+=2) {
    if (ncoarse_[1][ox2+1][1]) {
      int j, fj, fjg;
      if (ox2 > 0) j = ngh + 1, fj = ngh + 1, fjg = ngh + 2;
      else         j = ngh - 1, fj = ngh,     fjg = ngh - 1;
      Real ccval = cbuf_(0, ck, j, ci);
      Real gx1m = ccval - cbuf_(0, ck, j, ci-1);
      Real gx1p = cbuf_(0, ck, j, ci+1) - ccval;
      Real gx1c = 0.125*(SIGN(gx1m) + SIGN(gx1p))*std::min(std::abs(gx1m),
                                                           std::abs(gx1p));
      Real gx3m = ccval - cbuf_(0, ck-1, j, ci);
      Real gx3p = cbuf_(0, ck+1, j, ci) - ccval;
      Real gx3c = 0.125*(SIGN(gx3m) + SIGN(gx3p))*std::min(std::abs(gx3m),
                                                           std::abs(gx3p));
      dst(0, l, fjg, l) = ot*(2.0*(ccval - gx1c - gx3c) + u(0, l, fj, l));
      dst(0, l, fjg, r) = ot*(2.0*(ccval + gx1c - gx3c) + u(0, l, fj, r));
      dst(0, r, fjg, l) = ot*(2.0*(ccval - gx1c + gx3c) + u(0, r, fj, l));
      dst(0, r, fjg, r) = ot*(2.0*(ccval + gx1c + gx3c) + u(0, r, fj, r));
    }
  }

  // x3face
  for (int ox3=-1; ox3<=1; ox3+=2) {
    if (ncoarse_[ox3+1][1][1]) {
      int k, fk, fkg;
      if (ox3 > 0) k = ngh + 1, fk = ngh + 1, fkg = ngh + 2;
      else         k = ngh - 1, fk = ngh,     fkg = ngh - 1;
      Real ccval = cbuf_(0, k, cj, ci);
      Real gx1m = ccval - cbuf_(0, k, cj, ci-1);
      Real gx1p = cbuf_(0, k, cj, ci+1) - ccval;
      Real gx1c = 0.125*(SIGN(gx1m) + SIGN(gx1p))*std::min(std::abs(gx1m),
                                                           std::abs(gx1p));
      Real gx2m = ccval - cbuf_(0, k, cj-1, ci);
      Real gx2p = cbuf_(0, k, cj+1, ci) - ccval;
      Real gx2c = 0.125*(SIGN(gx2m) + SIGN(gx2p))*std::min(std::abs(gx2m),
                                                           std::abs(gx2p));
      dst(0, fkg, l, l) = ot*(2.0*(ccval - gx1c - gx2c) + u(0, fk, l, l));
      dst(0, fkg, l, r) = ot*(2.0*(ccval + gx1c - gx2c) + u(0, fk, l, r));
      dst(0, fkg, r, l) = ot*(2.0*(ccval - gx1c + gx2c) + u(0, fk, r, l));
      dst(0, fkg, r, r) = ot*(2.0*(ccval + gx1c + gx2c) + u(0, fk, r, r));
    }
  }

  return;
}


//----------------------------------------------------------------------------------------
//! \fn void MGGravityDriver::ScaleMultipoleCoefficients()
//  \brief scale coefficients for multipole expansion

void MGGravityDriver::ScaleMultipoleCoefficients() {
  for (int i = 0; i < nmpcoeff_; ++i)
    mpcoeff_(i) *= four_pi_G_;
}
