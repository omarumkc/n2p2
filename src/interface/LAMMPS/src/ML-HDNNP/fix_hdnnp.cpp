/* ----------------------------------------------------------------------
   LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
   http://lammps.sandia.gov, Sandia National Laboratories
   Steve Plimpton, sjplimp@sandia.gov

   Copyright (2003) Sandia Corporation.  Under the terms of Contract
   DE-AC04-94AL85000 with Sandia Corporation, the U.S. Government retains
   certain rights in this software.  This software is distributed under
   the GNU General Public License.

   See the README file in the top-level LAMMPS directory.
------------------------------------------------------------------------- */

#include "fix_hdnnp.h"
#include "pair_hdnnp_4g.h"
#include <iostream>
#include <gsl/gsl_multimin.h>
#include <mpi.h>
#include <cmath>
#include <cstring>
#include "pair_hdnnp.h"
#include "kspace_hdnnp.h"
#include "atom.h"
#include "comm.h"
#include "domain.h" // check for periodicity
#include "neighbor.h"
#include "neigh_list.h"
#include "neigh_request.h"
#include "update.h"
#include "force.h"
#include "group.h"
#include "pair.h"
#include "memory.h"
#include "error.h"
#include "utils.h"
#include <chrono> //time

using namespace LAMMPS_NS;
using namespace FixConst;
using namespace std::chrono;
using namespace nnp;

#define EV_TO_KCAL_PER_MOL 14.4
#define SQR(x) ((x)*(x))
#define CUBE(x) ((x)*(x)*(x))
#define MIN_NBRS 100
#define SAFE_ZONE      1.2
#define DANGER_ZONE    0.90
#define MIN_CAP        50


FixHDNNP::FixHDNNP(LAMMPS *lmp, int narg, char **arg) :
        Fix(lmp, narg, arg),
        hdnnp         (nullptr),
        kspacehdnnp   (nullptr),
        list          (nullptr),
        pertype_option(nullptr),
        hdnnpflag     (0      ),
        kspaceflag    (0      ),
        ngroup        (0      ),
        periodic      (false  ),
        qRef          (0.0    ),
        Q             (nullptr),
        type_all      (nullptr),
        type_loc      (nullptr),
        qall          (nullptr),
        qall_loc      (nullptr),
        dEdQ_all      (nullptr),
        dEdQ_loc      (nullptr),
        xx            (nullptr),
        xy            (nullptr),
        xz            (nullptr),
        xx_loc        (nullptr),
        xy_loc        (nullptr),
        xz_loc        (nullptr)

{
    if (narg<9 || narg>11) error->all(FLERR,"Illegal fix hdnnp command");

    nevery = utils::inumeric(FLERR,arg[3],false,lmp);
    if (nevery <= 0) error->all(FLERR,"Illegal fix hdnnp command");

    int len = strlen(arg[8]) + 1;
    pertype_option = new char[len];
    strcpy(pertype_option,arg[8]);

    hdnnp = nullptr;
    kspacehdnnp = nullptr;
    hdnnp = (PairHDNNP4G *) force->pair_match("^hdnnp/4g",0);
    kspacehdnnp = (KSpaceHDNNP *) force->kspace_match("^hdnnp",0);

    hdnnp->chi = nullptr;
    hdnnp->hardness = nullptr;
    hdnnp->sigmaSqrtPi = nullptr;
    hdnnp->gammaSqrt2 = nullptr;
    hdnnp->screening_info = nullptr;

    // User-defined minimization parameters used in pair_hdnnp as well
    // TODO: read only on proc 0 and then Bcast ?
    hdnnp->grad_tol = utils::numeric(FLERR,arg[4],false,lmp); //tolerance for gradient check
    hdnnp->min_tol = utils::numeric(FLERR,arg[5],false,lmp); //tolerance
    hdnnp->step = utils::numeric(FLERR,arg[6],false,lmp); //initial hdnnp->step size
    hdnnp->maxit = utils::inumeric(FLERR,arg[7],false,lmp); //maximum number of iterations
    hdnnp->minim_init_style = utils::inumeric(FLERR,arg[10],false,lmp); //initialization style

    // TODO: check these initializations and allocations
    int natoms = atom->natoms;

    // TODO: these are not necessary in case of periodic systems
    memory->create(xx,atom->natoms,"fix_hdnnp:xx");
    memory->create(xy,atom->natoms,"fix_hdnnp:xy");
    memory->create(xz,atom->natoms,"fix_hdnnp:xz");
    memory->create(xx_loc,atom->natoms,"fix_hdnnp:xx_loc");
    memory->create(xy_loc,atom->natoms,"fix_hdnnp:xy_loc");
    memory->create(xz_loc,atom->natoms,"fix_hdnnp:xz_loc");
    memory->create(type_loc,atom->natoms,"fix_hdnnp:type_loc");
    memory->create(type_all,atom->natoms,"fix_hdnnp:type_all");
    
}

/* ---------------------------------------------------------------------- */

FixHDNNP::~FixHDNNP()
{
    if (copymode) return;

    delete[] pertype_option;
    
    memory->destroy(hdnnp->chi);
    memory->destroy(hdnnp->hardness);
    memory->destroy(hdnnp->sigmaSqrtPi);
    memory->destroy(hdnnp->gammaSqrt2);

    memory->destroy(qall);
    memory->destroy(qall_loc);
    memory->destroy(dEdQ_loc);
    memory->destroy(dEdQ_all);
    
    memory->destroy(xx);
    memory->destroy(xy);
    memory->destroy(xz);
    memory->destroy(xx_loc);
    memory->destroy(xy_loc);
    memory->destroy(xz_loc);
    memory->destroy(type_loc);
    memory->destroy(type_all);
    
}

void FixHDNNP::post_constructor()
{
    pertype_parameters(pertype_option);

}

int FixHDNNP::setmask()
{
    int mask = 0;
    mask |= PRE_FORCE;
    mask |= PRE_FORCE_RESPA;
    mask |= MIN_PRE_FORCE;
    return mask;
}

void FixHDNNP::init()
{
    if (!atom->q_flag)
        error->all(FLERR,"Fix hdnnp requires atom attribute q");

    ngroup = group->count(igroup);
    if (ngroup == 0) error->all(FLERR,"Fix hdnnp group has no atoms");

    // need a half neighbor list w/ Newton off and ghost neighbors
    // built whenever re-neighboring occurs

    //int irequest = neighbor->request(this,instance_me);
    //neighbor->requests[irequest]->pair = 0;
    //neighbor->requests[irequest]->fix = 1;
    //neighbor->requests[irequest]->newton = 2;
    //neighbor->requests[irequest]->ghost = 1;

    isPeriodic();

    //if (periodic) neighbor->add_request(this, NeighConst::REQ_DEFAULT);
    //else neighbor->add_request(this, NeighConst::REQ_FULL);
    neighbor->add_request(this, NeighConst::REQ_FULL |
                                NeighConst::REQ_GHOST |
                                NeighConst::REQ_NEWTON_OFF);

    // TODO : do we really need a full NL in periodic cases ?
    //if (periodic) { // periodic : full neighborlist
    //    neighbor->requests[irequest]->half = 0;
    //    neighbor->requests[irequest]->full = 1;
    //}

    allocate_QEq();
}

void FixHDNNP::pertype_parameters(char *arg)
{
    if (strcmp(arg,"hdnnp") == 0) {
        hdnnpflag = 1;
        Pair *pair = force->pair_match("hdnnp",0);
        if (pair == NULL) error->all(FLERR,"No pair hdnnp for fix hdnnp");
    }
}

// Allocate QEq arrays
void FixHDNNP::allocate_QEq()
{
    int ne = atom->ntypes;
    memory->create(hdnnp->chi,atom->natoms,"qeq:hdnnp->chi");
    memory->create(hdnnp->hardness,ne,"qeq:hdnnp->hardness");
    memory->create(hdnnp->sigmaSqrtPi,ne,"qeq:hdnnp->sigmaSqrtPi");
    memory->create(hdnnp->gammaSqrt2,ne,ne,"qeq:hdnnp->gammaSqrt2");
    memory->create(hdnnp->screening_info,4,"qeq:screening");
    memory->create(qall,atom->natoms,"qeq:qall");
    memory->create(qall_loc,atom->natoms,"qeq:qall_loc");
    memory->create(dEdQ_loc,atom->natoms,"qeq:dEdQ_loc");
    memory->create(dEdQ_all,atom->natoms,"qeq:dEdQ_all");
    
    // Initialization
    for (int i =0; i < atom->natoms; i++){
        hdnnp->chi[i] = 0.0;
    }
    for (int i = 0; i < ne; i++){
	hdnnp->hardness[i] = 0.0;
	hdnnp->sigmaSqrtPi[i] = 0.0;
	for (int j = 0; j < ne; j++){
		hdnnp->gammaSqrt2[i][j] = 0.0;
	}
    }
    for (int i = 0; i < 4; i++){
	    hdnnp->screening_info[i] = 0.0;
    }
}

// Deallocate QEq arrays
void FixHDNNP::deallocate_QEq() {

    memory->destroy(hdnnp->chi);
    memory->destroy(hdnnp->hardness);
    memory->destroy(hdnnp->sigmaSqrtPi);
    memory->destroy(hdnnp->gammaSqrt2);
    memory->destroy(hdnnp->screening_info);
    memory->destroy(qall);
    memory->destroy(qall_loc);
    memory->destroy(dEdQ_loc);
    memory->destroy(dEdQ_all);

}

void FixHDNNP::init_list(int /*id*/, NeighList *ptr)
{
    list = ptr;
}

void FixHDNNP::setup_pre_force(int vflag)
{
    pre_force(vflag);
}

void FixHDNNP::min_setup_pre_force(int vflag)
{
    setup_pre_force(vflag);
}

void FixHDNNP::min_pre_force(int vflag)
{
    pre_force(vflag);
}

// Main calculation routine, runs before pair->compute at each timehdnnp->step
void FixHDNNP::pre_force(int /*vflag*/) {

    double *q = atom->q;
    double Qtot_loc = 0.0;
    int n = atom->nlocal;
    int j,jmap;

    //deallocate_QEq();
    //allocate_QEq();
    
    if(periodic) {
        //force->kspace->setup();
	kspacehdnnp->ewald_sfexp();
    }
    
    // Calculate atomic electronegativities \Chi_i
    calculate_electronegativities();
    
    // Calculate the current total charge
    for (int i = 0; i < n; i++) {
        //std::cout << q[i] << '\n';
        Qtot_loc += q[i];
    }
    MPI_Allreduce(&Qtot_loc,&qRef,1,MPI_DOUBLE,MPI_SUM,world);

    //TODO
    calculate_erfc_terms();
    
    std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();
    // Minimize QEq energy and calculate atomic charges
    calculate_QEqCharges();
    std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
    //std::cout << "iQEq time (s) = " << std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count() / 1000000.0 << std::endl;

}

void FixHDNNP::calculate_electronegativities()
{
    // Run first set of atomic NNs
    process_first_network();

    // Read QEq arrays from n2p2 into LAMMPS
    hdnnp->interface.getQEqParams(hdnnp->chi,hdnnp->hardness,hdnnp->sigmaSqrtPi,hdnnp->gammaSqrt2,qRef);

    // Read screening function information from n2p2 into LAMMPS
    hdnnp->interface.getScreeningInfo(hdnnp->screening_info); //TODO: read function type
}

// Runs interface.process for electronegativities
void FixHDNNP::process_first_network()
{
    if(hdnnp->interface.getNnpType() == InterfaceLammps::NNPType::HDNNP_4G)
    {
        // Set number of local atoms and add element.
        hdnnp->interface.setLocalAtoms(atom->nlocal,atom->type);

        // Set tags of local atoms.
        hdnnp->interface.setLocalTags(atom->tag);

        // Transfer local neighbor list to HDNNP interface.
        hdnnp->transferNeighborList();

        // Run the first NN for electronegativities
        hdnnp->interface.process();
    }
    else{ //TODO
        error->all(FLERR,"This fix style can only be used with a 4G-HDNNP.");
    }
}

// This routine is called once. Complementary error function terms are calculated and stored
void FixHDNNP::calculate_erfc_terms()
{
    int i,j,jmap;
    int nlocal = atom->nlocal;
    int *tag = atom->tag;
    int *type = atom->type;

    double **x = atom->x;
    double eta;

    int maxnumneigh = 0;
    // TODO is this already available in LAMMPS ?
    for (int i = 0; i < nlocal; i++)
    {
      int nneigh = list->numneigh[i];
      if (nneigh > maxnumneigh) maxnumneigh = nneigh;
    }
    
    //maxnumneigh = 100000;
    // allocate and initialize
    memory->create(hdnnp->erfc_val,nlocal+1,maxnumneigh+1,"fix_hdnnp:erfc_val");

    for (int i = 0; i < nlocal; i++){
        for (int j = 0; j < maxnumneigh; j++)		
            hdnnp->erfc_val[i][j] = 0.0;
    }

    if (periodic)
    {
        eta = 1 / kspacehdnnp->g_ewald; // LAMMPS truncation
        double sqrt2eta = (sqrt(2.0) * eta);
        for (i = 0; i < nlocal; i++)
        {
            for (int jj = 0; jj < list->numneigh[i]; ++jj) {
                j = list->firstneigh[i][jj];
                j &= NEIGHMASK;
                jmap = atom->map(tag[j]);
                double const dx = x[i][0] - x[j][0];
                double const dy = x[i][1] - x[j][1];
                double const dz = x[i][2] - x[j][2];
                double const rij = sqrt(SQR(dx) + SQR(dy) + SQR(dz)) * hdnnp->cflength;
                hdnnp->erfc_val[i][jj] = (erfc(rij/sqrt2eta) - erfc(rij/hdnnp->gammaSqrt2[type[i]-1][type[jmap]-1])) / rij;
            }
        }
    }
}

// QEq energy function, $E_{QEq}$
double FixHDNNP::QEq_f(const gsl_vector *v)
{
    int i,j,jmap;
    int *tag = atom->tag;
    int *type = atom->type;
    int nlocal = atom->nlocal;
    int nall = atom->natoms;

    double **x = atom->x;
    double *q = atom->q;

    double E_qeq_loc,E_qeq;
    double E_elec_loc,E_scr_loc,E_scr;
    double E_real,E_recip,E_self; // for periodic examples
    double iiterm,ijterm;

    double eta;

    if(periodic) eta = 1 / kspacehdnnp->g_ewald; // LAMMPS truncation

    E_qeq = 0.0;
    E_qeq_loc = 0.0;
    E_scr = 0.0;
    E_scr_loc = 0.0;
    hdnnp->E_elec = 0.0;

    if (periodic)
    {
        double sqrt2eta = (sqrt(2.0) * eta);
        E_recip = 0.0;
        E_real = 0.0;
        E_self = 0.0;
        for (i = 0; i < nlocal; i++) // over local atoms
        {
            double const qi = gsl_vector_get(v, tag[i]-1);
            double qi2 = qi * qi;
            // Self term
            E_self += qi2 * (1 / (2.0 * hdnnp->sigmaSqrtPi[type[i]-1]) - 1 / (sqrt(2.0 * M_PI) * eta));
            E_qeq_loc += hdnnp->chi[i] * qi + 0.5 * hdnnp->hardness[type[i]-1] * qi2;
            E_scr -= qi2 / (2.0 * hdnnp->sigmaSqrtPi[type[i]-1]); // self screening term TODO:parallel ?
            // Real Term
            // TODO: we loop over the full neighbor list, could this be optimized ?
            for (int jj = 0; jj < list->numneigh[i]; ++jj) {
                j = list->firstneigh[i][jj];
                j &= NEIGHMASK;
                jmap = atom->map(tag[j]);
                double const qj = gsl_vector_get(v, tag[j]-1); // mapping based on tags
                double const dx = x[i][0] - x[j][0];
                double const dy = x[i][1] - x[j][1];
                double const dz = x[i][2] - x[j][2];
                double const rij = sqrt(SQR(dx) + SQR(dy) + SQR(dz)) * hdnnp->cflength;
                //double erfcRij = (erfc(rij / sqrt2eta) - erfc(rij / hdnnp->gammaSqrt2[type[i]-1][type[jmap]-1])) / rij;
                //double real = 0.5 * qi * qj * erfcRij;
                double real = 0.5 * qi * qj * hdnnp->erfc_val[i][jj];
                E_real += real;
                if (rij <= hdnnp->screening_info[2]) {
                    E_scr += 0.5 * qi * qj *
                            erf(rij/hdnnp->gammaSqrt2[type[i]-1][type[jmap]-1])*(hdnnp->screening_f(rij) - 1) / rij;
                }
            }
        }
        // Reciprocal Term
        if (kspacehdnnp->ewaldflag) // Ewald Sum
        {
            // Calls the charge equilibration energy calculation routine in the KSpace base class
            // Returns reciprocal energy
            E_recip = kspacehdnnp->compute_ewald_eqeq(v);
        }
        else if (kspacehdnnp->pppmflag) // PPPM
        {
            //TODO: add contributions to Eqeq for PPPM style
            kspacehdnnp->particle_map();
            kspacehdnnp->make_rho_qeq(v); // map my particle charge onto my local 3d density grid
            E_recip = kspacehdnnp->compute_pppm_eqeq(); // TODO: WIP
        }
        hdnnp->E_elec = E_real + E_self; // do not add E_recip, it is already global
        E_qeq_loc += hdnnp->E_elec;
    }else
    {
        // first loop over local atoms
        for (i = 0; i < nlocal; i++) {
            double const qi = gsl_vector_get(v,tag[i]-1);
            double qi2 = qi * qi;
            // add i terms
            iiterm = qi2 * (1 / (2.0 * hdnnp->sigmaSqrtPi[type[i]-1]) + (0.5 * hdnnp->hardness[type[i]-1]));
            E_qeq_loc += iiterm + hdnnp->chi[i] * qi;
            hdnnp->E_elec += iiterm;
            E_scr -= iiterm; //TODO parallel ?
            // Looping over all atoms (parallelization of necessary arrays has been done beforehand)
            for (j = 0; j < nall; j++) {
                double const qj = gsl_vector_get(v, j);
                double const dx = xx[j] - x[i][0];
                double const dy = xy[j] - x[i][1];
                double const dz = xz[j] - x[i][2];
                double const rij = sqrt(SQR(dx) + SQR(dy) + SQR(dz)) * hdnnp->cflength;
                if (rij != 0.0) //TODO check
                {
                    ijterm = (erf(rij / hdnnp->gammaSqrt2[type[i]-1][type_all[j]-1]) / rij);
                    E_qeq_loc += 0.5 * ijterm * qi * qj;
                    hdnnp->E_elec += ijterm;
                    if(rij <= hdnnp->screening_info[2]) {
                        E_scr += ijterm * (hdnnp->screening_f(rij) - 1);
                    }
                }
            }
        }
    }

    //TODO: add communication steps for E_elec !!!
    hdnnp->E_elec = hdnnp->E_elec + E_scr; // add screening energy

    MPI_Allreduce(&E_qeq_loc,&E_qeq,1,MPI_DOUBLE,MPI_SUM,world); // MPI_SUM of local QEQ contributions

    if (periodic) E_qeq += E_recip; // adding already all-reduced reciprocal part now

    //fprintf(stderr, "Erecip = %24.16E\n", E_recip);
    return E_qeq;
}

// QEq energy function - wrapper
double FixHDNNP::QEq_f_wrap(const gsl_vector *v, void *params)
{
    return static_cast<FixHDNNP*>(params)->QEq_f(v);
}

// QEq energy gradient, $\partial E_{QEq} / \partial Q_i$
void FixHDNNP::QEq_df(const gsl_vector *v, gsl_vector *dEdQ)
{
    int i,j,jmap;
    int nlocal = atom->nlocal;
    int nall = atom->natoms;
    int *tag = atom->tag;
    int *type = atom->type;

    double **x = atom->x;
    double *q = atom->q;

    double grad;
    double grad_sum_loc;
    double grad_sum;
    double grad_recip; // reciprocal space contribution to the gradient
    double grad_i;
    double jsum; //summation over neighbors & kspace respectively
    double recip_sum;

    double eta;

    if (periodic) eta = 1 / kspacehdnnp->g_ewald; // LAMMPS truncation


    grad_sum = 0.0;
    grad_sum_loc = 0.0;
    if (periodic)
    {
        double sqrt2eta = (sqrt(2.0) * eta);
        for (i = 0; i < nlocal; i++) // over local atoms
        {
            double const qi = gsl_vector_get(v,tag[i]-1);
            // Reciprocal contribution
            if (kspacehdnnp->ewaldflag)
            {
                grad_recip = 0.0;
                for (int k = 0; k < kspacehdnnp->kcount; k++) // over k-space
                {
                    grad_recip += 2.0 * kspacehdnnp->kcoeff[k] *
                            (kspacehdnnp->sf_real[k] * kspacehdnnp->sfexp_rl[k][i] +
                             kspacehdnnp->sf_im[k] *   kspacehdnnp->sfexp_im[k][i]);
                }
            }
            else if (kspacehdnnp->pppmflag)
            {
                //TODO: calculate dEdQ_i for a given local atom i
                grad_recip = kspacehdnnp->compute_pppm_dEdQ(i);
            }
            // Real contribution - over neighbors
            jsum = 0.0;
            for (int jj = 0; jj < list->numneigh[i]; ++jj) {
                j = list->firstneigh[i][jj];
                j &= NEIGHMASK;
                double const qj = gsl_vector_get(v, tag[j]-1);
                //jmap = atom->map(tag[j]);
                //double const dx = x[i][0] - x[j][0];
                //double const dy = x[i][1] - x[j][1];
                //double const dz = x[i][2] - x[j][2];
                //double const rij = sqrt(SQR(dx) + SQR(dy) + SQR(dz)) * hdnnp->cflength;
                //double erfcRij = (erfc(rij / sqrt2eta) - erfc(rij / hdnnp->gammaSqrt2[type[i]-1][type[jmap]-1]));
                //jsum += qj * erfcRij / rij;
                jsum += qj * hdnnp->erfc_val[i][jj];
            }
            grad = jsum + grad_recip + hdnnp->chi[i] + hdnnp->hardness[type[i]-1]*qi +
                   qi * (1/(hdnnp->sigmaSqrtPi[type[i]-1])- 2/(eta * sqrt(2.0 * M_PI)));
            grad_sum_loc += grad;
            dEdQ_loc[tag[i]-1] = grad; // fill gradient array based on tags instead of local IDs
        }
    }else
    {
        // first loop over local atoms
        for (i = 0; i < nlocal; i++) { // TODO: indices
            double const qi = gsl_vector_get(v,tag[i]-1);
            // second loop over 'all' atoms
            jsum = 0.0;
            // Looping over all atoms (parallelization of necessary arrays has been done beforehand)
            for (j = 0; j < nall; j++) {
                {
                    double const qj = gsl_vector_get(v, j);
                    double const dx = xx[j] - x[i][0];
                    double const dy = xy[j] - x[i][1];
                    double const dz = xz[j] - x[i][2];
                    double const rij = sqrt(SQR(dx) + SQR(dy) + SQR(dz)) * hdnnp->cflength;
                    if (rij != 0.0) jsum += qj * erf(rij / hdnnp->gammaSqrt2[type[i]-1][type_all[j]-1]) / rij;
                }
            }
            grad = hdnnp->chi[i] + hdnnp->hardness[type[i]-1]*qi + qi/(hdnnp->sigmaSqrtPi[type[i]-1]) + jsum;
            grad_sum_loc += grad;
            dEdQ_loc[tag[i]-1] = grad;
            //gsl_vector_set(dEdQ,i,grad);
        }
    }

    MPI_Allreduce(dEdQ_loc,dEdQ_all,atom->natoms,MPI_DOUBLE,MPI_SUM,world);
    MPI_Allreduce(&grad_sum_loc,&grad_sum,1,MPI_DOUBLE,MPI_SUM,world);

    // Gradient projection
    for (i = 0; i < nall; i++){
        grad_i = dEdQ_all[i];
        gsl_vector_set(dEdQ,i,grad_i - (grad_sum)/nall);
    }
}

// QEq energy gradient - wrapper
void FixHDNNP::QEq_df_wrap(const gsl_vector *v, void *params, gsl_vector *df)
{
    static_cast<FixHDNNP*>(params)->QEq_df(v, df);
}

// QEq f*df, $E_{QEq} * (\partial E_{QEq} / \partial Q_i)$
void FixHDNNP::QEq_fdf(const gsl_vector *v, double *f, gsl_vector *df)
{
    *f = QEq_f(v);
    QEq_df(v, df);
}

// QEq f*df - wrapper
void FixHDNNP::QEq_fdf_wrap(const gsl_vector *v, void *params, double *E, gsl_vector *df)
{
    static_cast<FixHDNNP*>(params)->QEq_fdf(v, E, df);
}

// Main minimization routine
void FixHDNNP::calculate_QEqCharges()
{
    size_t iter = 0;
    int status;
    int i,j;
    int nsize;
    int nlocal;
    int *tag = atom->tag;

    double *q = atom->q;
    double qsum_it;
    double gradsum;
    double qi;
    double df,alpha;

    nsize = atom->natoms; // total number of atoms
    nlocal = atom->nlocal; // total number of atoms

    gsl_vector *Q; // charge vector
    QEq_minimizer.n = nsize; // it should be n_all in the future
    QEq_minimizer.f = &QEq_f_wrap; // function pointer f(x)
    QEq_minimizer.df = &QEq_df_wrap; // function pointer df(x)
    QEq_minimizer.fdf = &QEq_fdf_wrap;
    QEq_minimizer.params = this;

    //fprintf(stderr, "q[%d] = %24.16E\n", atom->tag[0], q[0]);

    // Allocation
    //memory->create(qall,atom->natoms,"qeq:qall");
    //memory->create(qall_loc,atom->natoms,"qeq:qall_loc");
    //memory->create(dEdQ_loc,atom->natoms,"qeq:dEdQ_loc");
    //memory->create(dEdQ_all,atom->natoms,"qeq:dEdQ_all");
    
    // Initialization
    for (int i =0; i < atom->natoms; i++){
        qall[i] = 0.0;
        qall_loc[i] = 0.0;
        dEdQ_loc[i] = 0.0;
        dEdQ_all[i] = 0.0;
    }

    for (int i = 0; i < atom->nlocal; i++) {
        qall_loc[tag[i] - 1] = q[i];
    }

    MPI_Allreduce(qall_loc,qall,atom->natoms,MPI_DOUBLE,MPI_SUM,world);

    if (!periodic) // we need global position arrays in nonperiodic systems
    {
        // Initialize global arrays
        for (int i = 0; i < atom->natoms; i++){
            type_loc[i] = 0;
            xx_loc[i] = 0.0;
            xy_loc[i] = 0.0;
            xz_loc[i] = 0.0;
            xx[i] = 0.0;
            xy[i] = 0.0;
            xz[i] = 0.0;
        }

        // Create global sparse arrays here
        for (int i = 0; i < atom->nlocal; i++){
            xx_loc[tag[i]-1] = atom->x[i][0];
            xy_loc[tag[i]-1] = atom->x[i][1];
            xz_loc[tag[i]-1] = atom->x[i][2];
            type_loc[tag[i]-1] = atom->type[i];
        }
        MPI_Allreduce(xx_loc,xx,atom->natoms,MPI_DOUBLE,MPI_SUM,world);
        MPI_Allreduce(xy_loc,xy,atom->natoms,MPI_DOUBLE,MPI_SUM,world);
        MPI_Allreduce(xz_loc,xz,atom->natoms,MPI_DOUBLE,MPI_SUM,world);
        MPI_Allreduce(type_loc,type_all,atom->natoms,MPI_DOUBLE,MPI_SUM,world);
    }


    // Set the initial guess vector 
    Q = gsl_vector_alloc(nsize);
    for (i = 0; i < nsize; i++)
    {
	if (hdnnp->minim_init_style == 0)
	{
	    gsl_vector_set(Q, i, 0.0);
	}else if (hdnnp->minim_init_style == 1)
	{
	    gsl_vector_set(Q, i, qall[i]);   
	}else
	{
	    error->all(FLERR,"Invalid minimization style. Allowed values are 0 and 1.");
	}
    }

    //memory->destroy(qall);
    //memory->destroy(qall_loc);

    // TODO: is bfgs2 standard ?
    T = gsl_multimin_fdfminimizer_vector_bfgs2;
    s = gsl_multimin_fdfminimizer_alloc(T, nsize);
    gsl_multimin_fdfminimizer_set(s, &QEq_minimizer, Q, hdnnp->step, hdnnp->min_tol); // tol = 0 might be expensive ???
    do
    {
        //fprintf(stderr, "eqeq-iter %zu\n", iter);
        iter++;
        qsum_it = 0.0;
        gradsum = 0.0;
        status = gsl_multimin_fdfminimizer_iterate(s);
        // Projection (enforcing constraints)
        // TODO: could this be done more efficiently ?
        for(i = 0; i < nsize; i++) {
            qsum_it = qsum_it + gsl_vector_get(s->x, i); // total charge after the minimization hdnnp->step
        }
        for(i = 0; i < nsize; i++) {
            qi = gsl_vector_get(s->x,i);
            gsl_vector_set(s->x,i, qi - (qsum_it-qRef)/nsize); // charge projection
        }
        status = gsl_multimin_test_gradient(s->gradient, hdnnp->grad_tol); // check for convergence

        // TODO: dump also iteration time ?
        //if (status == GSL_SUCCESS) printf ("Minimum charge distribution is found at iteration : %zu\n", iter);
    }
    while (status == GSL_CONTINUE && iter < hdnnp->maxit);

    // Read calculated atomic charges back into local arrays for further calculations
    for (int i = 0; i < atom->nlocal; i++){
        q[i] = gsl_vector_get(s->x,tag[i]-1);
    }

    // Deallocation
    gsl_multimin_fdfminimizer_free(s);
    gsl_vector_free(Q);
    //memory->destroy(dEdQ_loc);
    //memory->destroy(dEdQ_all);
}

// Check if the system is periodic
void FixHDNNP::isPeriodic()
{
    if (domain->nonperiodic == 0) periodic = true;
    else                          periodic = false;
}
