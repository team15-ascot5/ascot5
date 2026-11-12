/**
 * Implements "diag_moment.h".
 */
#include "diag_moment.h"
#include "consts.h"
#include "data/bfield.h"
#include "defines.h"
#include "marker.h"
#include "parallel.h"
#include "utils/mathlib.h"
#include "utils/physlib.h"
#include <stdlib.h>

void DiagMoment_offload(DiagMoment *moment)
{
    SUPPRESS_UNUSED_WARNING(moment);
    GPU_MAP_TO_DEVICE(
        moment->values[0:moment->axis.n*moment->num_moments],
        moment->chosen_moments[0:moment->axis.n*moment->num_moments])
}

void DiagMoment_onload(DiagMoment *moment)
{
    SUPPRESS_UNUSED_WARNING(moment);
    GPU_MAP_FROM_DEVICE(
        moment->values[0:moment->axis.n*moment->num_moments],
        moment->chosen_moments[0:moment->axis.n*moment->num_moments])
}

void DiagMoment_update_go(
    DiagMoment *moment, Bfield *bfield, MarkerGyroOrbit *mrk_f,
    MarkerGyroOrbit *mrk_i)
{
    GPU_PARALLEL_LOOP_ALL_LEVELS
    for (size_t i = 0; i < mrk_f->size; i++)
    {
        HistAxis *axis = &(moment->axis);
        real rho = mrk_f->rho[i];
        real computed_moments[MOMENT_NUM];

        /*Norm of the magnetic field |B|*/
        real bnorm = math_normc(mrk_f->br[i], mrk_f->bphi[i], mrk_f->bz[i]);

        /*Norm of the momentum p |p|*/
        real pnorm = math_normc(mrk_f->br[i], mrk_f->bphi[i], mrk_f->bz[i]);

        /*Momentum p along the magnetic field B. p \dot B / |B|*/
        real ppar =
            (
                mrk_f->p_r[i] * mrk_f->br[i] + 
                mrk_f->p_phi[i] * mrk_f->bphi[i] +
                mrk_f->p_z[i] * mrk_f->bz[i]
            ) / bnorm;
        /*Momentum p along the magnetic field B at the start of the timestep. p \dot B / |B|*/
        real ppar_i =
            (
                mrk_i->p_r[i] * mrk_i->br[i] + 
                mrk_i->p_phi[i] * mrk_i->bphi[i] +
                mrk_i->p_z[i] * mrk_i->bz[i]
            ) / bnorm;

        
        /*Parallel velocity*/
        real vpar = ppar / mrk_f->mass;
        /*Pitch xi*/
        real xi = ppar / pnorm; 

        real gamma = physlib_gamma_pnorm(mrk_f->mass, pnorm);

        /*Total kinetic energy from mass and gamma.*/
        real ekin = physlib_Ekin_gamma(mrk_f->mass, gamma);

        /*Change in total kinetic energy.*/
        real delta_e = ekin - physlib_Ekin_gamma(mrk_f->mass, gamma);

        /*Length of the timestep.*/
        real delta_t = (mrk_f->time[i] - mrk_i->time[i]);

        /*Psi at the start and end of the timestep.*/
        real psi_f, psi_i;
        Bfield_eval_psi(&psi_f, mrk_f->r[i], mrk_f->phi[i], mrk_f->z[i], mrk_f->time[i], bfield); 
        Bfield_eval_psi(&psi_i, mrk_i->r[i], mrk_i->phi[i], mrk_i->z[i], mrk_i->time[i], bfield); 
        /*Change in psi during the timestep.*/
        real delta_psi = psi_f - psi_i;

        /*Change in parallel momentum during the timestep.*/
        real delta_ppar = ppar - ppar_i; 

        real charge = CONST_E * mrk_f->charge[i];
        /**Change in momentum conjugate to phi.*/
        real delta_P_phi = 
             phys_ptoroid_fo(charge, mrk_f->r[i], mrk_f->p_phi[i], psi_f)
            -phys_ptoroid_fo(charge, mrk_i->r[i], mrk_i->p_phi[i], psi_i);

        /*Weight of the marker at the end of the timestep w.*/
        real marker_weight_w = mrk_f->weight[i];

        for(size_t moment_index = 0; moment_index < moment->num_moments; moment_index++)
        {
            MomentType current_moment = moment->chosen_moments[moment_index];
            switch(current_moment)
            {
                case MOMENT_DENSITY: 
                    computed_moments[current_moment] = marker_weight_w * delta_t; /**Particle density W_n = w*\delta t */
                    break;
                case MOMENT_ENERGYDENSITY: 
                    computed_moments[current_moment] = ekin * marker_weight_w * delta_t; /**Energy density W_e = E*w*\delta t */
                    break;
                case MOMENT_ENERGYDENSITYPAR: 
                    computed_moments[current_moment] = ekin * xi * xi * marker_weight_w * delta_t; /**Parallel energy density W_e_|| = E*xi^2*w*\delta t */
                    break;
                case MOMENT_JPAR: 
                    computed_moments[current_moment] = CONST_E * vpar * marker_weight_w * delta_t; /**Parallel current W_j_|| = e*v_||*w*\delta t */
                    break;
                case MOMENT_CPDEP: 
                    computed_moments[current_moment] = -delta_e * marker_weight_w; /**Collisional power deposition to plasma W_P,c=-\delta E_d*w */
                    break;
                case MOMENT_TORTORQ: 
                    computed_moments[current_moment] = -delta_e * delta_psi * marker_weight_w; /**Toroidal torque j x B, W_\tau,jxB=-e*\delta\psi_p*w */
                    break;
                case MOMENT_TORCTORQ: 
                    computed_moments[current_moment] = -mrk_f->r[i]*(delta_ppar*mrk_f->bphi[i]/bnorm); /**Toroidal torque j x B, W_\tau,jxB=-e*\delta\psi_p*w */
                    break;
                case MOMENT_TORCTORQPPHI: 
                    computed_moments[current_moment] = -CONST_E * delta_P_phi * marker_weight_w; /**Toroidal torque from changes in toroidal canonical momentum W_\tau,p_\phi = -e*\delta P_\phi*w*/
                    break;
                default: 
                    computed_moments[current_moment] = 0;
                    break;
            }
        }

        int valid = (rho >= axis->min && rho <= axis->max);

        size_t bin_index = math_bin_index(rho, axis->n, axis->min, axis->max);
        size_t bin_index_offset = valid ? bin_index*moment->num_moments: 0;


        for(size_t moment_index = 0; moment_index < moment->num_moments; moment_index++)
        {
            size_t selected_moment = moment->chosen_moments[moment_index];
            GPU_ATOMIC
            moment->values[bin_index_offset + moment_index] += valid * computed_moments[selected_moment];
        }
    }
}

void DiagMoment_update_gc(
    DiagMoment *moment, Bfield *bfield, MarkerGuidingCenter *mrk_f,
    MarkerGuidingCenter *mrk_i)
{
    GPU_PARALLEL_LOOP_ALL_LEVELS
    for (size_t i = 0; i < mrk_f->size; i++)
    {
        HistAxis *axis = &(moment->axis);
        real rho = mrk_f->rho[i];
        real computed_moments[MOMENT_NUM];

        /*Norm of the magnetic field |B|*/
        real bnorm = math_normc(mrk_f->br[i], mrk_f->bphi[i], mrk_f->bz[i]);

        /*Norm of the magnetic field |B|*/
        real bnorm_i = math_normc(mrk_i->br[i], mrk_i->bphi[i], mrk_i->bz[i]);

        /*Norm of the momentum p |p|*/
        real pnorm = math_normc(mrk_f->br[i], mrk_f->bphi[i], mrk_f->bz[i]);
        /*Norm of the momentum p |p| at the start of the timestep.*/
        real pnorm_i = math_normc(mrk_f->br[i], mrk_f->bphi[i], mrk_f->bz[i]);

        /*Momentum p along the magnetic field B. p \dot B / |B|*/
        real ppar = mrk_f->ppar[i];
        /*Momentum p along the magnetic field B. p \dot B / |B|*/
        real ppar_i = mrk_i->ppar[i];
        
        /*Parallel velocity*/
        real vpar = ppar / mrk_f->mass;

        /*Pitch xi*/
        real xi = ppar / pnorm;


        /*Lorentz factor from mass*/
        real gamma = physlib_gamma_pnorm(mrk_f->mass, pnorm);

        /*Total kinetic energy from mass and gamma.*/
        real ekin = physlib_Ekin_gamma(mrk_f->mass, gamma);

        /*Change in total kinetic energy.*/
        real delta_e = ekin - physlib_Ekin_gamma(mrk_f->mass, gamma);

        /*Length of the timestep.*/
        real delta_t = (mrk_f->time[i] - mrk_i->time[i]);

        /*Psi at the start and end of the timestep.*/
        real psi_f, psi_i;
        Bfield_eval_psi(&psi_f, mrk_f->r[i], mrk_f->phi[i], mrk_f->z[i], mrk_f->time[i], bfield); 
        Bfield_eval_psi(&psi_i, mrk_i->r[i], mrk_i->phi[i], mrk_i->z[i], mrk_i->time[i], bfield); 
        /*Change in psi during the timestep.*/
        real delta_psi = psi_f - psi_i;

        /*Change in parallel momentum during the timestep.*/
        real delta_ppar = ppar - ppar_i; 

        real charge = CONST_E * mrk_f->charge[i];
        real delta_P_phi = 
             phys_ptoroid_gc(charge, mrk_f->r[i], ppar, psi_f, bnorm, mrk_f->bphi[i])
            -phys_ptoroid_gc(charge, mrk_i->r[i], ppar_i, psi_i, bnorm, mrk_f->bphi[i]);


        /*Weight of the marker at the end of the timestep w.*/
        real marker_weight_w = mrk_f->weight[i];

        for(size_t moment_index = 0; moment_index < moment->num_moments; moment_index++)
        {
            MomentType current_moment = moment->chosen_moments[moment_index];
            switch(current_moment)
            {
                case MOMENT_DENSITY: 
                    computed_moments[current_moment] = marker_weight_w * delta_t; /**Particle density W_n = w*\delta t */
                    break;
                case MOMENT_ENERGYDENSITY: 
                    computed_moments[current_moment] = ekin * marker_weight_w * delta_t; /**Energy density W_e = E*w*\delta t */
                    break;
                case MOMENT_ENERGYDENSITYPAR: 
                    computed_moments[current_moment] = ekin * xi * xi * marker_weight_w * delta_t; /**Parallel energy density W_e_|| = E*xi^2*w*\delta t */
                    break;
                case MOMENT_JPAR: 
                    computed_moments[current_moment] = CONST_E * vpar * marker_weight_w * delta_t; /**Parallel current W_j_|| = e*v_||*w*\delta t */
                    break;
                case MOMENT_CPDEP: 
                    computed_moments[current_moment] = -delta_e * marker_weight_w; /**Collisional power deposition to plasma W_P,c=-\delta E_d*w */
                    break;
                case MOMENT_TORTORQ: 
                    computed_moments[current_moment] = -delta_e * delta_psi * marker_weight_w; /**Toroidal torque j x B, W_\tau,jxB=-e*\delta\psi_p*w */
                    break;
                case MOMENT_TORCTORQ: 
                    computed_moments[current_moment] = -mrk_f->r[i]*(delta_ppar*mrk_f->bphi[i]/bnorm); /**Toroidal torque j x B, W_\tau,jxB=-e*\delta\psi_p*w */
                    break;
                case MOMENT_TORCTORQPPHI: 
                    computed_moments[current_moment] = -CONST_E * delta_P_phi * marker_weight_w; /**Toroidal torque from changes in toroidal canonical momentum W_\tau,p_\phi = -e*\delta P_\phi*w*/
                    break;
                default: 
                    computed_moments[current_moment] = 0;
                    break;
            }
        }

        int valid = (rho >= axis->min && rho <= axis->max);

        size_t bin_index = math_bin_index(rho, axis->n, axis->min, axis->max);
        size_t bin_index_offset = valid ? bin_index*moment->num_moments: 0;


        for(size_t moment_index = 0; moment_index < moment->num_moments; moment_index++)
        {
            size_t selected_moment = moment->chosen_moments[moment_index];
            GPU_ATOMIC
            moment->values[bin_index_offset + moment_index] += valid * computed_moments[selected_moment];
        }
    }
}
