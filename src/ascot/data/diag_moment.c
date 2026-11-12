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
        for(size_t moment_index = 0; moment_index < moment->num_moments; moment_index++){
            MomentType current_moment = moment->chosen_moments[moment_index];

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


            real pperp = sqrt(
                mrk_f->p_r[i] * mrk_f->p_r[i] + mrk_f->p_phi[i] * mrk_f->p_phi[i] +
                mrk_f->p_z[i] * mrk_f->p_z[i] - ppar * ppar);

            real charge = mrk_f->charge[i] * CONST_E;
            real gamma = physlib_gamma_pnorm(mrk_f->mass, pnorm);
            real ekin = physlib_Ekin_gamma(mrk_f->mass, gamma);
            switch(current_moment)
            {
                case MOMENT_DENSITY: 
                    computed_moments[current_moment] = 1.0;
                    break;
                case MOMENT_CHARGEDENSITY: 
                    computed_moments[current_moment] = charge;
                    break;
                case MOMENT_ENERGYDENSITY: 
                    computed_moments[current_moment] = ekin;
                    break;
                case MOMENT_PPAR: 
                    computed_moments[current_moment] = ppar;
                    break;
                case MOMENT_PPERP: 
                    computed_moments[current_moment] = pperp;
                    break;
                case MOMENT_JPAR: 
                    computed_moments[current_moment] = phys();
                    break;
                default: 
                    computed_moments[current_moment] = 0;
                    break;

            }
        }

        int valid = (rho >= axis->min && rho <= axis->max);

        size_t bin_index = math_bin_index(rho, axis->n, axis->min, axis->max);
        size_t bin_index_offset = valid ? bin_index*moment->num_moments: 0;

        real weight = mrk_f->weight[i] * (mrk_f->time[i] - mrk_i->time[i]);

        for(size_t moment_index = 0; moment_index < moment->num_moments; moment_index++)
        {
            size_t selected_moment = moment->chosen_moments[moment_index];
            GPU_ATOMIC
            moment->values[bin_index_offset + moment_index] += valid * weight * computed_moments[selected_moment];
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
        for(size_t moment_index = 0; moment_index < moment->num_moments; moment_index++){
            MomentType current_moment = moment->chosen_moments[moment_index];
            switch(current_moment)
            {
                case MOMENT_DENSITY: 
                    computed_moments[current_moment] = 1.0;
                    break;
                case MOMENT_CHARGEDENSITY: 
                    computed_moments[current_moment] = mrk_f->charge[i] * CONST_E;
                    break;
                default: 
                    computed_moments[current_moment] = 0;
                    break;

            }
        }

        int valid = (rho >= axis->min && rho <= axis->max);

        size_t bin_index = math_bin_index(rho, axis->n, axis->min, axis->max);
        size_t bin_index_offset = valid ? bin_index*moment->num_moments: 0;

        real weight = mrk_f->weight[i] * (mrk_f->time[i] - mrk_i->time[i]);

        for(size_t moment_index = 0; moment_index < moment->num_moments; moment_index++)
        {
            size_t selected_moment = moment->chosen_moments[moment_index];
            GPU_ATOMIC
            moment->values[bin_index_offset + moment_index] += valid * weight * computed_moments[selected_moment];
        }
    }
}
