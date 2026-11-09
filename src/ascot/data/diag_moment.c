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
        computed_moments[MOMENT_DENSITY] = 1.0;

        int valid = 1;
        valid *= (rho >= axis->min && rho <= axis->max);

        size_t bin_index = math_bin_index(rho, axis->n, axis->min, axis->max);
        size_t bin_index_offset = valid ? bin_index*moment->num_moments: 0;

        real weight = mrk_f->weight[i] * (mrk_f->time[i] - mrk_i->time[i]);

        for(size_t moment_index = 0; moment_index < moment->num_moments; moment_index++)
        {
            size_t selected_moment = moment->chosen_moments[moment_index];
            GPU_ATOMIC
            moment->values[bin_index_offset + selected_moment] += valid * weight * computed_moments[selected_moment];
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
        computed_moments[MOMENT_DENSITY] = 1.0;

        int valid = 1;
        valid *= (rho >= axis->min && rho <= axis->max);

        size_t bin_index = math_bin_index(rho, axis->n, axis->min, axis->max);
        size_t bin_index_offset = valid ? bin_index*moment->num_moments: 0;

        real weight = mrk_f->weight[i] * (mrk_f->time[i] - mrk_i->time[i]);

        for(size_t moment_index = 0; moment_index < moment->num_moments; moment_index++)
        {
            size_t selected_moment = moment->chosen_moments[moment_index];
            GPU_ATOMIC
            moment->values[bin_index_offset + selected_moment] += valid * weight * computed_moments[selected_moment];
        }
    }
}
