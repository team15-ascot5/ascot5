import numpy as np
import unyt

from scipy.interpolate import interpn
from scipy.integrate   import quad

from a5py import Ascot
from a5py.physlib import aeq
from a5py.engine.interpolate import evaluate
from a5py.data.bfield import BfieldAnalytical

from .template import InputTemplate


class TransmuteAnalyticalBfieldToSplines(InputTemplate):


    def __init__(
            self, ascot: Ascot, field: BfieldAnalytical, nr: int, rlim, nz: int,
            zlim, nphi: int | None = None,
            ):
        """Transmute analytical magnetic field to spline-interpolated one.

        The result is either BfieldSpline2D or BfieldSpline3D depending on
        whether the analytical field has ripple or not.

        Parameters
        ----------
        ascot : :class:`Ascot`
            The Ascot object where the input will be created.
        field : BfieldAnalytical
            The analytical field to transmute.
        nr : int
            Number of R grid points.
        rlim : array_like (2,)
            Limits of the R grid.
        nz : int
            Number of z grid points.
        zlim : array_like (2,)
            Limits of the z grid.
        nphi : int, optional
            Number of toroidal grid points per ripple period.

            Required when the field has ripple. The ripple repeats with the
            period 360 deg / nripple, so the grid only covers one period.
        """
        rgrid = np.linspace(rlim[0], rlim[1], nr)
        zgrid = np.linspace(zlim[0], zlim[1], nz)
        if field.nripple:
            if nphi is None:
                raise ValueError(
                    "The field has ripple, so nphi must be given."
                )
            period = 360 / field.nripple
            phigrid = np.linspace(0, period, nphi + 1)[:-1] * unyt.deg
        else:
            phigrid = 0*unyt.deg

        # The evaluated arrays are (R, phi, z, t)
        psi, br, bphi, bz, psidr, psidz = (
            arr[..., 0] for arr in evaluate(
                rgrid, phigrid, zgrid, 0.*unyt.s, "psi", "br", "bphi", "bz",
                "psidr", "psidz", grid=True, bfield=field,
            )
        )
        # The splines add the poloidal field from psi to the tabulated BR and
        # Bz, so only the rest of them (e.g. a perturbation) is tabulated.
        # evaluate returns the psi derivatives without units, so they are
        # given here.
        dpsi = unyt.Wb / unyt.rad / unyt.m
        r = rgrid[:, None, None]
        br = br + (np.asarray(psidz) * dpsi * unyt.rad / r).to("T")
        bz = bz - (np.asarray(psidr) * dpsi * unyt.rad / r).to("T")

        # psi is axisymmetric in the splines
        data = {
            "rgrid": rgrid,
            "zgrid": zgrid,
            "psi": psi[:, 0, :],
            "psilimits": field.psilimits,
            "axisrz": field.axisrz,
        }
        if not field.nripple:
            data.update(bphi=bphi[:, 0, :], br=br[:, 0, :], bz=bz[:, 0, :])
            super().__init__(ascot, "bfieldspline2d", data)
            return

        data.update(phigrid=phigrid, bphi=bphi, br=br, bz=bz)
        super().__init__(ascot, "bfieldspline3d", data)


class ToroidallyAveragedBfield(InputTemplate):

    def convert_B_3DS(**kwargs):
        """Convert :class:`B_3DS` input to `B_2DS` input.

        This function takes a toroidal average of Bphi and sets BR and Bz to
        zero.

        Parameters
        ----------
        **kwargs
            Arguments passed to :meth:`B_3DS.write_hdf5` excluding ``fn`` and
            ``desc``.

        Returns
        -------
        out : dict
            :class:`B_3DS` converted as an input for :meth:`write_hdf5`.
        """
        bphi = np.mean(kwargs["bphi"], axis=1)
        br   = np.mean(kwargs["bphi"], axis=1) * 0
        bz   = np.mean(kwargs["bphi"], axis=1) * 0
        return {
            "rmin":kwargs["b_rmin"], "rmax":kwargs["b_rmax"],
            "nr":kwargs["b_nr"], "zmin":kwargs["b_zmin"],
            "zmax":kwargs["b_zmax"], "nz":kwargs["b_nz"],
            "axisr":kwargs["axisr"], "axisz":kwargs["axisz"],
            "psi0":kwargs["psi0"], "psi1":kwargs["psi1"],
            "psi":kwargs["psi"], "br":br, "bz":bz,
            "bphi":bphi}
