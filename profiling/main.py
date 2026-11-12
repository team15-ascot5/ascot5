import unyt
import numpy as np

from a5py import Ascot, SimulationOptions
from a5py.templates import PremadeMagneticField, TransmuteAnalyticalBfieldToSplines

MARKER_COUNT = 500   # increase for larger simulation
MAX_MILEAGE = 5e-4   # increase for longer simulation

ascot = Ascot()

analytical = PremadeMagneticField(
    ascot,
    field="iter-circular",
    axisymmetric=True,
).create_input()

TransmuteAnalyticalBfieldToSplines(
    ascot, field=analytical, nr=400, rlim=unyt.unyt_array([4, 8.2], "m"),
    nz=800, zlim=unyt.unyt_array([-4, 4], "m")
).create_input().activate()

ascot.data.create_efieldcartesian(
    exyz=unyt.unyt_array([0, 0, 0], "V/m"),
)

ascot.data.create_guidingcentermarker(
    species="He4",
    r=(6.2 + 1.8 * np.random.random(MARKER_COUNT)) * unyt.m,
    z=0.0 * unyt.m,
    ekin=3.5e6 * unyt.eV,
    pitch=-1 + 2 * np.random.random(MARKER_COUNT),
)

options = SimulationOptions.from_dict(
    physics={
        "enable_orbit_following": True,
    },
    endconditions={
        "activate_simulation_time_limits": True,
        "max_mileage": MAX_MILEAGE,
    },
    histograms=[
        {
            "dimensions": [
                ("r", 6.2 * unyt.m, 8.0 * unyt.m, 10),
            ],
            "charge_interval": (
                2,
                2,
            ),
        },
    ],
)

# gc-fixed
# options.simulation.mode = "guiding-center"
# options.simulation.enable_adaptive = False
# options.simulation.timestep = 1.0e-7

# gyro-orbit
options.simulation.mode = "gyro-orbit"
options.simulation.timestep = 1.0e-8

# gc-adaptive (seems to loop infinitely...)
# options.simulation.mode = "guiding-center"
# options.simulation.enable_adaptive = True
# options.simulation.timestep = 1.0e-7

run = ascot.simulate(params=options)

# Check that all markers reach the end
# print(run.getstate("time", state="end"))

# import matplotlib.pyplot as plt
# dist = run.getdist(0, ("r"))
# plt.plot(dist.distribution)
# plt.show()
