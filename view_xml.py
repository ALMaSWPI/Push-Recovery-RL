import mujoco
import time
from mujoco.viewer import launch_passive

model = mujoco.MjModel.from_xml_path("inverted_pendulum.xml")
data = mujoco.MjData(model)

with launch_passive(model, data) as viewer:
    while viewer.is_running():
        mujoco.mj_step(model, data)
        viewer.sync()
        time.sleep(0.01)  # slow down for smooth visuals
