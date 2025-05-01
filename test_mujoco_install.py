import mujoco
from mujoco.viewer import launch_passive


model = mujoco.MjModel.from_xml_path("inverted_pendulum.xml")
data = mujoco.MjData(model)

launch_passive(model, data)
input("Press Enter to exit...")

