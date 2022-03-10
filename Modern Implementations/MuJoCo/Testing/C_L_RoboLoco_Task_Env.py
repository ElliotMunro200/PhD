#Defining a task environment for a Robotic Locomotion agent
import PIL.Image
import numpy as np
import matplotlib.pyplot as plt

from dm_control import mjcf
from dm_control import composer
from dm_control.composer import variation
from dm_control.composer.observation import observable

from dm_control.locomotion.arenas import floors
from dm_control.composer.variation import noises
from dm_control.composer.variation import distributions

class Button(composer.Entity):
  """A button Entity which changes colour when pressed with certain force."""
  def _build(self, target_force_range=(5, 10)):
    self._min_force, self._max_force = target_force_range
    self._mjcf_model = mjcf.RootElement()
    self._geom = self._mjcf_model.worldbody.add(
        'geom', type='cylinder', size=[0.25, 0.02], rgba=[1, 0, 0, 1])
    self._site = self._mjcf_model.worldbody.add(
        'site', type='cylinder', size=self._geom.size*1.01, rgba=[1, 0, 0, 0])
    self._sensor = self._mjcf_model.sensor.add('touch', site=self._site)
    self._num_activated_steps = 0

  def _build_observables(self):
    return ButtonObservables(self)

  @property
  def mjcf_model(self):
    return self._mjcf_model
  # Update the activation (and colour) if the desired force is applied.
  def _update_activation(self, physics):
    current_force = physics.bind(self.touch_sensor).sensordata[0]
    self._is_activated = (current_force >= self._min_force and
                          current_force <= self._max_force)
    physics.bind(self._geom).rgba = (
        [0, 1, 0, 1] if self._is_activated else [1, 0, 0, 1])
    self._num_activated_steps += int(self._is_activated)

  def initialize_episode(self, physics, random_state):
    self._reward = 0.0
    self._num_activated_steps = 0
    self._update_activation(physics)

  def after_substep(self, physics, random_state):
    self._update_activation(physics)

  @property
  def touch_sensor(self):
    return self._sensor

  @property
  def num_activated_steps(self):
    return self._num_activated_steps


class ButtonObservables(composer.Observables):
  """A touch sensor which averages contact force over physics substeps."""
  @composer.observable
  def touch_force(self):
    return observable.MJCFFeature('sensordata', self._entity.touch_sensor,
                                  buffer_size=NUM_SUBSTEPS, aggregator='mean')

class UniformCircle(variation.Variation):
  """A uniformly sampled horizontal point on a circle of radius `distance`."""
  def __init__(self, distance):
    self._distance = distance
    self._heading = distributions.Uniform(0, 2*np.pi)

  def __call__(self, initial_value=None, current_value=None, random_state=None):
    distance, heading = variation.evaluate(
        (self._distance, self._heading), random_state=random_state)
    return (distance*np.cos(heading), distance*np.sin(heading), 0)

class PressWithSpecificForce(composer.Task):

  def __init__(self, RoboLoco):
    self._roboloco = RoboLoco
    self._arena = floors.Floor()
    self._arena.add_free_entity(self._roboloco)
    self._arena.mjcf_model.worldbody.add('light', pos=(0, 0, 4))
    self._arena.mjcf_model.worldbody.add('camera', name="cam0", pos=(0, -1, .7), xyaxes=(1,0,0,0,1,2))
    self._button = Button()
    self._arena.attach(self._button)

    # Configure initial poses
    self._roboloco_initial_pos = (0, 0, 0.15)
    self._roboloco_ini_act_pos = (0,0,0,0,0,0,0,0)
    self._roboloco_ini_vel = (0,0,0)
    self._roboloco_ini_ang_vel = (0,0,0)
    button_distance = distributions.Uniform(0.5, .75)
    self._button_initial_pos = UniformCircle(button_distance)

    # Configure variators
    self._mjcf_variator = variation.MJCFVariator()
    self._physics_variator = variation.PhysicsVariator()

    # Configure and enable observables
    pos_corrptor = noises.Additive(distributions.Normal(scale=0.01))
    self._roboloco.observables.joint_positions.corruptor = pos_corrptor
    self._roboloco.observables.joint_positions.enabled = True
    vel_corruptor = noises.Multiplicative(distributions.LogNormal(sigma=0.01))
    self._roboloco.observables.joint_velocities.corruptor = vel_corruptor
    self._roboloco.observables.joint_velocities.enabled = True
    self._button.observables.touch_force.enabled = True

    def to_button(physics):
      button_pos, _ = self._button.get_pose(physics)
      return self._roboloco.global_vector_to_local_frame(physics, button_pos)

    self._task_observables = {}
    self._task_observables['button_position'] = observable.Generic(to_button)

    for obs in self._task_observables.values():
      obs.enabled = True
    self.control_timestep = NUM_SUBSTEPS * self.physics_timestep

  @property
  def root_entity(self):
    return self._arena

  @property
  def task_observables(self):
    return self._task_observables

  def initialize_episode_mjcf(self, random_state):
    self._mjcf_variator.apply_variations(random_state)

  def initialize_episode(self, physics, random_state):
    self._physics_variator.apply_variations(physics, random_state)
    roboloco_pos, button_pos = variation.evaluate(
        (self._roboloco_initial_pos, self._button_initial_pos),
        random_state=random_state)
    self._roboloco.set_pose(physics, position=roboloco_pos) # randomized
    self._button.set_pose(physics, position=button_pos) # randomized
    all_joints = self._roboloco.mjcf_model.find_all('joint', exclude_attachments=False)
    physics.bind(all_joints).qpos = self._roboloco_ini_act_pos
    self._roboloco.set_velocity(physics, velocity=self._roboloco_ini_vel, angular_velocity=self._roboloco_ini_ang_vel)

  def get_reward(self, physics):
    return self._button.num_activated_steps / NUM_SUBSTEPS

global NUM_SUBSTEPS
NUM_SUBSTEPS = 25 # The number of physics substeps per control timestep.

if __name__ == "__main__":
    from C_L_RoboLoco_Agent import RoboLoco

    random_state = np.random.RandomState(42)

    BODY_RADIUS = 0.1
    BODY_SIZE = (BODY_RADIUS, BODY_RADIUS, BODY_RADIUS / 2)
    NUM_LEGS = 4

    roboloco = RoboLoco(num_legs=NUM_LEGS, body_size=BODY_SIZE, body_radius=BODY_RADIUS)
    task = PressWithSpecificForce(roboloco)
    env = composer.Environment(task, random_state=random_state)

    env.reset()
    img = PIL.Image.fromarray(env.physics.render())
    plt.imshow(img)
    print("done")