#The main workflow of the CartPole example: instantiate env(task(environment+agent models))
#Composer
#1.Define agent
#2.Define environment
#3.Define task
#Learning
#1.Define learning network
#2.Define net update
#3.Define learning flow
###########################
import inspect
import numpy as np
from dm_control import mujoco
from dm_control import mjcf
from dm_control import composer
from dm_control import suite
###########################
#Graphics-related
import matplotlib.animation as animation
import matplotlib.pyplot as plt
#The basic mujoco wrapper.
from dm_control import mujoco
#Access to enums and MuJoCo library functions.
from dm_control.mujoco.wrapper.mjbindings import enums
import wandb
###########################
import torch.optim as optim
###########################
from C_L_RoboLoco_Agent import RoboLoco
from C_L_RoboLoco_Task_Env import PressWithSpecificForce
from C_L_DQN import ReplayBuffer
from C_L_DQN import DQNet
from C_L_DQN import DQNet_update
from C_L_DQN import epsilon_by_frame
from C_L_DQN import train_log
from Matplotlib_Animation import display_video
###########################
def train_RoboLoco_DQNet(env, model, replay_buffer, batch_size, optimizer, num_frames):
    wandb.watch(model, log="all", log_freq=10)#"Hooks into the torch model (DQN) to collect gradients and the topology."
    frames = []
    losses = []
    all_rewards = []
    episode_reward = 0
    episode_counter = 0
    time_step = env.reset() #env is the specific environment task defined on the domain.
    #here specifically it is the task 'balance' defined on domain 'cartpole'.
    #env.reset() starts a new episode and returns the first 'TimeStep'(step-type, reward, discount, observation).
    state = time_step.observation #directory of variable values that needs values extracted.
    positions = [x for x in state['unnamed_model/joint_positions'][0]]
    velocities = [x for x in state['unnamed_model/joint_velocities'][0]]
    state = [*positions, *velocities] #extraction.
    #defining the project run for loop of frames. Each frame there is: action selection; state transition; observation;
    #end of episode check; replay buffer push; logging of: rewards (if end of episode), loss (if multiple of a 100th
    #of the way through the project run); DQN update if there are enough samples in the buffer.
    for frame_idx in range(1, num_frames + 1):
        print("frame index:"+str(frame_idx))
        epsilon = epsilon_by_frame(frame_idx) #value of decaying exponential as function of frame index.
        action = model.act(state, epsilon) #action selected as function of explore probability epsilon and the state.
        #print(action)
        _, reward, discount, obs = env.step(action) #action taken. returns step-type, reward, discount, obs.
        if reward == None: #so that rewards can be summed.
            reward = 0.0
        #new state is the extracted and ordered observation values.
        positions = [x for x in obs['unnamed_model/joint_positions'][0]]
        velocities = [x for x in obs['unnamed_model/joint_velocities'][0]]
        next_state = [*positions, *velocities]  # extraction.
        done = env._reset_next_step #Boolean variable of whether episode is finished or not.
        #Currently the episode is not terminating if the pole gets below a certain angle theta threshold.
        #This must be implemented.
        replay_buffer.push(state, action, reward, next_state, done) #pushing the sample to the replay buffer.

        print("discount: "+str(discount)) #1.0
        #print("done: " + str(done)) #False
        print("step-type: "+str(_)) #StepType.MID

        state = next_state #state variable update.
        episode_reward += reward #adding step rewards to total rewards.

        # frames capture for animation
        framerate=30
        if len(frames) < env._physics.data.time * framerate:
            #data.time resets each episode, so first episode recorded.
            #can add "+ config["ep_time_limit_secs"]*episode_counter) after data.time to have full frames animation.
            pixels = env._physics.render(scene_option=scene_option, camera_id='cam0')
            # rendering the physics model scene into pixels. cameras defined in the cartpole.xml file in suite.
            frames.append(pixels)  # building list of animation frames.

        if done:
            metric_name = "reward"
            train_log(metric_name, episode_reward, frame_idx) #logging the current total reward by frame index.
            #state = env.reset() #no reset needed, because done automatically through the 'TimeStep'.
            all_rewards.append(episode_reward) #tracking the total episode rewards across episodes.
            #only useful if we get multiple episodes per run.
            episode_reward = 0
            episode_counter += 1
        if len(replay_buffer) > batch_size:
            loss = DQNet_update(model, optimizer, replay_buffer, batch_size) #calculating the loss on a batch
            #of data from the replay buffer according to the DQN update function in 'DQN_Classes.py'.
            losses.append(loss.data) #tracking the loss data for each frame.
            if frame_idx % (num_frames/100) == 0:
                metric_name = "loss"
                train_log(metric_name,loss,frame_idx) #logging the loss data by frame if the frame index is a multiple
                #of a 100th of the total number of frames in the run.
    save_name = str(config["env_name"])+"_animation_"+str(config["num_total_frames"])+"_model_frames.mp4"
    print(save_name)
    display_video(save_name,frames)

if __name__ == "__main__":
    #defining the config file which holds defining characteristics about the WandB project run
    config = {
        "env_name": "MyRoboLoco",
        "buffer_size": 100,
        "batch_size": 24,
        "num_total_frames": 100,
        "ep_time_limit_secs": 1
    }
    #starts the WandB run. A WandB run is defined by the: project, config directory, job type, etc.
    run = wandb.init(
        project="C_L"+config["env_name"],  #name of project on WandB
        config=config,
        job_type="learning",
        #sync_tensorboard=True,  # auto-upload sb3's tensorboard metrics
        save_code=True,  # optional
    )
    ###########################
    #In my own conception of the dm_control RL pipeline I have split the steps into the Composer step
    #(defines agent, environment, task in that order) and the Learning step
    #(defines the agent training procedure on the given task in that environment).
    #The following are the steps followed when using the Composer dm_control library.

    #Composer
    #1.Define agent
    #agent = CartPole()
    #2.Define environment & 3.Define task
    #task = CartPoleTask(agent)
    #env = composer.Environment(task, random_state=np.random.RandomState(42))

    random_state = np.random.RandomState(42)
    NUM_SUBSTEPS = 25  # The number of physics substeps per control timestep.

    BODY_RADIUS = 0.1
    BODY_SIZE = (BODY_RADIUS, BODY_RADIUS, BODY_RADIUS / 2)
    NUM_LEGS = 4

    roboloco = RoboLoco(num_legs=NUM_LEGS, body_size=BODY_SIZE, body_radius=BODY_RADIUS)
    task = PressWithSpecificForce(roboloco)
    env = composer.Environment(task, random_state=random_state)

    # Visualize the joint axis
    scene_option = mujoco.wrapper.core.MjvOption()
    scene_option.flags[enums.mjtVisFlag.mjVIS_JOINT] = True
    ###########################
    #Learning
    #1.Define learning network
    num_inputs = 16  #the number of CartPole state variables: (x, v, theta, omega) = (cart+pole pos+vel).
    num_outputs = 2  #the number of the CartPole action variable options: left and right. A = argmax_a_Q(s,a)
    #print(len(env.physics.data.qpos)) #2 position variables.
    #print(len(env.physics.data.qvel)) #2 velocity variables.
    print(env.action_spec().shape[0]) #1 action variable (discrete valued: left and right).
    #defining the agent Q network architecture. Inputs the full state, outputs the action-values Q(s,a).
    net = DQNet(num_inputs, num_outputs)

    #2.Define net update & 3.Define learning flow
    optimizer = optim.Adam(net.parameters()) #choosing the optimizer (Adam) which uses the DQN loss gradients
    #in a particular way to compute backpropagation.
    buffer_size = config["buffer_size"] #defining the max size of the replay buffer.
    replay_buffer = ReplayBuffer(buffer_size) #instantiating the replay buffer.
    batch_size = config["batch_size"] #defining the batch size
    num_frames = config["num_total_frames"] #defining the run length (total frames over all episodes).
    #training the agent DQN as a function of: env[domain (agent+environment physics)+task on the domain],
    #DQN architecture, replay buffer size, batch size, optimizer, total length of run in frames.
    train_RoboLoco_DQNet(env, net, replay_buffer, batch_size, optimizer, num_frames)
    ###########################
    run.finish() #ending the WandB project run. Logging completed.
    ###########################