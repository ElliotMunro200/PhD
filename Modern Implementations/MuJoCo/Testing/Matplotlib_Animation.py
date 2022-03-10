# Graphics-related
import matplotlib.animation as animation
import matplotlib.pyplot as plt
# The basic mujoco wrapper.
from dm_control import mujoco
# Access to enums and MuJoCo library functions.
from dm_control.mujoco.wrapper.mjbindings import enums

# function that takes in the complete set of frames with framerate and makes, saves, displays animation.
def display_video(save_name, frames, framerate=30):
    height, width, _ = frames[0].shape # finding the height and width of animation figure size from frame size.
    dpi = 70 # dots per inch (resolution).
    # orig_backend = matplotlib.get_backend()
    # matplotlib.use('Agg')  # Switch to headless 'Agg' to inhibit figure rendering.
    fig, ax = plt.subplots(1, 1, figsize=(width / dpi, height / dpi), dpi=dpi) # making size of plot accordingly.
    # matplotlib.use(orig_backend)  # Switch back to the original backend.
    ax.set_axis_off() # turn x and y axis off.
    ax.set_aspect('equal') # same scaling for x and y.
    ax.set_position([0, 0, 1, 1]) # sets the axes position.
    im = ax.imshow(frames[0]) # display data as an image, i.e., on a 2D regular raster.
    # function to update the frame data on the animation figure axes.
    def update(frame):
        im.set_data(frame)
        return [im]

    interval = 1000 / framerate # delay between frames in milliseconds.
    anim = animation.FuncAnimation(fig=fig, func=update, frames=frames,
                                   interval=interval, blit=True, repeat=False)
    # "fig: the figure object used to get needed events, such as draw or resize"
    # "func: The function to call at each frame. The first argument will be the next value in frames."
    # "frames: Source of data to pass func and each frame of the animation."
    # "interval: delay between frames in milliseconds."
    # "blit: Whether blitting is used to optimize drawing."
    # (Blitting speeds up by rendering all non-changing graphic elements into a background image once.)
    # "repeat: Whether the animation repeats when the sequence of frames is completed."
    anim.save(save_name, fps=framerate, extra_args=['-vcodec', 'libx264'])
    # saving the animation with the filename: save_name, with extra arguments.
    # plt.show() # this command is not required for some reason.
    return interval

if __name__ == "__main__":

    #xml physics model
    swinging_body = """
    <mujoco>
      <worldbody>
        <light name="top" pos="0 0 1"/>
        <body name="box_and_sphere" euler="0 0 -30">
          <joint name="swing" type="hinge" axis="1 -1 0" pos="-.2 -.2 -.2"/>
          <geom name="red_box" type="box" size=".2 .2 .2" rgba="1 0 0 1"/>
          <geom name="green_sphere" pos=".2 .2 .2" size=".1" rgba="0 1 0 1"/>
        </body>
      </worldbody>
    </mujoco>
    """
    # physics model made from xml code
    physics = mujoco.Physics.from_xml_string(swinging_body)

    duration = 2  # (seconds)
    framerate = 30  # (Hz)

    # Visualize the joint axis
    #scene_option = mujoco.wrapper.core.MjvOption()
    #scene_option.flags[enums.mjtVisFlag.mjVIS_JOINT] = True

    # Simulate and display video.
    frames = []
    physics.reset()  # Reset state and time
    # while the physics model time is less than the specified duration, physics model steps,
    # then frames added to frames list as framerate requires.
    while physics.data.time < duration:
        physics.step()

        if len(frames) < physics.data.time * framerate:
            pixels = physics.render(scene_option=scene_option) # rendering the physics model scene into pixels.
            frames.append(pixels) # building list of animation frames.
    save_name = 'basic_animation.mp4',
    display_video(save_name, frames, framerate) # inputting the fully collected frames to the animation function.

