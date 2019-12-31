import matplotlib.pyplot as plt
from matplotlib import animation


def visualize_trajectory(trajectory):
    """Visualizes trajectory by imshow animation.

    Args:
        trajectory: numpy.array
            Shape `[num_frames, num_x_pixals, num_y_pixals]`.

    Returns: animation.FuncAnimation
    """
    fig = plt.figure()
    ax = plt.axes()
    img = ax.imshow(trajectory[0], cmap='gray')

    def init():
        img.set_data([[]])
        return img,

    def animate(i):
        frame = trajectory[i]
        img.set_data(frame)
        return img,

    anim = animation.FuncAnimation(fig, animate, init_func=init,
                                   frames=len(trajectory), blit=True)
    return anim
