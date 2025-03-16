import matplotlib.pyplot as plt
import numpy as np

def plot_results(solution, save_path=None, show=True):
    """
    Plot the results of the G-FOLD solver.
    
    Args:
        solution (dict): Solution from GFoldSolver.solve()
        save_path (str, optional): Path to save the figure
        show (bool): Whether to show the plot
        
    Returns:
        matplotlib.figure.Figure: The created figure
    """
    positions = solution["positions"]
    velocities = solution["velocities"]
    thrusts = solution["normalized_thrusts"]
    time_points = solution["time_points"]
    
    x = positions[:, 0]
    y = positions[:, 1]
    z = positions[:, 2]
    
    fig = plt.figure(figsize=(12, 10))
    
    # 3D trajectory plot
    ax = fig.add_subplot(221, projection='3d')
    ax.plot(x, y, z, "red", label="path")
    
    # Calculate and plot the glide slope cone
    initial_height = positions[0, 2]
    sin_glide_slope = 0  # Default value, can be passed from the solver
    
    if initial_height > 0:
        cosx = np.sqrt(1-np.square(sin_glide_slope))
        theta = np.arange(0, 2*np.pi, np.pi/100)
        radius = np.sqrt(1-np.square(cosx))/cosx * initial_height if cosx > 0 else initial_height
        z_values = np.arange(0, initial_height, initial_height/20)
        
        for count, zval in enumerate(z_values):
            x_vals = np.cos(theta) * (count+1)/20 * radius
            y_vals = np.sin(theta) * (count+1)/20 * radius
            ax.plot(x_vals, y_vals, zval, '#0000FF55')
    
    ax.legend(loc="center left", bbox_to_anchor=(2, 0.5))
    ax.set_xlim3d(-initial_height*0.6, initial_height*0.6)
    ax.set_ylim3d(-initial_height*0.6, initial_height*0.6)
    ax.set_zlim3d(0, initial_height*1.2)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('3D Trajectory')
    
    # Velocity plot
    ax = fig.add_subplot(222)
    v = np.linalg.norm(velocities, axis=1)
    ax.plot(time_points, velocities[:, 2], label="Z velocity")
    ax.plot(time_points, v, label="Total velocity")
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Velocity (m/s)')
    ax.set_title('Velocity Profile')
    ax.legend()
    
    # Thrust plot
    ax = fig.add_subplot(223)
    ax.plot(time_points, thrusts*100, label="thrust %")
    ax.set_ylim(0, 100)
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Thrust (%)')
    ax.set_title('Thrust Profile')
    ax.legend()
    
    # Ground trajectory
    ax = fig.add_subplot(224)
    ax.plot(positions[:, 0], positions[:, 1], label="ground trajectory")
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_title('Ground Track')
    ax.legend()
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
    
    if show:
        plt.show()
        
    return fig
