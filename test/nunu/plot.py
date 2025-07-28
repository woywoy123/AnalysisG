import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import time
import os

def live_visualization(opt_log="optimization_log.csv", ellipse_log="ellipses.csv"):
    # Read optimization log
    df = pd.read_csv(opt_log)
    n_ellipses = (len(df.columns) - 5) // 4
    
    # Read ellipse data
    ellipse_df = pd.read_csv(ellipse_log)
    unique_ellipses = ellipse_df['ellipse_index'].unique()
    
    # Prepare plot
    fig = plt.figure(figsize=(14, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # Determine plot limits
    all_points = []
    for i in range(n_ellipses):
        all_points.extend(df[[f'point_{i}_x', f'point_{i}_y', f'point_{i}_z']].values)
    all_points = np.array(all_points)
    
    ellipse_points = []
    for idx in unique_ellipses:
        ellipse_points.extend(ellipse_df[ellipse_df['ellipse_index'] == idx][['x','y','z']].values)
    ellipse_points = np.array(ellipse_points)
    
    all_points = np.vstack([all_points, ellipse_points])
    max_val = np.max(np.abs(all_points)) * 1.2
    
    ax.set_xlim([-max_val, max_val])
    ax.set_ylim([-max_val, max_val])
    ax.set_zlim([-max_val, max_val])
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('Multi-Ellipse Optimization')
    
    # Plot ellipses (static)
    ellipse_plots = []
    colors = plt.cm.tab10.colors
    for idx in unique_ellipses:
        ellipse_data = ellipse_df[ellipse_df['ellipse_index'] == idx]
        x = ellipse_data['x'].values
        y = ellipse_data['y'].values
        z = ellipse_data['z'].values
        
        # Close the ellipse
        x = np.append(x, x[0])
        y = np.append(y, y[0])
        z = np.append(z, z[0])
        
        color = colors[idx % len(colors)]
        plot, = ax.plot(x, y, z, c=color, alpha=0.4, linewidth=1.5, 
                        label=f'Ellipse {int(idx)}')
        ellipse_plots.append(plot)
    
    # Setup dynamic elements
    point_scatters = []
    centroid_scatter = ax.scatter([], [], [], c='red', s=100, label='Centroid')
    connection_lines = []
    
    for i in range(n_ellipses):
        scat = ax.scatter([], [], [], c=colors[i % len(colors)], 
                         s=80, marker='o', edgecolors='black', 
                         label=f'Point {i}')
        point_scatters.append(scat)
    
    plt.legend()
    plt.tight_layout()
    plt.ion()
    plt.show()
    
    # Animation loop
    for i in range(len(df)):
        iteration = df['iteration'].iloc[i]
        objective = df['objective'].iloc[i]
        centroid = [df['centroid_x'].iloc[i], 
                   df['centroid_y'].iloc[i], 
                   df['centroid_z'].iloc[i]]
        
        ax.set_title(f'Iteration {iteration}, Objective: {objective:.4f}')
        
        # Update points
        points = []
        for j in range(n_ellipses):
            x = df[f'point_{j}_x'].iloc[i]
            y = df[f'point_{j}_y'].iloc[i]
            z = df[f'point_{j}_z'].iloc[i]
            points.append([x, y, z])
            point_scatters[j]._offsets3d = ([x], [y], [z])
        
        # Update centroid
        centroid_scatter._offsets3d = ([centroid[0]], [centroid[1]], [centroid[2]])
        
        # Update connections
        for line in connection_lines:
            line.remove()
        connection_lines = []
        
        for point in points:
            line = ax.plot([point[0], centroid[0]], 
                          [point[1], centroid[1]], 
                          [point[2], centroid[2]], 
                          c='gray', alpha=0.5, linestyle='--')
            connection_lines.append(line[0])
        
        # Highlight current point on each ellipse
        for j, plot in enumerate(ellipse_plots):
            # Find closest point on ellipse to current point
            ellipse_data = ellipse_df[ellipse_df['ellipse_index'] == j]
            ellipse_points = ellipse_df[ellipse_df['ellipse_index'] == j][['x','y','z']].values
            current_point = np.array(points[j])
            
            # Find closest point index
            distances = np.linalg.norm(ellipse_points - current_point, axis=1)
            closest_idx = np.argmin(distances)
            
            # Get the point
            closest_point = ellipse_points[closest_idx]
            
            # Update highlight
            if j < len(point_scatters):
                point_scatters[j]._offsets3d = ([closest_point[0]], 
                                               [closest_point[1]], 
                                               [closest_point[2]])
        
        plt.draw()
        plt.pause(0.1)  # Control animation speed
        
    plt.ioff()
    plt.show()

if __name__ == "__main__":
    live_visualization("optimization_log.csv", "ellipses.csv")
