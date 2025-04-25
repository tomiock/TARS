import matplotlib.pyplot as plt
import numpy as np

# Define the number of points for each set
num_points_set1 = 8
num_points_set2 = 8 # A different number of points for the second set

# Generate sample data for the first set of points ('o' marker)
np.random.seed(0) # for reproducibility
x_data_set1 = np.random.rand(num_points_set1) * 10
y_data_set1 = np.random.rand(num_points_set1) * 10
z_data_set1 = np.random.rand(num_points_set1) * 5 + 6 # Ensure points are above the plane

# Generate random colors for the first set
colors_set1 = np.random.rand(num_points_set1, 3) # Generate RGB colors

# Generate sample data for the second set of points ('x' marker)
np.random.seed(1) # Use a different seed for different random data
x_data_set2 = np.random.rand(num_points_set2) * 10
y_data_set2 = np.random.rand(num_points_set2) * 10
z_data_set2 = np.random.rand(num_points_set2) * 2 # Varying z-range and starting height

# Generate random colors for the second set
colors_set2 = np.random.rand(num_points_set2, 3) # Generate RGB colors

# Create a new figure and a 3D axes
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

# Plot the first set of points ('o' marker) with random colors
ax.scatter(x_data_set1, y_data_set1, z_data_set1, c=colors_set1, marker='o', label='Set 1 (o)')

# Plot the second set of points ('x' marker) with random colors
ax.scatter(x_data_set2, y_data_set2, z_data_set2, c=colors_set2, marker='x', label='Set 2 (x)')


# Define the horizontal plane
plane_z = 5 # The height of the plane
x_plane = np.linspace(0, 10, 20)
y_plane = np.linspace(0, 10, 20)
X_plane, Y_plane = np.meshgrid(x_plane, y_plane)
Z_plane = np.full_like(X_plane, plane_z)

# Plot the horizontal plane with visible grid lines
ax.plot_surface(X_plane, Y_plane, Z_plane,
                color='lightgray',
                alpha=0.7,
                edgecolor='black',
                linewidth=0.2)

# --- Add perpendicular lines from points in Set 1 to the plane ---
for i in range(num_points_set1):
    point_coords = [x_data_set1[i], y_data_set1[i], z_data_set1[i]]
    projection_coords = [x_data_set1[i], y_data_set1[i], plane_z]

    ax.plot([point_coords[0], projection_coords[0]],
            [point_coords[1], projection_coords[1]],
            [point_coords[2], projection_coords[2]],
            c='gray', linestyle='--', linewidth=0.8, alpha=0.7)

# --- Add perpendicular lines from points in Set 2 to the plane ---
for i in range(num_points_set2):
    point_coords = [x_data_set2[i], y_data_set2[i], z_data_set2[i]]
    projection_coords = [x_data_set2[i], y_data_set2[i], plane_z]

    ax.plot([point_coords[0], projection_coords[0]],
            [point_coords[1], projection_coords[1]],
            [point_coords[2], projection_coords[2]],
            c='gray', linestyle=':', linewidth=0.8, alpha=0.7) # Using a different color/style for clarity


# Set labels for the axes (optional)
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')

# Set limits for the axes (optional)
ax.set_xlim([0, 10])
ax.set_ylim([0, 10])
# Adjust z-limit to accommodate the highest points in both sets
ax.set_zlim([0, max(np.max(z_data_set1), np.max(z_data_set2)) + 1])


# Set the view angle (optional)
ax.view_init(elev=20, azim=-60)

ax.set_axis_off()

# Show the plot
plt.show()
