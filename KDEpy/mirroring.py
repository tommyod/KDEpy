from KDEpy import FFTKDE
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import time

def time_it(func):
    def wrapper(*args, **kwargs):
        t1 = time.time()
        result = func(*args, **kwargs)
        t2 = time.time()
        print(f"Time taken: {t2-t1:.6f} seconds")
        return result
    return wrapper


def mirror_data(data, boundaries,  pdf_values = None, decimals=10):
    """
    Mirrors the data based on the provided boundaries and adds up the values onto the unmirrored parts.
    Sets everything below or above the boundaries to 0 in the values.
    
    Parameters:
    data (np.ndarray): The input data array (N x D).
    boundaries (list): A list of tuples specifying the lower and upper boundaries for each dimension.
                       Use None for no boundary in that dimension.
                       Valid examples: 
                            boundaries = [[1, 10], [1, 10]]  # Both dimensions are bounded between 1 and 10
                `           boundaries = [[1, 10], [None, 10]]  # First dimension is bounded between 1 and 10, second dimension is bounded between -inf and 10
                            boundaries = [[1, 10], None]     # First dimension is bounded between 1 and 10, second dimension is unbounded
    pdf_values (np.ndarray): Array (N,) of the evaluated PDF at each gridpoint.    
    Returns:
    pd.DataFrame: The mirrored data array (N x D) , and column ['value'] containing the rescaled PDF.
    """

    def sum_together_np(mirrored_data, updated_values, decimals=10):
        """
        Group by unique rows in data and sum the corresponding values.
        
        Parameters:
        data (np.ndarray): The input data array (N x D).
        values (np.ndarray): The values array (N,).
        decimals (int): Number of decimal places to round to for grouping.
        
        Returns:
        np.ndarray: The grouped data array (M x D).
        np.ndarray: The summed values array (M,).
        """
        print('numpy')
        # Round the data to the specified number of decimals. Important! Otherwise, due to numerical precision, the numbers won't match
        rounded_data = np.round(mirrored_data, decimals=decimals)

        unique_data, indices = np.unique(rounded_data, axis=0, return_inverse=True) # Find unique rows and their indices
        summed_values = np.zeros(unique_data.shape[0]) # Preallocate the summed values array
        np.add.at(summed_values, indices, updated_values) # Use np.add.at to sum the values for each unique row
        
        return unique_data, summed_values

    if pdf_values is None:
        pdf_values = np.ones(data.shape[0])/data.shape[0] # If no PDF values are provided, assume uniform distribution.

    mirrored_data = data.copy()
    updated_values = pdf_values.copy()
    
    # Each boundary is 'folding' the data, and creating a copy, and adding the PDF values.    
    for dim, boundary in enumerate(boundaries):
        if boundary is not None:
            lower, upper = boundary
            if lower is not None:
                try:
                    closest_lower = np.max(data[data[:, dim] <= lower, dim]) # Mirror using the closest data point as pivot
                    lower_mirror = 2 * closest_lower - data[:, dim]
                    mirrored_points = np.column_stack([lower_mirror if i == dim else data[:, i] for i in range(data.shape[1])])
                    mirrored_data = np.vstack([mirrored_data, mirrored_points])
                    updated_values = np.concatenate([updated_values, pdf_values])
                    
                    # mirrored_data, updated_values = sum_together(mirrored_data, updated_values)

                    mirrored_data, updated_values = sum_together_np(mirrored_data, updated_values)
                except:
                    pass
            if upper is not None:
                try:
                    closest_upper = np.min(data[data[:, dim] >= upper, dim]) # Mirror using the closest data point as pivot
                    upper_mirror = 2 * closest_upper - data[:, dim]
                    mirrored_points = np.column_stack([upper_mirror if i == dim else data[:, i] for i in range(data.shape[1])])
                    mirrored_data = np.vstack([mirrored_data, mirrored_points])
                    updated_values = np.concatenate([updated_values, pdf_values])
                    
                    # mirrored_data, updated_values = sum_together(mirrored_data, updated_values)  

                    mirrored_data, updated_values = sum_together_np(mirrored_data, updated_values)

                except:
                    pass

    # # Have to do a second pass. After the mirroring. Drop values out of boundaries.
    for dim, boundary in enumerate(boundaries):
        if boundary is not None:
            lower, upper = boundary
            if lower is not None:
                try:
                    mask = mirrored_data[:, dim] >= lower
                    mirrored_data = mirrored_data[mask]
                    updated_values = updated_values[mask]
                except:
                    pass
            if upper is not None:
                try:
                    mask = mirrored_data[:, dim] <= upper
                    mirrored_data = mirrored_data[mask]
                    updated_values = updated_values[mask]
                except:
                    pass
    

    # Adjusting pdf to make the sum = 1
    updated_values = updated_values/updated_values.sum()

    return mirrored_data, updated_values 







#%%%%%%%%%%%%%%%%%%%%%%%%%  EXAMPLE DATA GENERATION  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

def generate_beta_distribution(a, b, loc, scale, size):
    # Generate two columns of beta distribution
    col1 = np.random.beta(a, b, size=size) * scale + loc
    col2 = np.random.beta(a, b, size=size) * scale + loc
    
    # Combine the columns into a 2D array
    data = np.column_stack((col1, col2))
    
    return data


# Example usage
a = 1.05
b = 3
loc = 1
scale = 10
size = 10000  # Number of samples

data = generate_beta_distribution(a, b, loc, scale, size)

#%%%%%%%%%%%%%%%%%%%%%%%%%  USAGE, with results of FFTKDE  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

# With grid points: 
grid_points = 100
X, Z = FFTKDE(bw=1).fit(data)((grid_points,grid_points))
x, y = np.unique(X[:, 0]), np.unique(X[:, 1])

N = 8  # Number of contours, for contour plot

# ORIGINAL, UNMIRRORED KDE, beautiful plot, by kdepy
# Plot the contours
z = Z.reshape(grid_points, grid_points).T
plt.contour(x, y, z, N, colors="k")
plt.contourf(x, y, z, N, cmap="PuBu")
plt.plot(data[:, 0], data[:, 1], "ok", ms=2)
plt.yticks([])
plt.xticks([])
plt.title('Original KDE, Kernel passing boundaries')
plt.show()



#%%%%%%%%%%%%%%%%%%%%%%%%%  USAGE, with all boundaries  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

boundaries = [[1, 10], [1, 10]]  # Both dimensions are bounded between 1 and 10 

# With grid points: 
grid_points = 100
X, Z = FFTKDE(bw=1).fit(data)((grid_points,grid_points))
x, y = np.unique(X[:, 0]), np.unique(X[:, 1])

N = 8  # Number of contours, for contour plot

# ORIGINAL, UNMIRRORED KDE, tricontour
x, y = X[:, 0], X[:, 1]
z = Z
# Plot the contours
# Create subplots
fig, axs = plt.subplots(1, 2, figsize=(10, 8))
axs[0].tricontour(x, y, z, N, colors="k")
axs[0].plot(data[:, 0], data[:, 1], "ok", ms=2)
axs[0].set_yticks([])
axs[0].set_xticks([])
axs[0].set_title('Original KDE')

# MODIFIED, MIRRORED KDE, tricontour
# Define boundaries for mirroring
mirrored_data, updated_pdf = mirror_data(X, boundaries, Z)
x, y = mirrored_data[:,0], mirrored_data[:,1]
z = updated_pdf

# Plot the contours
axs[1].tricontour(x, y, z, N, colors="k")
axs[1].plot(data[:, 0], data[:, 1], "ok", ms=2)
axs[1].set_yticks([])
axs[1].set_xticks([])
axs[1].set_title('Mirrored KDE')

plt.show()



#%%%%%%%%%%%%%%%%%%%%%%%%%  USAGE, unbouded lower limit second variable %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

boundaries = [[1, 10], [None, 10]]  # Both dimensions are bounded between 1 and 10 

# With grid points: 
grid_points = 100
X, Z = FFTKDE(bw=1).fit(data)((grid_points,grid_points))
x, y = np.unique(X[:, 0]), np.unique(X[:, 1])

N = 8  # Number of contours, for contour plot

# ORIGINAL, UNMIRRORED KDE, tricontour
x, y = X[:, 0], X[:, 1]
z = Z
# Plot the contours
# Create subplots
fig, axs = plt.subplots(1, 2, figsize=(10, 8))
axs[0].tricontour(x, y, z, N, colors="k")
axs[0].plot(data[:, 0], data[:, 1], "ok", ms=2)
axs[0].set_yticks([])
axs[0].set_xticks([])
axs[0].set_title('Original KDE')

# MODIFIED, MIRRORED KDE, tricontour
# Define boundaries for mirroring
mirrored_data, updated_pdf = mirror_data(X, boundaries, Z)
x, y = mirrored_data[:,0], mirrored_data[:,1]
z = updated_pdf
# Plot the contours
axs[1].tricontour(x, y, z, N, colors="k")
axs[1].plot(data[:, 0], data[:, 1], "ok", ms=2)
axs[1].set_yticks([])
axs[1].set_xticks([])
axs[1].set_title('Mirrored KDE over 1 boundary only (-inf to 10 in second dimension)')

plt.show()




#%%%%%%%%%%%%%%%%%%%%%%%%%  USAGE,unbounded second covariate %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

boundaries = [[1, 10], None]  # Both dimensions are bounded between 1 and 10 

# With grid points: 
grid_points = 100
X, Z = FFTKDE(bw=1).fit(data)((grid_points,grid_points))
x, y = np.unique(X[:, 0]), np.unique(X[:, 1])

N = 8  # Number of contours, for contour plot

# ORIGINAL, UNMIRRORED KDE, tricontour
x, y = X[:, 0], X[:, 1]
z = Z
# Plot the contours
# Create subplots
fig, axs = plt.subplots(1, 2, figsize=(10, 8))
axs[0].tricontour(x, y, z, N, colors="k")
axs[0].plot(data[:, 0], data[:, 1], "ok", ms=2)
axs[0].set_yticks([])
axs[0].set_xticks([])
axs[0].set_title('Original KDE')

# MODIFIED, MIRRORED KDE, tricontour
# Define boundaries for mirroring
mirrored_data, updated_pdf = mirror_data(X, boundaries, Z)
x, y = mirrored_data[:,0], mirrored_data[:,1]
z = updated_pdf
# Plot the contours
axs[1].tricontour(x, y, z, N, colors="k")
axs[1].plot(data[:, 0], data[:, 1], "ok", ms=2)
axs[1].set_yticks([])
axs[1].set_xticks([])
axs[1].set_title('Mirrored KDE over 1 dimension only (-inf to inf in second dimension)')

plt.show()


#%%%%%%%%%%%%%%%%%%%%%%%%%  RESAMPLING and MIRRORING  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

boundaries = [[1, 10], [1, 10]]  # Both dimensions are bounded between 1 and 10 


from KDEpy.bw_selection import silvermans_rule, improved_sheather_jones
# Get the standard deviation of the kernel functions
# Silverman assumes normality of data - use ISJ with much data instead
kernel_std = [silvermans_rule(data[:,column].reshape(-1, 1)) for column in range(data.shape[1])]

# (1) First resample original data, then (2) add noise from kernel
size = 5000
resampled_data = data[np.random.choice(data.shape[0], size=size, replace=True)]
resampled_data = resampled_data + np.random.randn(size, data.shape[1]) * kernel_std

mirrored_data, _ = mirror_data(resampled_data, boundaries)
print(mirrored_data.shape)

# Plot the results
plt.scatter(data[:,0], data[:,1], alpha=0.5, label='Original Data')
plt.scatter(resampled_data[:,0], resampled_data[:,1], c='b', alpha=0.5, label='Unbounded resampling')
plt.scatter(mirrored_data[:, 0], mirrored_data[:, 1], c='r', alpha=0.5, label='Bounded Mirrored resampling')
plt.title('Bounded and unbounded KDE')
plt.tight_layout(); 
plt.legend(loc='upper left');
plt.show()


