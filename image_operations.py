import os
import json
import numpy as np
import pandas as pd
import os
from scipy.ndimage import rotate


class ImageOperations(object):

	def __init__(self, filepath):
		self.filepath = filepath

		# todo: add error handling on filepath
		trace_df_in = pd.read_xml(path_or_buffer=f'{filepath}.gom_volume', xpath='//volume/volume_size')
		self.x, self.y, self.z = trace_df_in.loc[0, 'x'], trace_df_in.loc[0, 'y'], trace_df_in.loc[0, 'z']
		trace_df_voxel_size = pd.read_xml(path_or_buffer=f'{filepath}.gom_volume', xpath='//volume/voxel_size')
		self.x_voxel, self.y_voxel, self.z_voxel = trace_df_voxel_size.loc[0, 'x'], trace_df_voxel_size.loc[0, 'y'], trace_df_voxel_size.loc[0, 'z']
		self.array_1D = np.fromfile(
			open(f'{filepath}.raw'),
			dtype=np.uint16,
			count=self.x * self.y * self.z)

	"""
	array modification methods
	"""

	# todo: make cutoff_value, threshold algorithmically determined
	@staticmethod
	def clean_artifacts(
			array: np.ndarray,
			cutoff_value: int,
			threshold: int
	) -> np.ndarray:
		# get normalizing values from array
		arr_max = np.max(array)
		arr_min = np.min(array)
		arr_range = arr_max - arr_min

		# set new zero value and apply to array
		zero_val = int(cutoff_value * 65535 / arr_range)  # todo: check math against this normalized array
		array[array < threshold] = zero_val

		return array

	@staticmethod
	def resize_array(
			array: np.ndarray,
			target_shape: tuple
	) -> np.ndarray:
		# Calculate the difference in shape for each axis
		shape_diff = np.subtract(target_shape, array.shape)

		# Calculate the padding for each side of each axis
		pad_width = [(int(np.ceil(diff / 2)), int(np.floor(diff / 2))) for diff in shape_diff]

		# Pad the array with zeros
		resized_array = np.pad(array, pad_width, mode='constant')

		return resized_array

	@staticmethod
	# https://stackoverflow.com/questions/37532184/downsize-3d-matrix-by-averaging-in-numpy-or-alike
	def compress_volume(
			volume: np.ndarray,
			block_size: tuple
	) -> np.ndarray:
		# volume is the 3D input array
		# block_size is the block size on which averaging is to be performed

		m, n, r = np.array(volume.shape) // block_size
		new_shape = ((m, n, r) + (np.array(volume.shape) % block_size > 0)) * block_size

		volume_i = ImageOperations.resize_array(volume, new_shape)
		m, n, r = np.array(volume_i.shape) // block_size

		return volume_i.reshape(m, block_size[0], n, block_size[1], r, block_size[2]).mean((1, 3, 5))

	@staticmethod
	def expand_array(
			array: np.ndarray,
			expansion_size: tuple
	) -> np.ndarray:
		# todo: error handling that array dimensions = expansion_size dimensions

		expanded_array = array.copy()

		for i in range(len(expansion_size)):
			# Expand array along each dimension using numpy.repeat()
			expanded_array = np.repeat(expanded_array, expansion_size[i], axis=i)

		return expanded_array

	@staticmethod
	def remove_edge_pixels(
			volume: np.ndarray,
			edge_threshold: int,
			zero_value: int,
			upper_threshold: int = 60000
	) -> np.ndarray:
		# todo: error handling on array dimensions

		# Find the bright pixels
		bright_pixels = np.argwhere(volume > edge_threshold)

		# Initialize a boolean mask to track marked pixels
		marked_pixels = np.zeros_like(volume, dtype=bool)

		# Define function for region-growing
		def region_grow(seed: tuple):
			stack = [seed]
			while stack:
				pixel = stack.pop()
				marked_pixels[pixel] = True

				# Get the neighboring pixels
				x, y, z = pixel
				neighbors = [(x + 1, y, z), (x - 1, y, z), (x, y + 1, z), (x, y - 1, z), (x, y, z + 1), (x, y, z - 1)]

				# Add unmarked bright neighboring pixels to the stack
				for neighbor in neighbors:
					if (
							0 <= neighbor[0] < volume.shape[0]
							and 0 <= neighbor[1] < volume.shape[1]
							and 0 <= neighbor[2] < volume.shape[2]
							and volume[neighbor] > zero_value
							and not marked_pixels[neighbor]
							and volume[neighbor] < upper_threshold
					):
						stack.append(neighbor)

		# Apply region-growing starting from pixels at the edges
		for pixel in bright_pixels:
			x, y, z = pixel
			if (
					(
						x == 0 or x == volume.shape[0] - 1 or
						y == 0 or y == volume.shape[1] - 1 or
						z == 0 or z == volume.shape[2] - 1
					)
					and tuple(pixel) not in marked_pixels
			):
				region_grow(tuple(pixel))  # Convert pixel to tuple before using it as a seed

		return ~marked_pixels  # returns the inverse of the binary array

	"""
	instance functions
	"""

	def __enter__(self):
		return self

	def get_voxel_size(self):
		return self.x_voxel, self.y_voxel, self.z_voxel

	def clean_white_px(
			self,
			threshold: int = 60000
	):
		self.array_1D[self.array_1D > threshold] = 0

	def normalize_array(
			self,
			chunk_size: int = 1000000
	) -> np.ndarray:
		# self.clean_white_px()

		# construct normalizing values
		arr_max = np.max(self.array_1D)
		arr_min = np.min(self.array_1D)
		arr_range = arr_max - arr_min

		# determine number of chunks
		num_elements = self.array_1D.size
		num_chunks = int(np.ceil(num_elements / chunk_size))

		# normalize the chunks
		normalized_chunks = []
		for i in range(num_chunks):
			# Calculate the start and end indices for the current chunk
			start_idx = i * chunk_size
			end_idx = min(start_idx + chunk_size, num_elements)

			# Extract the current chunk
			chunk = self.array_1D[start_idx:end_idx]

			# Convert the chunk to floating-point type
			chunk_float = chunk.astype(np.float64)

			# Normalize the chunk by subtracting the minimum value and dividing by the range
			normalized_chunk = 65535 * (chunk_float - arr_min) / arr_range

			# Append the normalized chunk to the list
			normalized_chunks.append(np.array(normalized_chunk, dtype=np.uint16))

		# Concatenate the normalized chunks into a single array
		return np.concatenate(normalized_chunks)

	def array_to_volume(
			self,
			array: np.ndarray
	) -> np.ndarray:
		array.shape = (self.z, self.y, self.x)
		return array

	def __exit__(self, exc_type, exc_val, exc_tb):
		return


class DataAugmentation(object):
	def __init__(self, array: np.ndarray):
		self.array = array

	@staticmethod
	def rotate(volume, angle_x, angle_y, angle_z, order=3, mode='constant', cval=0):
		"""
		Rotate a 3D volume along the x, y, and z axes.

		Parameters:
			volume (numpy.ndarray): The 3D volume as a NumPy array.
			angle_x (float): The rotation angle in degrees around the x-axis.
			angle_y (float): The rotation angle in degrees around the y-axis.
			angle_z (float): The rotation angle in degrees around the z-axis.
			order (int): The order of interpolation. Default is 3, which indicates cubic interpolation.
			mode (str): Points outside the boundaries of the input are filled according to the given mode.
						See scipy.ndimage.rotate documentation for available modes. Default is 'constant'.
			cval (float): The constant value used for points outside the boundaries when mode is 'constant'.
						  Default is 0.

		Returns:
			numpy.ndarray: The rotated 3D volume as a NumPy array.
		"""
		# Ensure the input volume is a numpy array
		volume = np.array(volume)

		# Perform rotation along each axis
		rotated_volume = volume.copy()
		rotated_volume = rotate(rotated_volume, angle_x, axes=(1, 2), order=order, mode=mode, cval=cval)
		rotated_volume = rotate(rotated_volume, angle_y, axes=(0, 2), order=order, mode=mode, cval=cval)
		rotated_volume = rotate(rotated_volume, angle_z, axes=(0, 1), order=order, mode=mode, cval=cval)

		return rotated_volume

	@staticmethod
	def scale():
		pass

	@staticmethod
	def flip():
		pass

	@staticmethod
	def translate():
		pass

	@staticmethod
	def gaussian_noise():
		pass


def generate_2d_slices(volume: np.ndarray, plane: str) -> dict:
	if plane not in ['XY', 'XZ', 'YZ']:
		raise ValueError("Invalid plane. Use 'XY', 'XZ', or 'YZ'.")

	slices = []
	shape = volume.shape

	if plane == 'XY':
		for z in range(shape[2]):
			slice_order_label = z
			slice_2d = volume[:, :, z]
			slices.append((slice_order_label, slice_2d))

	elif plane == 'XZ':
		for y in range(shape[1]):
			slice_order_label = y
			slice_2d = volume[:, y, :]
			slices.append((slice_order_label, slice_2d))

	elif plane == 'YZ':
		for x in range(shape[0]):
			slice_order_label = x
			slice_2d = volume[x, :, :]
			slices.append((slice_order_label, slice_2d))

	return {'slices': slices, 'shape': shape, 'plane': plane}  # return dict which retains original volume shape


def reconstruct_3d_array(slices, shape: tuple, plane: str) -> np.ndarray:
	if len(shape) != 3:
		raise ValueError("The shape of the 3D array should be a tuple with three dimensions.")

	if plane not in ['XY', 'XZ', 'YZ']:
		raise ValueError("Invalid plane. Use 'XY', 'XZ', or 'YZ'.")

	volume = np.zeros(shape)

	for axis, slice_2d in slices:
		if plane == 'XY':
			volume[:, :, axis] = slice_2d

		elif plane == 'XZ':
			volume[:, axis, :] = slice_2d

		elif plane == 'YZ':
			volume[axis, :, :] = slice_2d

	return volume


def process_json_files(plane, directory, shape):

	def get_color(defect_type):
		if defect_type == "Void":
			return (0, 0, 255)  # Blue
		elif defect_type == "Crack":
			return (0, 255, 0)  # Green
		else:
			return (0, 0, 0)  # Default color (Black)

	if plane not in ["XY", "XZ", "YZ"]:
		raise ValueError("Invalid 'plane' value. Use 'XY', 'XZ', or 'YZ'.")

	volume = np.zeros(shape + (4,), dtype=np.uint8)

	for filename in os.listdir(directory):
		if filename.startswith(f"{plane}_slice_") and filename.endswith(".json"):
			file_path = os.path.join(directory, filename)

			with open(file_path, "r") as json_file:
				data = json.load(json_file)
				slice_number = int(filename.split("_")[2].split('.json')[0])

				for defect in data["DefectResult"]:
					color = get_color(defect["DefectName"])
					opacity = int(defect["DefectProb"] * 255)  # Scale to [0, 255]

					x_start = int(defect["X"])
					x_end = x_start + int(defect["DefectWidth"])
					y_start = int(defect["Y"])
					y_end = y_start + int(defect["DefectHeight"])

					if plane == "XY":
						volume[y_start:y_end, x_start:x_end, slice_number, :3] = color
						volume[y_start:y_end, x_start:x_end, slice_number, 3] = opacity
					elif plane == "XZ":
						volume[y_start:y_end, slice_number, x_start:x_end, :3] = color
						volume[y_start:y_end, slice_number, x_start:x_end, 3] = opacity
					elif plane == "YZ":
						volume[slice_number, y_start:y_end, x_start:x_end, :3] = color
						volume[slice_number, y_start:y_end, x_start:x_end, 3] = opacity

	return volume


# Try storing float in 3D array
def create_annotated_volume(plane, directory, shape):
	if plane not in ["XY", "XZ", "YZ"]:
		raise ValueError("Invalid plane. Must be 'XY', 'XZ', or 'YZ'.")

	volume = np.zeros(shape, dtype=np.float64)

	for filename in os.listdir(directory):
		if filename.startswith(f"{plane}_slice_") and filename.endswith(".json"):
			file_path = os.path.join(directory, filename)
			with open(file_path, "r") as json_file:
				data = json.load(json_file)
				slice_number = int(filename.split("_")[2].split('.json')[0])

		# if filename.startswith(f"{plane}_slice_"):
		#     slice_number = int(filename.split("_")[-1].split(".")[0])
		#     with open(os.path.join(directory, filename), 'r') as file:
		#         data = json.load(file)

				for defect in data["DefectResult"]:
					x = defect["X"]
					y = defect["Y"]
					width = defect["DefectWidth"]
					height = defect["DefectHeight"]
					prob = defect["DefectProb"]

					if plane == "XY":
						volume[x:x+width, y:y+height, slice_number] = prob
					elif plane == "XZ":
						volume[y:y+height, slice_number, x:x+width] = prob
					elif plane == "YZ":
						volume[slice_number, y:y+height, x:x+width] = prob

	return volume


def json_to_df(filepath):
	with open(filepath) as json_data:
		data = json.load(json_data)
		return pd.DataFrame(data["DefectResult"],
						  columns=["DefectName", "X", "Y", "DefectWidth", "DefectHeight", "DefectProb"])


def import_json_files_to_df(directory: str) -> pd.DataFrame:
	# df = pd.DataFrame(columns=["plane", "slice_num", "DefectName", "X", "Y", "DefectWidth", "DefectHeight", "DefectProb"])

	dfs = []

	for filename in os.listdir(directory):
		if filename.endswith(".json"):
			file_path = os.path.join(directory, filename)

			df = json_to_df(filepath=file_path)
			df['plane'] = filename.split("_")[0]
			df['slice_number'] = int(filename.split("_")[2].split('.json')[0])

			dfs.append(df)

	return pd.concat(dfs, ignore_index=True)


def transform_defect_df(row):
	if row['plane'] == 'XY':
		x_start, x_end = row['Y'], row['Y'] + row['DefectHeight']
		y_start, y_end = row['X'], row['X'] + row['DefectWidth']
		z_start, z_end = row['slice_number'], row['slice_number']
	elif row['plane'] == 'XZ':
		x_start, x_end = row['Y'], row['Y'] + row['DefectHeight']
		y_start, y_end = row['slice_number'], row['slice_number']
		z_start, z_end = row['X'], row['X'] + row['DefectWidth']
	elif row['plane'] == 'YZ':
		x_start, x_end = row['slice_number'], row['slice_number']
		y_start, y_end = row['Y'], row['Y'] + row['DefectHeight']
		z_start, z_end = row['X'], row['X'] + row['DefectWidth']
	else:
		# Handle any other cases if needed
		x_start, x_end, y_start, y_end, z_start, z_end = None, None, None, None, None, None

	return pd.Series({
		'defect_type': row['DefectName'],
		'plane': row['plane'],
		'defect_prob': row['DefectProb'],
		'x_start': x_start,
		'x_end': x_end,
		'y_start': y_start,
		'y_end': y_end,
		'z_start': z_start,
		'z_end': z_end
	})


def convert_ranges_to_coordinates(row):
	x_values = np.arange(row['x_start'], row['x_end'] + 1)
	y_values = np.arange(row['y_start'], row['y_end'] + 1)
	z_values = np.arange(row['z_start'], row['z_end'] + 1)

	coordinates = [(x, y, z) for x in x_values for y in y_values for z in z_values]

	return pd.Series([row['defect_type'], row['plane'], row['defect_prob'], coordinates],
					 index=['defect_type', 'plane', 'defect_prob', 'coordinate'])


def compile_coordinates_df(dataframe: pd.DataFrame) -> pd.DataFrame:
	# Apply the function to each row in the DataFrame
	df = dataframe.apply(convert_ranges_to_coordinates, axis=1)

	# Explode the coordinate lists to individual rows
	df = df.explode('coordinate')

	# Reset the index of the new DataFrame
	return df.reset_index(drop=True)


def collapse_defect_probs(dataframe: pd.DataFrame) -> pd.DataFrame:
	df = dataframe.copy()

	# Step 1: Drop the 'plane' column
	df.drop(columns='plane', inplace=True)

	# Step 2: Compile defect_probs into tuples
	compiled_df = df.groupby(['defect_type', 'coordinate'])['defect_prob'].apply(list)

	# Fill missing entries with zeros
	compiled_df = compiled_df.apply(lambda x: x + [0] * (3 - len(x)) if len(x) < 3 else x)

	# Step 3: Reset the index and rename columns
	compiled_df = compiled_df.reset_index()
	compiled_df.columns = ['defect_type', 'coordinate', 'defect_prob']

	return compiled_df


def power_mean(dataframe: pd.DataFrame, column_name: str, power: int) -> pd.DataFrame:
    """
    Apply a power mean function to a specified column in a DataFrame.

    Parameters:
    df (DataFrame): The input DataFrame.
    column_name (str): The name of the column containing arrays.
    power (float): The power value for the power mean.

    Returns:
    DataFrame: A new DataFrame with the power mean applied to the specified column.
    """

    def compute_power_mean(arr):
        # Ensure the array contains at least one element
        if len(arr) == 0:
            return np.nan  # Return NaN for empty arrays

        # Calculate the power mean
        powered_sum = sum([x ** power for x in arr])
        return (powered_sum / len(arr)) ** (1 / power)

    # Create a new column with the power means applied
    dataframe['power_mean_' + column_name] = dataframe[column_name].apply(compute_power_mean)

    return dataframe


def generate_defect_volumes(defects_dir: str, power: int, shape: tuple, volume_dtype: str = 'uint8') -> np.ndarray:
	if 'int' in volume_dtype:
		try:
			max_value = np.iinfo(volume_dtype).max
		except ValueError:
			print('Invalid datatype for return volume')
			return []

	df = import_json_files_to_df(directory=defects_dir)
	df = df.apply(transform_defect_df, axis=1)
	df = compile_coordinates_df(dataframe=df)
	df = collapse_defect_probs(dataframe=df)

	# apply power mean to defect probabilities
	df_agg = power_mean(dataframe=df, column_name='defect_prob', power=3)

	if 'int' in volume_dtype:
		df_agg['array_result'] = (df_agg['power_mean_defect_prob']*max_value).astype(volume_dtype)
	else:
		df_agg['array_result'] = df_agg['power_mean_defect_prob']

	coordinates = df_agg['coordinate'].tolist()
	defect_values = df_agg['array_result'].tolist()

	volume = np.zeros(shape, dtype=volume_dtype)
	volume[tuple(zip(*coordinates))] = defect_values

	return volume






"""
def convert_df_to_arrays(dataframe: pd.DataFrame, shape):
	# TODO: schema error handling

	df = dataframe.apply(transform_defect_df, axis=1)

	arrays = {}
	for plane in df['plane'].unique():
		volume = np.zeros(shape, dtype='float64')

		mask = df['plane'] == plane

		for index, row in df.loc[mask].iterrows():
			volume[
				row['x_start']:row['x_end'],
				row['y_start']:row['y_end'],
				row['z_start']:row['z_end']
			] = row['defect_prob']

		
		# establish x, y ranges according to volume shape
		if plane == 'XY':
			x_start, x_end = df.loc[mask, 'Y'], df.loc[mask, 'Y'] + df.loc[mask, 'DefectHeight']
			y_start, y_end = df.loc[mask, 'X'], df.loc[mask, 'X'] + df.loc[mask, 'DefectWidth']
			z_start, z_end = df.loc[mask, 'slice_number'], df.loc[mask, 'slice_number']
		if plane == 'XZ':
			x_start, x_end = df.loc[mask, 'Y'], df.loc[mask, 'Y'] + df.loc[mask, 'DefectHeight']
			y_start, y_end = df.loc[mask, 'slice_number'], df.loc[mask, 'slice_number']
			z_start, z_end = df.loc[mask, 'X'], df.loc[mask, 'X'] + df.loc[mask, 'DefectWidth']
		if plane == 'YZ':
			x_start, x_end = df.loc[mask, 'slice_number'], df.loc[mask, 'slice_number']
			y_start, y_end = df.loc[mask, 'Y'], df.loc[mask, 'Y'] + df.loc[mask, 'DefectHeight']
			z_start, z_end = df.loc[mask, 'X'], df.loc[mask, 'X'] + df.loc[mask, 'DefectWidth']

		# apply defect mapping
		for i in range(1, len(x_start.index)):
			print(plane, i)
			volume[x_start[i]:x_end[i], y_start[i]:y_end[i], z_start[i]:z_end[i]] = df.loc[mask, 'DefectProb'][i]
		

		arrays[plane] = volume

	return arrays
"""

