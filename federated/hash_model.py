import hashlib
import numpy as np


def hash_model_parameters(parameters):
    """
    Generate a deterministic SHA-256 hash
    from model parameter arrays.

    Parameters
    ----------
    parameters : list of numpy.ndarray
        Model weights and biases in a fixed order.

    Returns
    -------
    str
        64-character hexadecimal SHA-256 hash.
    """

    hasher = hashlib.sha256()

    for parameter in parameters:
        # Convert to NumPy array
        array = np.asarray(parameter)

        # Make sure the memory representation is deterministic
        array = np.ascontiguousarray(array)

        # Add raw bytes to the hash
        hasher.update(array.tobytes())

    return hasher.hexdigest()


if __name__ == "__main__":
    # Simple test parameters
    parameters = [
        np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32),
        np.array([1.0, 2.0], dtype=np.float32)
    ]

    model_hash = hash_model_parameters(parameters)

    print("SHA-256 Model Hash:")
    print(model_hash)

    print("\nHash length:", len(model_hash))

    if len(model_hash) == 64:
        print("SUCCESS: Valid SHA-256 hash generated.")
    else:
        print("ERROR: Invalid hash length.")