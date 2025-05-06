def callback(
    data: NDArray[...], timestamps: NDArray[np.float64], info: mne.Info
) -> tuple[NDArray[...], NDArray[np.float64]]:
    """A callback function.

    Parameters
    ----------
    data : NDArray[...]
        Data array of shape (n_times, n_channels).
    timestamps : NDArray[np.float64]
        Timestamp array of shape (n_times,).

    Returns
    -------
    data : NDArray[...]
        The modified data array of shape (n_times, n_channels).
    timestamps : NDArray[np.float64]
        The modified timestamp array of shape (n_times,).
    """
    # implement your callback function here and return the modified data and timestamps
    return data, timestamps
