:orphan:

.. include:: ../links.inc

Differences with pylsl
======================

Faster chunk pull
~~~~~~~~~~~~~~~~~

Arguably the most important difference, pulling a chunk with
:meth:`~bsl.lsl.StreamInlet.pull_chunk` is much faster than with its
`pylsl <lsl python_>`_ counterpart. `pylsl <lsl python_>`_. loads the retrieved samples
one by one in a list of list, `here <pylsl pull_chunk_>`_.

.. code-block:: python

    # return results (note: could offer a more efficient format in the
    # future, e.g., a numpy array)
    num_samples = num_elements / num_channels
    if dest_obj is None:
        samples = [
            [data_buff[s * num_channels + c] for c in range(num_channels)]
            for s in range(int(num_samples))
        ]
        if self.channel_format == cf_string:
            samples = [[v.decode("utf-8") for v in s] for s in samples]
            free_char_p_array_memory(data_buff, max_values)

The creation of the variable ``samples`` is expensive and is performed in linear time
``O(n)``, scaling with the number of values. Instead, ``numpy`` can be used to pull
the entire buffer at once with :func:`numpy.frombuffer`.

.. code-block:: python

    samples = np.frombuffer(
        data_buffer, dtype=self._dtype)[:n_samples_data].reshape(-1, self._n_channels
    )

Now, ``samples`` is created in constant time ``O(1)``. The performance gain varies
depending on the number of values pulled, for instance retrieving 1024 samples with
65 channels in double precision (``float64``) is ~1800 times slower with
`pylsl <lsl python_>`_.:

* 7.87 ms ± 58 µs with ``pylsl``
* 4.35 µs ± 134 ns with ``bsl.lsl``

.. _pylsl pull_chunk: https://github.com/labstreaminglayer/pylsl/blob/16a4198087936386e866d7239bfde32d1fef6d6b/pylsl/pylsl.py#L862-L870
