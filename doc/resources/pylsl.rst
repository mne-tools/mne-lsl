:orphan:

.. include:: ../links.inc

Differences with pylsl
======================

Faster chunk pull
~~~~~~~~~~~~~~~~~

Arguably the most important difference, pulling a chunk of numerical data with
:meth:`~bsl.lsl.StreamInlet.pull_chunk` is much faster than with its
`pylsl <lsl python_>`_ counterpart. `pylsl <lsl python_>`_ loads the retrieved samples
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
`pylsl <lsl python_>`_:

* 7.87 ms ± 58 µs with ``pylsl``
* 4.35 µs ± 134 ns with ``bsl.lsl``

Note that this performance improvement is absent for ``string`` based streams.

Convenience methods
~~~~~~~~~~~~~~~~~~~

A :class:`~bsl.lsl.StreamInfo` has several convenience methods to retrieve and set
channel attributes: names, types, units.

.. hlist::
    :columns: 2

    * :meth:`~bsl.lsl.StreamInfo.get_channel_names`
    * :meth:`~bsl.lsl.StreamInfo.get_channel_types`
    * :meth:`~bsl.lsl.StreamInfo.get_channel_units`
    * :meth:`~bsl.lsl.StreamInfo.set_channel_names`
    * :meth:`~bsl.lsl.StreamInfo.set_channel_types`
    * :meth:`~bsl.lsl.StreamInfo.set_channel_units`

Those methods eliminate the need to interact with the ``XMLElement`` underlying tree,
present in the :py:attr:`bsl.lsl.StreamInfo.desc` property.

Improve arguments
~~~~~~~~~~~~~~~~~

The arguments of a `~bsl.lsl.StreamInfo`, `~bsl.lsl.StreamInlet`,
`~bsl.lsl.StreamOutlet` support a wider variety of types. For instance:

* ``dtype``, which correspond to the ``channel_format`` in `pylsl <lsl python_>`_, can
  be provided as a string or as a supported :class:`numpy.dtype`, e.g. ``np.int8``.
* ``processing_flags`` can be provided as strings instead of the underlying integer
  mapping.

Overall, the arguments are checked in ``bsl.lsl``. Any type or value mistake will raise
an helpful error message.

Unique resolve function
~~~~~~~~~~~~~~~~~~~~~~~

`pylsl <lsl python_>`_ has several stream resolving functions:

* ``resolve_streams`` which resolves all streams on the network.
* ``resolve_byprop`` which resolves all streams with a specific value for a given
  property.
* ``resolve_bypred`` which resolves all streams with a given predicate.

:func:`bsl.lsl.resolve_streams` simplifies stream resolution with a unique function with
similar functionalities.

.. _pylsl pull_chunk: https://github.com/labstreaminglayer/pylsl/blob/16a4198087936386e866d7239bfde32d1fef6d6b/pylsl/pylsl.py#L862-L870
