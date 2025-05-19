:orphan:

.. include:: ../links.inc

Differences with pylsl
======================

Faster chunk pull
-----------------

Arguably the most important difference, pulling a chunk of numerical data with
:meth:`~mne_lsl.lsl.StreamInlet.pull_chunk` is faster than with its
`pylsl <lsl python_>`_ counterpart, except if using the argument ``dest_obj``.
By default, `pylsl <lsl python_>`_ loads the retrieved samples one by one in a list of
list, `here <pylsl pull_chunk_>`_.

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
65 channels in double precision (``float64``) takes:

* 3.91 ± 0.12 ms with ``pylsl`` (default behavior: list comprehension)
* 419.08 ± 38.10 ns with ``mne_lsl.lsl`` (:func:`numpy.frombuffer`)

Increasing the number of channels to 650, simulating an higher sample count yields:

* 46.56 ± 1.06 ms per loop with ``pylsl`` (default behavior: list comprehension)
* 424.29 ± 36.41 ns with ``mne_lsl.lsl`` (:func:`numpy.frombuffer`)

.. dropdown:: Timeit python code
    :animate: fade-in-slide-down

    .. coode-block:: python

        import ctypes
        import timeit

        import numpy as np

        n_samples = 1024
        n_channels = 650
        max_values = n_samples * n_channels

        # create a data buffer, 'retrieved from liblsl'
        data_buffer = (ctypes.c_double * max_values)()
        for i in range(max_values):
            data_buffer[i] = float(i)


        def method_list_comprehension():
            """List comprehension, default method for pylsl."""
            samples = [
                [data_buffer[s * n_channels + c] for c in range(n_channels)]
                for s in range(n_samples)
            ]
            return samples


        def method_frombuffer():
            """Numpy frombuffer, default method for mne-lsl."""
            samples = np.frombuffer(data_buffer, dtype=np.float64)[:max_values].reshape(
                n_samples, n_channels
            )
            return samples


        repeat = 5
        number = 100
        for func in (method_list_comprehension, method_frombuffer):
            timer = timeit.Timer(func)
            results = timer.repeat(repeat, number=number)
            times = [t / number for t in results]
            best = min(times)
            worst = max(times)
            avg = sum(times) / len(times)

            # format with the corret unit
            if best < 1e-6:
                unit, factor = "ns", 1e9
            elif best < 1e-3:
                unit, factor = "µs", 1e6
            elif best < 1:
                unit, factor = "ms", 1e3
            else:
                unit, factor = "s", 1

            formatted_time = f"{best * factor:.2f} ± {(avg - best) * factor:.2f} {unit}"
            plural = "s" if number != 1 else ""
            print(f"{number} loop{plural}, best of {repeat}: {formatted_time} per loop")

Note that ``pylsl`` pulling function support a ``dest_obj`` argument described as::

    A Python object that supports the buffer interface.
    If this is provided then the dest_obj will be updated in place and the samples list
    returned by this method will be empty. It is up to the caller to trim the buffer to
    the appropriate number of samples. A numpy buffer must be order='C'.

If a :class:`~numpy.ndarray` is used as ``dest_obj``, the memory re-allocation step
described above is skipped, yielding better performance than ``mne_lsl.lsl`` at the constant
of code complexity as the user is now responsible for the memory management.

.. note::

    This performance improvement is absent for ``string`` based streams. Follow
    :issue:`225` for more information.

Convenience methods
-------------------

A :class:`~mne_lsl.lsl.StreamInfo` has several convenience methods to retrieve and set
channel attributes: names, types, units.

.. hlist::
    :columns: 2

    * :meth:`~mne_lsl.lsl.StreamInfo.get_channel_names`
    * :meth:`~mne_lsl.lsl.StreamInfo.get_channel_types`
    * :meth:`~mne_lsl.lsl.StreamInfo.get_channel_units`
    * :meth:`~mne_lsl.lsl.StreamInfo.set_channel_names`
    * :meth:`~mne_lsl.lsl.StreamInfo.set_channel_types`
    * :meth:`~mne_lsl.lsl.StreamInfo.set_channel_units`

Those methods eliminate the need to interact with the ``XMLElement`` underlying tree,
present in the :py:attr:`mne_lsl.lsl.StreamInfo.desc` property. The description can even
be set or retrieved directly from a :class:`~mne.Info` object with
:meth:`~mne_lsl.lsl.StreamInfo.set_channel_info` and
:meth:`~mne_lsl.lsl.StreamInfo.get_channel_info`.

Improve arguments
-----------------

The arguments of a :class:`~mne_lsl.lsl.StreamInfo`, :class:`~mne_lsl.lsl.StreamInlet`,
:class:`~mne_lsl.lsl.StreamOutlet` support a wider variety of types. For instance:

* ``dtype``, which correspond to the ``channel_format`` in `pylsl <lsl python_>`_, can
  be provided as a string or as a supported :class:`numpy.dtype`, e.g. ``np.int8``.
* ``processing_flags`` can be provided as strings instead of the underlying integer
  mapping.

Overall, the arguments are checked in ``mne_lsl.lsl``. Any type or value mistake will
raise an helpful error message.

Unique resolve function
-----------------------

`pylsl <lsl python_>`_ has several stream resolving functions:

* ``resolve_streams`` which resolves all streams on the network.
* ``resolve_byprop`` which resolves all streams with a specific value for a given
  property.
* ``resolve_bypred`` which resolves all streams with a given predicate.

:func:`mne_lsl.lsl.resolve_streams` simplifies stream resolution with a unique function
with similar functionalities.

.. _pylsl pull_chunk: https://github.com/labstreaminglayer/pylsl/blob/16a4198087936386e866d7239bfde32d1fef6d6b/pylsl/pylsl.py#L862-L870
