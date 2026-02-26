===========
Quick Start
===========

RadDB allows to download weather radar data from various cloud buckets of several
meteorological offices.
To download the data, it is necessary to create the RadDB configuration file.

Create the RadDB configuration file
---------------------------------------

The RadDB configuration file records the directory on your local machine where to
save the radar data of interest.

To facilitate the creation of the configuration file, you can adapt and run the following script in Python.
The configuration file will be created in the user's home directory under the name ``.config_raddb.yaml``.

.. code-block:: python

    import raddb

    base_dir = "<path/to/a/local/directory/>"  # where to download all RADAR data
    raddb.define_configs(
        base_dir=base_dir,
    )

Now please close and restart the python session to make sure that the configuration file is correctly loaded.
You can check that the configuration file has been correctly created with:

.. code-block:: python

    import raddb

    configs = raddb.read_configs()
    print(configs)


Search the radar of interest
----------------------------------------

To list the available radars, you can use the following command:

.. code-block:: python

    import raddb

    raddb.available_radars()


You can also search which radars of a specific network are available during
a given time period:


.. code-block:: python

    raddb.available_radars(network="NEXRAD", start_time="1992-01-01", end__time="1993-01-01")


The available radar networks can be listed using:

.. code-block:: python

    raddb.available_networks()


Download the data
--------------------

To download the data, you can adapt the following code snippet:

.. code-block:: python

    import raddb

    radar = "KABR"
    network = "NEXRAD"

    start_time = "2021-02-01 12:00:00"
    end_time = "2021-02-01 13:00:00"

    raddb.download(
        network=network,
        radar=radar,
        start_time=start_time,
        end_time=end_time,
    )

Search the data
--------------------

RadDB enables to search files files which have been downloaded locally,
or files which are located into a cloud bucket. To search for file locally,
specify ``procol="local"``, while to retrieve the file path of cloud bucket files,
specify ``procol="s3"``.

.. code-block:: python

    # Search for files on cloud bucket
    filepaths = raddb.find_files(
        network=network,
        radar=radar,
        start_time=start_time,
        end_time=end_time,
        fs_args=fs_args,
        protocol="s3",
        verbose=verbose,
    )
    print(filepaths)

    # Search for files locally
    filepaths = raddb.find_files(
        network=network,
        radar=radar,
        start_time=start_time,
        end_time=end_time,
        fs_args=fs_args,
        protocol="local",
        verbose=verbose,
    )
    print(filepaths)


RadDB provide an utility to group filepaths by temporal interval,
radar volume identifiers, etc.

.. code-block:: python

    dict_filepaths = raddb.group_filepaths(filepaths, network=network, groups="volume_identifier")
    dict_filepaths = raddb.group_filepaths(filepaths, network=network, groups=["day", "hour"])


Open the data
----------------

RadDB enables to open radars files into various objects by simply providing a
local or cloud file path.

- ``raddb.open_datatree(filepath, network)`` opens a file into a ``xarray.DataTree`` object using the appropriate ``xradar`` reader. Typically, an ``xarray.DataTree`` object contains multiple radar sweeps.

- ``raddb.open_dataset(filepath, network, sweep="sweep_0")`` opens a file and extract a single radar sweep into a ``xarray.Dataset`` object. The name of the radar sweep must be known beforehand !

- ``raddb.open_pyart(filepath, network)`` opens a file into a ``pyart.Radar``  object.


Further documentation
--------------------------

For radar data processing, please have a look at
`xradar <https://docs.openradarscience.org/projects/xradar/en/stable/>`_,
`pyart <https://arm-doe.github.io/pyart/>`_ and
`wradlib <https://docs.wradlib.org/en/latest/>`_ software.


If you are not familiar with `xarray <http://xarray.pydata.org/en/stable/>`_,
`numpy <https://numpy.org/doc/stable/index.html>`_,
`pandas <https://pandas.pydata.org/>`_, and
`dask <https://docs.dask.org/en/stable/array.html>`_,
it is highly suggested to first have a look also at the documentation of these software.
