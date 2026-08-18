
Welcome to RadDB !
========================


**RadDB** archives weather radar volumes as compact **Parquet** tables — one row per
radar gate — and gives you a small fluent interface to load, filter, crop, cut
cross-sections and plot them.

It is **network-agnostic**: any `xarray.DataTree <https://docs.xarray.dev/en/stable/generated/xarray.DataTree.html>`__
following the `xradar <https://docs.openradarscience.org/projects/xradar/en/stable/>`__
coordinate layout can be archived, whether it comes from NEXRAD, ODIM, IRIS or any
other network.

Each radar is stored once as a static look-up table holding the per-gate geometry, plus
one Parquet file per volume holding only the variables that change. Gates without echo are
dropped at archive time, so a whole campaign stays small enough to query like a dataframe —
no re-reading of the source files, and no fixed grid imposed on the measurements.


**Ready to jump in?**

Consider joining the `Open Radar Science Discourse Group <https://openradar.discourse.group/>`__ to say hi or ask questions.
It's a great place to connect with others and get support.


Documentation
==============

.. toctree::
   :maxdepth: 2

   02_installation
   03_quickstart
   06_contributors_guidelines
   07_maintainers_guidelines
   08_authors


Package Reference
==================

.. toctree::
   :maxdepth: 1

   Modules <api/modules>


Indices and tables
===================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
