Building the clifford documentation
====================================

One-time setup
--------------

Install Sphinx and the `furo` theme into your environment::

    pip install sphinx furo

(If you use a virtualenv for the project, activate it first.)


Building the HTML docs
----------------------

From this directory (``/home/dajonesy/engine/docs``)::

    make html

The output goes into ``_build/html/``.  Open ``_build/html/index.html``
in a browser to read it.

If anything goes sideways with ``make``, the equivalent direct command is::

    sphinx-build -M html . _build


Rebuilding cleanly
------------------

::

    make clean
    make html


Adding a new example page
-------------------------

1. Drop an ``.rst`` file in ``examples/`` (e.g. ``examples/my_example.rst``).
2. Add its filename (without the ``.rst``) to the ``examples`` toctree
   in ``index.rst``.
3. Run ``make html`` again.


Notes on what gets built
------------------------

- **API pages** under ``api/generated/`` are auto-generated from your
  module docstrings.  You do *not* hand-edit these — they get regenerated
  every time ``autosummary`` runs.  Edit the docstrings in your ``.py``
  files instead.

- **Example pages** under ``examples/`` are hand-written.  Use them for
  worked code and tutorials.

- **index.rst** is your landing page.  Edit it to change what shows
  up on the front page.


Layout
------

::

    docs/
    ├── conf.py                    Sphinx configuration
    ├── index.rst                  landing page
    ├── Makefile                   build entry point (``make html``)
    ├── README.rst                 this file
    ├── examples/                  hand-written example pages
    │   └── inverting_7d.rst
    └── api/                       auto-generated API reference
        ├── clifford.rst           top-level API toc
        └── generated/             autosummary stubs (don't hand-edit)
