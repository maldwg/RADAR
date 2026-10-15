Development and documentation
=============================

Package and command
-------------------

The distribution, Python package, command, and log namespace are ``radar``.
The framework name is **RADAR**, expanded as **Reproducible Acquisition,
Discovery, Assessment, and Reporting**. The module entry point is
``python -m radar``. Production modules live under ``src/radar/``; provider
adapters live under ``src/radar/adapters/``.

Repository
----------

The canonical repository is `maldwg/RADAR <https://github.com/maldwg/RADAR>`_.
For an existing checkout, update its remote after the repository rename:

.. code-block:: bash

   git remote set-url origin git@github.com:maldwg/RADAR.git

For a checkout using HTTPS, use ``https://github.com/maldwg/RADAR.git`` instead.
The local checkout directory can keep its existing name.

Run the tests
-------------

The regression suite uses local fixtures and injected API/model clients.

.. code-block:: bash

   MPLCONFIGDIR=/tmp/radar-mpl PYTHONPATH=src \
     python -m unittest discover -s tests -v

For focused adapter and shared-paper checks:

.. code-block:: bash

   PYTHONPATH=src python -m unittest tests.test_adapters tests.test_shared_paper_studies

The project checks docstrings on production and test functions. Test changes
against the existing benchmark profiles rather than silently changing expected
scientific labels to make unrelated checks pass.

Documentation layout
--------------------

``README.md`` is the repository landing page. Detailed documentation uses native
reStructuredText in ``docs/``. ``index.rst`` owns navigation, ``conf.py`` selects
the Read the Docs theme, and ``requirements.txt`` pins the documentation build
dependencies. Guides use explicit Sphinx cross-references and downloadable
configuration/source files. Architecture figures are generated from
``radar.architecture`` and included in the architecture guide.

.. code-block:: bash

   python -m pip install -r docs/requirements.txt
   python -m sphinx -n -W --keep-going -b html docs docs/_build/html

Alternatively, install the project with its ``docs`` extra. The documentation
can be built without provider credentials, network retrieval, runtime model
dependencies, or Ollama. The API reference imports only the provider contracts.

Brand assets
------------

The compact RADAR symbol is available as a
:download:`scalable SVG <../assets/branding/radar-mark.svg>`, a
:download:`512-pixel PNG <../assets/branding/radar-mark.png>`, and a
:download:`multi-resolution favicon <../assets/branding/radar-favicon.ico>`.
The README displays the symbol at 80 pixels. The documentation uses the same
mark as its logo and favicon. Use the SVG for landing pages and the ICO for
browser icons; the ICO contains 16, 32, 48, 64, 128, and 256-pixel variants.

Read the Docs
-------------

The repository's ``.readthedocs.yaml`` selects Python 3.11 on Ubuntu 24.04,
installs the pinned documentation dependencies, and builds with warnings treated
as errors. Read the Docs checks out the repository and builds the site itself.
Its GitHub integration receives repository events, synchronizes branches and
tags, and builds active versions. No separate HTML upload or synchronization
workflow is required for this setup.

One-time hosted setup
~~~~~~~~~~~~~~~~~~~~~

1. Connect your GitHub account to Read the Docs and grant its GitHub App access
   to this repository.
2. Create a Read the Docs project by selecting ``maldwg/RADAR``. Keep the
   configuration file at ``.readthedocs.yaml`` and select the intended default
   branch for the ``latest`` documentation version.
3. Trigger the first build and confirm it succeeds. Subsequent pushes to the
   configured branch trigger hosted builds through the integration.
4. Under **Settings → Pull request builds**, enable **Build pull requests for
   this project** if it is not already enabled. Read the Docs then publishes
   hosted previews for new or updated pull requests.

When adding releases, activate the desired tag versions in Read the Docs or
configure its automation rules. Not every synchronized branch or tag is
automatically an active documentation version.

For an existing Read the Docs project, verify that its repository URL points to
``https://github.com/maldwg/RADAR.git`` and its GitHub integration still has
access. The Read the Docs project slug and hosted documentation address are
separate settings; a GitHub repository rename does not select a new docs address.

If hosted builds stop updating, check the project's GitHub integration and its
build/event logs. Older webhook-based integrations can be re-synced from the
project's **Integrations** settings. With the GitHub App, repository events are
subscribed automatically.

GitHub Actions checks
~~~~~~~~~~~~~~~~~~~~~

``.github/workflows/documentation.yml`` runs on pushes, pull requests, and manual
dispatch through GitHub's **Actions → Documentation → Run workflow** control.
It uses the same Python version and pinned dependencies as Read the Docs, builds
with strict Sphinx reference checks and warnings treated as errors, and uploads
the successful HTML build as ``radar-documentation-html`` for 14 days.

Download and extract that artifact, then serve it locally to inspect the site:

.. code-block:: bash

   python -m http.server --directory path/to/extracted-artifact 8000

This check needs no Read the Docs token, provider credentials, or model service.
Hosted publication is handled independently by the Read the Docs integration;
the GitHub artifact is a downloadable preview. The workflow becomes active once
it is committed and pushed. It does not create or connect a Read the Docs project.

The configuration follows the
`Read the Docs v2 reference <https://docs.readthedocs.com/platform/stable/config-file/v2.html>`_.
The local validation command uses the
`Sphinx build options <https://www.sphinx-doc.org/en/master/man/sphinx-build.html>`_.
Automatic publication follows the
`Read the Docs Git integration <https://docs.readthedocs.com/platform/stable/reference/git-integration.html>`_,
and hosted previews follow its
`pull request build guide <https://docs.readthedocs.com/platform/stable/guides/pull-requests.html>`_.
