Provider contract reference
===========================

Provider implementations satisfy these protocols structurally. Inheriting a
shared implementation class is not required. The registry checks required
members and advertised optional capabilities when it creates an adapter;
behavior and method signatures also require appropriate static checks and tests.

.. automodule:: radar.adapters.contracts
   :members:
   :member-order: bysource

See :doc:`adapters` for registration, provider configuration,
and the limitations of catalogue workflow support.

Plain inference
---------------

.. automodule:: radar.common.plain_llm
   :members:
   :member-order: bysource

Independent quality
-------------------

.. automodule:: radar.common.quality
   :members:
   :member-order: bysource

.. automodule:: radar.common.quality_config
   :members:
