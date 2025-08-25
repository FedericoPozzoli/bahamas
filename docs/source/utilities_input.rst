run_input
================================

.. automodule:: utilities.run_input
   :members:
   :exclude-members: 
   :undoc-members:
   :show-inheritance:


To generate input files for inference using the script, use the following command:

.. code-block:: bash

   bahamas_input --file /path/to/file --chunk_T 2 --n_chunks 54 --nseg 1000 --path /path/to/output

Alternatively, you can use the executable entrypoint:

.. code-block:: bash
   
   bahamas-input --help
