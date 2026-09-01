GR00T
=====

`GR00T N1.6 <https://github.com/NVIDIA/Isaac-GR00T/>`_ is a pre-trained robotic
foundation model. No fine-tuning or separate model download is required. The weights
are fetched from `HuggingFace <https://huggingface.co/nvidia/GR00T-N1.6-DROID>`_
when the policy server starts for the first time.

Start a GR00T policy server
---------------------------

The closed-loop policy connects to a GR00T policy server in a separate process. The
server runs from the
`Isaac-GR00T <https://github.com/NVIDIA/Isaac-GR00T/tree/e29d8fc50b0e4745120ae3fb72447986fe638aa6>`_
submodule pinned at commit ``e29d8fc``. Populate it if needed:

.. code-block:: bash

   git submodule update --init submodules/Isaac-GR00T

If you run Arena from its native ``uv`` environment, install the GR00T client
package:

.. tab-set::

   .. tab-item:: Source
      :selected:

      .. code-block:: bash

         uv sync --group gr00t-client

   .. tab-item:: Wheel

      .. code-block:: bash

         uv sync --no-default-groups --group isaaclab-from-wheel --group gr00t-client

Then start the server from the repository root in a separate shell:

.. todo::

   The ``submodules/Isaac-GR00T`` submodule will be removed after the policy
   config refactor. After that, users will set up a separate GR00T repository
   checkout and launch the server from there.

.. code-block:: bash

   cd submodules/Isaac-GR00T
   uv run python gr00t/eval/run_gr00t_server.py \
     --model-path nvidia/GR00T-N1.6-DROID \
     --embodiment-tag OXE_DROID \
     --device cuda --host 127.0.0.1 --port 5555

GR00T N1.6-DROID provides its own modality configuration, so the command does not need
``--modality-config-path``. The first launch downloads the model weights; later launches
reuse the local cache. Leave this server running.


Run GR00T with the Experiment Runner
------------------------------------

Arena includes a one-Run YAML configuration for the first rollout. It selects the
DROID environment, connects the GR00T policy to the server, and stops after three episodes.

.. dropdown:: Configuration file (``droid_pnp_gr00t_experiment.yaml``)
   :animate: fade-in

   .. literalinclude:: ../../../../isaaclab_arena_environments/experiment_configs/droid_pnp_gr00t_experiment.yaml
      :language: yaml

GR00T N1.6-DROID uses absolute joint positions. The YAML therefore selects
``droid_abs_joint_pos`` and enables the cameras required by the policy. The natural-language
instruction belongs to the environment builder, while the server connection belongs to the
policy.

Open another shell, enter the Arena container, and start the rollout with the Experiment Runner:

.. code-block:: bash

   python isaaclab_arena/evaluation/experiment_runner.py \
     --viz kit \
     --experiment_config isaaclab_arena_environments/experiment_configs/droid_pnp_gr00t_experiment.yaml

The Kit window shows the DROID arm acting on GR00T commands. After every episode, Arena
reports whether the pick-and-place task succeeded.

If the server runs on another host or port, override the declared policy value. For example:

.. code-block:: bash

   python isaaclab_arena/evaluation/experiment_runner.py \
     --viz kit \
     --experiment_config isaaclab_arena_environments/experiment_configs/droid_pnp_gr00t_experiment.yaml \
     runs.droid_pnp_gr00t.policy.remote_port=5556

Evaluate several object variations
----------------------------------

The multi-Run YAML evaluates nine combinations of pick-up object, destination, HDR background,
and language instruction. Its ``shared`` mapping keeps the GR00T policy and rollout settings in
one place. Every Run lists only what changes.

.. dropdown:: Configuration file (``droid_pnp_srl_gr00t_experiment.yaml``)
   :animate: fade-in

   .. literalinclude:: ../../../../isaaclab_arena_environments/experiment_configs/droid_pnp_srl_gr00t_experiment.yaml
      :language: yaml

Start all nine Runs with one command:

.. code-block:: bash

   python isaaclab_arena/evaluation/experiment_runner.py \
     --viz kit \
     --experiment_config isaaclab_arena_environments/experiment_configs/droid_pnp_srl_gr00t_experiment.yaml

The runner executes the nine Runs in YAML order. It keeps one SimulationApp open, but builds a
fresh environment for every Run.

.. figure:: ../../../images/gr00t_droid_3x3_grid.gif
   :width: 100%
   :alt: 3x3 grid of GR00T N1.6 DROID Runs across different objects, backgrounds, and destinations
   :align: center

   Nine closed-loop Runs of GR00T N1.6 on the DROID embodiment. Each cell changes the
   pick-up object, HDR background, and destination.

When all Runs finish, Arena prints a summary table followed by a metrics report:

.. dropdown:: Example Run summary and metrics
   :animate: fade-in

   Metric values can vary between evaluations. This is example output:

   .. code-block:: text

      +---------------------------------------+-----------+-------------------------+----------+-----------+--------------+--------------+
      |               Run Name                |  Status   |       Policy Type       | Num Envs | Num Steps | Num Episodes | Num Rebuilds |
      +---------------------------------------+-----------+-------------------------+----------+-----------+--------------+--------------+
      |   droid_pnp_srl_gr00t_billiard_hall   | completed | gr00t_remote_closedloop |    1     |   None    |      3       |      1       |
      |    droid_pnp_srl_gr00t_blue_block     | completed | gr00t_remote_closedloop |    1     |   None    |      3       |      1       |
      | droid_pnp_srl_gr00t_alphabet_soup_can | completed | gr00t_remote_closedloop |    1     |   None    |      3       |      1       |
      |      droid_pnp_srl_gr00t_orange       | completed | gr00t_remote_closedloop |    1     |   None    |      3       |      1       |
      |       droid_pnp_srl_gr00t_lemon       | completed | gr00t_remote_closedloop |    1     |   None    |      3       |      1       |
      | droid_pnp_srl_gr00t_tomato_sauce_can  | completed | gr00t_remote_closedloop |    1     |   None    |      3       |      1       |
      |  droid_pnp_srl_gr00t_mustard_bottle   | completed | gr00t_remote_closedloop |    1     |   None    |      3       |      1       |
      |     droid_pnp_srl_gr00t_sugar_box     | completed | gr00t_remote_closedloop |    1     |   None    |      3       |      1       |
      |        droid_pnp_srl_gr00t_mug        | completed | gr00t_remote_closedloop |    1     |   None    |      3       |      1       |
      +---------------------------------------+-----------+-------------------------+----------+-----------+--------------+--------------+

      ======================================================================
      METRICS SUMMARY
      ======================================================================

      droid_pnp_srl_gr00t_alphabet_soup_can:
        num_episodes                            3
        object_moved_rate                  0.0000
        success_rate                       0.0000

      droid_pnp_srl_gr00t_billiard_hall:
        num_episodes                            3
        object_moved_rate                  0.3333
        success_rate                       0.0000

      droid_pnp_srl_gr00t_blue_block:
        num_episodes                            3
        object_moved_rate                  0.0000
        success_rate                       0.0000

      droid_pnp_srl_gr00t_lemon:
        num_episodes                            3
        object_moved_rate                  1.0000
        success_rate                       0.6667

      ...
      ======================================================================

These results show that zero-shot deployment of robotic foundation models remains
challenging. Recent
`[robolab] <https://gitlab-master.nvidia.com/xuningy/robolab/-/blob/main/docs/analysis.md>`_
results compare GR00T with other vision-language-action models.


View rollouts as an HTML report
-------------------------------

The runner builds an HTML report for the complete evaluation. Add ``--record_camera_video`` to
record one video per camera and episode, then use
``--serve_evaluation_report`` to open the report through a local HTTP server:

.. code-block:: bash

   python isaaclab_arena/evaluation/experiment_runner.py \
     --viz kit \
     --experiment_config isaaclab_arena_environments/experiment_configs/droid_pnp_srl_gr00t_experiment.yaml \
     --output_base_dir ./output \
     --record_camera_video \
     --serve_evaluation_report

You can rebuild and serve a report later by pointing the standalone tool at the output
directory. It selects the most recent evaluation:

.. code-block:: bash

   python isaaclab_arena/visualization/report.py --video_dir ./output


Run GR00T N1.7 instead of N1.6
------------------------------

The ``submodules/Isaac-GR00T`` checkout decides which model family is available: GR00T N1.7 renamed
the DROID embodiment tag and dropped the ``GR1`` and ``OXE_DROID`` tags, so one checkout cannot
serve both. The submodule is pinned to ``n1.7-release``; the N1.6 configuration files are kept for
users who pin the submodule back to the N1.6 line.

Grant access to the backbone
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

GR00T N1.7 loads ``nvidia/Cosmos-Reason2-2B`` as its backbone, and that repository is gated. Request
access once on its `Hugging Face page <https://huggingface.co/nvidia/Cosmos-Reason2-2B>`__, then
authenticate on the machine that runs the server:

.. code-block:: bash

   hf auth login   # or: export HF_TOKEN=<your token>

Without this the server downloads the GR00T weights and then fails with
``Cannot access gated repo ... nvidia/Cosmos-Reason2-2B``.

Start the N1.7 server
~~~~~~~~~~~~~~~~~~~~~

Install the server dependencies for this platform and start the server from the submodule. Use
``uv pip install`` and the virtual environment's interpreter rather than ``uv run``: the release
tag declares an ``aarch64`` wheel that is stored in Git LFS, and any command that re-resolves the
whole project fails on that placeholder unless ``git lfs pull`` has been run.

.. code-block:: bash

   cd submodules/Isaac-GR00T
   uv pip install -e .
   .venv/bin/python gr00t/eval/run_gr00t_server.py \
     --model-path nvidia/GR00T-N1.7-3B \
     --embodiment-tag OXE_DROID_RELATIVE_EEF_RELATIVE_JOINT \
     --device cuda --host 127.0.0.1 --port 5555

DROID is a pretrain tag for N1.7, so ``nvidia/GR00T-N1.7-3B`` runs zero-shot. Swap in
``nvidia/GR00T-N1.7-DROID`` for the finetuned checkpoint.

Run the rollout
~~~~~~~~~~~~~~~

Every DROID environment ships an N1.7 Experiment alongside its N1.6 one. From the Arena container:

.. code-block:: bash

   python isaaclab_arena/evaluation/experiment_runner.py \
     --headless --enable_cameras --record_viewport_video \
     --experiment_config isaaclab_arena_environments/experiment_configs/droid_peg_insertion_gr00t_n17_experiment.yaml \
     --output_base_dir ./output

Arena obtains the resolved modality config from the running server rather than assuming that the
embodiment tag uniquely determines it. For example, the N1.7 base checkpoint asks for two video
frames per camera (``delta_indices`` ``[-15, 0]``), while a post-trained checkpoint can ask for only
the current frame. :class:`~isaaclab_arena_gr00t.policy.video_history.VideoHistoryBuffer` records
every control step and serves the exact history requested. N1.7 also conditions on an ``eef_9d``
end-effector state built in the DROID/Polymetis base frame by
``isaaclab_arena_gr00t/utils/droid_eef.py``. The server converts its relative end-effector and joint
actions back to absolute values before replying, so ``droid_abs_joint_pos`` consumes them directly.


Next steps
----------

To go beyond the pre-trained GR00T N1.6 foundation model, such as fine-tuning on your own
teleoperation data, see :doc:`/pages/example_workflows/imitation_learning/index` for the
complete imitation-learning workflows.
