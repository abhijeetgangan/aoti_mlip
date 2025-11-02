"""ASE Calculator wrapper around an AOTInductor-compiled MatterSim model."""

import torch
from ase.atoms import Atoms
from ase.calculators.calculator import Calculator
from ase.constraints import full_3x3_to_voigt_6_stress

from aoti_mlip.models.mattersim_modules.dataloader.build import batch_to_dict, build_dataloader
from aoti_mlip.utils.batch_info import batch_to_tuples


class MatterSimCalculator(Calculator):
    """ASE Calculator backed by an AOT-compiled MatterSim model.

    Implements ``energy``, ``free_energy``, ``forces``, and ``stress``.
    """

    implemented_properties = ["energy", "free_energy", "forces", "stress"]

    def __init__(
        self,
        model_path: str,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        **kwargs,
    ):
        """Initialize the calculator.

        Args:
            model_path: Filesystem path to the packaged ``.pt2`` model.
            device: Device string to run inference on (e.g., ``"cpu"``, ``"cuda"``).
            **kwargs: Additional ``ase.calculators.calculator.Calculator`` kwargs.
        """
        super().__init__(**kwargs)
        self.model_path = model_path
        self.device = torch.device(device)
        self.model = torch._inductor.aoti_load_package(model_path)
        self.metadata = self.model.get_metadata()

    def calculate(
        self,
        atoms: Atoms,
        properties: list | None = None,
        system_changes: list | None = None,
    ):
        """Run a calculation and populate ``self.results``.

        Args:
            atoms: The input ``ase.Atoms`` structure.
            properties: Properties to compute; defaults to ``["energy"]``.
            system_changes: List of changes that trigger recalculation; defaults
                to standard ASE triggers.

        Returns:
            None. Results are stored in ``self.results`` with keys:
            ``energy``, ``free_energy``, ``forces``, and ``stress``.
        """

        all_changes = [
            "positions",
            "numbers",
            "cell",
            "pbc",
            "initial_charges",
            "initial_magmoms",
        ]

        properties = properties or ["energy"]
        system_changes = system_changes or all_changes
        super().calculate(atoms=atoms, properties=properties, system_changes=system_changes)

        cutoff = float(self.metadata["cutoff"])
        threebody_cutoff = float(self.metadata["threebody_cutoff"])

        dataloader = build_dataloader(
            positions_list=[atoms.get_positions()],
            cell_list=[atoms.get_cell()],
            pbc_list=[atoms.get_pbc()],
            atomic_numbers_list=[atoms.get_atomic_numbers()],
            cutoff=cutoff,
            threebody_cutoff=threebody_cutoff,
            batch_size=1,
        )
        graph_batch = next(iter(dataloader))
        input_dict = {
            k: v.to(self.device) if isinstance(v, torch.Tensor) else v
            for k, v in batch_to_dict(graph_batch).items()
        }
        input_tuple = batch_to_tuples(input_dict)
        results = self.model(*input_tuple)
        self.results.update(
            energy=results["energy"].detach().cpu().numpy()[0],
            free_energy=results["energy"].detach().cpu().numpy()[0],
            forces=results["forces"].detach().cpu().numpy(),
            stress=full_3x3_to_voigt_6_stress(results["stress"].detach().cpu().numpy()[0]),
        )
