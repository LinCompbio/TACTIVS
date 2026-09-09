"""Known-active reference pools for benchmark and user inference."""

from __future__ import annotations

import csv
import random
import sys
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Protocol

import numpy as np
import torch
from rdkit import Chem
from rdkit.Chem.rdDistGeom import ETKDGv3, EmbedMultipleConfs
from tqdm.auto import tqdm

from .cache import TargetData, external_whiten


@dataclass(frozen=True)
class MoleculeRecord:
    parent_id: str
    canonical_smiles: str


@dataclass
class ReferencePool:
    parent_ids: np.ndarray
    parent_counts: np.ndarray
    raw_embeddings: torch.Tensor | None
    source: str
    canonical_smiles: np.ndarray | None = None

    @classmethod
    def from_cached_ids(
        cls, parent_ids: list[str], source: str = "cached_references"
    ) -> "ReferencePool":
        if not parent_ids:
            raise ValueError("reference pool is empty")
        return cls(
            parent_ids=np.asarray(parent_ids, dtype=str),
            parent_counts=np.zeros(len(parent_ids), dtype=np.int64),
            raw_embeddings=None,
            source=source,
        )

    @classmethod
    def sample_labeled(
        cls, target: TargetData, k: int, seed: int = 0
    ) -> "ReferencePool":
        if target.parent_labels is None:
            raise ValueError("benchmark reference sampling requires labels")
        active_ids = sorted(
            str(parent_id)
            for parent_id, label in zip(target.parent_ids, target.parent_labels)
            if label == 1
        )
        if k < 1:
            raise ValueError("K must be positive")
        if len(active_ids) < k:
            raise ValueError(
                f"target {target.target_id} has {len(active_ids)} actives, fewer than K={k}"
            )
        random.Random(seed).shuffle(active_ids)
        return cls.from_cached_ids(active_ids[:k], source="benchmark_actives")

    def save(self, path: Path) -> None:
        if self.raw_embeddings is None:
            raise ValueError("only encoded reference pools can be saved")
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path,
            parent_ids=self.parent_ids,
            parent_counts=self.parent_counts,
            raw_embeddings=self.raw_embeddings.cpu().numpy(),
            canonical_smiles=(
                np.asarray([], dtype=str)
                if self.canonical_smiles is None
                else self.canonical_smiles
            ),
        )

    @classmethod
    def load(cls, path: Path, device: torch.device) -> "ReferencePool":
        with np.load(path, allow_pickle=False) as data:
            pool = cls(
                parent_ids=np.asarray(data["parent_ids"]).astype(str),
                parent_counts=np.asarray(data["parent_counts"], dtype=np.int64),
                raw_embeddings=torch.as_tensor(
                    np.asarray(data["raw_embeddings"], dtype=np.float32), device=device
                ),
                source="external_actives",
                canonical_smiles=np.asarray(data["canonical_smiles"]).astype(str),
            )
        if not len(pool.canonical_smiles):
            pool.canonical_smiles = None
        pool.validate()
        return pool

    def validate(self) -> None:
        if not len(self.parent_ids):
            raise ValueError("reference pool is empty")
        if len(set(map(str, self.parent_ids))) != len(self.parent_ids):
            raise ValueError("reference pool contains duplicate molecule IDs")
        if self.raw_embeddings is None:
            return
        if len(self.parent_counts) != len(self.parent_ids):
            raise ValueError("reference conformer counts do not match molecule IDs")
        if np.any(self.parent_counts < 1):
            raise ValueError("each reference molecule needs at least one conformer")
        if self.parent_counts.sum() != len(self.raw_embeddings):
            raise ValueError("reference conformer counts do not match embeddings")

    def materialize(
        self, target: TargetData, projection: torch.Tensor | None
    ) -> tuple[TargetData, list[str]]:
        """Place cached or external actives in the library used by the scorer."""
        self.validate()
        reference_ids = list(map(str, self.parent_ids))
        if self.raw_embeddings is None:
            return replace(target, parent_labels=None), reference_ids
        if target.raw_embeddings is None or projection is None:
            raise ValueError(
                "external references require raw candidate embeddings and a whitener"
            )
        collisions = set(reference_ids) & set(map(str, target.parent_ids))
        if collisions:
            raise ValueError(
                f"reference IDs already exist in the candidate library: {sorted(collisions)}"
            )

        references = self.raw_embeddings.to(
            device=target.raw_embeddings.device, dtype=target.raw_embeddings.dtype
        )
        if references.shape[1] != target.raw_embeddings.shape[1]:
            raise ValueError("reference and candidate embedding dimensions differ")
        raw = torch.cat((target.raw_embeddings, references))
        counts = np.concatenate((target.parent_counts, self.parent_counts))
        starts = np.concatenate(([0], np.cumsum(counts[:-1]))).astype(np.int64)
        return (
            TargetData(
                target_id=target.target_id,
                embeddings=external_whiten(raw, projection, counts),
                parent_ids=np.concatenate((target.parent_ids.astype(str), self.parent_ids)),
                parent_starts=starts,
                parent_counts=counts,
                raw_embeddings=raw,
                projection=projection,
            ),
            reference_ids,
        )


class ConformerEncoder(Protocol):
    def encode(self, conformers: list[Chem.Mol]) -> torch.Tensor: ...


def read_molecules(path: Path) -> list[MoleculeRecord]:
    path = Path(path)
    records: list[MoleculeRecord] = []
    if path.suffix.lower() == ".csv":
        with path.open(newline="") as handle:
            reader = csv.DictReader(handle)
            fields = set(reader.fieldnames or [])
            smiles_field = "canonical_smiles" if "canonical_smiles" in fields else "smiles"
            missing = {"parent_molecule_id", smiles_field} - fields
            if missing:
                raise ValueError(f"molecule CSV missing columns: {sorted(missing)}")
            records = [
                _molecule_record(row["parent_molecule_id"], row[smiles_field])
                for row in reader
            ]
    elif path.suffix.lower() in {".sdf", ".mol"}:
        molecules = (
            Chem.SDMolSupplier(str(path), removeHs=False)
            if path.suffix.lower() == ".sdf"
            else [Chem.MolFromMolFile(str(path), removeHs=False)]
        )
        for index, molecule in enumerate(molecules, 1):
            if molecule is None:
                raise ValueError(f"invalid molecule {index} in {path}")
            parent_id = (
                molecule.GetProp("_Name").strip()
                if molecule.HasProp("_Name") and molecule.GetProp("_Name").strip()
                else f"molecule_{index}"
            )
            records.append(
                _molecule_record(
                    parent_id,
                    Chem.MolToSmiles(Chem.RemoveHs(molecule), isomericSmiles=True),
                )
            )
    else:
        raise ValueError("molecule input must be CSV, SDF, or MOL")
    if not records:
        raise ValueError("molecule input is empty")
    parent_ids = [record.parent_id for record in records]
    if len(parent_ids) != len(set(parent_ids)):
        raise ValueError("molecule IDs must be unique")
    return records


def _molecule_record(parent_id: str, smiles: str) -> MoleculeRecord:
    parent_id = str(parent_id).strip()
    if not parent_id:
        raise ValueError("molecule ID is empty")
    molecule = Chem.MolFromSmiles(str(smiles).strip())
    if molecule is None:
        raise ValueError(f"invalid SMILES for {parent_id}")
    return MoleculeRecord(
        parent_id,
        Chem.MolToSmiles(molecule, canonical=True, isomericSmiles=True),
    )


def generate_conformers(
    records: list[MoleculeRecord], conformers_per_molecule: int = 10, seed: int = 0
) -> tuple[list[Chem.Mol], np.ndarray]:
    if conformers_per_molecule < 1:
        raise ValueError("conformers_per_molecule must be positive")
    conformers = []
    counts = []
    for record in tqdm(records, desc="Generating conformers", unit="molecule", leave=False):
        molecule = Chem.AddHs(Chem.MolFromSmiles(record.canonical_smiles))
        params = ETKDGv3()
        params.useRandomCoords = True
        params.randomSeed = seed % (2**31 - 1)
        conformer_ids = EmbedMultipleConfs(
            molecule, numConfs=conformers_per_molecule, params=params
        )
        if not conformer_ids:
            raise RuntimeError(f"conformer generation failed for {record.parent_id}")
        heavy = Chem.RemoveHs(molecule)
        for conformer_id in range(heavy.GetNumConformers()):
            conformer = Chem.Mol(heavy)
            conformer.RemoveAllConformers()
            conformer.AddConformer(
                Chem.Conformer(heavy.GetConformer(conformer_id)), assignId=True
            )
            conformer.SetProp("_Name", f"{record.parent_id}#{conformer_id}")
            conformers.append(conformer)
        counts.append(heavy.GetNumConformers())
    return conformers, np.asarray(counts, dtype=np.int64)


def build_reference_pool(
    records: list[MoleculeRecord],
    encoder: ConformerEncoder,
    conformers_per_molecule: int = 10,
    seed: int = 0,
) -> ReferencePool:
    conformers, counts = generate_conformers(records, conformers_per_molecule, seed)
    pool = ReferencePool(
        parent_ids=np.asarray([record.parent_id for record in records]),
        parent_counts=counts,
        raw_embeddings=encoder.encode(conformers),
        source="external_actives",
        canonical_smiles=np.asarray([record.canonical_smiles for record in records]),
    )
    pool.validate()
    return pool


class EPTConformerEncoder:
    """EPT ligand encoder used to build candidate and reference caches."""

    def __init__(
        self,
        checkpoint: Path,
        device: torch.device,
        batch_size: int = 128,
    ) -> None:
        self.device = device
        self.batch_size = batch_size
        ept_root = Path(__file__).parent / "_vendor" / "ept"
        if str(ept_root) not in sys.path:
            sys.path.insert(0, str(ept_root))

        import models  # noqa: F401
        import models.wrappers  # noqa: F401
        from data.converter.blocks_to_data import blocks_to_data
        from data.converter.rdkit_to_blocks import rdkit_to_blocks
        from data.mmap_dataset import MMAPDataset
        from models.wrappers.denoise_pretrain import Denoise

        checkpoint = Path(checkpoint)
        if not checkpoint.is_file():
            raise FileNotFoundError(f"EPT checkpoint not found: {checkpoint}")
        saved_model = torch.load(checkpoint, map_location="cpu", weights_only=False)
        encoder_config = dict(saved_model.encoder_config)
        encoder_config["efficient"] = False
        model = Denoise(
            encoder=encoder_config,
            graph_constructor=saved_model.graph_config,
            n_noise_level=len(saved_model.sigmas),
            rot_n_noise_level=len(saved_model.rot_sigmas),
            noise_type=saved_model.noise_type,
            pred_type=saved_model.pred_type,
        )
        model.load_state_dict(saved_model.state_dict(), strict=True)

        self.blocks_to_data = blocks_to_data
        self.rdkit_to_blocks = rdkit_to_blocks
        self.collate = MMAPDataset.collate_fn
        self.model = model.eval().to(device)

    @torch.no_grad()
    def encode(self, conformers: list[Chem.Mol]) -> torch.Tensor:
        outputs = []
        for begin in tqdm(
            range(0, len(conformers), self.batch_size),
            desc="Encoding conformers",
            unit="batch",
            leave=False,
        ):
            examples = []
            for molecule in conformers[begin : begin + self.batch_size]:
                blocks = self.rdkit_to_blocks(molecule, using_hydrogen=False)
                if not blocks:
                    raise ValueError(f"EPT conversion failed for {molecule.GetProp('_Name')}")
                example = self.blocks_to_data(blocks)
                example["label"] = 0
                example["pair_id"] = molecule.GetProp("_Name")
                examples.append(example)
            batch = {
                key: value.to(self.device) if torch.is_tensor(value) else value
                for key, value in self.collate(examples).items()
            }
            outputs.append(_encode_graph_repr(self.model, batch))
        return torch.cat(outputs)


@torch.no_grad()
def _encode_graph_repr(model: torch.nn.Module, batch: dict) -> torch.Tensor:
    graph = model.graph_constructor.forward(
        unit_type=batch["A"],
        unit_pos=batch["X"],
        num_nodes=batch["lengths"],
        unit_position_ids=batch["atom_positions"],
        segment_ids=batch["segment_ids"],
        block_type=batch["B"],
        block_num_units=batch["block_lengths"],
    )
    block_type = graph.block_type
    positions = model.normalize(
        graph.unit_pos, block_type, graph.unit2block, graph.batch_ids
    )
    positions, _ = model.update_global_block(positions, block_type, graph.unit2block)
    edges = graph.edges
    keep = (block_type[edges[0]] != model.global_block_id) & (
        block_type[edges[1]] != model.global_block_id
    )
    edges, edge_attr = edges[:, keep], graph.edge_attr[keep]
    _, _, graph_repr, _ = model.encoder(
        graph.unit_features,
        positions,
        graph.unit2block,
        graph.batch_ids,
        edges,
        edge_attr,
    )
    return graph_repr.detach().float().cpu()
