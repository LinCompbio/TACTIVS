"""Reference-pool construction for labelled benchmarks and uploaded molecules."""

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
from rdkit.Chem.rdDistGeom import EmbedMultipleConfs, ETKDGv3

from .cache import TargetData, external_whiten
from .episodes import stable_seed


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
    ) -> ReferencePool:
        if not parent_ids:
            raise ValueError("reference pool is empty")
        return cls(
            parent_ids=np.asarray(list(map(str, parent_ids))),
            parent_counts=np.zeros(len(parent_ids), dtype=np.int64),
            raw_embeddings=None,
            source=source,
        )

    @classmethod
    def sample_labeled(
        cls, target: TargetData, k: int, seed: int = 0
    ) -> ReferencePool:
        if target.parent_labels is None:
            raise ValueError("labelled sampling requires parent labels")
        active_ids = sorted(
            str(parent)
            for parent, label in zip(target.parent_ids, target.parent_labels)
            if int(label) == 1
        )
        if k < 1:
            raise ValueError("K must be positive")
        if len(active_ids) < k:
            raise ValueError(
                f"target {target.target_id} has {len(active_ids)} actives, fewer than K={k}"
            )
        rng = random.Random(stable_seed(seed, f"fixed_random:{target.target_id}"))
        rng.shuffle(active_ids)
        return cls.from_cached_ids(active_ids[:k], source="labelled_sample")

    def save(self, path: Path) -> None:
        if self.raw_embeddings is None:
            raise ValueError("only encoded external reference pools can be saved")
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path,
            parent_ids=self.parent_ids,
            parent_counts=self.parent_counts,
            raw_embeddings=self.raw_embeddings.detach().cpu().numpy(),
            source=np.asarray(self.source),
            canonical_smiles=(
                np.asarray([], dtype=str)
                if self.canonical_smiles is None
                else self.canonical_smiles
            ),
        )

    @classmethod
    def load(cls, path: Path, device: torch.device) -> ReferencePool:
        with np.load(path, allow_pickle=False) as data:
            parent_ids = np.asarray(data["parent_ids"]).astype(str)
            counts = np.asarray(data["parent_counts"], dtype=np.int64)
            raw = torch.as_tensor(
                np.asarray(data["raw_embeddings"], dtype=np.float32), device=device
            )
            smiles = np.asarray(data["canonical_smiles"]).astype(str)
            source = str(np.asarray(data["source"]).item())
        pool = cls(
            parent_ids=parent_ids,
            parent_counts=counts,
            raw_embeddings=raw,
            source=source,
            canonical_smiles=smiles if len(smiles) else None,
        )
        pool.validate()
        return pool

    def validate(self) -> None:
        if not len(self.parent_ids):
            raise ValueError("reference pool is empty")
        if len(set(map(str, self.parent_ids))) != len(self.parent_ids):
            raise ValueError("reference pool contains duplicate parent IDs")
        if self.raw_embeddings is None:
            return
        if len(self.parent_counts) != len(self.parent_ids):
            raise ValueError("reference pool counts do not match parent IDs")
        if np.any(self.parent_counts < 1):
            raise ValueError("every external reference requires at least one conformer")
        if int(self.parent_counts.sum()) != len(self.raw_embeddings):
            raise ValueError("reference conformer counts do not match embeddings")

    def materialize(
        self, target: TargetData, projection: torch.Tensor | None
    ) -> tuple[TargetData, list[str]]:
        self.validate()
        reference_ids = list(map(str, self.parent_ids))
        if self.raw_embeddings is None:
            return replace(target, parent_labels=None), reference_ids
        if projection is None:
            raise ValueError("external references require a whitening projection")
        if target.raw_embeddings is None:
            raise ValueError("external references require raw candidate embeddings")
        collisions = sorted(set(reference_ids) & set(map(str, target.parent_ids)))
        if collisions:
            raise ValueError(f"external reference IDs collide with library IDs: {collisions}")
        raw_references = self.raw_embeddings.to(
            device=target.raw_embeddings.device, dtype=target.raw_embeddings.dtype
        )
        if raw_references.shape[1] != target.raw_embeddings.shape[1]:
            raise ValueError(
                "reference and candidate embedding dimensions do not match"
            )
        raw = torch.cat((target.raw_embeddings, raw_references), dim=0)
        counts = np.concatenate((target.parent_counts, self.parent_counts))
        starts = np.concatenate(([0], np.cumsum(counts[:-1]))).astype(np.int64)
        embeddings = external_whiten(raw, projection, counts)
        combined = TargetData(
            target_id=target.target_id,
            embeddings=embeddings,
            parent_ids=np.concatenate((target.parent_ids.astype(str), self.parent_ids)),
            parent_starts=starts,
            parent_counts=counts,
            parent_labels=None,
            raw_embeddings=raw,
            projection=projection,
        )
        return combined, reference_ids


class ConformerEncoder(Protocol):
    def encode(self, conformers: list[Chem.Mol]) -> torch.Tensor: ...


def read_uploaded_molecules(path: Path) -> list[MoleculeRecord]:
    path = Path(path)
    suffix = path.suffix.lower()
    records: list[MoleculeRecord] = []
    if suffix == ".csv":
        with path.open(newline="") as handle:
            reader = csv.DictReader(handle)
            fields = set(reader.fieldnames or [])
            smiles_field = "canonical_smiles" if "canonical_smiles" in fields else "smiles"
            required = {"parent_molecule_id", smiles_field}
            missing = required - fields
            if missing:
                raise ValueError(f"reference CSV missing columns: {sorted(missing)}")
            for row in reader:
                records.append(
                    _canonical_record(row["parent_molecule_id"], row[smiles_field])
                )
    elif suffix in {".sdf", ".mol"}:
        supplier = (
            Chem.SDMolSupplier(str(path), removeHs=False)
            if suffix == ".sdf"
            else [Chem.MolFromMolFile(str(path), removeHs=False)]
        )
        for index, molecule in enumerate(supplier):
            if molecule is None:
                raise ValueError(f"invalid molecule {index + 1} in {path}")
            parent_id = (
                molecule.GetProp("_Name").strip()
                if molecule.HasProp("_Name") and molecule.GetProp("_Name").strip()
                else f"reference_{index + 1}"
            )
            smiles = Chem.MolToSmiles(
                Chem.RemoveHs(molecule), canonical=True, isomericSmiles=True
            )
            records.append(_canonical_record(parent_id, smiles))
    else:
        raise ValueError("uploaded references must be CSV, SDF, or MOL")
    if not records:
        raise ValueError("uploaded reference file contains no molecules")
    parent_ids = [record.parent_id for record in records]
    if len(parent_ids) != len(set(parent_ids)):
        raise ValueError("uploaded reference IDs must be unique")
    return records


def _canonical_record(parent_id: str, smiles: str) -> MoleculeRecord:
    parent_id = str(parent_id).strip()
    if not parent_id:
        raise ValueError("reference parent ID is empty")
    molecule = Chem.MolFromSmiles(str(smiles).strip())
    if molecule is None:
        raise ValueError(f"invalid reference SMILES for {parent_id}")
    canonical = Chem.MolToSmiles(molecule, canonical=True, isomericSmiles=True)
    return MoleculeRecord(parent_id=parent_id, canonical_smiles=canonical)


def generate_conformers(
    records: list[MoleculeRecord], conformers_per_molecule: int = 10, seed: int = 0
) -> tuple[list[Chem.Mol], np.ndarray]:
    if conformers_per_molecule < 1:
        raise ValueError("conformers_per_molecule must be positive")
    conformers: list[Chem.Mol] = []
    counts = []
    for record in records:
        molecule = Chem.AddHs(Chem.MolFromSmiles(record.canonical_smiles))
        params = ETKDGv3()
        params.useRandomCoords = True
        params.randomSeed = int(stable_seed(seed, record.parent_id) % (2**31 - 1))
        conformer_ids = list(
            EmbedMultipleConfs(
                molecule, numConfs=conformers_per_molecule, params=params
            )
        )
        if not conformer_ids:
            raise RuntimeError(f"ETKDGv3 failed for reference {record.parent_id}")
        heavy = Chem.RemoveHs(molecule)
        count = 0
        for conformer_id in range(heavy.GetNumConformers()):
            single = Chem.Mol(heavy)
            conformer = Chem.Conformer(heavy.GetConformer(conformer_id))
            single.RemoveAllConformers()
            single.AddConformer(conformer, assignId=True)
            single.SetProp("_Name", f"{record.parent_id}#{conformer_id}")
            conformers.append(single)
            count += 1
        counts.append(count)
    return conformers, np.asarray(counts, dtype=np.int64)


def build_uploaded_pool(
    records: list[MoleculeRecord],
    encoder: ConformerEncoder,
    conformers_per_molecule: int = 10,
    seed: int = 0,
) -> ReferencePool:
    conformers, counts = generate_conformers(records, conformers_per_molecule, seed)
    embeddings = encoder.encode(conformers)
    pool = ReferencePool(
        parent_ids=np.asarray([record.parent_id for record in records]),
        parent_counts=counts,
        raw_embeddings=embeddings,
        source="uploaded_molecules",
        canonical_smiles=np.asarray([record.canonical_smiles for record in records]),
    )
    pool.validate()
    return pool


class EPTConformerEncoder:
    """Adapter for the frozen ligand-only EPT graph-representation path."""

    def __init__(
        self,
        ept_root: Path,
        ranking_root: Path,
        checkpoint: Path,
        device: torch.device,
        batch_size: int = 128,
    ):
        self.device = device
        self.batch_size = batch_size
        root = Path(ept_root).resolve()
        ranking_root = Path(ranking_root).resolve()
        if not root.exists():
            raise FileNotFoundError(f"EPT source root not found at {root}")
        if not ranking_root.exists():
            raise FileNotFoundError(f"EPT ranking source tree not found at {ranking_root}")
        sys.path.insert(0, str(root))
        sys.path.insert(0, str(ranking_root))
        # Register the upstream modules required to deserialize the legacy checkpoint.
        import models
        import models.wrappers  # noqa: F401 -- required by checkpoint unpickling
        from data.converter.blocks_to_data import blocks_to_data
        from data.converter.rdkit_to_blocks import rdkit_to_blocks
        from data.mmap_dataset import MMAPDataset

        self.blocks_to_data = blocks_to_data
        self.rdkit_to_blocks = rdkit_to_blocks
        self.collate = MMAPDataset.collate_fn
        self.model = torch.load(checkpoint, map_location="cpu", weights_only=False)
        _patch_legacy_checkpoint_flags(self.model)
        self.model.eval().to(device)

    @torch.no_grad()
    def encode(self, conformers: list[Chem.Mol]) -> torch.Tensor:
        outputs = []
        for begin in range(0, len(conformers), self.batch_size):
            examples = []
            for molecule in conformers[begin : begin + self.batch_size]:
                blocks = self.rdkit_to_blocks(molecule, using_hydrogen=False)
                if not blocks:
                    raise ValueError(f"EPT conversion failed for {molecule.GetProp('_Name')}")
                example = self.blocks_to_data(blocks)
                # The upstream collator requires this inert field; inference ignores it.
                example["label"] = 0
                example["pair_id"] = molecule.GetProp("_Name")
                examples.append(example)
            batch = self.collate(examples)
            batch = {
                key: value.to(self.device) if torch.is_tensor(value) else value
                for key, value in batch.items()
            }
            outputs.append(_encode_graph_repr(self.model, batch))
        return torch.cat(outputs, dim=0)


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
    positions, _ = model.update_global_block(
        positions, block_type, graph.unit2block
    )
    edges = graph.edges
    keep = torch.logical_and(
        block_type[edges[0]] != model.global_block_id,
        block_type[edges[1]] != model.global_block_id,
    )
    edges, edge_attr = (edges.T[keep]).T, graph.edge_attr[keep]
    _, _, graph_repr, _ = model.encoder(
        graph.unit_features,
        positions,
        graph.unit2block,
        graph.batch_ids,
        edges,
        edge_attr,
    )
    return graph_repr.detach().float().cpu()


def _patch_legacy_checkpoint_flags(model: torch.nn.Module) -> None:
    """Fill attributes absent from the frozen epoch-49 checkpoint schema."""
    try:
        from models.TransAllAtom import xtrans_act

        has_attention = bool(getattr(xtrans_act, "xformers_enable", False)) and hasattr(
            xtrans_act, "attn_func"
        )
    except (ImportError, AttributeError):
        has_attention = False
    for module in model.modules():
        name = module.__class__.__name__
        if name == "SelfAttnLayer" and not has_attention:
            module.efficient = False
        if name != "Transformer":
            continue
        for attribute in (
            "use_ieconv",
            "ieconv_share_edge_feat",
            "zero_conv",
            "activation_checkpointing",
        ):
            if not hasattr(module, attribute):
                setattr(module, attribute, False)
        if not has_attention and hasattr(module, "efficient"):
            module.efficient = False
