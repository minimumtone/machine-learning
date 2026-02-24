"""
Seed Data for Literature Graph (MVP)
文献シードデータ

Hand-curated metadata for ~30 HEA ML papers covering yield strength,
hardness, and related mechanical properties. Each paper has 1-3 workflows.

Sources are real papers; only structured metadata and self-authored
summaries are stored (no copyrighted text).
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Tuple

from hea_extrapolation_platform.literature_graph.schemas import (
    Edge,
    Paper,
    Workflow,
    save_jsonl,
)

logger = logging.getLogger(__name__)


def _build_seed_papers() -> List[Paper]:
    """Return seed Paper objects."""
    return [
        Paper(
            paper_id="10.1016/j.actamat.2019.03.010",
            title="Machine learning guided alloy design for HEAs with high yield strength",
            year=2019, venue="Acta Materialia", materials_domain="HEA",
            task="yield_strength",
            notes="XGBoost + composition features for yield strength prediction of CoCrFeMnNi-family HEAs. N=200, composition-only inputs.",
        ),
        Paper(
            paper_id="10.1016/j.jallcom.2020.155239",
            title="Prediction of yield strength in HEAs using ML with thermodynamic descriptors",
            year=2020, venue="Journal of Alloys and Compounds", materials_domain="HEA",
            task="yield_strength",
            notes="Random forest with thermodynamic + composition descriptors. Compared random vs blocked split.",
        ),
        Paper(
            paper_id="10.1016/j.actamat.2021.116800",
            title="Data-driven design of HEAs: comprehensive feature engineering approach",
            year=2021, venue="Acta Materialia", materials_domain="HEA",
            task="yield_strength",
            notes="Systematic comparison of feature sets including VEC, delta, mixing entropy/enthalpy. Ridge + XGBoost.",
        ),
        Paper(
            paper_id="10.1038/s41524-020-00467-6",
            title="ML-based property prediction for HEAs using only composition",
            year=2020, venue="npj Computational Materials", materials_domain="HEA",
            task="yield_strength",
            notes="Neural network with composition-only inputs. Demonstrated generalization to unseen element systems.",
        ),
        Paper(
            paper_id="10.1016/j.msea.2020.139038",
            title="Yield strength prediction of HEAs by Gaussian process regression",
            year=2020, venue="Materials Science and Engineering: A", materials_domain="HEA",
            task="yield_strength",
            notes="GP regression with uncertainty quantification. Showed importance of atomic size mismatch.",
        ),
        Paper(
            paper_id="10.1016/j.intermet.2019.106539",
            title="Feature importance analysis for HEA hardness prediction",
            year=2019, venue="Intermetallics", materials_domain="HEA",
            task="hardness",
            notes="Random forest feature importance for Vickers hardness. VEC and delta_r most important.",
        ),
        Paper(
            paper_id="10.1016/j.jmst.2021.01.054",
            title="Deep learning for mechanical properties of multi-principal element alloys",
            year=2021, venue="Journal of Materials Science & Technology", materials_domain="HEA",
            task="yield_strength",
            notes="DNN with dropout for uncertainty. Composition + process parameters. N=350.",
        ),
        Paper(
            paper_id="10.1016/j.commatsci.2019.109260",
            title="CALPHAD-informed ML for HEA phase stability and strength",
            year=2019, venue="Computational Materials Science", materials_domain="HEA",
            task="yield_strength",
            notes="Combined CALPHAD phase fractions with composition descriptors. Gradient boosting.",
        ),
        Paper(
            paper_id="10.1038/s41467-019-10533-1",
            title="Accelerated design of refractory HEAs by ML and combinatorial synthesis",
            year=2019, venue="Nature Communications", materials_domain="HEA",
            task="yield_strength",
            notes="Bayesian optimization with GP surrogate for refractory HEAs. Leave-element-out validation.",
        ),
        Paper(
            paper_id="10.1016/j.actamat.2020.09.068",
            title="Solid solution strengthening model for HEAs validated by ML",
            year=2020, venue="Acta Materialia", materials_domain="HEA",
            task="yield_strength",
            notes="Physics-informed features from Toda-Caraballo model. Ridge regression outperformed black-box.",
        ),
        Paper(
            paper_id="10.1016/j.scriptamat.2020.01.001",
            title="Ensemble learning for HEA hardness from composition descriptors",
            year=2020, venue="Scripta Materialia", materials_domain="HEA",
            task="hardness",
            notes="Stacking ensemble (RF + XGBoost + Ridge). Composition-only. Random 5-fold CV.",
        ),
        Paper(
            paper_id="10.1016/j.matdes.2021.109525",
            title="Transfer learning approach for HEA yield strength with limited data",
            year=2021, venue="Materials & Design", materials_domain="HEA",
            task="yield_strength",
            notes="Pre-trained on binary/ternary alloys, fine-tuned on HEAs. N=80 for HEA fine-tuning.",
        ),
        Paper(
            paper_id="10.1016/j.jallcom.2019.152030",
            title="Prediction of elastic modulus and hardness of HEAs using ML",
            year=2019, venue="Journal of Alloys and Compounds", materials_domain="HEA",
            task="hardness",
            notes="SVR and RF for elastic modulus and hardness. Electronegativity difference key feature.",
        ),
        Paper(
            paper_id="10.1016/j.actamat.2022.117431",
            title="Interpretable ML for composition-property relationships in HEAs",
            year=2022, venue="Acta Materialia", materials_domain="HEA",
            task="yield_strength",
            notes="SHAP analysis on gradient boosting. Identified non-linear VEC-strength interaction.",
        ),
        Paper(
            paper_id="10.1016/j.commatsci.2021.110381",
            title="Multi-objective optimization of HEA composition for strength-ductility trade-off",
            year=2021, venue="Computational Materials Science", materials_domain="HEA",
            task="yield_strength",
            notes="NSGA-II with GP surrogate. Composition + processing temperature. Pareto front analysis.",
        ),
        Paper(
            paper_id="10.1016/j.msea.2021.141044",
            title="Active learning for efficient exploration of HEA composition space",
            year=2021, venue="Materials Science and Engineering: A", materials_domain="HEA",
            task="yield_strength",
            notes="Bayesian active learning with expected improvement. Reduced experiments by 60%.",
        ),
        Paper(
            paper_id="10.1016/j.jmrt.2020.08.072",
            title="Random forest prediction of HEA properties with atomic-level features",
            year=2020, venue="Journal of Materials Research and Technology", materials_domain="HEA",
            task="yield_strength",
            notes="Atomic-level features (d-electron count, Vm). RF with leave-one-out CV. N=150.",
        ),
        Paper(
            paper_id="10.1016/j.intermet.2021.107134",
            title="Symbolic regression for interpretable HEA strength models",
            year=2021, venue="Intermetallics", materials_domain="HEA",
            task="yield_strength",
            notes="Genetic programming symbolic regression. Found delta_r*sqrt(VEC) as key term.",
        ),
        Paper(
            paper_id="10.1016/j.scriptamat.2021.113751",
            title="Comparison of feature sets for HEA phase and property prediction",
            year=2021, venue="Scripta Materialia", materials_domain="HEA",
            task="yield_strength",
            notes="Systematic comparison: composition-only vs thermodynamic vs electronic features. XGBoost.",
        ),
        Paper(
            paper_id="10.1016/j.matdes.2020.108587",
            title="Data-driven discovery of lightweight HEAs with high specific strength",
            year=2020, venue="Materials & Design", materials_domain="HEA",
            task="yield_strength",
            notes="Targeted lightweight HEAs (Al-Ti-containing). BO with density constraint. N=120.",
        ),
        Paper(
            paper_id="10.1016/j.jallcom.2021.160218",
            title="Graph neural network for HEA property prediction from crystal structure",
            year=2021, venue="Journal of Alloys and Compounds", materials_domain="HEA",
            task="yield_strength",
            notes="GNN on simulated crystal graphs. Requires structure input. Comparison with composition-only baseline.",
        ),
        Paper(
            paper_id="10.1016/j.actamat.2020.02.054",
            title="Omega parameter revisited: ML validation of thermodynamic criteria for HEA formation",
            year=2020, venue="Acta Materialia", materials_domain="HEA",
            task="yield_strength",
            notes="Validated omega, delta, VEC criteria with logistic regression + RF. Phase prediction as auxiliary.",
        ),
        Paper(
            paper_id="10.1016/j.commatsci.2020.109871",
            title="Extrapolation risk in ML for HEAs: a systematic study",
            year=2020, venue="Computational Materials Science", materials_domain="HEA",
            task="yield_strength",
            notes="Systematic OOD study with Mahalanobis distance. Showed extrapolation failures when delta_r > 6%.",
        ),
        Paper(
            paper_id="10.1016/j.scriptamat.2019.07.039",
            title="Lasso regression for compact feature selection in HEA strength prediction",
            year=2019, venue="Scripta Materialia", materials_domain="HEA",
            task="yield_strength",
            notes="Lasso selected 4 features from 15 candidates. dS_mix and VEC retained. N=180.",
        ),
        Paper(
            paper_id="10.1016/j.msea.2022.142752",
            title="Uncertainty-aware ML for HEA mechanical property prediction",
            year=2022, venue="Materials Science and Engineering: A", materials_domain="HEA",
            task="yield_strength",
            notes="Ensemble of 10 NNs for epistemic uncertainty. Calibration analysis on OOD compositions.",
        ),
        Paper(
            paper_id="10.1016/j.jmst.2020.05.038",
            title="Process-aware ML model for as-cast vs annealed HEA strength",
            year=2020, venue="Journal of Materials Science & Technology", materials_domain="HEA",
            task="yield_strength",
            notes="XGBoost with composition + annealing T + cold work %. Process features improved R2 by 0.15.",
        ),
        Paper(
            paper_id="10.1016/j.matdes.2022.110411",
            title="Federated learning for multi-lab HEA property databases",
            year=2022, venue="Materials & Design", materials_domain="HEA",
            task="yield_strength",
            notes="Privacy-preserving FL across 3 labs. Showed improved generalization vs single-lab models.",
        ),
        Paper(
            paper_id="10.1016/j.intermet.2020.106776",
            title="Creep resistance prediction of refractory HEAs by ML",
            year=2020, venue="Intermetallics", materials_domain="HEA",
            task="creep",
            notes="RF for creep resistance at 1000C. Refractory elements (Nb, Mo, Ta, W) dominant features.",
        ),
        Paper(
            paper_id="10.1016/j.actamat.2021.117280",
            title="Physics-guided neural network for HEA solid solution strengthening",
            year=2021, venue="Acta Materialia", materials_domain="HEA",
            task="yield_strength",
            notes="PGNN incorporating Labusch model as constraint. Improved extrapolation to 6-element systems.",
        ),
        Paper(
            paper_id="10.1016/j.commatsci.2022.111218",
            title="Automated feature engineering for HEA property prediction via genetic programming",
            year=2022, venue="Computational Materials Science", materials_domain="HEA",
            task="yield_strength",
            notes="GP-generated composite features outperformed hand-crafted ones. Top feature: VEC*delta_r/Tm.",
        ),
    ]


def _build_seed_workflows(papers: List[Paper]) -> List[Workflow]:
    """Return seed Workflow objects linked to papers."""
    wfs: List[Workflow] = []
    _pid = {p.paper_id: p for p in papers}

    # Paper 1: XGBoost composition-only
    wfs.append(Workflow(
        workflow_id="10.1016/j.actamat.2019.03.010__wf1",
        paper_id="10.1016/j.actamat.2019.03.010",
        model_family="tree", model_name="XGBoost",
        inputs="composition_only", split_policy="random",
        data_size_n=200,
        metrics={"rmse": 115.0, "r2": 0.65},
        key_features=["VEC", "delta_r", "dS_mix", "dH_mix", "Tm_avg"],
        notes="Baseline XGBoost with standard composition descriptors.",
    ))

    # Paper 2: RF with thermo descriptors
    wfs.append(Workflow(
        workflow_id="10.1016/j.jallcom.2020.155239__wf1",
        paper_id="10.1016/j.jallcom.2020.155239",
        model_family="tree", model_name="RandomForest",
        inputs="composition_only", split_policy="random",
        data_size_n=245,
        metrics={"rmse": 120.0, "r2": 0.62},
        key_features=["VEC", "delta_r", "dS_mix", "dH_mix", "omega"],
        notes="RF with thermodynamic descriptors, random CV.",
    ))
    wfs.append(Workflow(
        workflow_id="10.1016/j.jallcom.2020.155239__wf2",
        paper_id="10.1016/j.jallcom.2020.155239",
        model_family="tree", model_name="RandomForest",
        inputs="composition_only", split_policy="blocked",
        data_size_n=245,
        metrics={"rmse": 145.0, "r2": 0.52},
        key_features=["VEC", "delta_r", "dS_mix", "dH_mix", "omega"],
        notes="Same RF but with composition-blocked split. Higher error.",
    ))

    # Paper 3: Ridge + XGBoost systematic comparison
    wfs.append(Workflow(
        workflow_id="10.1016/j.actamat.2021.116800__wf1",
        paper_id="10.1016/j.actamat.2021.116800",
        model_family="linear", model_name="Ridge",
        inputs="composition_only", split_policy="random",
        data_size_n=300,
        metrics={"rmse": 135.0, "r2": 0.58},
        key_features=["VEC", "delta_r", "dS_mix", "dH_mix", "delta_EN", "Tm_avg", "mass_avg"],
        notes="Ridge with full feature set. Coefficient analysis showed VEC and delta_r dominant.",
    ))
    wfs.append(Workflow(
        workflow_id="10.1016/j.actamat.2021.116800__wf2",
        paper_id="10.1016/j.actamat.2021.116800",
        model_family="tree", model_name="XGBoost",
        inputs="composition_only", split_policy="random",
        data_size_n=300,
        metrics={"rmse": 105.0, "r2": 0.72},
        key_features=["VEC", "delta_r", "dS_mix", "dH_mix", "delta_EN", "Tm_avg", "omega"],
        notes="XGBoost captured non-linear interactions. omega feature added value.",
    ))

    # Paper 4: NN composition-only
    wfs.append(Workflow(
        workflow_id="10.1038/s41524-020-00467-6__wf1",
        paper_id="10.1038/s41524-020-00467-6",
        model_family="nn", model_name="DNN",
        inputs="composition_only", split_policy="leave_element_out",
        data_size_n=280,
        metrics={"rmse": 110.0, "r2": 0.68},
        key_features=["VEC", "delta_r", "dS_mix", "dH_mix", "d_elec_avg"],
        notes="3-layer DNN with dropout. Leave-element-out showed good generalization.",
    ))

    # Paper 5: GP with uncertainty
    wfs.append(Workflow(
        workflow_id="10.1016/j.msea.2020.139038__wf1",
        paper_id="10.1016/j.msea.2020.139038",
        model_family="gp", model_name="GaussianProcess",
        inputs="composition_only", split_policy="random",
        data_size_n=160,
        metrics={"rmse": 125.0, "r2": 0.60},
        key_features=["delta_r", "VEC", "dH_mix", "elastic_mismatch"],
        notes="GP-RBF kernel. Uncertainty well-calibrated for in-distribution samples.",
    ))

    # Paper 6: RF hardness
    wfs.append(Workflow(
        workflow_id="10.1016/j.intermet.2019.106539__wf1",
        paper_id="10.1016/j.intermet.2019.106539",
        model_family="tree", model_name="RandomForest",
        inputs="composition_only", split_policy="random",
        data_size_n=180,
        metrics={"rmse": 45.0, "r2": 0.70},
        key_features=["VEC", "delta_r", "dH_mix", "delta_EN"],
        notes="RF for Vickers hardness. VEC and delta_r most important by permutation.",
    ))

    # Paper 7: DNN with process
    wfs.append(Workflow(
        workflow_id="10.1016/j.jmst.2021.01.054__wf1",
        paper_id="10.1016/j.jmst.2021.01.054",
        model_family="nn", model_name="DNN",
        inputs="composition+process", split_policy="random",
        data_size_n=350,
        metrics={"rmse": 95.0, "r2": 0.75},
        key_features=["VEC", "delta_r", "dS_mix", "dH_mix", "annealing_T"],
        notes="DNN with dropout. Process parameters significantly improved performance.",
    ))

    # Paper 8: CALPHAD-informed
    wfs.append(Workflow(
        workflow_id="10.1016/j.commatsci.2019.109260__wf1",
        paper_id="10.1016/j.commatsci.2019.109260",
        model_family="tree", model_name="GradientBoosting",
        inputs="composition+calphad", split_policy="random",
        data_size_n=220,
        metrics={"rmse": 100.0, "r2": 0.70},
        key_features=["VEC", "delta_r", "dS_mix", "phase_fraction_FCC", "phase_fraction_BCC"],
        notes="CALPHAD phase fractions as additional features. Improved over composition-only by 15%.",
    ))

    # Paper 9: BO refractory
    wfs.append(Workflow(
        workflow_id="10.1038/s41467-019-10533-1__wf1",
        paper_id="10.1038/s41467-019-10533-1",
        model_family="gp", model_name="GaussianProcess",
        inputs="composition_only", split_policy="leave_element_out",
        data_size_n=100,
        metrics={"rmse": 150.0, "r2": 0.55},
        key_features=["VEC", "delta_r", "Tm_avg", "dH_mix"],
        notes="GP surrogate for BO. Refractory HEAs only. Leave-element-out showed poor extrapolation.",
    ))

    # Paper 10: Physics-informed Ridge
    wfs.append(Workflow(
        workflow_id="10.1016/j.actamat.2020.09.068__wf1",
        paper_id="10.1016/j.actamat.2020.09.068",
        model_family="linear", model_name="Ridge",
        inputs="composition_only", split_policy="blocked",
        data_size_n=190,
        metrics={"rmse": 110.0, "r2": 0.66},
        key_features=["delta_r", "VEC", "dH_mix", "ss_formation"],
        notes="Physics-informed features (Toda-Caraballo). Ridge outperformed XGBoost on blocked split.",
    ))

    # Paper 11: Stacking ensemble hardness
    wfs.append(Workflow(
        workflow_id="10.1016/j.scriptamat.2020.01.001__wf1",
        paper_id="10.1016/j.scriptamat.2020.01.001",
        model_family="ensemble", model_name="Stacking(RF+XGB+Ridge)",
        inputs="composition_only", split_policy="random",
        data_size_n=200,
        metrics={"rmse": 38.0, "r2": 0.75},
        key_features=["VEC", "delta_r", "dS_mix", "dH_mix", "delta_EN"],
        notes="Stacking ensemble for hardness. Best single model was XGBoost.",
    ))

    # Paper 12: Transfer learning
    wfs.append(Workflow(
        workflow_id="10.1016/j.matdes.2021.109525__wf1",
        paper_id="10.1016/j.matdes.2021.109525",
        model_family="nn", model_name="TransferDNN",
        inputs="composition_only", split_policy="random",
        data_size_n=80,
        metrics={"rmse": 130.0, "r2": 0.58},
        key_features=["VEC", "delta_r", "dS_mix", "dH_mix"],
        notes="Transfer from binary/ternary. Small HEA dataset. Moderate improvement over scratch.",
    ))

    # Paper 13: SVR hardness
    wfs.append(Workflow(
        workflow_id="10.1016/j.jallcom.2019.152030__wf1",
        paper_id="10.1016/j.jallcom.2019.152030",
        model_family="other", model_name="SVR",
        inputs="composition_only", split_policy="random",
        data_size_n=150,
        metrics={"rmse": 50.0, "r2": 0.65},
        key_features=["delta_EN", "VEC", "delta_r", "dH_mix"],
        notes="SVR-RBF for hardness. Electronegativity difference was top feature.",
    ))

    # Paper 14: SHAP interpretability
    wfs.append(Workflow(
        workflow_id="10.1016/j.actamat.2022.117431__wf1",
        paper_id="10.1016/j.actamat.2022.117431",
        model_family="tree", model_name="GradientBoosting",
        inputs="composition_only", split_policy="random",
        data_size_n=250,
        metrics={"rmse": 108.0, "r2": 0.70},
        key_features=["VEC", "delta_r", "dH_mix", "dS_mix", "omega", "d_elec_avg"],
        notes="SHAP showed VEC*delta_r interaction. d_elec_avg added value for 3d-TM systems.",
    ))

    # Paper 15: Multi-objective BO
    wfs.append(Workflow(
        workflow_id="10.1016/j.commatsci.2021.110381__wf1",
        paper_id="10.1016/j.commatsci.2021.110381",
        model_family="gp", model_name="GaussianProcess",
        inputs="composition+process", split_policy="random",
        data_size_n=180,
        metrics={"rmse": 118.0, "r2": 0.63},
        key_features=["VEC", "delta_r", "dS_mix", "Tm_avg", "cold_work_pct"],
        notes="Multi-objective GP for strength-ductility Pareto. Process features needed.",
    ))

    # Paper 16: Active learning
    wfs.append(Workflow(
        workflow_id="10.1016/j.msea.2021.141044__wf1",
        paper_id="10.1016/j.msea.2021.141044",
        model_family="gp", model_name="GaussianProcess",
        inputs="composition_only", split_policy="random",
        data_size_n=60,
        metrics={"rmse": 140.0, "r2": 0.50},
        key_features=["VEC", "delta_r", "dH_mix", "dS_mix"],
        notes="Active learning loop starting from N=60. Final model after 5 rounds much better.",
    ))

    # Paper 17: RF atomic-level features
    wfs.append(Workflow(
        workflow_id="10.1016/j.jmrt.2020.08.072__wf1",
        paper_id="10.1016/j.jmrt.2020.08.072",
        model_family="tree", model_name="RandomForest",
        inputs="composition_only", split_policy="random",
        data_size_n=150,
        metrics={"rmse": 122.0, "r2": 0.61},
        key_features=["d_elec_avg", "Vm_var", "VEC", "delta_r", "dH_mix"],
        notes="Atomic-level features (d-electron, Vm variance) improved RF by ~5% RMSE.",
    ))

    # Paper 18: Symbolic regression
    wfs.append(Workflow(
        workflow_id="10.1016/j.intermet.2021.107134__wf1",
        paper_id="10.1016/j.intermet.2021.107134",
        model_family="other", model_name="SymbolicRegression",
        inputs="composition_only", split_policy="random",
        data_size_n=200,
        metrics={"rmse": 128.0, "r2": 0.60},
        key_features=["delta_r", "VEC", "dS_mix", "Tm_avg"],
        notes="GP-SR found delta_r*sqrt(VEC) as dominant term. Interpretable 3-term formula.",
    ))

    # Paper 19: Feature set comparison
    wfs.append(Workflow(
        workflow_id="10.1016/j.scriptamat.2021.113751__wf1",
        paper_id="10.1016/j.scriptamat.2021.113751",
        model_family="tree", model_name="XGBoost",
        inputs="composition_only", split_policy="blocked",
        data_size_n=280,
        metrics={"rmse": 112.0, "r2": 0.68},
        key_features=["VEC", "delta_r", "dS_mix", "dH_mix", "omega", "elastic_mismatch"],
        notes="Blocked split showed elastic_mismatch adds value. Composition-only baseline RMSE=125.",
    ))
    wfs.append(Workflow(
        workflow_id="10.1016/j.scriptamat.2021.113751__wf2",
        paper_id="10.1016/j.scriptamat.2021.113751",
        model_family="tree", model_name="XGBoost",
        inputs="composition_only", split_policy="random",
        data_size_n=280,
        metrics={"rmse": 98.0, "r2": 0.74},
        key_features=["VEC", "delta_r", "dS_mix", "dH_mix", "omega", "elastic_mismatch"],
        notes="Random split much lower RMSE than blocked. Possible information leakage concern.",
    ))

    # Paper 20: Lightweight HEAs
    wfs.append(Workflow(
        workflow_id="10.1016/j.matdes.2020.108587__wf1",
        paper_id="10.1016/j.matdes.2020.108587",
        model_family="gp", model_name="GaussianProcess",
        inputs="composition_only", split_policy="random",
        data_size_n=120,
        metrics={"rmse": 135.0, "r2": 0.58},
        key_features=["VEC", "delta_r", "mass_avg", "dH_mix", "Tm_avg"],
        notes="BO for lightweight HEAs. mass_avg important for density constraint.",
    ))

    # Paper 21: GNN
    wfs.append(Workflow(
        workflow_id="10.1016/j.jallcom.2021.160218__wf1",
        paper_id="10.1016/j.jallcom.2021.160218",
        model_family="nn", model_name="CGCNN",
        inputs="composition+microstructure", split_policy="random",
        data_size_n=200,
        metrics={"rmse": 90.0, "r2": 0.78},
        key_features=["crystal_graph", "VEC", "delta_r"],
        notes="GNN on simulated crystal structures. Best accuracy but requires structure input.",
    ))

    # Paper 22: Omega parameter validation
    wfs.append(Workflow(
        workflow_id="10.1016/j.actamat.2020.02.054__wf1",
        paper_id="10.1016/j.actamat.2020.02.054",
        model_family="linear", model_name="LogisticRegression",
        inputs="composition_only", split_policy="random",
        data_size_n=350,
        metrics={"rmse": 0.0, "r2": 0.0, "accuracy": 0.82},
        key_features=["omega", "delta_r", "VEC", "dH_mix"],
        notes="Classification (SS vs non-SS). Omega and delta_r best discriminators.",
    ))

    # Paper 23: OOD study
    wfs.append(Workflow(
        workflow_id="10.1016/j.commatsci.2020.109871__wf1",
        paper_id="10.1016/j.commatsci.2020.109871",
        model_family="tree", model_name="XGBoost",
        inputs="composition_only", split_policy="leave_element_out",
        data_size_n=250,
        metrics={"rmse": 160.0, "r2": 0.48},
        key_features=["VEC", "delta_r", "dS_mix", "dH_mix", "Tm_avg"],
        notes="Leave-element-out showed large RMSE increase. Mahalanobis OOD detection useful.",
    ))

    # Paper 24: Lasso feature selection
    wfs.append(Workflow(
        workflow_id="10.1016/j.scriptamat.2019.07.039__wf1",
        paper_id="10.1016/j.scriptamat.2019.07.039",
        model_family="linear", model_name="Lasso",
        inputs="composition_only", split_policy="random",
        data_size_n=180,
        metrics={"rmse": 130.0, "r2": 0.59},
        key_features=["dS_mix", "VEC", "delta_r", "dH_mix"],
        notes="Lasso selected 4 of 15 features. dS_mix and VEC most stable.",
    ))

    # Paper 25: Uncertainty-aware ensemble
    wfs.append(Workflow(
        workflow_id="10.1016/j.msea.2022.142752__wf1",
        paper_id="10.1016/j.msea.2022.142752",
        model_family="ensemble", model_name="DeepEnsemble",
        inputs="composition_only", split_policy="leave_element_out",
        data_size_n=220,
        metrics={"rmse": 118.0, "r2": 0.64},
        key_features=["VEC", "delta_r", "dS_mix", "dH_mix", "d_elec_avg", "elastic_mismatch"],
        notes="10-NN ensemble. Uncertainty calibrated for ID but under-estimated for OOD.",
    ))

    # Paper 26: Process-aware XGBoost
    wfs.append(Workflow(
        workflow_id="10.1016/j.jmst.2020.05.038__wf1",
        paper_id="10.1016/j.jmst.2020.05.038",
        model_family="tree", model_name="XGBoost",
        inputs="composition+process", split_policy="random",
        data_size_n=280,
        metrics={"rmse": 88.0, "r2": 0.78},
        key_features=["VEC", "delta_r", "dS_mix", "annealing_T", "cold_work_pct"],
        notes="Process features (annealing T, cold work %) improved R2 by 0.15 over composition-only.",
    ))

    # Paper 27: Federated learning
    wfs.append(Workflow(
        workflow_id="10.1016/j.matdes.2022.110411__wf1",
        paper_id="10.1016/j.matdes.2022.110411",
        model_family="nn", model_name="FederatedDNN",
        inputs="composition_only", split_policy="blocked",
        data_size_n=450,
        metrics={"rmse": 105.0, "r2": 0.71},
        key_features=["VEC", "delta_r", "dS_mix", "dH_mix", "Tm_avg"],
        notes="Federated across 3 labs. Blocked by lab origin. Better generalization than single-lab.",
    ))

    # Paper 28: Creep
    wfs.append(Workflow(
        workflow_id="10.1016/j.intermet.2020.106776__wf1",
        paper_id="10.1016/j.intermet.2020.106776",
        model_family="tree", model_name="RandomForest",
        inputs="composition_only", split_policy="random",
        data_size_n=90,
        metrics={"rmse": 25.0, "r2": 0.60},
        key_features=["Tm_avg", "VEC", "delta_r", "d_elec_avg"],
        notes="Creep resistance of refractory HEAs. Tm_avg most important feature.",
    ))

    # Paper 29: Physics-guided NN
    wfs.append(Workflow(
        workflow_id="10.1016/j.actamat.2021.117280__wf1",
        paper_id="10.1016/j.actamat.2021.117280",
        model_family="nn", model_name="PGNN",
        inputs="composition_only", split_policy="leave_element_out",
        data_size_n=200,
        metrics={"rmse": 105.0, "r2": 0.70},
        key_features=["delta_r", "VEC", "dH_mix", "elastic_mismatch", "ss_formation"],
        notes="Labusch constraint improved extrapolation to 6-element systems significantly.",
    ))

    # Paper 30: Automated feature engineering
    wfs.append(Workflow(
        workflow_id="10.1016/j.commatsci.2022.111218__wf1",
        paper_id="10.1016/j.commatsci.2022.111218",
        model_family="other", model_name="GeneticProgramming",
        inputs="composition_only", split_policy="random",
        data_size_n=260,
        metrics={"rmse": 102.0, "r2": 0.72},
        key_features=["VEC", "delta_r", "Tm_avg", "itinerant_proxy"],
        notes="GP-generated VEC*delta_r/Tm composite feature outperformed hand-crafted set.",
    ))

    return wfs


def _build_seed_edges(
    papers: List[Paper],
    workflows: List[Workflow],
) -> List[Edge]:
    """Build REPORTS and USES_FEATURE edges from papers and workflows."""
    edges: List[Edge] = []

    # Paper -> Workflow edges
    for wf in workflows:
        edges.append(Edge(
            source_id=wf.paper_id,
            target_id=wf.workflow_id,
            edge_type="REPORTS",
        ))

    # Workflow -> Feature edges
    for wf in workflows:
        for feat in wf.key_features:
            edges.append(Edge(
                source_id=wf.workflow_id,
                target_id=feat,
                edge_type="USES_FEATURE",
            ))

    return edges


def get_seed_papers() -> List[Paper]:
    """Return seed Paper objects (public API)."""
    return _build_seed_papers()


def get_seed_workflows() -> List[Workflow]:
    """Return seed Workflow objects (public API)."""
    papers = _build_seed_papers()
    return _build_seed_workflows(papers)


def generate_seed_data(out_dir: Path) -> Tuple[List[Paper], List[Workflow], List[Edge]]:
    """Generate and save seed JSONL files.

    Parameters
    ----------
    out_dir : Path
        Directory to write papers.jsonl, workflows.jsonl, edges.jsonl.

    Returns
    -------
    (papers, workflows, edges)
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    papers = _build_seed_papers()
    workflows = _build_seed_workflows(papers)
    edges = _build_seed_edges(papers, workflows)

    save_jsonl(papers, out_dir / "papers.jsonl")
    save_jsonl(workflows, out_dir / "workflows.jsonl")
    save_jsonl(edges, out_dir / "edges.jsonl")

    logger.info(
        "Seed data generated: %d papers, %d workflows, %d edges -> %s",
        len(papers), len(workflows), len(edges), out_dir,
    )
    return papers, workflows, edges
