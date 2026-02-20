# IDeLM‑Surgery‑Generator

The **IDeLM‑Surgery‑Generator** is a publicly available synthetic surgical data generator developed under the **IDeLM – Intelligent Decision Models** research program. Its purpose is to provide a reproducible, anonymized, and structurally realistic benchmark environment for evaluating intelligent scheduling algorithms, learning‑based optimization approaches, and simulation models in surgical operations.

Real surgical scheduling data cannot be shared due to personal information, operational sensitivity, and institutional constraints. This generator addresses that challenge by reconstructing a fully synthetic system based on structural and statistical patterns observed in real hospital workflows and established literature. All outputs are generated from scratch with no patient‑level or staff‑level data, ensuring complete anonymity while preserving realistic dynamics.

---

## 🔍 Overview

The generator produces a comprehensive multi‑component representation of an operating room (OR) environment, including:

- **Synthetic surgery type–surgeon frequencies** via Dirichlet distributions with complexity-based specialization  
- **Log‑normal duration distributions** with surgeon-specific speed multipliers and specialization effects  
- **Priority parameters** (operate‑by deadlines and rescheduling flexibility) based on operational complexity  
- **ICU and ward admission probabilities** with lognormal length‑of‑stay distributions  
- **Master surgical schedules** representing surgeon-room-weekday desirability scores  
- **Synthetic waiting lists** with realistic patient attributes sampled from all generated distributions  

This makes the generator suitable for:

- Learning‑based decision models and reinforcement learning for scheduling  
- Intelligent optimization and hybrid ML‑optimization pipelines  
- Rolling‑horizon scheduling simulations  
- Algorithm benchmarking and stress‑testing across diverse case mixes  
- Comparison of scheduling policies under varying demand and capacity constraints  

---

## 📦 Features

- **Fully synthetic and anonymized** — safe for public sharing, no real patient or staff data  
- **Literature-grounded** — based on case mix classification frameworks (Leeftink & Hans 2018) and established postoperative care patterns  
- **Dirichlet-based generation** — consistent probabilistic approach across frequency, surgeon assignment, and schedule modules  
- **Complexity-driven parameters** — surgical complexity (duration + variability) drives priority, admission probability, and LOS  
- **Continuous linear mappings** — smooth relationships between complexity and outcomes
- **Configurable and scalable** — adjust OR capacity, case mix size, surgeon pool, and all distribution parameters  
- **Modular design** — each component can be extended or replaced independently  
- **Benchmark‑oriented** — consistent structure for repeated experiments with full reproducibility  
- **Compatible with ML workflows** — supports learning scheduling behavior, predicting durations, and evaluating data‑driven methods  

---

## 📁 Components Generated

| Component | Method | Description |
|----------|--------|-------------|
| **Baseline Parameters** | Lognormal sampling | Surgery type duration parameters (μ, σ, γ) and operational complexity scores |
| **Surgery Frequencies** | Dirichlet distribution | Case mix composition and surgeon specialization with complexity-based concentration |
| **Duration Model** | 3-parameter lognormal | Surgeon-specific duration parameters with speed multipliers and specialization effects |
| **Priority Model** | Linear + noise | Operate-by deadlines and allowed rescheduling changes based on complexity |
| **Admission Model** | Linear + noise | ICU/ward admission probabilities and lognormal LOS parameters based on complexity |
| **Master Schedule** | Dirichlet per surgeon | Surgeon-room-weekday desirability scores with workload-based concentration |
| **Waiting List** | Multi-stage sampling | Synthetic patients with all attributes sampled from above distributions |

---

## 🚀 Installation

The project uses a standard `src/` layout with Python packaging via `pyproject.toml`.

```bash
git clone https://github.com/HI-IDN/IDeLM-Surgery-Generator
cd IDeLM-Surgery-Generator

# (optional) create a virtual environment
python -m venv .venv
source .venv/bin/activate 

# install dependencies
pip install -e .
```

**Requirements:**
- Python 3.10+
- NumPy
- Pydantic (for parameter validation)

---

## 📂 Project Layout

```
IDeLM-Surgery-Generator/
├── pyproject.toml           # project metadata and dependencies
├── README.md                # this file
├── src/
│   ├── __init__.py
│   ├── main.py              # entrypoint: python -m src.main
│   ├── generate_all_data.py # orchestrates full data generation pipeline
│   ├── models.py            # core dataclasses (Surgery, DurationCell, etc.)
│   ├── type_aliases.py      # shared type definitions
│   └── generators/          # modular generators
│       ├── __init__.py
│       ├── helpers.py       # baseline parameter generation, complexity scores
│       ├── params.py        # all parameter classes (Pydantic models)
│       ├── admission.py     # ICU/ward admission and LOS parameters
│       ├── duration.py      # surgeon-specific duration parameters
│       ├── frequency.py     # case mix and surgeon assignment frequencies
│       ├── priority.py      # operate-by deadlines and rescheduling flexibility
│       ├── schedule.py      # master surgical schedule (room-day preferences)
│       └── waiting_list.py  # synthetic patient generation
└── examples/                # (to be added) usage examples
```

---

## 🔧 Usage

### Basic Usage

Run the default data generation pipeline with built-in parameters:

```bash
python -m src.main
```

### Programmatic Usage

```python
import numpy as np
from src.generators import (
    generate_baseline_parameters,
    generate_frequency_data,
    generate_duration_data,
    generate_priority_data,
    generate_admission_data,
    generate_schedule,
    generate_waiting_list,
)
from src.generators.params import (
    FrequencyParams,
    DurationParams,
    PriorityParams,
    AdmissionParams,
    ScheduleParams,
)

# Set up
rng = np.random.default_rng(seed=42)
operation_cards = [f"OperationCard_{i}" for i in range(20)]
surgeons = list(range(10))
rooms = [f"OR_{i}" for i in range(5)]
weekdays = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"]

# Step 1: Generate baseline parameters (shared across modules)
mu_t, sigma_t, gamma_t, complexity = generate_baseline_parameters(
    num_operation_cards=len(operation_cards),
    or_capacity=480.0,
    rng=rng,
)

# Step 2: Generate frequency data
freq_data = generate_frequency_data(
    operation_cards=operation_cards,
    surgeons=surgeons,
    complexity_scores=complexity,
    params=FrequencyParams(),
    rng=rng,
)

# Step 3: Generate duration data
dur_data = generate_duration_data(
    frequency_data=freq_data,
    mu_t=mu_t,
    sigma_t=sigma_t,
    gamma_t=gamma_t,
    params=DurationParams(),
    rng=rng,
)

# Step 4: Generate priority data
priority_data = generate_priority_data(
    operation_cards=operation_cards,
    complexity_scores=complexity,
    params=PriorityParams(),
    rng=rng,
)

# Step 5: Generate admission data
admission_data = generate_admission_data(
    operation_cards=operation_cards,
    complexity_scores=complexity,
    params=AdmissionParams(),
    rng=rng,
)

# Step 6: Generate master surgical schedule
schedule = generate_schedule(
    frequency_data=freq_data,
    rooms=rooms,
    weekdays=weekdays,
    params=ScheduleParams(),
    rng=rng,
)

# Step 7: Generate waiting list
waiting_list = generate_waiting_list(
    n=100,
    frequency_data=freq_data,
    duration_data=dur_data,
    priority_data=priority_data,
    admission_data=admission_data,
    rng=rng,
)

print(f"Generated {len(waiting_list)} synthetic patients")
print(f"Schedule has {len(schedule)} (surgeon, room, day) assignments")
```

### Customizing Parameters

All distribution parameters can be customized:

```python
# More focused surgeon specialization
freq_params = FrequencyParams(
    complexity_scaling=2.0,  # Strong complexity-based specialization
)

# Tighter priority windows
priority_params = PriorityParams(
    operate_by_min=7,   # Minimum 1 week
    operate_by_max=60,  # Maximum 2 months
)

# Higher ICU admission rates
admission_params = AdmissionParams(
    p_icu_max=0.50,  # Up to 50% ICU admission for complex cases
)

# More focused surgeon schedules
schedule_params = ScheduleParams(
    base_concentration=0.5,      # Lower = more focused
    sparsity_threshold=0.05,     # Remove assignments < 5%
)
```

---

## 📊 Key Design Decisions

### 1. **Operational Complexity as a Central Metric**

Following Leeftink & Hans (2018), we compute an operational complexity score for each surgery type:

```
complexity = 0.5 × (duration / OR_capacity) + 0.5 × (CV)
```

Where:
- `duration / OR_capacity` captures relative surgery length
- `CV = std_dev / mean` captures duration variability

This complexity score drives:
- Surgeon specialization (complex surgeries → fewer surgeons dominate)
- Priority assignment (complex surgeries → shorter deadlines, less flexibility)
- Postoperative needs (complex surgeries → higher ICU probability, longer LOS)

**Justification:** While operational complexity ≠ clinical complexity, procedure duration is a validated predictor of ICU admission and LOS in the literature. This simplification is appropriate for synthetic benchmark generation.

### 2. **Dirichlet-Based Generation**

We use Dirichlet distributions for compositional data (frequencies, surgeon assignments, schedules):

**Why?** 
- Standard for compositional data (Aitchison 1982)
- Conjugate prior for multinomial (Bayesian framework)
- Interpretable concentration parameter
- Ensures non-negativity and sum-to-one constraints

### 3. **Continuous Linear Mappings**

For priority and admission parameters, we use:

```
param = min + complexity × (max - min) + noise
```

### 4. **Lognormal Distributions**

Surgery durations and length of stay use 3-parameter lognormal:

```
X = γ + exp(N(μ, σ))
```

**Why?**
- Well-established for surgery durations (Strum et al. 2000, May et al. 2000)
- Heavy-tailed (captures long outliers)
- Non-negative support
- Three parameters allow flexible fitting

---

## 📘 Documentation

Comprehensive module-level docstrings describe:
- Input/output formats
- Parameter meanings and defaults
- Mathematical formulations
- Usage examples

For detailed API documentation, see the docstrings in each module.

---

## 📚 References

The generator is grounded in established literature:

- **Leeftink, G., & Hans, E. W.** (2018). Case mix classification and a benchmark set for surgery scheduling. *Journal of Scheduling*, 21(1), 17-33.
- **Aitchison, J.** (1982). The statistical analysis of compositional data. *Journal of the Royal Statistical Society: Series B*, 44(2), 139-177.
- **Strum, D. P., May, J. H., & Vargas, L. G.** (2000). Modeling the uncertainty of surgical procedure times. *Anesthesiology*, 92(4), 1160-1167.

Additional references on ICU admission prediction, LOS modeling, and surgical prioritization informed the design.

---

## 📄 How to Cite

If you use the **IDeLM‑Surgery‑Generator** in academic work, please cite the accompanying paper:

```bibtex
@article{idelm2025surgery,
  title={IDeLM-Surgery-Generator: A Synthetic Benchmark for Surgical Scheduling},
  author={TODO},
  journal={TODO},
  year={2025},
  note={Available at: https://github.com/HI-IDN/IDeLM-Surgery-Generator}
}
```

*(Citation details to be updated upon publication)*

---

## 🤝 Contributing

Contributions, feature requests, and discussions are welcome. Please:
- Open an issue for bug reports or feature requests
- Submit a pull request for code contributions
- Follow existing code style and add tests where applicable

---

## 📬 Contact

For questions or collaboration inquiries, please contact the IDeLM project team or open an issue on GitHub.

---

## 📝 License

*(Add license information here - e.g., MIT, Apache 2.0, etc.)*

---

## 🙏 Acknowledgments

This generator was developed as part of the IDeLM (Intelligent Decision Models) research program. We thank the healthcare professionals and researchers who provided insights into surgical scheduling workflows.