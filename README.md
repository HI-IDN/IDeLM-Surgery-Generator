
# IDeLM‑Surgery‑Generator

The **IDeLM‑Surgery‑Generator** is a publicly available synthetic surgical data generator developed under the **IDeLM – Intelligent Decision Models** research program. Its purpose is to provide a reproducible, anonymized, and structurally realistic benchmark environment for evaluating intelligent scheduling algorithms, learning‑based optimization approaches, and simulation models in surgical operations.

Real surgical scheduling data cannot be shared due to personal information, operational sensitivity, and institutional constraints. This generator addresses that challenge by reconstructing a fully synthetic system based on structural and statistical patterns observed in real hospital workflows. All outputs are generated from scratch with no patient‑level or staff‑level data, ensuring complete anonymity while preserving realistic dynamics.

---

## 🔍 Overview

The generator produces a comprehensive multi‑component representation of an operating room (OR) environment, including:

- Synthetic surgery type–surgeon frequencies  
- Surgeon workload and specialization profiles  
- Log‑normal duration distributions for each procedure  
- Priority rules (operate‑by targets and allowed plan changes)  
- ICU and inpatient ward admission probabilities and length‑of‑stay distributions  
- Slot‑based OR schedules derived from surgeon frequencies  
- A synthetic waiting list with realistic attributes  

This makes the generator suitable for:

- Learning‑based decision models  
- Intelligent optimization and hybrid ML‑optimization pipelines  
- Rolling‑horizon scheduling simulations  
- Algorithm benchmarking and stress‑testing  
- Comparison of scheduling policies under varying demand and capacity  

---

## 📦 Features

- **Fully synthetic and anonymized** — safe for public sharing  
- **Statistically informed** — based on realistic workload patterns, duration behavior, and postoperative pathways  
- **Configurable and scalable** — expand the system by increasing the number of ORs, patients, or downstream capacity  
- **Modular design** — each component can be extended independently  
- **Benchmark‑oriented** — consistent structure for repeated experiments  
- **Compatible with ML workflows** — supports learning scheduling behavior, predicting durations, and evaluating data‑driven methods  

---

## 📁 Components Generated

| Component | Description |
|----------|-------------|
| Surgery frequencies | Power‑law‑based frequency model across procedures |
| Surgeon profiles | Specialization groups and log‑normal activity levels |
| Duration model | Procedure‑specific log‑normal duration parameters |
| Priority model | Operate‑by targets and allowed plan changes |
| Postoperative model | ICU/ward admission probabilities and LOS distributions |
| Schedule | Slot‑based OR assignment by room and weekday |
| Waiting‑list entries | Synthetic requests with realistic features |
| Initial schedule | Multi‑week OR plan based on patterns and fullness |

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

## 📂 Project Layout

```
IDeLM-Surgery-Generator/
├── pyproject.toml       # project metadata and dependencies
├── src/
│   ├── __init__.py
│   ├── main.py          # entrypoint: python -m src.main
│   ├── generate_all_data.py
│   ├── models.py        # core dataclasses
│   ├── type_aliases.py  # shared type definitions
│   └── generators/      # modular generators
│       ├── params.py
│       ├── frequency_data.py
│       ├── duration_data.py
│       ├── schedule.py
│       ├── priority.py
│       ├── admission_data.py
│       └── waiting_list.py
```

## 🔧 Usage

Run the default data generation pipeline (uses built‑in parameters):

```bash
python -m src.main
```

Example usage scripts will be included in the `examples/` directory.

## 📘 Documentation
Module‑level docstrings describe inputs/outputs; fuller docs and examples will be added.

## 📄 How to Cite
If you use the **IDeLM‑Surgery‑Generator** in academic work, please cite the accompanying paper:
``
TODO
``

## 🤝 Contributing
Contributions, feature requests, and discussions are welcome.
Please open an issue or submit a pull request.

## 📬 Contact
For questions or collaboration inquiries, please contact the IDeLM project team.
