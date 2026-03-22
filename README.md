# python-env-crispie

Python environment build — a Docker/Apptainer container image providing a
reproducible scientific Python stack (NumPy, pandas, Matplotlib, scikit-learn)
built and published automatically via GitHub Actions.

---

## Contents

- [Container image](#container-image)
- [Python environment](#python-environment)
- [Quickstart: sklearn training example](#quickstart-sklearn-training-example)
  - [Running with Docker (macOS or Linux)](#running-with-docker-macos-or-linux)
  - [Running with Apptainer on Amarel HPC](#running-with-apptainer-on-amarel-hpc)
  - [Running as a SLURM batch job on Amarel](#running-as-a-slurm-batch-job-on-amarel)
- [Building the image locally](#building-the-image-locally)
- [Repository structure](#repository-structure)

---

## Container image

The container is published to the GitHub Container Registry (GHCR) on every
push to `main`:

```
ghcr.io/paulgarias/python-env-crispie:latest
```

You do not need to build the image yourself.  Pull it directly with Docker
or convert it to a Singularity Image File (`.sif`) with Apptainer.

---

## Python environment

| Package | Purpose |
|---------|---------|
| `numpy` | Numerical arrays |
| `pandas` | Tabular data |
| `matplotlib` | Plotting |
| `scikit-learn` | Machine learning |
| `joblib` | Model serialisation (bundled with scikit-learn) |

---

## Quickstart: sklearn training example

The repository includes `sklearn_train.py`, a self-contained scikit-learn
example that trains a `RandomForestClassifier` on the built-in Iris dataset,
performs 5-fold cross-validation, evaluates on a held-out test set, saves the
trained model (`results/iris_rf_model.joblib`), and writes a confusion-matrix
plot (`results/confusion_matrix.png`).

### Running with Docker (macOS or Linux)

**Prerequisites:** [Docker Desktop](https://www.docker.com/products/docker-desktop/)
installed and running.

```bash
# 1. Clone the repository
git clone https://github.com/paulgarias/python-env-crispie.git
cd python-env-crispie

# 2. Pull the pre-built container image
docker pull ghcr.io/paulgarias/python-env-crispie:latest

# 3. Run the training script
#    -v mounts the current directory into /workspace inside the container
#    --rm removes the container after it exits
docker run --rm \
  -v "$(pwd)":/workspace \
  -w /workspace \
  ghcr.io/paulgarias/python-env-crispie:latest \
  python sklearn_train.py --output-dir results
```

After the run completes, inspect the outputs:

```bash
ls results/
# iris_rf_model.joblib
# confusion_matrix.png
```

Optional CLI flags:

| Flag | Default | Description |
|------|---------|-------------|
| `--output-dir` | `results` | Directory for model and plot files |
| `--n-estimators` | `100` | Number of trees in the Random Forest |
| `--random-state` | `42` | Random seed for reproducibility |

---

### Running with Apptainer on Amarel HPC

Amarel uses [Apptainer](https://apptainer.org/) (formerly Singularity) to run
containers without root privileges.  Run the steps below after logging in via
SSH (key-pair authentication recommended — add your public key to
`~/.ssh/authorized_keys` on Amarel and your private key entry to
`~/.ssh/config` on your Mac):

```sshconfig
# ~/.ssh/config  (macOS)
Host amarel
  HostName amarel.hpc.rutgers.edu
  User your_netid
  IdentityFile ~/.ssh/id_rsa
```

**Step 1 — Pull the image (once, from a login node)**

Store the image in `/projects` so it persists beyond the 90-day scratch purge
and can be shared across your research group.

```bash
mkdir -p /projects/$USER/containers
apptainer pull /projects/$USER/containers/python_env_crispie.sif \
  docker://ghcr.io/paulgarias/python-env-crispie:latest
```

**Step 2 — Clone the repository on Amarel**

```bash
git clone https://github.com/paulgarias/python-env-crispie.git
cd python-env-crispie
```

**Step 3 — Run interactively (quick test)**

```bash
# Secure an interactive session first
srun --partition=main --nodes=1 --ntasks=1 --mem=4G --time=00:30:00 --pty bash

# Then run the script inside the container
apptainer exec \
  --bind "$PWD":/workspace \
  /projects/$USER/containers/python_env_crispie.sif \
  python /workspace/sklearn_train.py --output-dir /workspace/results
```

---

### Running as a SLURM batch job on Amarel

Save the following as `run_sklearn.sh` in your working directory, then submit
with `sbatch run_sklearn.sh`.

```bash
#!/bin/bash
#SBATCH --job-name=sklearn_iris
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH --partition=main
#SBATCH --output=logs/sklearn_%j.out
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=your_netid@rutgers.edu

# --- Stage to scratch for fast I/O ---
SCRATCH=/scratch/$USER/$SLURM_JOB_ID
mkdir -p "$SCRATCH" logs

cp sklearn_train.py "$SCRATCH"/
cd "$SCRATCH"

# --- Run inside the Apptainer container ---
apptainer exec \
  --bind "$SCRATCH":/workspace \
  /projects/$USER/containers/python_env_crispie.sif \
  python /workspace/sklearn_train.py \
    --output-dir /workspace/results \
    --n-estimators 200

# --- Archive results back to home ---
cp -r "$SCRATCH"/results ~/results/sklearn_$SLURM_JOB_ID

echo "Job complete.  Results in ~/results/sklearn_$SLURM_JOB_ID"
```

Submit the job:

```bash
sbatch run_sklearn.sh
```

Monitor progress:

```bash
squeue -u $USER
tail -f logs/sklearn_<JOBID>.out
```

Check resource efficiency after completion:

```bash
seff <JOBID>
```

---

## Building the image locally

If you need to modify the environment or test a change to the Dockerfile:

```bash
# From the repository root
docker build -t python-env-crispie:local -f docker/Dockerfile .

# Run the local build
docker run --rm \
  -v "$(pwd)":/workspace \
  -w /workspace \
  python-env-crispie:local \
  python sklearn_train.py
```

Pushing a tagged release to GHCR is handled automatically by the
`.github/workflows/build-and-push.yml` GitHub Actions workflow on every push
to `main`.

---

## Repository structure

```
python-env-crispie/
├── .github/
│   └── workflows/
│       └── build-and-push.yml   # CI: builds and pushes image to GHCR
├── docker/
│   └── Dockerfile               # Container definition
├── requirements.txt             # Python dependencies
├── sklearn_train.py             # Example training script (this guide)
└── README.md
```

---

*Maintained by Paul G. Arias, Ph.D. — Rutgers Office of Advanced Research Computing*