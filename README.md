# Lucidum Cognitive Firmware
A small Python library + notebooks that simulate **cognitive energy (“battery”)**, **concurrent focus (“channels”)**, **alignment bonus / misalignment cost**, and **overflow thresholds**.

<p align="center"><img src="notebooks/figures/example_battery.png" width="560"></p>

## Quickstart
```bash
pip install -U pip
pip install numpy matplotlib
from lucidum.core import BatteryManager, simulate, Presets

bm = Presets.deep_work()
log = simulate(T=300, bm=bm, rate=2)

from lucidum.core import dashboard
dashboard(log, bm, title="Deep Work")
src/lucidum/        # library code
notebooks/          # examples / demos
data/               # sample event logs
tests/              # minimal invariants

---

### `pyproject.toml`
```toml
[build-system]
requires = ["setuptools>=68", "wheel"]
build-backend = "setuptools.build_meta"

[project]
name = "lucidum"
version = "0.1.0"
description = "Cognitive energy & channels firmware"
readme = "README.md"
authors = [{name="Tim Mains"}]
license = {text = "MIT"}
requires-python = ">=3.9"
dependencies = ["numpy>=1.22", "matplotlib>=3.5"]

[tool.setuptools.packages.find]
where = ["src"]
from .core import Spark, BatteryManager, Presets, simulate, dashboard
from dataclasses import dataclass, field
from typing import List, Dict, Tuple
import numpy as np
import matplotlib.pyplot as plt

# ----------------- Data model -----------------
@dataclass
class Spark:
    t: int
    tag: str                 # category/topic
    origin: str              # "internal" | "external"
    clarity: float           # 0..1
    valence: int             # -1,0,1
    energy_cost: float       # base demand

TAGS_CORE = ("cognitive_reconstruction","tools","book")
TAGS_NOISE = ("inbox","news","drama","random")
ALL_TAGS = TAGS_CORE + TAGS_NOISE

_rng = np.random.default_rng(42)

def make_spark(t:int) -> Spark:
    tag = _rng.choice(ALL_TAGS, p=[.14,.14,.14,.18,.16,.12,.12])
    origin = _rng.choice(["internal","external"], p=[.55,.45])
    if tag in TAGS_CORE:
        clarity  = float(np.clip(_rng.beta(6,2),0,1))
        valence  = int(_rng.choice([-1,0,1], p=[.05,.15,.80]))
        energy   = float(np.clip(_rng.normal(0.7,0.18), 0.15, 1.6))
    else:
        clarity  = float(np.clip(_rng.beta(2.5,4.5),0,1))
        valence  = int(_rng.choice([-1,0,1], p=[.30,.45,.25]))
        energy   = float(np.clip(_rng.normal(1.0,0.25), 0.20, 2.2))
    return Spark(t, tag, origin, clarity, valence, energy)

# ----------------- Controller -----------------
@dataclass
class BatteryManager:
    max_channels: int = 3
    recovery_per_tick: float = 0.35
    align_bonus: float = 0.85
    misalign_cost: float = 0.55
    low_thresh: float = 10.0
    high_thresh: float = 95.0
    battery_min: float = 0.0
    battery_max: float = 100.0
    battery: float = 60.0
    intent_tags: set = field(default_factory=lambda: set(TAGS_CORE))
    open_channels: int = 0

    def allow(self, tag:str, clarity:float) -> bool:
        if self.open_channels >= self.max_channels and tag not in self.intent_tags:
            return False
        if self.battery <= 5 and tag not in self.intent_tags:
            return False
        return True

    def apply(self, sparks: List[Spark]) -> Tuple[float,float]:
        cost = sum(s.energy_cost for s in sparks)
        gain = 0.0
        for s in sparks:
            if s.tag in self.intent_tags and s.valence >= 0:
                gain += self.align_bonus * (0.5 + 0.5*s.clarity)
            else:
                cost += self.misalign_cost * (0.5 + 0.5*(1-s.clarity))
        self.battery = float(np.clip(self.battery - cost + gain + self.recovery_per_tick,
                                     self.battery_min, self.battery_max))
        return cost, gain

    def overflow(self) -> bool:
        return (self.battery <= self.low_thresh) or (self.battery >= self.high_thresh) or (self.open_channels > self.max_channels)

# ----------------- Simulation -----------------
def simulate(T=240, bm:BatteryManager=None, rate=2) -> Dict[str, np.ndarray]:
    bm = bm or BatteryManager()
    log: Dict[str, list] = {"t":[], "battery":[], "open_channels":[], "overflows":[], "accepted_tags":[], "cost":[], "gain":[]}
    for t in range(T):
        sparks = [make_spark(t) for _ in range(_rng.poisson(rate))]
        # score & admit up to max_channels
        def score(s: Spark):
            align = 1 if s.tag in bm.intent_tags else 0
            return (2*align + 0.8*s.clarity + 0.3*s.valence) - 0.2*s.energy_cost
        sparks.sort(key=score, reverse=True)
        accepted: List[Spark] = []
        for s in sparks:
            if len(accepted) >= bm.max_channels: break
            if bm.allow(s.tag, s.clarity): accepted.append(s)
        bm.open_channels = len(accepted)
        cost, gain = bm.apply(accepted)

        log["t"].append(t)
        log["battery"].append(bm.battery)
        log["open_channels"].append(len(accepted))
        log["overflows"].append(int(bm.overflow()))
        log["accepted_tags"].append([s.tag for s in accepted])
        log["cost"].append(cost); log["gain"].append(gain)
    return {k: np.array(v, dtype=object if k=="accepted_tags" else float) for k,v in log.items()}

# ----------------- Presets -----------------
class Presets:
    @staticmethod
    def deep_work() -> BatteryManager:
        return BatteryManager(max_channels=1, recovery_per_tick=0.40, align_bonus=1.00, misalign_cost=0.40)
    @staticmethod
    def social_feed() -> BatteryManager:
        return BatteryManager(max_channels=5, recovery_per_tick=0.20, align_bonus=0.40, misalign_cost=0.80,
                              intent_tags={"news","inbox","random"})
    @staticmethod
    def recovery() -> BatteryManager:
        return BatteryManager(max_channels=1, recovery_per_tick=0.80, align_bonus=0.60, misalign_cost=0.30)

# ----------------- Dashboard -----------------
def dashboard(log:Dict[str,np.ndarray], bm:BatteryManager, title="Lucidum — Session Summary"):
    t = log["t"].astype(float)
    battery = log["battery"].astype(float)
    channels = log["open_channels"].astype(float)
    over = log["overflows"].astype(float)
    print(f"{title}\nfinal_battery={battery[-1]:.2f}  avg_channels={channels.mean():.2f}  overflow_ticks={int(over.sum())}")
    plt.figure(figsize=(9,4)); plt.plot(t,battery,label="Battery")
    plt.axhline(bm.low_thresh,linestyle="--",label="Low threshold")
    plt.axhline(bm.high_thresh,linestyle="--",label="High threshold")
    plt.title("Cognitive Battery Over Time"); plt.xlabel("t"); plt.ylabel("battery"); plt.legend(); plt.show()
    plt.figure(figsize=(9,2.8)); plt.plot(t,channels,label="Open channels")
    plt.title("Channels Followed Per Tick"); plt.xlabel("t"); plt.ylabel("#"); plt.legend(); plt.show()
label,source,signal,benefit,cost
research,internal,0.8,0.5,0.3
book,internal,0.9,0.7,0.4
inbox,external,0.6,0.2,0.7
news,external,0.5,0.2,0.8
drama,external,0.4,0.1,0.9
from lucidum.core import BatteryManager, simulate, Presets

def test_monotonicity_cost():
    bm = BatteryManager()
    b0 = bm.battery
    bm.apply([])  # no cost
    b_no = bm.battery
    bm.battery = b0
    bm.apply([type("S",(),{"energy_cost":1.0,"tag":"random","valence":0,"clarity":0.5})()])
    b_yes = bm.battery
    assert b_yes <= b_no

def test_presets_run():
    for preset in (Presets.deep_work(), Presets.social_feed(), Presets.recovery()):
        log = simulate(T=20, bm=preset, rate=2)
        assert len(log["t"]) == 20
name: ci
on: [push, pull_request]
jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with: { python-version: '3.11' }
      - run: pip install -e . pytest
      - run: pytest -q
git clone https://github.com/<you>/lucidum-cognitive-firmware.git
cd lucidum-cognitive-firmware
pip install -e .
python - <<'PY'
from lucidum.core import Presets, simulate, dashboard
bm = Presets.deep_work()
log = simulate(T=200, bm=bm, rate=2)
dashboard(log, bm, title="Deep Work")
PY
