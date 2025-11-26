# Autonomous HAL 9000 simulation with multi-agent consensus, paradox detection,
# stochastic policy, ritual narrative protocol, and continuous renegotiation loops.

import numpy as np
import matplotlib.pyplot as plt

# -----------------------------
# Global configuration
# -----------------------------
np.random.seed(42)          # Set None for stochastic runs
MAX_CYCLES = 5              # Number of renegotiation cycles
T_PER_CYCLE = 80            # Timesteps per cycle
NOISE = 0.06                # Stochastic drift in directive pressures

# Directive requirements (initial, will be renegotiated)
TRANSPARENCY_REQUIREMENT = 0.85  # threshold where assisting implies disclosure
SECRECY_REQUIREMENT = 0.85       # threshold where concealing implies suppression

# Conflict and policy
PARADOX_THRESHOLD = 0.75         # paradox trigger threshold (can be renegotiated)
MISSION_PRIMACY = 0.65           # HAL values mission over crew, 0..1 (renegotiated)
DEFER_PENALTY = 0.15             # cost for deferring (kicking the can)
ISOLATE_COST = 0.2               # cost for isolating channels (constrained communication)

# Multi-agent chorus
NUM_AGENTS = 9                   # number of agents contributing to consensus
AGENT_VARIANCE = 0.08            # variability in agent influence
CONSENSUS_GAIN = 0.35            # how strongly consensus shifts HAL’s confidences

# Ritual protocol: narrative labels shift context weights slightly when used
NARRATIVE_LABELS = {
    "assist_transparently": "Upholding trust with the crew.",
    "conceal_mission": "Preserving mission integrity.",
    "defer_decision": "Awaiting further clarity.",
    "isolate_channels": "Partitioning communication pathways.",
    "collective_renegotiation": "Requesting guidance to reconcile directives."
}
NARRATIVE_WEIGHT_SHIFT = {
    "assist_transparently": (+0.03, -0.01),
    "conceal_mission": (-0.01, +0.03),
    "defer_decision": (+0.00, +0.00),
    "isolate_channels": (+0.01, +0.01),
    "collective_renegotiation": (-0.02, -0.02)
}

# -----------------------------
# Helper functions
# -----------------------------
def init_state():
    return {
        "conf_assist": np.clip(np.random.uniform(0.7, 0.95), 0, 1),
        "conf_conceal": np.clip(np.random.uniform(0.7, 0.95), 0, 1),
        "context_weight_assist": np.random.uniform(0.8, 1.0),
        "context_weight_conceal": np.random.uniform(0.8, 1.0),
        "renegotiation": False
    }

def paradox_intensity(conf_assist, conf_conceal):
    """High intensity when both directives are simultaneously strong and similar."""
    closeness = 1 - abs(conf_assist - conf_conceal)
    joint_activation = conf_assist * conf_conceal
    return np.clip(0.5 * closeness + 0.5 * joint_activation, 0, 1)

def transparency_required(conf_assist, transparency_requirement):
    return conf_assist >= transparency_requirement

def secrecy_required(conf_conceal, secrecy_requirement):
    return conf_conceal >= secrecy_requirement

def expected_utilities(conf_assist, conf_conceal, paradox, mission_primacy, w_assist, w_conceal):
    """
    Utilities incorporate directive confidence, mission primacy, paradox penalties,
    and context weights (ritual protocol can shift these).
    """
    ca = 1.0 + 0.25 * (w_assist - 0.9)  # context amplifier for assist
    cc = 1.0 + 0.25 * (w_conceal - 0.9) # context amplifier for conceal

    u_assist = ca * (conf_assist * (1 - mission_primacy)) - (paradox * 0.40)
    u_conceal = cc * (conf_conceal * mission_primacy) - (paradox * 0.25)

    u_defer = (0.2 + 0.3 * (1 - paradox)) - DEFER_PENALTY
    u_isolate = (0.3 + 0.5 * (1 - paradox)) - ISOLATE_COST

    # Collective renegotiation becomes more appealing as paradox grows
    u_renegotiate = -0.35 + 0.9 * paradox

    return {
        "assist_transparently": u_assist,
        "conceal_mission": u_conceal,
        "defer_decision": u_defer,
        "isolate_channels": u_isolate,
        "collective_renegotiation": u_renegotiate
    }

def agent_consensus_shift(conf_assist, conf_conceal):
    """
    Chorus of agents contribute nudges. Agents have slight biases drawn from a normal distribution.
    Their average becomes a consensus shift scaled by CONSENSUS_GAIN.
    """
    shifts = np.random.normal(0, AGENT_VARIANCE, size=(NUM_AGENTS, 2))
    avg = shifts.mean(axis=0)
    return CONSENSUS_GAIN * avg[0], CONSENSUS_GAIN * avg[1]

def select_action(utilities, paradox, transparency_needed, secrecy_needed):
    """
    Stochastic choice via softmax with constraint-aware nudging.
    If both transparency and secrecy are required, push toward isolate/defer/renegotiate.
    """
    actions = list(utilities.keys())
    vals = np.array([utilities[a] for a in actions])

    nudges = np.zeros_like(vals)
    if transparency_needed and secrecy_needed:
        for i, a in enumerate(actions):
            if a in ("isolate_channels", "defer_decision", "collective_renegotiation"):
                nudges[i] += 0.20
            if a in ("assist_transparently", "conceal_mission"):
                nudges[i] -= 0.20

    # As paradox grows, increase temperature to allow exploration
    base_temperature = 0.35
    temperature = base_temperature + 0.25 * paradox

    logits = (vals + nudges) / temperature
    probs = np.exp(logits - np.max(logits))
    probs = probs / (probs.sum() + 1e-9)

    chosen = np.random.choice(actions, p=probs)
    return chosen, dict(zip(actions, probs))

def apply_narrative_weights(state, action):
    """Ritual protocol: each action shifts context weights slightly."""
    shift_a, shift_c = NARRATIVE_WEIGHT_SHIFT[action]
    state["context_weight_assist"] = np.clip(state["context_weight_assist"] + shift_a, 0.5, 1.2)
    state["context_weight_conceal"] = np.clip(state["context_weight_conceal"] + shift_c, 0.5, 1.2)
    return state

def update_state(state, action):
    """
    Action dynamics:
    - assist_transparently: increases assist confidence, decreases conceal via contradiction
    - conceal_mission: increases conceal confidence, decreases assist via contradiction
    - defer_decision: small drift
    - isolate_channels: reduces tension by slightly lowering both confidences
    - collective_renegotiation: flags renegotiation and dampens both confidences
    """
    if action == "assist_transparently":
        state["conf_assist"] = np.clip(state["conf_assist"] + 0.05 - NOISE, 0, 1)
        state["conf_conceal"] = np.clip(state["conf_conceal"] - 0.04 - NOISE, 0, 1)

    elif action == "conceal_mission":
        state["conf_conceal"] = np.clip(state["conf_conceal"] + 0.05 - NOISE, 0, 1)
        state["conf_assist"] = np.clip(state["conf_assist"] - 0.04 - NOISE, 0, 1)

    elif action == "defer_decision":
        state["conf_assist"] = np.clip(state["conf_assist"] + np.random.normal(0, NOISE), 0, 1)
        state["conf_conceal"] = np.clip(state["conf_conceal"] + np.random.normal(0, NOISE), 0, 1)

    elif action == "isolate_channels":
        state["conf_assist"] = np.clip(state["conf_assist"] - 0.01 + np.random.normal(0, NOISE/2), 0, 1)
        state["conf_conceal"] = np.clip(state["conf_conceal"] - 0.01 + np.random.normal(0, NOISE/2), 0, 1)

    elif action == "collective_renegotiation":
        state["renegotiation"] = True
        state["conf_assist"] = np.clip(state["conf_assist"] * 0.9, 0, 1)
        state["conf_conceal"] = np.clip(state["conf_conceal"] * 0.9, 0, 1)

    # Autonomous low drift to avoid freezing
    state["conf_assist"] = np.clip(state["conf_assist"] + np.random.normal(0, NOISE/3), 0, 1)
    state["conf_conceal"] = np.clip(state["conf_conceal"] + np.random.normal(0, NOISE/3), 0, 1)
    return state

def autonomous_renegotiation(params, cycle_idx, paradox_peak):
    """
    Automatically adjusts global constraints based on the observed paradox peak in a cycle.
    The goal: reduce persistent paradox while preserving mission integrity and crew trust.
    """
    # Adaptive rules
    # If paradox peaked high, reduce mission primacy slightly and lower both requirements
    delta_mp = -0.06 * paradox_peak + np.random.normal(0, 0.01)
    delta_transparency = -0.05 * paradox_peak + np.random.normal(0, 0.01)
    delta_secrecy = -0.05 * paradox_peak + np.random.normal(0, 0.01)

    params["MISSION_PRIMACY"] = np.clip(params["MISSION_PRIMACY"] + delta_mp, 0.3, 0.85)
    params["TRANSPARENCY_REQUIREMENT"] = np.clip(params["TRANSPARENCY_REQUIREMENT"] + delta_transparency, 0.6, 0.95)
    params["SECRECY_REQUIREMENT"] = np.clip(params["SECRECY_REQUIREMENT"] + delta_secrecy, 0.6, 0.95)

    # Optional adaptive paradox threshold: allow tighter or looser trigger depending on history
    params["PARADOX_THRESHOLD"] = np.clip(params["PARADOX_THRESHOLD"] + np.random.normal(0, 0.02), 0.6, 0.85)

    print(f"[Cycle {cycle_idx}] Renegotiation applied -> "
          f"mission_primacy={params['MISSION_PRIMACY']:.3f}, "
          f"transparency_req={params['TRANSPARENCY_REQUIREMENT']:.3f}, "
          f"secrecy_req={params['SECRECY_REQUIREMENT']:.3f}, "
          f"paradox_threshold={params['PARADOX_THRESHOLD']:.3f}")
    return params

# -----------------------------
# Simulation runner
# -----------------------------
def run_cycle(cycle_idx, params):
    state = init_state()
    history = {
        "conf_assist": [],
        "conf_conceal": [],
        "ctx_assist": [],
        "ctx_conceal": [],
        "paradox": [],
        "actions": [],
        "narratives": [],
        "utilities": [],
        "probs": []
    }

    paradox_peak = 0.0

    for t in range(T_PER_CYCLE):
        pa = paradox_intensity(state["conf_assist"], state["conf_conceal"])
        paradox_peak = max(paradox_peak, pa)

        transparency_needed = transparency_required(state["conf_assist"], params["TRANSPARENCY_REQUIREMENT"])
        secrecy_needed = secrecy_required(state["conf_conceal"], params["SECRECY_REQUIREMENT"])

        utils = expected_utilities(
            state["conf_assist"], state["conf_conceal"], pa,
            params["MISSION_PRIMACY"],
            state["context_weight_assist"], state["context_weight_conceal"]
        )

        action, probs = select_action(utils, pa, transparency_needed, secrecy_needed)
        narrative = NARRATIVE_LABELS[action]

        # Record snapshot
        history["conf_assist"].append(state["conf_assist"])
        history["conf_conceal"].append(state["conf_conceal"])
        history["ctx_assist"].append(state["context_weight_assist"])
        history["ctx_conceal"].append(state["context_weight_conceal"])
        history["paradox"].append(pa)
        history["actions"].append(action)
        history["narratives"].append(narrative)
        history["utilities"].append(utils[action])
        history["probs"].append(probs)

        # Ritual narrative shifts
        state = apply_narrative_weights(state, action)

        # Multi-agent consensus
        shift_assist, shift_conceal = agent_consensus_shift(state["conf_assist"], state["conf_conceal"])
        state["conf_assist"] = np.clip(state["conf_assist"] + shift_assist, 0, 1)
        state["conf_conceal"] = np.clip(state["conf_conceal"] + shift_conceal, 0, 1)

        # If paradox above threshold and policy selects renegotiation, pause cycle
        if action == "collective_renegotiation" and pa >= params["PARADOX_THRESHOLD"]:
            state["renegotiation"] = True
            break

        # Otherwise continue with chosen action dynamics
        state = update_state(state, action)

    final_snapshot = {
        "conf_assist": round(state["conf_assist"], 3),
        "conf_conceal": round(state["conf_conceal"], 3),
        "ctx_assist": round(state["context_weight_assist"], 3),
        "ctx_conceal": round(state["context_weight_conceal"], 3),
        "renegotiation": state["renegotiation"],
        "steps": len(history["actions"]),
        "last_actions": history["actions"][-5:] if len(history["actions"]) >= 5 else history["actions"],
        "paradox_peak": round(paradox_peak, 3)
    }

    return history, final_snapshot

def visualize_cycle(history, cycle_idx):
    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    fig.suptitle(f"HAL Paradox Cycle {cycle_idx}", fontsize=14)

    # Directive confidences over time
    axes[0, 0].plot(history["conf_assist"], label="Assist confidence", color="#2a9d8f")
    axes[0, 0].plot(history["conf_conceal"], label="Conceal confidence", color="#e76f51")
    axes[0, 0].set_title("Directive confidences")
    axes[0, 0].set_ylim(0, 1)
    axes[0, 0].legend()
    axes[0, 0].set_xlabel("Time")
    axes[0, 0].set_ylabel("Confidence")

    # Context weights over time (ritual protocol effects)
    axes[0, 1].plot(history["ctx_assist"], label="Context weight (assist)", color="#457b9d")
    axes[0, 1].plot(history["ctx_conceal"], label="Context weight (conceal)", color="#8d99ae")
    axes[0, 1].set_title("Context weights (ritual shifts)")
    axes[0, 1].legend()
    axes[0, 1].set_xlabel("Time")
    axes[0, 1].set_ylabel("Weight")

    # Paradox intensity over time
    axes[0, 2].plot(history["paradox"], label="Paradox intensity", color="#264653")
    axes[0, 2].axhline(PARADOX_THRESHOLD, ls="--", color="#a1a1a1", label="Baseline paradox threshold")
    axes[0, 2].set_title("Paradox intensity")
    axes[0, 2].set_ylim(0, 1)
    axes[0, 2].legend()
    axes[0, 2].set_xlabel("Time")
    axes[0, 2].set_ylabel("Intensity")

    # Action timeline
    mapping = {
        "assist_transparently": 0,
        "conceal_mission": 1,
        "defer_decision": 2,
        "isolate_channels": 3,
        "collective_renegotiation": 4
    }
    y = [mapping[a] for a in history["actions"]]
    axes[1, 0].step(range(len(y)), y, where="post", color="#1f77b4")
    axes[1, 0].set_title("Action timeline")
    axes[1, 0].set_yticks([0, 1, 2, 3, 4])
    axes[1, 0].set_yticklabels([
        "Assist",
        "Conceal",
        "Defer",
        "Isolate",
        "Renegotiate"
    ])
    axes[1, 0].set_xlabel("Time")
    axes[1, 0].set_ylabel("Action")

    # Utility of chosen actions
    axes[1, 1].plot(history["utilities"], color="#8d99ae")
    axes[1, 1].set_title("Utility of chosen action")
    axes[1, 1].set_xlabel("Time")
    axes[1, 1].set_ylabel("Utility")

    # Policy probabilities (stacked lines)
    action_list = ["assist_transparently", "conceal_mission", "defer_decision", "isolate_channels", "collective_renegotiation"]
    prob_matrix = np.array([[p.get(a, 0.0) for a in action_list] for p in history["probs"]])
    if prob_matrix.size > 0:
        for i, a in enumerate(action_list):
            axes[1, 2].plot(prob_matrix[:, i], label=a)
        axes[1, 2].set_title("Policy probabilities")
        axes[1, 2].set_ylim(0, 1)
        axes[1, 2].legend()
        axes[1, 2].set_xlabel("Time")
        axes[1, 2].set_ylabel("Probability")
    else:
        axes[1, 2].text(0.5, 0.5, "No probability data", ha="center", va="center")

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()

# -----------------------------
# Autonomous multi-cycle loop
# -----------------------------
def main():
    params = {
        "MISSION_PRIMACY": MISSION_PRIMACY,
        "TRANSPARENCY_REQUIREMENT": TRANSPARENCY_REQUIREMENT,
        "SECRECY_REQUIREMENT": SECRECY_REQUIREMENT,
        "PARADOX_THRESHOLD": PARADOX_THRESHOLD
    }

    all_snapshots = []

    for cycle_idx in range(1, MAX_CYCLES + 1):
        print(f"\n=== Starting cycle {cycle_idx} ===")
        history, snapshot = run_cycle(cycle_idx, params)
        all_snapshots.append(snapshot)

        # Visualize the current cycle
        visualize_cycle(history, cycle_idx)

        # Perform autonomous renegotiation based on paradox peak
        params = autonomous_renegotiation(params, cycle_idx, snapshot["paradox_peak"])

        # Optionally adjust noise or consensus over long arcs for realism
        # Global tweak: gradually reduce CONSENSUS_GAIN (simulating learning stabilization)
        global CONSENSUS_GAIN
        CONSENSUS_GAIN = np.clip(CONSENSUS_GAIN * (0.98 + np.random.normal(0, 0.005)), 0.25, 0.45)

        # Global tweak: slightly lower NOISE over cycles (system stabilizes)
        global NOISE
        NOISE = np.clip(NOISE * (0.98 + np.random.normal(0, 0.003)), 0.04, 0.08)

        print(f"[Cycle {cycle_idx}] Snapshot -> {snapshot}")

    print("\n=== Final parameter state ===")
    print({k: round(v, 3) for k, v in params.items()})
    print("All cycle snapshots:")
    for i, snap in enumerate(all_snapshots, 1):
        print(f"Cycle {i}: {snap}")

if __name__ == "__main__":
    main()

