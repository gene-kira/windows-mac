@dataclass
class ProbeResult:
    path: List[str]
    eta_s: float
    confidence: float
    hops: int
    target_threshold: float

class Probe:
    """
    Prometheus-style probe that explores ‘tunnels’ (lead–lag causal paths)
    and reports the best path to threshold crossing with ETA and confidence.
    """
    def __init__(self, graph: LeadLagGraph, bands: Dict[str, Dict[str, Tuple[List[float], List[float]]]], hz: float = 10.0):
        self.graph = graph
        self.bands = bands
        self.hz = hz

    def _eta_local(self, node: str, target_threshold: float) -> Optional[Tuple[float, float]]:
        lo, hi = self.bands.get(node, {}).get("M", ([], []))
        if not lo or not hi:
            return None
        return hazard_eta(target_threshold, lo, hi, hz=self.hz)

    def scan(self, start: str, target_threshold: float, max_depth: int = 5) -> Optional[ProbeResult]:
        """
        DFS with path confidence accumulation and ETA aggregation.
        Prefers paths with higher minimum edge confidence and tighter local bands.
        """
        best: Optional[ProbeResult] = None
        visited: Set[str] = set()

        def dfs(node: str, depth: int, acc_eta: float, acc_conf: float, path: List[str]):
            nonlocal best
            visited.add(node)
            path.append(node)

            local_eta = self._eta_local(node, target_threshold)
            if local_eta:
                eta_s, eta_conf = local_eta
                # Aggregate ETA and confidence: confidence is bottleneck along path
                total_eta = acc_eta + eta_s
                total_conf = min(acc_conf, eta_conf)
                candidate = ProbeResult(path=list(path), eta_s=total_eta, confidence=total_conf, hops=len(path)-1, target_threshold=target_threshold)
                if best is None or candidate.confidence > best.confidence or (
                    abs(candidate.confidence - best.confidence) < 1e-6 and candidate.eta_s < best.eta_s
                ):
                    best = candidate

            if depth >= max_depth:
                path.pop()
                return

            for nxt, edge in self.graph.edges.get(node, {}).items():
                if nxt in visited:
                    continue
                # Path confidence reduces by edge confidence
                next_conf = min(acc_conf, edge.confidence)
                # ETA accumulates lag in seconds
                next_eta = acc_eta + max(0.0, edge.lag_s)
                dfs(nxt, depth+1, next_eta, next_conf, path)

            path.pop()
            visited.remove(node)

        # Start with neutral confidence; let the path and local bands determine its value
        dfs(start, 0, 0.0, 1.0, [])
        return best

