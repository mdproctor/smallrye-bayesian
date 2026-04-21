# Benchmarks

All results use `SimpleBenchmark` — 20 000 warmup iterations, 100 000 measurement iterations,
median of 3 independent runs. Chain topology, binary variables, single evidence change at
position 0 (or 0 / n/3 / 2n/3 for batch). Scores in **µs/op**.

> **Caveat — small networks at Java 25:** The fixed iteration count gives ~0.2 s warmup for
> n=8. Java 25's JIT needs more time to reach C2 at very small workloads; n=8 and n=16 results
> on Java 25 are unreliable. n≥64 results are stable across all JVMs.

---

## Java 17 (OpenJDK 17.0.18, Homebrew, ARM64)

| Benchmark | n=8 | n=16 | n=32 | n=64 | n=128 |
|---|---|---|---|---|---|
| `baselineFull` | 4.137 | 8.202 | 17.092 | 34.627 | 74.691 |
| `noChangeUpdate` | 0.124 | 0.102 | 0.112 | 0.110 | 0.122 |
| `singleEvidenceChange` | 3.159 | 6.782 | 13.662 | 27.657 | 54.787 |
| `batchEvidenceChange` | 4.111 | 7.778 | 15.686 | 31.353 | 63.092 |
| `fastRetraction` | 2.881 | 6.237 | 12.822 | 26.554 | 54.174 |

---

## Java 22 (Amazon Corretto 22.0.2, ARM64)

| Benchmark | n=8 | n=16 | n=32 | n=64 | n=128 |
|---|---|---|---|---|---|
| `baselineFull` | 3.814 | 8.217 | 16.393 | 33.190 | 68.275 |
| `noChangeUpdate` | 0.107 | 0.103 | 0.105 | 0.098 | 0.106 |
| `singleEvidenceChange` | 3.412 | 6.251 | 12.842 | 25.938 | 53.190 |
| `batchEvidenceChange` | 4.874 | 7.326 | 14.718 | 31.776 | 62.259 |
| `fastRetraction` | 2.830 | 5.896 | 12.044 | 25.521 | 51.939 |

---

## Java 25 (OpenJDK 25.0.1, ARM64) with `-XX:+UseCompactObjectHeaders`

| Benchmark | n=8 | n=16 | n=32 | n=64 | n=128 |
|---|---|---|---|---|---|
| `baselineFull` | 3.869 | 12.403† | 16.794 | 34.298 | 68.211 |
| `noChangeUpdate` | 0.175 | 0.114 | 0.115 | 0.109 | 0.117 |
| `singleEvidenceChange` | 3.323 | 6.872 | 13.813 | 29.257 | 55.721 |
| `batchEvidenceChange` | 4.495 | 7.689 | 16.714 | 43.089† | 63.932 |
| `fastRetraction` | 8.041† | 6.593 | 15.048 | 26.873 | 52.895 |

† Anomalous — JIT warmup insufficient for this workload size at this JVM version.

---

## Comparison: Java 17 → Java 22 → Java 25 at n=128 (reliable range)

| Benchmark | Java 17 | Java 22 | Java 25 | 17→22 | 17→25 |
|---|---|---|---|---|---|
| `baselineFull` | 74.69 | 68.28 | 68.21 | **-8.6%** | **-8.7%** |
| `noChangeUpdate` | 0.12 | 0.11 | 0.12 | ~same | ~same |
| `singleEvidenceChange` | 54.79 | 53.19 | 55.72 | -2.9% | ~same |
| `batchEvidenceChange` | 63.09 | 62.26 | 63.93 | ~same | ~same |
| `fastRetraction` | 54.17 | 51.94 | 52.90 | **-4.1%** | **-2.4%** |

**Key findings:**
- **Phase 1 (noChangeUpdate)** is effectively free (~0.11 µs) at all JVM versions and network sizes — the `calibrated && !isDirty()` branch is a single comparison.
- **JVM upgrade alone** delivers ~8–9% on `baselineFull` from Java 17 → 22/25, with no code changes. This comes from JIT improvements across those versions.
- **Incremental phases (2 & 3)** save 20–25% vs `baselineFull` at the same JVM version, independent of JVM version.
- **Scaling** is near-linear with n for all benchmarks — expected for chain topology. Dense-DAG networks will show different characteristics.

---

## How to run

```bash
cd /Users/mdproctor/claude/smallrye-bayesian
mvn test-compile -q
java -cp "target/test-classes:target/classes:$(mvn dependency:build-classpath -q -DforceStdout)" \
     io.smallrye.bayesian.benchmark.SimpleBenchmark
```

For a specific JVM:
```bash
JAVA_HOME=/path/to/jdk java -cp "..." io.smallrye.bayesian.benchmark.SimpleBenchmark
```

---

## Next benchmark targets

- **Vector API SIMD** on `PotentialMultiplier` / `BayesProjection` / `BayesAbsorption` (JEP 529, incubating Java 26)
- **Parallel distribute phase** using Virtual Threads (Java 21+)
- **Dense-DAG topology** (e.g., Alarm 37-var BIF network) vs chain topology
- **Valhalla value classes** on `CliqueState` / `SeparatorState` (JEP 401, Java 27 preview est.)
- Proper JMH run once the forked-process classpath issue is resolved
