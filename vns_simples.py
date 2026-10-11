"""
vns_simples.py
==============
Implementação simplificada e autocontida da Busca em Vizinhança Variável (VNS)
para planejamento de trajetórias bidimensionais de VANTs.

Acompanha o Trabalho de Conclusão de Curso "Planejamento de Trajetórias
Bidimensionais para VANTs por Otimização" (Daniel de Sousa da Silva, Instituto
de Computação, UFRJ). Reúne em um único arquivo, sem dependências além de NumPy
(e Matplotlib, opcional, para a figura), o que no trabalho está distribuído em
vários módulos:

  - os três objetivos: distância (J1), energia (J2) e indicador de suavidade (J3);
  - a verificação exata de colisão entre segmento e obstáculo;
  - as penalidades de colisão, de energia e de ângulo de curva;
  - as seis vizinhanças do VNS e os dois refinamentos determinísticos;
  - o cenário padrão do trabalho.

Uso:
    python3 vns_simples.py                 # semente 42, 500 iterações
    python3 vns_simples.py --semente 7 --iteracoes 200 --figura rota.png

A versão usada nos experimentos do trabalho tem as mesmas fórmulas, com as
funções numéricas compiladas, e produz a mesma trajetória para a mesma semente.
"""
from __future__ import annotations

import argparse
import math

import numpy as np

# ── Ambiente (cenário padrão do trabalho) ────────────────────────────────────
ORIGEM, DESTINO = (5.0, 5.0), (95.0, 95.0)
LIMITE_MIN, LIMITE_MAX = 0.0, 100.0
CIRCULOS = [(30, 30, 8), (55, 50, 10), (75, 25, 7), (40, 70, 9)]          # (cx, cy, raio)
RETANGULOS = [(60, 60, 75, 80), (15, 55, 30, 70), (45, 15, 60, 30)]        # (xmin, ymin, xmax, ymax)

# ── Plataforma de referência (modelo energético) ─────────────────────────────
PESO = 20.0                 # N (massa de 2,04 kg)
AREA_ROTOR = 0.503          # m^2
VEL_PONTA = 120.0           # m/s
ARRASTO_FUSELAGEM = 0.6
DENSIDADE_AR = 1.225        # kg/m^3
SOLIDEZ = 0.05
POT_PERFIL = 79.85          # W
K_INDUZIDA = 0.1
V_CRUZEIRO = 10.0           # m/s
ACEL_LATERAL_MAX = 2.0      # m/s^2  (raio mínimo de curva: 10^2 / 2 = 50 m)

# ── Formulação ───────────────────────────────────────────────────────────────
W1, W2, W3 = 0.40, 0.35, 0.25
E_MAX = 3000.0                       # J
THETA_MAX = math.radians(30.0)
PEN_COLISAO, PEN_ENERGIA, PEN_ANGULO = 1e4, 5e3, 200.0
K_MIN, K_MAX = 2, 15                 # waypoints intermediários

# ── Parâmetros do VNS ────────────────────────────────────────────────────────
ITERACOES = 500
TENTATIVAS_BUSCA_LOCAL = 50
SIGMA_DESLOCAR, SIGMA_INSERIR = 5.0, 3.0
N_RECOMB = 5


# ═════════════════════════════════════════════════════════════════════════════
# Objetivos
# ═════════════════════════════════════════════════════════════════════════════

def j1(wp) -> float:
    """Distância total: soma dos comprimentos dos segmentos."""
    return sum(math.dist(wp[i], wp[i + 1]) for i in range(len(wp) - 1))


def potencia(v: float) -> float:
    """Potência de propulsão de asa rotativa em função da velocidade (Zeng et al., 2019)."""
    v0 = math.sqrt(PESO / (2.0 * DENSIDADE_AR * AREA_ROTOR))
    p1 = (1.0 + K_INDUZIDA) * PESO ** 1.5 / math.sqrt(2.0 * DENSIDADE_AR * AREA_ROTOR)
    perfil = POT_PERFIL * (1.0 + 3.0 * v * v / VEL_PONTA ** 2)
    interno = math.sqrt(1.0 + v ** 4 / (4.0 * v0 ** 4)) - v * v / (2.0 * v0 * v0)
    induzida = p1 * math.sqrt(max(interno, 0.0))
    parasita = 0.5 * ARRASTO_FUSELAGEM * DENSIDADE_AR * SOLIDEZ * AREA_ROTOR * v ** 3
    return perfil + induzida + parasita


def j2(wp) -> float:
    """
    Energia total. A velocidade de cada segmento é a de cruzeiro, reduzida
    quando a curvatura de Menger no vértice que o encerra exige: v = sqrt(a / kappa).
    O último segmento não tem vértice seguinte e é percorrido em cruzeiro.
    """
    n, total = len(wp), 0.0
    for i in range(n - 1):
        d = math.dist(wp[i], wp[i + 1])
        v = V_CRUZEIRO
        if i < n - 2:
            (ax, ay), (bx, by), (cx, cy) = wp[i], wp[i + 1], wp[i + 2]
            denom = math.dist(wp[i], wp[i + 1]) * math.dist(wp[i + 1], wp[i + 2]) * math.dist(wp[i], wp[i + 2])
            cruz = (bx - ax) * (cy - ay) - (by - ay) * (cx - ax)
            kappa = 2.0 * abs(cruz) / max(denom, 1e-9) if denom > 1e-9 else 0.0
            v = min(v, math.sqrt(ACEL_LATERAL_MAX / max(kappa, 1e-9)))
        total += potencia(v) * d / v
    return total


def j3(wp) -> float:
    """Indicador de suavidade: soma das normas das segundas diferenças."""
    return sum(math.hypot(wp[i + 1][0] - 2 * wp[i][0] + wp[i - 1][0],
                          wp[i + 1][1] - 2 * wp[i][1] + wp[i - 1][1])
               for i in range(1, len(wp) - 1))


# ═════════════════════════════════════════════════════════════════════════════
# Restrições
# ═════════════════════════════════════════════════════════════════════════════

def segmento_colide(x1, y1, x2, y2) -> bool:
    """Interseção exata do segmento com algum obstáculo (sem amostragem)."""
    dx, dy = x2 - x1, y2 - y1
    comp2 = dx * dx + dy * dy
    for cx, cy, r in CIRCULOS:                       # ponto do segmento mais próximo do centro
        t = 0.0 if comp2 == 0.0 else min(1.0, max(0.0, ((cx - x1) * dx + (cy - y1) * dy) / comp2))
        if (x1 + t * dx - cx) ** 2 + (y1 + t * dy - cy) ** 2 <= r * r:
            return True
    for xmin, ymin, xmax, ymax in RETANGULOS:        # recorte de Liang-Barsky
        t0, t1, dentro = 0.0, 1.0, True
        for p, o, lo, hi in ((dx, x1, xmin, xmax), (dy, y1, ymin, ymax)):
            q0, q1 = lo - o, hi - o
            if abs(p) < 1e-12:
                if q0 > 0.0 or q1 < 0.0:
                    dentro = False
                    break
            else:
                a, b = sorted((q0 / p, q1 / p))
                t0, t1 = max(t0, a), min(t1, b)
                if t0 > t1:
                    dentro = False
                    break
        if dentro:
            return True
    return False


def colisoes(wp) -> int:
    """Número de segmentos da trajetória que interceptam algum obstáculo."""
    return sum(segmento_colide(*wp[i], *wp[i + 1]) for i in range(len(wp) - 1))


def excesso_angular(wp) -> float:
    """Soma, em radianos, do que cada mudança de direção excede THETA_MAX."""
    total = 0.0
    for i in range(1, len(wp) - 1):
        ux, uy = wp[i][0] - wp[i - 1][0], wp[i][1] - wp[i - 1][1]
        vx, vy = wp[i + 1][0] - wp[i][0], wp[i + 1][1] - wp[i][1]
        nu, nv = math.hypot(ux, uy), math.hypot(vx, vy)
        if nu < 1e-9 or nv < 1e-9:
            continue
        curva = math.acos(min(1.0, max(-1.0, (ux * vx + uy * vy) / (nu * nv))))
        total += max(0.0, curva - THETA_MAX)
    return total


def custo(wp: np.ndarray) -> float:
    """Função escalar minimizada pelo VNS: soma ponderada dos objetivos mais penalidades."""
    p = wp.tolist()
    e = j2(p)
    return (W1 * j1(p) + W2 * e + W3 * j3(p)
            + PEN_COLISAO * colisoes(p)
            + PEN_ENERGIA * max(0.0, e - E_MAX)
            + PEN_ANGULO * excesso_angular(p))


# ═════════════════════════════════════════════════════════════════════════════
# Vizinhanças (cada uma devolve um vizinho aleatório da solução recebida)
# ═════════════════════════════════════════════════════════════════════════════

def _limitar(wp):
    wp = wp.copy()
    wp[1:-1] = np.clip(wp[1:-1], LIMITE_MIN, LIMITE_MAX)
    return wp


def n1_deslocar(wp, rng):
    wp = wp.copy()
    if len(wp) == 2:
        return wp
    i = int(rng.integers(1, len(wp) - 1))
    wp[i] += rng.normal(0.0, SIGMA_DESLOCAR, size=2)
    return _limitar(wp)


def n2_trocar(wp, rng):
    wp = wp.copy()
    if len(wp) - 2 < 2:
        return wp
    i, j = rng.choice(np.arange(1, len(wp) - 1), size=2, replace=False)
    wp[[i, j]] = wp[[j, i]]
    return wp


def n3_remover(wp, rng):
    if len(wp) - 2 <= K_MIN:
        return wp.copy()
    return np.delete(wp, int(rng.integers(1, len(wp) - 1)), axis=0)


def n4_inserir(wp, rng):
    """Insere um ponto perto do meio do segmento mais longo."""
    if len(wp) - 2 >= K_MAX:
        return wp.copy()
    seg = int(np.argmax(np.linalg.norm(np.diff(wp, axis=0), axis=1)))
    meio = (wp[seg] + wp[seg + 1]) / 2.0 + rng.normal(0.0, SIGMA_INSERIR, size=2)
    return np.insert(wp, seg + 1, np.clip(meio, LIMITE_MIN, LIMITE_MAX), axis=0)


def n5_inverter(wp, rng):
    """2-opt: inverte a ordem de um trecho de waypoints intermediários."""
    wp = wp.copy()
    if len(wp) - 2 < 2:
        return wp
    i, j = sorted(rng.choice(np.arange(1, len(wp) - 1), size=2, replace=False))
    wp[i: j + 1] = wp[i: j + 1][::-1]
    return wp


def n6_recombinar(wp, rng):
    """Várias trocas aleatórias em sequência."""
    wp = wp.copy()
    if len(wp) - 2 < 2:
        return wp
    for _ in range(N_RECOMB):
        i, j = rng.choice(np.arange(1, len(wp) - 1), size=2, replace=False)
        wp[[i, j]] = wp[[j, i]]
    return wp


VIZINHANCAS = [n1_deslocar, n2_trocar, n3_remover, n4_inserir, n5_inverter, n6_recombinar]


# ═════════════════════════════════════════════════════════════════════════════
# Refinamentos determinísticos, aplicados após cada melhoria
# ═════════════════════════════════════════════════════════════════════════════

def rotacoes_circulares(wp, f):
    """Testa todas as rotações cíclicas dos waypoints intermediários."""
    meio, melhor, f_melhor = wp[1:-1], wp.copy(), f
    for r in range(1, len(meio)):
        cand = np.vstack([wp[0:1], np.roll(meio, r, axis=0), wp[-1:]])
        fc = custo(cand)
        if fc < f_melhor:
            melhor, f_melhor = cand, fc
    return melhor, f_melhor


def realocacoes(wp, f):
    """Testa mover cada waypoint intermediário para cada outra posição."""
    melhor, f_melhor, melhorou = wp.copy(), f, True
    while melhorou:
        melhorou = False
        for i in range(1, len(melhor) - 1):
            ponto, sem_i = melhor[i], np.delete(melhor, i, axis=0)
            for j in range(1, len(sem_i) - 1):
                if j == i:
                    continue
                cand = np.insert(sem_i, j, ponto, axis=0)
                fc = custo(cand)
                if fc < f_melhor - 1e-9:
                    melhor, f_melhor, melhorou = cand, fc, True
                    break
            if melhorou:
                break
    return melhor, f_melhor


# ═════════════════════════════════════════════════════════════════════════════
# VNS
# ═════════════════════════════════════════════════════════════════════════════

def solucao_inicial(rng, n=5, sigma=8.0):
    """Reta entre origem e destino com n pontos intermediários perturbados."""
    o, d = np.array(ORIGEM), np.array(DESTINO)
    meio = o + np.outer(np.linspace(0.0, 1.0, n + 2)[1:-1], d - o)
    meio += rng.normal(0.0, sigma, size=meio.shape)
    return _limitar(np.vstack([o, meio, d]))


def busca_local(wp, f, vizinhanca, rng):
    for _ in range(TENTATIVAS_BUSCA_LOCAL):
        cand = vizinhanca(wp, rng)
        fc = custo(cand)
        if fc < f:
            wp, f = cand, fc
    return wp, f


def vns(semente=42, iteracoes=ITERACOES, verboso=True):
    """
    Para cada iteração, percorre as vizinhanças em ordem. Em cada uma, perturba
    a solução corrente, aplica busca local e, se houve melhoria, aceita, refina
    e volta à primeira vizinhança; caso contrário, passa à seguinte.
    """
    rng = np.random.default_rng(semente)
    atual = solucao_inicial(rng)
    f_atual = custo(atual)
    inicial, melhor, f_melhor = atual.copy(), atual.copy(), f_atual

    for it in range(iteracoes):
        k = 0
        while k < len(VIZINHANCAS):
            viz = VIZINHANCAS[k]
            perturbada = viz(atual, rng)
            cand, f_cand = busca_local(perturbada, custo(perturbada), viz, rng)
            if f_cand < f_atual:
                atual, f_atual = rotacoes_circulares(cand, f_cand)
                atual, f_atual = realocacoes(atual, f_atual)
                k = 0
                if f_atual < f_melhor:
                    melhor, f_melhor = atual.copy(), f_atual
            else:
                k += 1
        if verboso and (it % 100 == 0 or it == iteracoes - 1):
            p = melhor.tolist()
            print(f"iteração {it + 1:4d}/{iteracoes}  F={f_melhor:10.3f}  "
                  f"J1={j1(p):7.2f} m  J2={j2(p):8.2f} J  J3={j3(p):6.2f}  colisões={colisoes(p)}")
    return inicial, melhor


def desenhar(inicial, melhor, arquivo):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle, Rectangle

    fig, ax = plt.subplots(figsize=(6, 6))
    for cx, cy, r in CIRCULOS:
        ax.add_patch(Circle((cx, cy), r, fc="#d9d9d9", ec="#555555"))
    for x0, y0, x1, y1 in RETANGULOS:
        ax.add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0, fc="#d9d9d9", ec="#555555"))
    ax.plot(inicial[:, 0], inicial[:, 1], "--", color="#999999", label="solução inicial")
    ax.plot(melhor[:, 0], melhor[:, 1], "-o", color="#2a9d3f", ms=4, label="VNS")
    ax.plot(*ORIGEM, "ko")
    ax.plot(*DESTINO, "k*", ms=12)
    ax.set_xlim(LIMITE_MIN, LIMITE_MAX)
    ax.set_ylim(LIMITE_MIN, LIMITE_MAX)
    ax.set_aspect("equal")
    ax.grid(alpha=0.2)
    ax.legend()
    fig.tight_layout()
    fig.savefig(arquivo, dpi=150)


def main():
    ap = argparse.ArgumentParser(description="VNS simplificado para trajetória de VANT")
    ap.add_argument("--semente", type=int, default=42)
    ap.add_argument("--iteracoes", type=int, default=ITERACOES)
    ap.add_argument("--figura", type=str, default=None, help="arquivo PNG de saída (opcional)")
    args = ap.parse_args()

    inicial, melhor = vns(args.semente, args.iteracoes)
    p = melhor.tolist()
    print(f"\nWaypoints ({len(p)}):")
    for x, y in p:
        print(f"  ({x:6.2f}, {y:6.2f})")
    print(f"J1 = {j1(p):.2f} m   J2 = {j2(p):.2f} J   J3 = {j3(p):.2f}   "
          f"colisões = {colisoes(p)}   excesso angular = {math.degrees(excesso_angular(p)):.2f} graus")
    if args.figura:
        desenhar(inicial, melhor, args.figura)
        print(f"Figura gravada em {args.figura}")


if __name__ == "__main__":
    main()
