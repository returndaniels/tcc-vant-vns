# Planejamento de trajetória de VANT por VNS: implementação simplificada

Material de apoio ao Trabalho de Conclusão de Curso "Planejamento de Trajetórias
Bidimensionais para VANTs por Otimização", de Daniel de Sousa da Silva
(Instituto de Computação, UFRJ).

O trabalho compara seis algoritmos. Este repositório traz um deles, a Busca em
Vizinhança Variável (VNS), em um único arquivo e sem dependências além de NumPy,
para que o leitor possa executar e inspecionar o método sem o restante do
código.

## Conteúdo

| Arquivo | O que é |
|---|---|
| `vns_simples.py` | VNS completo: objetivos, verificação de colisão, penalidades, vizinhanças e o cenário padrão |
| `cenarios/scenarios_index.csv` | Obstáculos dos quatro cenários da campanha e sementes de geração dos cenários aleatórios |
| `exemplo.png` | Trajetória obtida com a semente 42 |

## Como executar

```
pip install numpy matplotlib
python3 vns_simples.py --figura rota.png
```

Opções: `--semente N` (padrão 42) e `--iteracoes N` (padrão 500). A execução
padrão leva cerca de meio minuto e termina com:

```
J1 = 130.45 m   J2 = 1643.87 J   J3 = 17.41   colisões = 0
```

Esse é o mesmo resultado da execução de semente 42 do VNS no cenário padrão
reportada no trabalho. A versão usada nos experimentos tem as mesmas fórmulas,
com as funções numéricas compiladas para ganhar velocidade.

## O que o código faz

Uma trajetória é uma sequência de waypoints entre a origem (5, 5) e o destino
(95, 95), em um mapa de 100 m por 100 m com obstáculos circulares e
retangulares. O VNS minimiza

```
F = 0,40 J1 + 0,35 J2 + 0,25 J3 + penalidades
```

em que J1 é a distância, J2 a energia estimada (a velocidade cai nas curvas de
raio menor que 50 m) e J3 um indicador de suavidade. As penalidades tratam
colisão, energia acima de 3000 J e mudança de direção acima de 30 graus por
vértice. A colisão é verificada por interseção geométrica exata entre segmento
e obstáculo.

## Formato de `scenarios_index.csv`

Uma linha por cenário. `circles_summary` lista os círculos como
`(cx, cy, r=raio)` e `rectangles_summary` lista os retângulos como
`[xmin, ymin → xmax, ymax]`, em metros. `gen_seed` é a semente que gerou os
cenários aleatórios.
