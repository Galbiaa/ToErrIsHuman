# Riscrizioni brevi - punti 1, 5, 7 (misura da paragrafo)

Riferimento: `Rapporto1_1.pdf`. Vale il criterio della patch §4.4: sostituzioni brevi, dentro i
paragrafi esistenti, senza sezioni nuove e senza allungare il documento.

---

## Punto 1 - p. 2, §3.1 (q_ir)

**Dove**: l'ultima frase prima di «Interpretazione». Testo attuale (1 riga):

> Equivalentemente `q_ir = P(Y ≠ R | immagini, rating)`, con `Y = TARGET`.

**Sostituisci con** (2 righe):

> `q_ir` è quindi uno **score di rischio d'errore** costruito da `p_i` e dal rating osservato, non la
> probabilità condizionata esatta `P(Y ≠ R | immagini, rating)`: l'identità richiederebbe che il rating
> non aggiungesse informazione sulla diagnosi oltre alle immagini, ipotesi smentita dai dati (AUROC del
> solo `rating` verso `TARGET` = 0,81).

**Variante senza numeri** (1 riga, se preferisci non aggiungere righe):

> `q_ir` è quindi uno **score di rischio d'errore** costruito da `p_i` e dal rating osservato: non è la
> probabilità condizionata esatta `P(Y ≠ R | immagini, rating)`.

In entrambi i casi, nel paragrafo **Interpretazione** cambia una sola parola:

> **Interpretazione.** `q_ir` è la **stima del rischio** d'errore ottenuta dal solo modello image-only
> (tramite `p_i`) e dal rating del rater. …

(il resto del paragrafo - baseline *image-only + rating*, rimando alle 10 feature di D - resta identico).
I paragrafi «Verifica della definizione» e «Avvertenza necessaria» non si toccano.

---

## Punto 5 - p. 21, §7 (una clausola)

**Dove**: prima frase del §7, dopo «…(AUROC pari a 0,72 per q_ir contro 0,63 per D)». Testo attuale:

> …il modello D apporta un netto beneficio nella calibrazione probabilistica (Brier pari a 0,152
> contro 0,210, con un BSS di +0,0139).

**Sostituisci con** (stessa lunghezza):

> …il modello D mostra una qualità probabilistica migliore rispetto a `q_ir` (Brier 0,152 contro 0,210;
> log-loss ed ECE inferiori), con un BSS **solo leggermente positivo** (+0,0139).

Il §6.5 a p. 18 non va toccato: dichiara già «Il guadagno rispetto alla baseline costante è però
marginale», «nessuno dei tre modelli è ben calibrato in senso assoluto» e «nessuna calibrazione
post-hoc è stata applicata».

---

## Punto 7 - p. 23, §9 (due micro-interventi)

**7a. Una clausola nel penultimo capoverso.** Testo attuale:

> La baseline q_ir prevale nella capacità di ordinamento delle decisioni con un AUROC di circa 0,72,
> mentre la soluzione tabellare D **garantisce probabilità meglio calibrate** e un Brier Skill Score
> positivo.

**Sostituisci con** (stessa lunghezza):

> La baseline q_ir prevale nella capacità di ordinamento delle decisioni con un AUROC di circa 0,72,
> mentre la soluzione tabellare D mostra una **qualità probabilistica migliore** (Brier ed ECE più bassi)
> e un Brier Skill Score **solo leggermente positivo**.

**7b. L'ultimo capoverso** (attualmente la raccomandazione operativa). Testo attuale:

> In conclusione, la scelta tra q_ir e D dipende dal contesto applicativo. Per il ranking e lo screening
> dei casi critici è raccomandata la baseline q_ir, mentre per stime probabilistiche calibrate da
> integrare in workflow clinici è preferibile la soluzione D. Gli sviluppi futuri potrebbero focalizzarsi
> sull'applicazione di algoritmi di calibrazione post-hoc.

**Sostituisci con** (stessa lunghezza, nessuna raccomandazione clinica):

> In conclusione, i risultati documentano un trade-off tra discriminazione e qualità probabilistica:
> `q_ir` ordina meglio le decisioni, D produce probabilità meno distanti dalle frequenze osservate,
> senza raggiungere una calibrazione assoluta soddisfacente. Con un solo dataset e nessuna coorte
> esterna non si può stabilire quale soluzione sia preferibile in un workflow clinico. Gli sviluppi
> futuri sono la calibrazione post-hoc e la valutazione su coorti indipendenti.

**7c. Una parola nella prima riga della sezione**: «sviluppata e **validata**» → «sviluppata e **valutata**».

---

## Bilancio di righe

| Punto | Righe aggiunte |
|---|---|
| 1 (§3.1) | +1 (versione con numero) oppure 0 (variante senza numeri) |
| 5 (§7) | 0 |
| 7 (§9) | 0 |
