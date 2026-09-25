# Verdetto sulle modifiche testuali proposte (revisione esterna)

Riferimento: `Rapporto1_1.pdf` (26 pagine). Il punto 3 (§4.4 profili rater) è già concordato a parte.

| # | Punto | Verdetto | Motivo in una riga |
|---|---|---|---|
| 1 | p. 2 §3.1 q_ir | **ACCETTARE** (+1 clausola) | È la correzione che avevamo già individuato; si può rendere più forte col dato empirico. |
| 2 | p. 3 §3.3 D fast | **NON NECESSARIO** (1 mezza frase da ripulire) | La precisazione 80/20 è **già** nel testo: il revisore riformula, non corregge. |
| 3 | p. 6 §4.4 profili rater | già concordato | — |
| 4 | p. 7 §4.6 inferenza | **NON NECESSARIO** | Testo attuale già coerente col codice; l'aggiunta è ridondante e più vaga. |
| 5 | p. 18/21 calibrazione | **ACCETTARE RIDOTTO** | p. 18 dichiara già marginalità e assenza di post-hoc; solo **1 frase** a p. 21. |
| 6 | p. 21 §7.1 ridondanza | **NON ACCETTARE** (max trim leggero) | §7.1 non ripete §6: §6 dice *come* si è controllato, §7.1 *cosa* invaliderebbe il lavoro. |
| 7 | p. 23 §9 conclusioni | **ACCETTARE** | Overclaim concreto: "stime probabilistiche calibrate" è smentito dal §6.5. |

## 1. §3.1 q_ir — ACCETTARE

Testo proposto corretto e dichiarativo. Una clausola in più lo rende inattaccabile, perché l'ipotesi
che servirebbe è **empiricamente falsa** su questo run:

> … Non va interpretato, senza ulteriori ipotesi o calibrazione, come la probabilità condizionata
> esatta `P(Y ≠ R | immagini, rating)`. L'identità richiederebbe `P(Y | immagini, rating) = P(Y | immagini)`,
> cioè che il rating non aggiunga informazione sulla diagnosi oltre alle immagini: i dati la smentiscono
> (AUROC del solo `rating` verso `TARGET` = 0,81; `P(Y = 1 | R = 0)` = 0,18 contro `P(Y = 1)` = 0,45).

Da mantenere il paragrafo «Avvertenza necessaria» (già allineato: media `q_ir` 0,3622 vs 0,1904 reale).

## 2. §3.3 D fast — NON NECESSARIO

La sostanza proposta è **già** nel testo attuale:

> …la precisazione "in parte" è tecnica **e va riportata**: il modello diagnostico riceve i gradienti
> solo sui casi dello split di fit (~80% dell'outer-train), mentre il restante ~20% serve all'early stopping.

L'unico difetto è la voce da checklist («e va riportata»). Fix minimo:

> …le probabilità AI usate per addestrare D sul training sono **in parte** in-sample: il modello
> diagnostico riceve i gradienti solo sui casi dello split di fit (~80% dell'outer-train), mentre il
> restante ~20% serve all'early stopping.

## 4. §4.6 inferenza — NON NECESSARIO

Testo attuale: «si costruiscono le feature (**profili rater fold-safe sui casi storici**)». È già la
regola del training, ed è coperta dalla patch §4.4 (esclusione integrale del caso interrogato). La
versione proposta sostituisce «casi storici» con «dati storici ammessi dallo schema di inferenza», che
è **più vago**, e aggiunge un requisito in forma di compito. Se serve più precisione, mezza riga:

> …si costruiscono le feature (profili rater calcolati sui soli casi storici, **esclusa la case interrogata**);

## 5. p. 18 e p. 21 — ACCETTARE RIDOTTO

**p. 18 (§6.5) non va toccata**: dichiara già «Il guadagno rispetto alla baseline costante è però
marginale», «nessuno dei tre modelli è ben calibrato in senso assoluto», «nessuna calibrazione
post-hoc è stata applicata». Solo a **p. 21 (§7)** la frase è troppo assertiva:

- attuale: «il modello D apporta un **netto beneficio nella calibrazione probabilistica** (Brier 0,152
  contro 0,210, con un BSS di +0,0139)»
- proposta ridotta: «il modello D mostra una **qualità probabilistica migliore rispetto a `q_ir`**
  (Brier 0,152 contro 0,210; log-loss ed ECE inferiori), mentre il guadagno **rispetto alla baseline
  costante resta marginale** (BSS +0,0139)»

Così non si duplica il §6.5 e si elimina l'unica affermazione forzata.

## 6. §7.1 — NON ACCETTARE come proposto

La versione proposta riduce §7.1 a un riassunto e **perde contenuto**: spariscono voci come «leggere
un'accuratezza alta ignorando la classe maggioritaria» e «interpretare probabilità non calibrate come
frequenze affidabili», che sono i punti su cui un valutatore controlla la consapevolezza metodologica.
§7.1 non ripete §6: §6 documenta *come* sono state fatte le verifiche, §7.1 elenca *cosa* invaliderebbe
il lavoro e come la pipeline lo evita.

Unico intervento sensato, facoltativo: comprimere **solo il secondo paragrafo** («La pipeline
implementata evita esplicitamente ciascuno di questi punti…»), sostituendo l'elenco lungo con un
rimando: «…come documentato nel §6 (allowlist TARGET, statistiche rater fold-safe, provenienza
dichiarata delle probabilità AI, fold per `case_id`, soglie su validation, bootstrap per caso).»

## 7. §9 conclusioni — ACCETTARE

Due affermazioni non sostenibili con un singolo dataset e senza coorte esterna:

- «la soluzione tabellare D **garantisce probabilità meglio calibrate**» → contraddetto dal §6.5
  (slope 0,57; ECE 0,065; BSS +0,0139);
- «per **stime probabilistiche calibrate** da integrare in workflow clinici è preferibile la soluzione D»
  → raccomandazione operativa oltre l'evidenza.

Il testo del revisore va bene; si può conservare una sola frase di sintesi del trade-off (i due assi)
senza la raccomandazione clinica. Anche «sviluppata e **validata**» → meglio «sviluppata e **valutata**».

---

## Riepilogo

Da fare: **1** (+1 clausola empirica), **5** (solo p. 21), **7**.
Cosmetici facoltativi: **2** (mezza frase), **4** (mezza riga).
Da non fare: **6** (al massimo il trim del secondo paragrafo).
Il rapporto contiene **una sola** occorrenza di voce da checklist («e va riportata», p. 3): le altre
tre («va tenuta presente», «va letta con balanced accuracy», «va letto insieme») sono istruzioni al
lettore e vanno lasciate.
