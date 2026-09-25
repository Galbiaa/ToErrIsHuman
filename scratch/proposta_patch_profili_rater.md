# Proposta di revisione - "Statistiche rater fold-safe" (§4.4) + verifica (§6.7)

Documento di lavoro. Riferimento: `Rapporto1_1.pdf` (versione corrente, 26 pagine).
Testo attuale nel PDF (pag. 6, §4.4):

> **Statistiche rater fold-safe.** Nell'Excel, rater-accuracy e rater-confidence sono medie
> globali su tutto il dataset: usarle in cross-validation leakerebbe il test. Per ogni outer fold
> si ricalcolano solo sui casi di training; sulle righe di training si usa leave-one-out
> (la decisione corrente non entra nella propria feature).

Il testo non e' sbagliato: e' **incompleto** su un punto che un revisore chiedera'. Non dice
**su quali casi** e' calcolato il profilo, ne' collega training e inferenza (dove i profili sono
dichiarati "sui casi storici").

---

## A) Patch del paragrafo §4.4 (da incollare al posto del testo attuale)

**Statistiche rater fold-safe.** Nell'Excel, rater-accuracy e rater-confidence sono medie globali su
tutto il dataset: usarle in cross-validation leakerebbe il test. Per ogni outer fold si ricalcolano
**solo sui casi di training**. Il profilo della riga (caso *i*, rater *r*) e' costruito **escludendo
l'intero caso *i***: sulle righe di training si usa leave-one-out rispetto alla decisione corrente,
che qui equivale a un leave-one-case-out perche' ogni rater valuta ogni caso una volta sola (§6.7).
Il profilo non contiene quindi mai decisioni dello stesso caso, e il protocollo coincide con quello
dell'inferenza, dove i profili usano solo i **casi storici** e il caso interrogato e' escluso
integralmente. `rater-accuracy` e' calcolata su `error-rating`, che su un caso nuovo non e'
disponibile per definizione: in nessuno stadio vengono usate le etichette di errore delle altre
decisioni sullo stesso caso (§6.7).

---

## B) Nuova sezione §6.7 (da inserire dopo §6.6, prima di §7)

### 6.7 Profili rater fold-safe: nessuna informazione dello stesso caso

`rater-accuracy` entra in D ed e' costruita su `error-rating`, cioe' sullo stesso bersaglio che D
deve predire: va quindi escluso che il profilo di una decisione contenga informazione su quella
decisione o sul caso a cui appartiene. La verifica e' stata fatta su tre fronti.

**1. Struttura dei dati.** 5551 righe = 427 casi x 13 rater; coppie `(case_id, rater_id)` duplicate:
**0**; decisioni per coppia: minimo 1, massimo 1; ogni rater copre tutti e 427 i casi. Esiste quindi
**una sola** decisione per coppia (rater, caso): escludere la decisione corrente equivale a
escludere l'intero caso, perche' per il rater *r* non esistono altre decisioni sul caso *i*.

**2. Confronto con un leave-one-case-out esplicito.** Le statistiche sono state ricalcolate
escludendo **tutte** le righe del caso corrente e confrontate con l'implementazione usata in
training (`methodology_guards.fold_safe_rater_stats`):

| Fold | righe train | righe test | max \|delta\| `rater-accuracy` | max \|delta\| `rater-confidence` |
|---:|---:|---:|---:|---:|
| 1 | 4433 | 1118 | 0,0 | 0,0 |
| 2 | 4433 | 1118 | 0,0 | 0,0 |
| 3 | 4446 | 1105 | 0,0 | 0,0 |
| 4 | 4446 | 1105 | 0,0 | 0,0 |
| 5 | 4446 | 1105 | 0,0 | 0,0 |

La differenza e' **esattamente zero**, in tutti i fold e per entrambe le feature.

**3. Composizione del pool di riferimento.** Per ogni decisione e' stato conteggiato quante
decisioni dello **stesso caso** entrano nel pool usato per calcolare il proprio profilo: **zero** sia
per le righe di training sia per quelle di test. Il pool contiene inoltre solo casi di training:
nessuna riga di test partecipa al calcolo.

**Coerenza con l'inferenza.** Su un caso nuovo il caso interrogato non entra mai nei profili:
l'inferenza usa come pool storico tutte le case tranne quella richiesta. La regola e' la stessa in
training, in validazione/test e in inferenza: "decisioni del rater su altri casi".

**Sul caso alternativo.** Far entrare nel profilo le altre decisioni dello stesso caso non e'
applicabile a `rater-accuracy`: la feature e' calcolata su `error-rating`, che su un caso nuovo e' il
bersaglio da predire e non e' noto. Un workflow che rendesse disponibili le **osservazioni** degli
altri rater sullo stesso caso (per esempio `rating-confidence`) potrebbe arricchire le feature di
contesto, non l'accuratezza del rater; sarebbe un requisito operativo diverso dall'implementazione
attuale, da dichiarare come tale.

Fonte: `scratch/verify_q_ir_and_rater_stats.py`, `outputs/D/rater_profiles_foldsafe_fold_{1..5}.csv`,
`models/D/scenario_D_fold_{1..5}.joblib`.

---

## C) Mezza riga in §7.1 (facoltativa)

Dove l'elenco cita "il calcolo di statistiche del rater utilizzando informazioni provenienti dal
test", aggiungere: **"o dallo stesso caso da predire (§6.7)"**.

---

## D) Perche' non usare il testo del revisore verbatim

1. E' scritto come **compito aperto** ("Va inoltre verificato esplicitamente che...") e lascia
   un'**alternativa aperta** ("Se invece il workflow reale rende disponibili..."): in un rapporto di
   consegna si legge "non l'abbiamo controllato". La verifica e' fatta: va **affermata** con l'esito
   e la fonte.
2. Il ramo alternativo, per `rater-accuracy`, **non e' applicabile**: la feature usa `error-rating`,
   che su un caso nuovo e' il target stesso. Escludere l'intero `case_id` non e' una scelta tra
   opzioni, e' l'unico protocollo ammissibile.
3. Il resto della sostanza del revisore e' corretto ed e' incorporato nei testi sopra.
