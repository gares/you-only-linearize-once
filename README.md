Notes and an Elpi formalization of Linear A, the intermediate language of
*You Only Linearize Once*. The repository also compares that paper with how
Tapenade handles linearity.

Files:

- [you-only-linearize-once.pdf](you-only-linearize-once.pdf): Radul, Paszke,
  Frostig, Johnson, Maclaurin, *You Only Linearize Once: Tangents Transpose to
  Gradients*, POPL 2023 ([arXiv:2204.10923](https://arxiv.org/abs/2204.10923)).
- [tapenade-rr7957.pdf](tapenade-rr7957.pdf): Hascoët, Pascual, *The Tapenade
  Automatic Differentiation tool: principles, model, and specification*,
  Inria research report RR-7957, 2012
  ([hal-00695839](https://inria.hal.science/hal-00695839)).
- The Elpi code, one file per language, transformation and utility (see
  [The Elpi code](#the-elpi-code)), with two entry points:
  [run-example.elpi](run-example.elpi) and [run-tests.elpi](run-tests.elpi),
  run by the [Makefile](Makefile).

## You Only Linearize Once (Radul et al., 2023)

The paper splits reverse-mode automatic differentiation (AD) into three
separate program transformations:

1. **J, forward mode (JVP).** This is the standard tangent transformation. Its
   only twist is that the tangent code goes into the *linear* fragment of the
   language.
2. **U, unzipping.** U separates the non-linear (primal) part of each
   expression and function from the linear (tangent) part. It turns `f` into
   `f.nonlin` and `f.lin`. The non-linear values that the linear part needs
   become extra outputs of `f.nonlin` and extra inputs of `f.lin`. This is the
   "tape".
3. **T, transposition.** T runs the linear part backward. Fan-out (`dup`)
   becomes addition, `drop` becomes a zero, and `v * u̇` stays `v * ü`.

So `grad = T ∘ U ∘ J`. Only J knows about derivatives, and only T knows about
running backwards. This is how `jax.grad` is built, and Dex does the same.

The boundary between the transformations is **Linear A**. It is a first-order,
A-normal-form language without recursion or control flow. Every binder,
argument list and result list is split into non-linear and linear parts
`(v ; v̇)`. Its type system is substructural:

- linear variables are used exactly once, with explicit `dup` and `drop`;
- non-linear variables are used at least once (dropping one is explicit,
  because it costs something).

Multiplication `v * v̇` is linear in its right argument only.

**Main results:**

- Well-typed Linear A expressions are algebraically linear in their linear
  inputs (Theorem 4.1).
- J is total, correct and work-preserving.
- U and T are total, correct, preserve work exactly, and preserve code size up
  to a constant factor.

All of this holds under an explicit cost model (Section 4.3). In that model
`drop` costs something and `dup` is free, which is what makes "fan-out
transposes to addition" preserve work.

**Side results:**

- **Checkpointing:** it falls out of unzipping. ULet may recompute a
  non-linear intermediate in the linear part instead of saving it, and T then
  handles that correctly.
- **Partial evaluation:** unzipping is an instance of it, so checkpointing
  ideas may carry over to partial evaluation in general.
- **Fewer rules:** an AD system needs `N + H + L` rules instead of `N + N`,
  with N primitives, of which H are higher-order and L are linear.

**Left open:**

- control flow, which would need sum types and probably dependent types;
- multilinearity;
- size-polymorphic arrays.

## Tapenade, RR-7957 (Hascoët and Pascual, 2012)

The report is the reference description of Tapenade's AD model. Tapenade is
Inria's source-to-source AD tool for Fortran and C.

- **Design choices (Section 3):**
  - source transformation rather than operator overloading;
  - one language-independent internal language, IL, on top of flow graphs
    and nested symbol tables;
  - each array is treated as a single unit by the analyses;
  - derivatives are linked to variables by name (`x` gets `xd` or `xb`);
  - one differentiated procedure per original procedure, valid for every
    calling context;
  - in adjoint mode, *Store-All* with *save-on-kill*: a value is saved just
    before it is overwritten.
- **Model (Section 4):**
  - The tangent code puts each derivative instruction just before its
    original instruction.
  - The adjoint code has two sweeps. The forward sweep is the original code
    plus `PUSH` of overwritten values and of control decisions. The backward
    sweep `POP`s them and runs the derivative instructions in reverse order.
  - Each procedure call becomes `FOO_FWD` / `FOO_BWD`. By default each call is
    checkpointed instead ("joint mode").
  - Refinements: activity analysis, implicit zeros, multi-directional
    tangents, splitting and merging of adjoint instructions,
    diff-liveness, TBR (To-Be-Recorded) analysis, recomputation, and
    checkpointing directives.
- **Specification (Section 5):** the AD-specific analyses are given as
  data-flow equations, on top of pointer analysis and in-out analysis:
  - Diff-dep;
  - Varied and Useful, which together give Active;
  - Dep;
  - Diff-live;
  - TBR;
  - Adj-out.

  The tangent and adjoint transformations are given as Natural Semantics
  inference rules on well-structured code. The authors note that these rules
  are "almost immediate" to implement in a logic language such as Prolog.
- **Cost (Section 2):** tangent code costs about 1–3× the original (`R_t`).
  Adjoint code costs about 5–10× (`R_a`), mostly because of data-flow
  reversal. Section 6 reports measurements on real applications.

## How Tapenade handles linearity

Tapenade never states linearity as an invariant. There is no linear type and
no separate linear program. The linear structure is still there, but it is
implicit, and several analyses rediscover it.

- **Linearity at the instruction level.** For an assignment `I_k`, the report
  writes the local Jacobian `f'_k(X_{k-1})` (Section 4.3, Figure 6). The
  tangent instruction is `Ẋ_k = f'_k × Ẋ_{k-1}` and the adjoint is
  `X̄_{k-1} = X̄_k × f'_k`. The adjoint is produced by transposing that small
  matrix-vector product directly.

  YOLO instead transposes a *program* in which linearity is checked by types.
  In Tapenade the transposition is built into the adjoint rules (Section 5.4),
  which are written separately from the tangent rules (Section 5.3). This is
  the `N + N` situation that YOLO avoids: the adjoint is derived from the
  original code, not from the tangent code.
- **The adjoint expression rule is transposition.** The adjoint of an
  expression pushes a cotangent `VAL` down to the leaves:
  - `EXPR1*EXPR2` sends `EXPR2*VAL` to the left and `EXPR1*VAL` to the right;
  - each active leaf gets `REF̄ := REF̄ + VAL`.

  An overwritten variable gets `REF̄ := 0.0`. These match YOLO's rules: fan-out
  (`dup`) becomes accumulation (`+`), and a killed value becomes a zero.
  Tapenade has no `drop`/`dup`: variables can be read many times and
  overwritten, so accumulation with `+=` and resetting take their place.
- **Activity analysis ≈ the linear/non-linear split.** A variable is *active*
  when it is both *varied* (depends differentiably on an input) and *useful*
  (influences an output differentiably). Only active variables get a
  derivative.

  This decides which values are tangents, the job Linear A does with its
  `;`-split syntax. The difference is that Tapenade finds it by conservative
  data-flow analysis on untyped imperative code ("active" means *possibly*
  active). It also merges calling contexts, so one differentiated procedure
  may receive dummy zero derivatives.
- **Diff-liveness ≈ unzipping's dead code.** Diff-liveness removes primal
  instructions whose results no derivative needs. In YOLO's terms, it keeps
  only the part of `f.nonlin` that `f.lin` actually reads.
- **TBR uses linearity explicitly.** This is the one place where the report
  appeals to linearity directly (Section 4.9). A value has to be saved only if
  it is used in a derivative instruction. If a variable "is always used in
  linear expressions, it never appears in the derivative computations", so it
  is not pushed.

  In Linear A terms, the values to record are the non-linear coefficients.
  Those are the left operands of `v * u̇` (and of primitives' JVPs), which ULet
  passes from `f.nonlin` to `f.lin`. TBR computes, with a flow analysis, the
  set that unzipping produces by construction.
- **The tape vs. residual variables.** Tapenade's forward sweep with
  `PUSH`/`POP` is unzipping with a stack as the store. YOLO keeps the residuals
  as ordinary variables, which makes their lifetimes longer (Section 4.3.2).
- **Checkpointing and recomputation.** Both systems trade storage for
  recomputation:
  - Tapenade stores snapshots `SNP = out(C) ∩ (TBR⁻(C) ∪ Diff-live⁻(C))` and
    recomputes cheap assignments instead of pushing them.
  - In YOLO this is the free choice in ULet between saving a non-linear
    intermediate and recomputing it inside the linear part.
- **What each covers that the other doesn't.**
  - Tapenade handles control flow (it records the branches taken), aliasing,
    pointers, arrays and whole Fortran/C programs. Linear A has none of these.
  - YOLO gives a mechanically checkable linearity invariant and proofs that
    work is preserved. Tapenade only reports measured slowdowns (`R_t`, `R_a`).
- **Multilinearity.** Neither system models it. In Tapenade, the derivative of
  `a*b` with both active is `ad*b + a*bd`, and each term is non-linear times
  linear. That is exactly YOLO's choice to make `*` linear in one argument.

## How to run

You need [Elpi](https://github.com/LPCIC/elpi) 3.8 or later, which added the
`data` / `symb` syntax. Install it with `opam install elpi`.

**Run.** There are two entry points, each with a `make` target:

```
make         # the running example: elpi -test run-example.elpi
make test    # all the tests:       elpi -test run-tests.elpi
```

Set `ELPI=…` to use another `elpi` binary (`make test ELPI=~/bin/elpi`).

The running example is `f(u) = −sin(u)` at `u = 0.5` (Figures 6, 9 and 10).
`make` prints each step in the paper's notation, checks that it type-checks,
and runs it with its cost:
1. `f`, in Linear A;
2. `J(f)`, its forward derivative, as the rules produce it (with nested
   lets);
3. `C(J(f))`, cleaned up in two parts, `flatten` then `copyprop`; the result
   is Figure 6;
4. `U(C(J(f)))`, unzipped into `f.nonlin` and `f.lin` in Linear B;
5. `T(U(C(J(f))))`: `f.lin` transposed into `f.linᵀ`, which computes the
   gradient `f′(0.5) = −cos(0.5)` (reverse mode);
6. `C(T(U(C(J(f)))))`, cleaned up again, on Linear B: `f.linᵀ` is now
   `f.lin` run backwards, as in Figure 10;
7. `L(C(J(f)))`, the erased derivative, which computes `f` and `f′` with no
   linear code.

`make test` runs the tests, in seven groups: Linear A (type checker,
interpreter, printer), J, C, L, Linear B, U and T. Each prints the programs it
builds, then `ok: …` lines, and finally `Success:`. A failing check prints
`FAILED: …` and makes `make test` fail, so it also works in a script or in
CI. The end of the running example looks like this:

```
== T(U(C(J(f)))): f^J.lin transposed (Figure 10); reverse mode ==

def f.nonlin(a : R ;) → (R, ⊗(R, R) ;) =
  ...
def f.linᵀ(a : ⊗(R, R) ; ċ : R) → (; R) =
  let (⊗(e, m) : ⊗(R, R) ;) = a in
  ...
(type-checks in Linear A)
f^J.linᵀ(tape ; 1) = [linA.real -0.877583] cost 2
  (the gradient f'(0.5) = −cos(0.5))
```

**Files and namespaces.** Each file holds one namespace, and queries use
qualified names. Each file accumulates the files it uses:

| File | Namespace | Contents |
|---|---|---|
| [linA.elpi](linA.elpi) | `linA` | Linear A syntax (Figure 3), and its free-variable helpers |
| [linB.elpi](linB.elpi) | `linB` | Linear B syntax (Section 6), and its inclusion in Linear A |
| [forward.elpi](forward.elpi) | `forward` | transformation J, forward derivative (Section 5) |
| [flatten.elpi](flatten.elpi) | `flatten` | cleanup C, part 1: let-floating |
| [copyprop.elpi](copyprop.elpi) | `copyprop` | cleanup C, part 2: copy propagation |
| [erase.elpi](erase.elpi) | `erase` | transformation L, linearity erasure (Section 4.5) |
| [unzip.elpi](unzip.elpi) | `unzip` | transformation U, unzipping, Linear A to Linear B (Section 6) |
| [transpose.elpi](transpose.elpi) | `transpose` | transformation T, transposition of Linear B (Section 7) |
| [lincheckA.elpi](lincheckA.elpi) | `lincheckA` | type checker of Linear A (Figure 4) |
| [evalA.elpi](evalA.elpi) | `evalA` | interpreter of Linear A, with the cost model (Section 4.3) |
| [ppA.elpi](ppA.elpi) | `pp` | printer of Linear A |
| [ppB.elpi](ppB.elpi) | `ppB` | printer of Linear B |
| [examples.elpi](examples.elpi) | `example`, `exampleB` | ready-made Linear A and Linear B terms |
| [test.elpi](test.elpi) | `test` | the checks; `test.all` runs them all |
| [running.elpi](running.elpi) | `running` | the running example; `running.show` prints it |
| [run-example.elpi](run-example.elpi) | — | entry point for `make`: `main` runs `running.show` |
| [run-tests.elpi](run-tests.elpi) | — | entry point for `make test`: `main` runs `test.all`; loads every file |

Linear B is checked, run and printed as Linear A, through the inclusion
`linB.to-linA`; `ppB` prints it directly.

**Run your own query.** Start `elpi run-tests.elpi`, which loads every
file, then type a query at the
`goal>` prompt, ending with a full stop. You can also pipe the query in:

```
echo "example.g G, lincheckA.of-def G Ts LTs Ss LSs, evalA.run-def G [linA.real 2.0] [linA.real 0.5] Vs LVs Cost." \
  | elpi run-tests.elpi
```

That query prints the term `G` and its types (`Ts LTs -> Ss LSs`, here
`[linA.r] [linA.r] -> [linA.r] [linA.r]`), then the results:
`Vs = [linA.real 4.0]`, `LVs = [linA.real 2.0]`, `Cost = 2`. When a query
succeeds, the interactive prompt asks `More? (Y/n)`. Answer `n` to stop.

The predicates you will use most:

| Query | Meaning |
|---|---|
| `lincheckA.of E Ts LTs` | type of an expression (it must be closed) |
| `lincheckA.of-def D Ts LTs Ss LSs` | check a definition: arguments `Ts ; LTs`, results `Ss ; LSs` |
| `lincheckA.of-prog P` | check a whole program |
| `evalA.run-def D Xs LXs Vs LVs Cost` | run one definition on non-linear inputs `Xs` and linear inputs `LXs` |
| `evalA.run P Xs LXs Vs LVs Cost` | run a program; its last definition is the entry point |
| `ppA.prog->string P S` | print a Linear A program in the paper's notation |
| `ppA.def->string D S` | print one Linear A definition (named `f`) |
| `ppA.expr->string E S` | print a closed Linear A expression |
| `ppB.prog->string B S` | print a Linear B program (`f.nonlin`, `f.lin`, …) |
| `ppB.tprog->string B S` | the same, for a transposed program (`f.linᵀ`) |
| `forward.dj-def D D'` / `forward.dj-prog P P'` | J of a non-linear definition / program |
| `forward.dj E E'` | J on an expression; each free `x` needs a clause `forward.dot x ẋ T` in the context |
| `flatten.flatten-prog P P'` / `copyprop.copyprop-prog P P'` | the two parts of C, on a Linear A program |
| `flatten.flatten-progB B B'` / `copyprop.copyprop-progB B B'` | the two parts of C, on a Linear B program |
| `erase.lerase-prog P P'` | L of a program (adds the helper functions) |
| `linB.to-linA B A` | a Linear B program as the Linear A program it is |
| `unzip.unzip-prog P B` | U of a Linear A program into a Linear B one |
| `unzip.unzip-def D N L Ss T` | U of one definition: `f.nonlin` `N`, `f.lin` `L`, result types `Ss`, tape type `T` |
| `transpose.transpose-prog B B'` | T of a Linear B program: each `f.lin` becomes `f.linᵀ` |
| `transpose.transpose E Dd Cs E'` | T on a linear expression: `Dd` its linear inputs (ordered, typed), `Cs` cotangents of its results |

To see an example in the paper's notation:

```
echo 'example.dsin D, ppA.def->string D S, print S.' | elpi run-tests.elpi
```

prints

```
def f(a : R ; ċ : R) → (R ; R) =
  let (e : R ;) = sin(a) in
  let (m : R ;) = cos(a) in
  let (; ṅ : R) = m * ċ in
  (e ; ṅ)
```

To differentiate a definition, print it, and run it at `x = 3` in direction
`ẋ = 1`:

```
echo 'example.sq D, forward.dj-def D DJ, ppA.def->string DJ S, print S, evalA.run-def DJ [linA.real 3.0] [linA.real 1.0] Vs LVs C.' \
  | elpi run-tests.elpi
```

To get `F(x)` and `F'(x)` with no linear code, as in Section 5, erase the
derivative and pass the direction as an ordinary argument:

```
echo 'example.k-prog P, forward.dj-prog P PJ, erase.lerase-prog PJ PE, ppA.prog->string PE S, print S, evalA.run PE [linA.real 3.0, linA.real 1.0] [] Vs _ C.' \
  | elpi run-tests.elpi
```

To print reverse mode, `C(T(U(C(J(P)))))`, for the program `k`:

```
echo 'example.k-prog P, forward.dj-prog P PJ, flatten.flatten-prog PJ P1, copyprop.copyprop-prog P1 PC, unzip.unzip-prog PC PU, transpose.transpose-prog PU PT, flatten.flatten-progB PT PT1, copyprop.copyprop-progB PT1 PTC, ppB.tprog->string PTC S, print S.' \
  | elpi run-tests.elpi
```

Values are written `linA.real 3.0` and `linA.vtup [linA.real 1.0, linA.real 2.0]`.
Types are `linA.r` and `linA.tup [linA.r, linA.r]`. Ready-made terms to try:

- the running example: `example.neg-sin`, `f(u) = −sin(u)`;
- for the type checker and interpreter: `example.g`, `example.dsin`,
  `example.swap`, `example.drop-dup`, `example.g-prog`, `example.lin-tup`,
  and the ill-typed `example.bad-twice` and `example.bad-unused`;
- non-linear inputs for `forward.dj`: `example.sin1`, `example.sq`,
  `example.h`, `example.m`, and the program `example.k-prog`;
- Linear B: `exampleB.dsin-nonlin`, `exampleB.dsin-lin`, and the program
  `exampleB.dsin-unzipped`.

**Reading the papers.** Both PDFs are in the repository root. The Tapenade
report's PDF has no usable text layer: `pdftotext` returns garbled characters,
so read it in a PDF viewer.

## The Elpi code

Done so far. Each part is in its own file and namespace (see the table
above). Each namespace starts with a `shorten linA.{…}` so its code can say
`let` instead of `linA.let`. The two exceptions are `evalA` and `test`, which
don't shorten `sin`, `cos` and `exp`, so that `is` can still use the builtin
functions; they write `linA.sin` and so on.

- **Syntax (Figure 3), namespace `linA`, in HOAS, using Elpi's `data` /
  `symb` / `func` declarations.**
  - Non-linear variables, linear variables and function names are separate
    kinds (`nl`, `lin`, `fn`). Each binder is an Elpi λ that carries the
    variable's type.
  - A binder spine is non-linear binders followed by linear ones:
    `nbind A` ends in `lbind A`, which ends in the body. `let` and `def` take an
    `nbind`. The non-linear unpacking let takes an `nbind` whose linear part
    must be empty. The linear unpacking let takes an `lbind`.
  - A program is `def D (f\ P)`. A function name is in scope only after its
    definition, so recursion cannot be written.
  - Not encoded: the environments Σ and Δ and the evaluation contexts E, which
    only the transformations in Sections 5–7 use.
  - Primitives: `sin`, `cos`, `exp`, `neg` (−), `plus`, `mul`. Figure 3
    lists sin, cos, exp, + and "…"; `mul` was added for the paper's example
    `g`, and `neg` for its running example `f(u) = −sin(u)`.
- **Type checker (Figure 4), namespace `lincheckA`.** `of E Ts LTs` stands for `Γ ; Γ̇ ⊢ e : (τ ; τ̇)`,
  with one clause per rule.
  - Each bound variable carries a fresh "used" mark, which is set when the
    variable occurs.
  - A linear variable can only be used while its mark is unset (exactly once,
    `⊎`). A non-linear variable can be marked any number of times (at least
    once, `∪`).
  - When a binder goes out of scope, its mark must be set.
  - `of-def` checks a definition and `of-prog` checks a whole program.
- **Interpreter with the Section 4.3 cost model, namespace `evalA`.**
  `eval E Vs LVs Cost` is big-step evaluation. Variables are bound through
  hypothetical `nlval` / `linval` clauses, and function names through
  `fndef`.
  - Costs: non-linear primitives cost 1. Linear `+` and `*` cost 1 per scalar
    in the result. `drop` costs its subexpression plus 1 per real scalar it
    discards. `let` and calls add up their parts. Everything else is free.
  - `run P Xs LXs Vs LVs Cost` runs a program. The paper does not say which
    function is a program's entry point; here it is the **last definition**.
    `run-def` runs a single definition.
- **Pretty printer, in the paper's notation, namespace `ppA`.** `prog->string`, `def->string`
  and `expr->string` turn the HOAS terms back into the paper's syntax:
  `let (a : R ; ċ : R) = e in`, `⊗(…)` for tuples, `(v ; v̇)` for returns,
  `def f(…) → (… ; …) =` for definitions, and `0̇_R`, `dup(ȧ)`, `drop(e)`.
  - Variable names come from their binding depth, using the 16 letters
    `a c e m n o p q r s u v w x y z`; deeper binders reuse them with a
    numeric suffix (`a1, c1, …`). Linear variables get a dot (`ȧ, ċ, ė`)
    from a combining character.
    Functions are named `f, g, h, f1, …` in program order.
  - Because names come from depth, a name can be reused once the earlier
    binder's scope has ended. The bound expression of a `let` is numbered
    past the `let`'s own binders, so its names never repeat the ones on the
    left of `=`.
  - Literals print with a trailing digit (`2.0`, not OCaml's `2.`).
  - A `let` whose bound expression takes several lines is printed as
    `let (…) =`, then the expression indented, then `in`.
  - The paper's arity indices (`e^{m,n}`, `f^{o,p}_{m,n}`) are not printed.
- **Forward derivative J (Section 5, Figure 7), namespace `forward`: `dj`.**
  `dj E E'` is `J(Σ | e) ⇝ e'`. `dj-def` implements JDef: it turns
  `def f(v:τ ;) → (σ ;)` into `def f^J(v:τ ; v̇:τ) → (σ ; σ)`. `dj-prog`
  differentiates every definition and renames each call `f(…)` to `f^J(…)`.
  - **Σ lives in the context.** It is one hypothetical clause `dot x ẋ T`
    per primal variable in scope, mapping it to its tangent and type.
    Extending Σ with `x ↦ ẋ` is loading `dot x ẋ T =!=> …`. `=!=>` works like
    `==>` (it puts the new clause in front of the older ones), and it also
    adds a cut at the end of the clause, so `dot` stops at the first match.
    A newer clause for the same `x` therefore shadows the older one, the way
    the extended Σ does in the rules. The file never uses `=>`: every
    hypothetical clause is loaded with `==>`, or with `=!=>` when it must
    shadow.
  - **Introducing dup:** a primal variable needed more than once gets its
    tangent duplicated with `dup`, and each copy is loaded as a newer `dot`
    clause for the part of the code that uses it. Two helpers do this:
    - `dup-each` handles JLet. A variable `u` free in both the bound
      expression and the body gets one `ẇ, ż = dup(u̇)`, with `ẇ` and `ż`
      fresh. `e1` runs under `u ↦ ẇ` and `e2` under `u ↦ ż`, as in Figure 7.
    - `dup-each` also handles JUnpackB, a tuple used again after it is
      unpacked: the unpack takes the first copy and the body the second.
    - `with-tans` handles repeated arguments in JRet, JApp, JTup and
      JPrimMul, for example `x * x`. A variable used `n` times gets a chain
      of `n − 1` dups.

    The paper describes these variants but leaves them out of Figure 7.
    Variables used on only one side keep their outer tangent, so Σ₁ and Σ₂
    of JLet never have to be built explicitly.
  - **Primitive rules:** JPrimSin and JPrimMul are as in Figure 7. The paper
    gives no rules for the other primitives, so these are derived:
    - `cos`: `(cos v ; (-1.0 * sin v) * v̇)`;
    - `exp`: `(exp v ; exp v * v̇)`;
    - `+`: `(v1 + v2 ; v̇1 + v̇2)`;
    - `−`: `(−v ; -1.0 * v̇)`. Figure 6 writes `−v̇`, but Linear A has no
      linear negation, so it is a scaling, which costs the same.
  - **A-normal form:** compound tangents such as `v1 * v̇2 + v2 * v̇1` are
    let-bound.
  - **Literals (JLit):** `l` becomes `(l ; 0̇_R)`.
  - **Input check:** `dj` has no rules for linear syntax, so it fails on
    input that is not purely non-linear.
- **Cleanup C, namespaces `flatten` and `copyprop`.** The paper writes J's
  results flat (Figure 6), but its rules nest them: JLet makes `J(e1)` the
  bound expression of a `let`, and `J(e1)` is usually a block of lets
  itself. Unzipping then turns each block's result into a trivial rebinding
  such as `let (m ;) = (c ;) in`. T nests lets too. C removes both, in two
  parts, and runs right after J and again right after T:
  - `flatten` floats lets out: `let x̄ = (let ȳ = a in b) in c` becomes
    `let ȳ = a in let x̄ = b in c`, and likewise when the bound expression is
    an unpacking let. In HOAS the variables `ȳ` cannot occur in `c`, so
    nothing needs renaming.
  - `copyprop` propagates copies: `let (x̄ ; ẏ) = (v̄ ; ẇ) in c` becomes `c`
    with `x̄ := v̄` and `ẏ := ẇ`, and likewise for a single variable. In HOAS
    the substitution is applying the binders to `v̄` and `ẇ`.
  - `let` and returns cost nothing, so C changes neither results nor cost.
    `C(J(f))` for the running example is exactly Figure 6 (with `−v̇`
    written as `-1.0 * v̇`).
  - C works on both languages: `flatten-prog` / `copyprop-prog` on Linear
    A, and `flatten-progB` / `copyprop-progB` on Linear B. On Linear B, a
    linear `let` can float out any binding form, and a non-linear `let`
    inside a linear expression floats out non-linear ones (they become
    `nlet` and `nunpack`). After T, C makes `f.linᵀ` flat: for the running
    example it is `f.lin` run backwards, as in Figure 10.
- **Linearity erasure L (Section 4.5), namespace `erase`: `lerase`.** The paper gives L only in
  prose. `lerase-prog` follows it:
  - **Helper functions:** for every type at which linear `+`, `*` or `0̇` is
    used, a fresh non-linear function is defined before the program. Tuples
    are handled by unpacking, applying the function for each component type,
    and repacking, so components get their own functions too.
  - **Calls:** linear `+`, `*` and `0̇` become calls to those functions.
  - **dup:** a `let` that binds the results of `dup(ẋ)` is removed, and both
    results are renamed to `x`. A `dup` anywhere else becomes `(x, x ;)`.
  - **Semicolons:** they move to the right. Linear binders, arguments, results
    and return types come after the non-linear ones, and the linear unpacking
    let becomes a non-linear one.
  - **Single definitions:** `lerase-prog` works on whole programs, because it
    adds definitions. Wrap a single definition as `def D x\ pnil`.
- **Linear B (Section 6, Definition 6.1), namespace `linB`.** Linear B is
  the sublanguage of Linear A in which (1) every function and expression
  returns only non-linear or only linear results, and (2) an expression with
  non-linear results reads no linear variable, not even to drop it. Both
  restrictions are in the syntax:
  - Expressions come in two kinds. `nexpr` has non-linear results: `ret`,
    `let`, `unpack`, `app`, `drop`, and `leaf`. `lexpr` has linear results:
    `lret`, `llet`, `lunpack`, `lapp`, `ldrop`, `lleaf`, plus `nlet` and
    `nunpack`, which bind non-linear intermediates inside a linear
    expression (as `f.lin` does in Figure 9).
  - An `nexpr` has no constructor that mentions a linear variable, so
    restriction (2) holds by construction.
  - Binder spines are all non-linear (`nabs`) or all linear (`labs`).
    Functions are non-linear (`nfn`, defined by `ndef`, `(v ;) → (σ ;)`) or
    linear (`lfn`, defined by `ldef`, `(v ; v̇) → (; σ̇)`).
  - Types and variables are those of Linear A. Leaves (variables, literals,
    tuples, primitives, and their linear counterparts) are Linear A leaves
    wrapped by `leaf` or `lleaf`, as in the paper's ULeaf and ULinLeaf.
  - `to-linA` is the inclusion of Linear B in Linear A, and it rejects a
    leaf of the wrong kind. Through it, Linear B programs are type-checked,
    run and printed with the Linear A tools.
- **Unzipping U (Section 6, Figure 8), namespace `unzip`.** `unzip-prog`
  turns a Linear A program into a Linear B one. Each definition `f` becomes
  `f.nonlin`, which computes the non-linear results and returns the tape,
  followed by `f.lin`, which reads the tape and computes the linear results.
  - **Representation:** `U(e) ⇝ E; e′; ė′` is a context `E`, a chain of
    non-linear `clet`/`cunpack` binders in HOAS, whose `hole` holds `e′` and
    `ė′`. `plug` replaces the hole, and composing contexts (as in ULet) is
    plugging one into the other.
  - **Rules:** one clause per rule of Figure 8: URet, ULet, UUnpack,
    ULinUnpack, UApp, ULeaf, ULinLeaf, UDrop, and UDef in `unzip-def`.
  - **Moving binders:** ULet and ULinUnpack bind `v̇` around `ė′` only, and
    UDef binds the tape variables inside `f.lin`'s linear parameters. In the
    input those variables are bound further out. A HOAS binder can't be
    moved by unification, so these rules copy `ė′` with a renaming. The
    renaming is `cp-nl x y` / `cp-lin ẋ ẏ`, loaded with `=!=>`, and every
    other variable is copied unchanged.
  - **The tape:** the variables bound by `f`'s non-linear parameters or by
    `E` that `ė′` reads, in binding order. `f.nonlin` returns them as one
    tuple, `f.lin` unpacks them from its first argument, and a call
    `f(v ; v̇)` becomes `f.nonlin(v ;)` and then `f.lin(tape ; v̇)` (UApp).
  - **Two simplifications:** ULet produces `let (;) = (;) in E` when the
    `let` binds no non-linear variable and its expression has no non-linear
    part. It produces `let (;) = (;) in ė` in the symmetric linear case. Both
    are left out. They are identities that cost nothing.
- **Transposition T (Section 7, Figure 11), namespace `transpose`.**
  `transpose-prog` keeps the non-linear functions and replaces each linear
  function `f.lin` by its transpose `f.linᵀ`, which reads cotangents of
  `f.lin`'s results and returns cotangents of its linear parameters. Calls
  to `f.lin` become calls to `f.linᵀ`. Reverse mode is `C ∘ T ∘ U ∘ C ∘ J`: run
  `f.nonlin(x ;)` for the results and the tape, then `f.linᵀ(tape ; ÿ)`.
  - `transpose E Dd Cs E'` is `T(Δ̇, Δ̈ | ė) ⇝ ë`. `Dd` lists the linear
    variables `ė` consumes, with types, in the order in which `ë` returns
    their cotangents. `Cs` lists one cotangent variable per result of `ė`.
  - `ë` never mentions `ė`'s linear variables, which only fix positions, so
    T needs no renaming.
  - **Rules:** one clause per rule of Figure 11: TRet, TLetLin, TLetNonLin,
    TApp, TLinVar, TLinZero, TLinTup, TLinAdd, TLinMul, TDup, TDropLin and
    TDef. TNonLin needs no clause, because non-linear expressions only occur
    under `nlet`, which TLetNonLin keeps as they are.
  - **Rules the paper omits:** TUnpack and TLinUnpack, which it says are
    like the `let` rules, are derived that way. A non-linear unpack is kept.
    A linear `let (; ⊗ẇ) = u̇ in ė` transposes `ė`, then packs the
    cotangents of `ẇ` into a tuple, the cotangent of `u̇`.
  - **Shuffles:** where a rule's cotangents come out in another order than
    the requested one (as in TLetLin or TApp), they are let-bound and
    returned in order. The paper leaves these shuffles implicit; they cost
    nothing.
- **Printer of Linear B, namespace `ppB`.** It prints Linear B directly in
  the paper's notation, using the helpers of `ppA`. A non-linear function
  followed by a linear one prints as `f.nonlin` and `f.lin`, as in
  Figure 9, and `tprog->string` prints transposed ones as `f.linᵀ`.
- **The running example, in `running.elpi`.** `f(u) = −sin(u)` goes through J,
  U, T and L, printed and run at `u = 0.5` (see "How to run").
- **Tests (`make test`, namespace `test`):**
  - The paper's `g(x ; ẏ) = (x² ; x²ẏ)` type-checks. Run as `g(3 ; 1)` it
    returns `(9 ; 9)` with cost 2.
  - A `drop`/`dup` example type-checks.
  - A linear variable used twice is rejected.
  - An unused non-linear variable is rejected.
  - A swap example with both kinds of unpacking type-checks and runs at
    cost 0.
  - The forward derivative of `sin` runs as `dsin(0 ; 2) = (0 ; 2)` with
    cost 3.
  - The examples are printed in the paper's notation.
  - J (`forward-tests`):
    - `sq(x) = x*x` gives `(9 ; 6)` at `(3 ; 1)`.
    - `h(x) = exp(x sin x)` matches the closed-form derivative at `0.5`.
    - A tuple example `m` covers JUnpackB and JLit.
    - A two-function program gives `(12 ; 7)`.
    - Every output type-checks in Linear A.
    - J rejects linear input.
    - J's cost is a constant factor over the primal (for example 1 → 4 for
      `sq`).
  - L (`erase-tests`):
    - `L(C(J(k)))(3, 1 ;)` gives `(12, 7 ;)`.
    - Linear `+`, `*` and `0̇` on tuples go through the generated helpers.
    - `dup` disappears and `lunpack` becomes `unpack`.
    - Every output type-checks.
    - Results and cost are the same as before erasure (Lemma 4.4).
  - Linear B (`linB-tests`):
    - The unzipped `dsin` of Figure 9 (`exampleB.dsin-unzipped`) type-checks
      in Linear A.
    - Running `f.nonlin(0.5 ;)` and then `f.lin(tape ; 2)` gives the same
      result as `dsin(0.5 ; 2)`.
    - A linear leaf in a non-linear expression is rejected, and so is a
      non-linear leaf in a linear one.
  - U (`unzip-tests`):
    - `U(C(J(sin)))` is exactly the hand-written unzipped `dsin` of Figure 9.
    - For `sq`, `h`, `m` and the two-function program `k`, `U(C(J(P)))`
      type-checks.
    - For each of them, running `f.nonlin(x ;)` and then
      `f.lin(tape ; ẋ)` gives the results of `J(P)(x ; ẋ)`, and the two
      costs add up exactly to J's cost (Theorem 6.2: unzipping preserves
      work).
  - C (`normalize-tests`): for the running example, `sq`, `h`, `m` and `k`:
    - `C(J(P))` type-checks, and has no nested let and no copy left;
    - it computes the same results as `J(P)`, at the same cost;
    - for the running example, it is exactly Figure 6 (`example.neg-sin-J`).
  - T (`transpose-tests`): for the running example, `sq`, `h`, `m` and
    `k`, with `ė = f.lin` and `ë = f.linᵀ` at the tape of `f.nonlin(x ;)`:
    - `T(U(C(J(P))))` type-checks;
    - `C` after T type-checks, leaves no nested let or copy, and gives the
      same results at the same cost;
    - `⟨ẋ, ë[x; ÿ]⟩ = ⟨ė[x; ẋ], ÿ⟩`: `ë` is the transpose of `ė`
      (Theorem 7.1, claim 3);
    - `W[ë] + L_in ≤ W[ė] + L_out`, where `L` counts the scalars in `ẋ` and
      `ÿ` (Theorem 7.1, claim 2).

Not done: checkpointing (the free choice in ULet to recompute a non-linear
value in the linear part instead of putting it on the tape).
