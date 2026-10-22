# L1

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
- [linear_a.elpi](linear_a.elpi): Linear A syntax in HOAS, its type checker,
  and an interpreter with the paper's cost model.

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

**Run the tests.** `main` type-checks and runs the examples:

```
elpi -test linear_a.elpi
```

Each test prints one line, then the examples are printed in the paper's
notation. Then come the tests of `dj` and `lerase`, which print each
transformed definition followed by `ok: …` lines, and finally `Success:`. A
failing check prints `FAILED: …`. The exit code is 0 when all tests pass and 1
when one fails, so the command also works in a script or in CI. Expected
output (shortened here):

```
g : [r] [r] -> [r] [r]
drop/dup example: ok
linear var used twice: rejected
unused non-linear var: rejected
unpack example: ok
program: ok
run g(3 ; 1) = [real 9.000000] [real 9.000000] cost 2
run dsin(0 ; 2) = [real 0.000000] [real 2.000000] cost 3
run drop/dup(5 ; 7) = [] [real 7.000000, real 7.000000] cost 1
run swap((1,2) ; (3,4)) = [real 2.000000, real 1.000000]
 [real 4.000000, real 3.000000] cost 0

def f(a : R ; ċ : R) → (R ; R) =
  let (e : R ;) = a * a in
  let (; ṁ : R) = e * ċ in
  (e ; ṁ)

def g(a : R ; ċ : R) → (R ; R) =
  f(a ; ċ)
...
--- forward derivative J (dj) ---
def f(a : R ; ċ : R) → (R ; R) =
  let (; ė : R, ṁ : R) = dup(ċ) in
  let (n : R ;) = a * a in
  let (; ȯ : R) = a * ṁ in
  let (; ṗ : R) = a * ė in
  let (; q̇ : R) = ȯ + ṗ in
  (n ; q̇)
ok: sq^J type-checks
sq^J(3 ; 1) = [real 9.000000] [real 6.000000] cost 4 (primal cost 1 )
ok: sq^J(3 ; 1) = (9 ; 6)
...
--- linearity erasure L (lerase) ---
...
ok: L(lt) computes the same result with the same cost
```

**Run your own query.** Start `elpi linear_a.elpi`, then type a query at the
`goal>` prompt, ending with a full stop. You can also pipe the query in:

```
echo "example-g G, of-def G Ts LTs Ss LSs, run-def G [real 2.0] [real 0.5] Vs LVs Cost." \
  | elpi linear_a.elpi
```

That query prints the term `G` and its types (`Ts LTs -> Ss LSs`, here
`[r] [r] -> [r] [r]`), then the results: `Vs = [real 4.0]`,
`LVs = [real 2.0]`, `Cost = 2`. When a query succeeds, the interactive prompt
asks `More? (Y/n)`. Answer `n` to stop.

The predicates you will use most:

| Query | Meaning |
|---|---|
| `of E Ts LTs` | type of an expression (it must be closed) |
| `of-def D Ts LTs Ss LSs` | check a definition: arguments `Ts ; LTs`, results `Ss ; LSs` |
| `of-prog P` | check a whole program |
| `run-def D Xs LXs Vs LVs Cost` | run one definition on non-linear inputs `Xs` and linear inputs `LXs` |
| `run P Xs LXs Vs LVs Cost` | run a program; its last definition is the entry point |
| `prog->string P S` | print a program in the paper's notation |
| `def->string D S` | print one definition (named `f`) |
| `expr->string E S` | print a closed expression |
| `dj-def D D'` / `dj-prog P P'` | forward derivative J of a non-linear definition / program |
| `dj Sigma E E'` | J on an expression; `Sigma` lists `sg x ẋ T` for each free variable |
| `lerase-prog P P'` | linearity erasure L of a program (adds the helper functions) |

To see an example in the paper's notation:

```
echo 'example-dsin D, def->string D S, print S.' | elpi linear_a.elpi
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
echo 'primal-sq D, dj-def D DJ, def->string DJ S, print S, run-def DJ [real 3.0] [real 1.0] Vs LVs C.' \
  | elpi linear_a.elpi
```

To get `F(x)` and `F'(x)` with no linear code, as in Section 5, apply `lerase`
to the derivative and pass the direction as an ordinary argument:

```
echo 'primal-prog P, dj-prog P PJ, lerase-prog PJ PE, prog->string PE S, print S, run PE [real 3.0, real 1.0] [] Vs _ C.' \
  | elpi linear_a.elpi
```

Values are written `real 3.0` and `vtup [real 1.0, real 2.0]`. Types are `r`
and `tup [r, r]`. Ready-made definitions to try:

- for the type checker and interpreter: `example-g`, `example-dsin`,
  `example-unpack`, `good-drop-dup`, `example-prog`, `example-lin-tup`, and
  the ill-typed `bad-twice` and `bad-unused`;
- non-linear inputs for `dj`: `primal-sq`, `primal-h`, `primal-m`, and the
  program `primal-prog`.

**Reading the papers.** Both PDFs are in the repository root. The Tapenade
report's PDF has no usable text layer: `pdftotext` returns garbled characters,
so read it in a PDF viewer.

## linear_a.elpi

Done so far:

- **Syntax (Figure 3), in HOAS, using Elpi's `data` / `symb` / `func`
  declarations.**
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
  - Primitives: `sin`, `cos`, `exp`, `plus`, `mul`. Figure 3 lists sin, cos,
    exp, + and "…"; `mul` was added for the paper's example.
- **Type checker (Figure 4).** `of E Ts LTs` stands for `Γ ; Γ̇ ⊢ e : (τ ; τ̇)`,
  with one clause per rule.
  - Each bound variable carries a fresh "used" mark, which is set when the
    variable occurs.
  - A linear variable can only be used while its mark is unset (exactly once,
    `⊎`). A non-linear variable can be marked any number of times (at least
    once, `∪`).
  - When a binder goes out of scope, its mark must be set.
  - `of-def` checks a definition and `of-prog` checks a whole program.
- **Interpreter with the Section 4.3 cost model.**
  `eval E Vs LVs Cost` is big-step evaluation. Variables are bound through
  hypothetical `nlval` / `linval` clauses, and function names through
  `fndef`.
  - Costs: non-linear primitives cost 1. Linear `+` and `*` cost 1 per scalar
    in the result. `drop` costs its subexpression plus 1 per real scalar it
    discards. `let` and calls add up their parts. Everything else is free.
  - `run P Xs LXs Vs LVs Cost` runs a program. The paper does not say which
    function is a program's entry point; here it is the **last definition**.
- **Pretty printer, in the paper's notation.** `prog->string`, `def->string`
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
- **Forward derivative J (Section 5, Figure 7): `dj`.**
  `dj Sigma E E'` is `J(Σ | e) ⇝ e'`. `dj-def` implements JDef: it turns
  `def f(v:τ ;) → (σ ;)` into `def f^J(v:τ ; v̇:τ) → (σ ; σ)`. `dj-prog`
  differentiates every definition and renames each call `f(…)` to `f^J(…)`.
  - Σ is an explicit list `sg x ẋ T` that maps each free variable to its
    tangent and type. It holds exactly the free variables of the expression,
    so each tangent is used exactly once.
  - **Introducing dup:** a primal variable needed more than once gets its
    tangent duplicated with `dup`. One helper, `with-tans`, does this for
    JLet (a variable free in both the bound expression and the body), for
    JUnpackB (a tuple used again after it is unpacked), and for repeated
    arguments in JRet, JApp, JTup and JPrimMul (for example `x * x`). The
    paper describes these variants but leaves them out of Figure 7.
  - **Primitive rules:** JPrimSin and JPrimMul are as in Figure 7. The paper
    gives no rules for the other primitives, so these are derived:
    - `cos`: `(cos v ; (-1.0 * sin v) * v̇)`;
    - `exp`: `(exp v ; exp v * v̇)`;
    - `+`: `(v1 + v2 ; v̇1 + v̇2)`.
  - **A-normal form:** compound tangents such as `v1 * v̇2 + v2 * v̇1` are
    let-bound.
  - **Literals (JLit):** `l` becomes `(l ; 0̇_R)`.
  - **Input check:** `dj` has no rules for linear syntax, so it fails on
    input that is not purely non-linear.
- **Linearity erasure L (Section 4.5): `lerase`.** The paper gives L only in
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
- **Tests (`main`):**
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
  - J (`test-dj`):
    - `sq(x) = x*x` gives `(9 ; 6)` at `(3 ; 1)`.
    - `h(x) = exp(x sin x)` matches the closed-form derivative at `0.5`.
    - A tuple example `m` covers JUnpackB and JLit.
    - A two-function program gives `(12 ; 7)`.
    - Every output type-checks in Linear A.
    - J rejects linear input.
    - J's cost is a constant factor over the primal (for example 1 → 4 for
      `sq`).
  - L (`test-lerase`):
    - `L(J(k))(3, 1 ;)` gives `(12, 7 ;)`.
    - Linear `+`, `*` and `0̇` on tuples go through the generated helpers.
    - `dup` disappears and `lunpack` becomes `unpack`.
    - Every output type-checks.
    - Results and cost are the same as before erasure (Lemma 4.4).

Next steps would be unzipping U (Section 6) and transposition T (Section 7).
Reverse mode would then be `T ∘ U ∘ J`, and the gradient could be checked
against `dj` on the examples.
