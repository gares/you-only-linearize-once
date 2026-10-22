# L1

Elpi formalization of Linear A, from *You Only Linearize Once*. See
README.md for the papers, the design, and the full list of predicates.

## Checking changes

```
make test
```

This must succeed and print no `FAILED:` line. It needs Elpi 3.8 or later.
`make` (the running example) must also still succeed.

## Layout

One file per language (`linA.elpi`, `linB.elpi`), one per transformation
(`forward.elpi` J, `flatten.elpi` and `copyprop.elpi` the cleanup C,
`erase.elpi` L, `unzip.elpi` U, `transpose.elpi` T), one
per utility (`lincheckA.elpi`, `evalA.elpi`, `ppA.elpi`, `ppB.elpi`), plus
`examples.elpi`, `test.elpi`, `running.elpi` (the running example), and
two entry points, `run-example.elpi` and `run-tests.elpi`, run by the
Makefile. Each file accumulates the files it uses, right after its header
comment; Elpi loads a file only once, however many files accumulate it. A
new file's tests go in `test.elpi` and are called from `test.all`.

Each file holds one namespace, named after the file (`examples.elpi` holds
`example` and `exampleB`; the two entry points hold only `main`). A namespace starts
with `shorten` for the names it uses. Don't shorten `sin`, `cos` or `exp` in
a namespace that uses them with `is`, because the shortened name hides the
builtin; write `linA.sin` there instead. `linA` and `linB` share names such
as `ret` and `let`, so a namespace that uses both shortens neither and
writes `linA.ret` / `linB.ret`.

## Elpi style

- **Declarations:** use the Elpi 3.8 syntax: `data t A.` (not `kind`),
  `symb c T.` (not `type`), and `func p In1, In2 -> Out1, Out2.` (not
  `pred p i:…, o:…`). Type higher-order arguments as `(func A -> B)`.
  `func` predicates are checked for determinacy, so overlapping clauses need
  cuts.
- **Hypothetical clauses:** never write `=>`. Load clauses with `==>`, or
  with `=!=>` when the clause needs a cut at the end of its body, for
  example to shadow an older clause for the same input. Both bind looser
  than `,`, so `pi x\ p x ==> g1, g2` scopes over both goals. After other
  goals in a conjunction, add parentheses: `a, (c =!=> g)`, because
  `a, c =!=> g` reads as `(a, c) =!=> g`.
- **Continuations:** a recursive predicate passes its continuation `K`
  unchanged to every recursive call. Never rebuild it as
  `(Ls\ R\ K [X|Ls] R)` at each step. Collect results in accumulators that
  grow at each step, then call `K` once at the end, reversing the
  accumulators if order matters. Keep a clean entry point that calls the
  recursive version: `p Xs K E :- p-acc Xs [] K E.`
- **No eta-expansions:** write a partial application, not a lambda that only
  forwards its arguments. Write `(dj-let Us E1 B)`, not
  `(Ws\ Zs\ R\ dj-let Us E1 B Ws Zs R)`. If the arguments come in the wrong
  order for that, reorder the predicate's parameters so the ones the
  continuation supplies come last.
